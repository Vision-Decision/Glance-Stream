"""Rank-16 routed adapters with fixed split-K arithmetic and measured readout tiling."""
import torch
import triton
import triton.language as tl

from glance_stream.engine.state import DEFAULT_LORA_TILE


def pack_a(a):
    """Split immutable FP32 bank weights once, at registration."""
    hi = a.bfloat16()
    rem = a - hi.float()
    lo = rem.bfloat16()
    return torch.stack((hi, lo, (rem - lo.float()).bfloat16()))


@triton.jit
def _rank_partial(X, A, row, split, END: tl.constexpr, K: tl.constexpr, XROW: tl.constexpr,
                  BK: tl.constexpr):
    rank = tl.arange(0, 16)
    k = split * BK + tl.arange(0, BK)
    x = tl.load(X + row[:, None] * XROW + k[None, :],
                (row[:, None] < END) & (k[None, :] < K), 0)
    ptr = A + rank[:, None] * K + k[None, :]
    a = tl.load(ptr, k[None, :] < K, 0)
    if A.dtype.element_ty == tl.bfloat16:
        hi = a
        lo = tl.load(ptr + 16*K, k[None, :] < K, 0)
        tail = tl.load(ptr + 32*K, k[None, :] < K, 0)
    else:
        # Active wrappers can be mutated by their owner between replays.
        hi = a.to(tl.bfloat16)
        rem = a - hi.to(tl.float32)
        lo = rem.to(tl.bfloat16)
        tail = (rem - lo.to(tl.float32)).to(tl.bfloat16)
    small = tl.dot(x, tl.trans(tail)) + tl.dot(x, tl.trans(lo))
    return small + tl.dot(x, tl.trans(hi))


@triton.jit
def _grouped_lora_rank(X, As, PART, GROUPS: tl.constexpr, PRODUCERS: tl.constexpr,
                       WIDE_GROUPS: tl.constexpr, RANK_ROWS: tl.constexpr, K: tl.constexpr,
                       XROW: tl.constexpr, SPLITS: tl.constexpr, BK: tl.constexpr):
    pid, split = tl.program_id(0), tl.program_id(1)
    for i in tl.static_range(len(GROUPS)):
        begin, stop, first, end, offset, n, scale = GROUPS[i]
        lo, hi = PRODUCERS[i]
        if (pid >= lo) & (pid < hi):
            # The host sets bit i for a tuned tile. Both extents are compile-time constants.
            if (WIDE_GROUPS >> i) & 1:
                r = (pid - lo) * RANK_ROWS + tl.arange(0, RANK_ROWS)
            else:
                r = (pid - lo) * 16 + tl.arange(0, 16)
            part = _rank_partial(X, As[i], first + r, split, end, K, XROW, BK)
            # Both producer shapes write the original 16-row split-K partial layout.
            dst = ((begin + r // 16) * SPLITS + split) * 256 + r % 16 * 16
            tl.store(PART + dst[:, None] + tl.arange(0, 16)[None, :], part,
                     r[:, None] < tl.cdiv(end - first, 16) * 16)


@triton.jit
def _tf32_rn(x):
    bits = tl.inline_asm_elementwise("cvt.rna.tf32.f32 $0, $1;", constraints="=r,f",
                                     args=[x], dtype=tl.int32, is_pure=True, pack=1)
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _pack_b(B, PACK, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    b = tl.load(B + i, i < SIZE, 0)
    hi = _tf32_rn(b)
    tl.store(PACK + i, hi, i < SIZE)
    tl.store(PACK + SIZE + i, _tf32_rn(b - hi), i < SIZE)


def pack_b(b):
    packed = b.new_empty((2, *b.shape))
    _pack_b[(triton.cdiv(b.numel(), 256),)](b, packed, b.numel(), 256)
    return packed


@triton.jit
def _hidden(PART, pid, SPLITS: tl.constexpr, PAD: tl.constexpr):
    split = tl.arange(0, PAD)
    element = tl.arange(0, 256)
    part = tl.load(PART + ((pid * SPLITS + split) * 256)[:, None] + element[None, :],
                   split[:, None] < SPLITS, 0)
    return tl.reshape(tl.sum(part, 0), (16, 16))


@triton.jit
def _output_tile(Y, B, RES, hh, hl, tile, stripe, FIRST: tl.constexpr, END: tl.constexpr,
                 OFFSET: tl.constexpr, N: tl.constexpr, SCALE: tl.constexpr, YROW: tl.constexpr,
                 BN: tl.constexpr, PACKED: tl.constexpr, RESROW: tl.constexpr):
    rank = tl.arange(0, 16)
    col = stripe * BN + tl.arange(0, BN)
    p = B + col[:, None] * 16 + rank[None, :]
    b = tl.load(p, col[:, None] < N, 0)
    # Two rounded TF32 components per operand, retaining the cross products.
    # The residual x residual term matters for FP32 accuracy near cancellation.
    if PACKED:
        bh = b
        bl = tl.load(p + N * 16, col[:, None] < N, 0)
    else:
        bh = _tf32_rn(b)
        bl = _tf32_rn(b - bh)
    small = tl.dot(hl, tl.trans(bl), input_precision="tf32")
    small += tl.dot(hl, tl.trans(bh), input_precision="tf32")
    small += tl.dot(hh, tl.trans(bl), input_precision="tf32")
    delta = small + tl.dot(hh, tl.trans(bh), input_precision="tf32")
    dtype: tl.constexpr = Y.dtype.element_ty
    scaled = (delta.to(dtype).to(tl.float32) * SCALE).to(dtype)
    row = FIRST + tile * 16 + tl.arange(0, 16)
    ptr = Y + row[:, None] * YROW + OFFSET + col[None, :]
    valid = (row[:, None] < END) & (col[None, :] < N)
    val = tl.load(ptr, valid, 0).to(tl.float32) + scaled.to(tl.float32)
    if RESROW:
        residual = tl.load(RES + row[:, None] * RESROW + col[None, :], valid, 0)
        # Preserve the BF16 store boundary between the adapter and residual add.
        val = residual.to(tl.float32) + val.to(dtype).to(tl.float32)
    tl.store(ptr, val, valid)


@triton.jit
def _grouped_lora_add(Y, Bs, PART, GROUPS: tl.constexpr, PACKED: tl.constexpr, YROW: tl.constexpr,
                      SPLITS: tl.constexpr, PAD: tl.constexpr, BN: tl.constexpr):
    pid, stripe = tl.program_id(0), tl.program_id(1)
    for i in tl.static_range(len(GROUPS)):
        begin, stop, first, end, offset, n, scale = GROUPS[i]
        if (pid >= begin) & (pid < stop) & (stripe * BN < n):
            h = _hidden(PART, pid, SPLITS, PAD)
            hh = _tf32_rn(h)
            _output_tile(Y, Bs[i], Y, hh, _tf32_rn(h - hh), pid-begin, stripe, first, end,
                         offset, n, scale, YROW, BN, PACKED[i], 0)


@triton.jit
def _grouped_lora_add_packed(X, As, Y, Bs, PART, RES, GROUPS: tl.constexpr, ENDS: tl.constexpr,
                             GAPS: tl.constexpr, RESROW: tl.constexpr,
                             YROW: tl.constexpr, SPLITS: tl.constexpr, PAD: tl.constexpr,
                             BN: tl.constexpr, STRIPES: tl.constexpr, FUSED: tl.constexpr,
                             K: tl.constexpr, XROW: tl.constexpr, BK: tl.constexpr):
    # A compact grid omits empty K/V stripes alongside the much wider Q projection.
    # Adjacent CTAs share immutable B stripes across the readout row tiles in L2.
    work = tl.program_id(0)
    for i in tl.static_range(len(GROUPS)):
        begin, stop, first, end, offset, n, scale = GROUPS[i]
        if (work >= ENDS[i]) & (work < ENDS[i+1]):
            tile = (work - ENDS[i]) % (stop - begin)
            stripe = (work - ENDS[i]) // (stop - begin) * STRIPES
            if FUSED:
                # One CTA keeps its 16x16 partials on chip for a group of B
                # stripes. Preserve the zero-padded split axis and its reduction.
                row = first + tile * 16 + tl.arange(0, 16)
                part = tl.full((PAD, 256), 0, tl.float32)
                split_ids = tl.arange(0, PAD)
                for split in range(SPLITS):
                    p = _rank_partial(X, As[i], row, split, end, K, XROW, BK)
                    part = tl.where(split_ids[:, None] == split, tl.reshape(p, (1, 256)), part)
                h = tl.reshape(tl.sum(part, 0), (16, 16))
            else:
                h = _hidden(PART, begin + tile, SPLITS, PAD)
            hh = _tf32_rn(h)
            hl = _tf32_rn(h - hh)
            for j in range(STRIPES):
                if (stripe + j) * BN < n:
                    _output_tile(Y, Bs[i], RES, hh, hl, tile, stripe + j, first, end, offset, n,
                                 scale, YROW, BN, True, RESROW)
    for i in tl.static_range(len(GAPS)):
        lo, hi, first, end, n = GAPS[i]
        if (work >= lo) & (work < hi):
            element = (work - lo) * 1024 + tl.arange(0, 1024)
            gap_row = first + element // n
            dst = Y + gap_row * YROW + element % n
            residual = tl.load(RES + gap_row * RESROW + element % n, gap_row < end, 0)
            base = tl.load(dst, gap_row < end, 0)
            tl.store(dst, residual.to(tl.float32) + base.to(tl.float32), gap_row < end)


def update(x, y, pairs, residual=None, tiling=DEFAULT_LORA_TILE):
    """Packed B holds the two TF32 components of an immutable registered bank."""
    groups, producers, aa, bb = [], [], [], []
    tiles = programs = wide_groups = 0
    for first, end, offset, a, b, scale in pairs:
        count = triton.cdiv(end - first, 16)
        if not count:
            continue
        groups.append((tiles, tiles+count, first, end, offset, b.size(-2), scale))
        # The 320 judgement rows have enough work to amortize all three A loads
        # over 32 rows. Keep the small box and shorter stress tiles at M16.
        m = tiling.rank_rows if b.ndim == 3 and end - first >= 320 else 16
        nprograms = triton.cdiv(end - first, m)
        if m != 16:
            wide_groups |= 1 << len(producers)
        producers.append((programs, programs + nprograms))
        programs += nprograms
        aa.append(a)
        bb.append(b)
        tiles += count
    if tiles:
        bk, bn = 256, 128
        splits = triton.cdiv(x.size(1), bk)
        packed = all(b.ndim == 3 for b in bb)
        fused = packed and tiling.fused and x.size(1) <= 4096
        partial = x  # Unused pointer in the single-launch specialization.
        if not fused:
            partial = x.new_empty((tiles, splits, 16, 16), dtype=torch.float32)
            _grouped_lora_rank[(programs, splits)](
                x, tuple(aa), partial, tuple(groups), tuple(producers), wide_groups, tiling.rank_rows,
                x.size(1), x.stride(0), splits, bk, num_stages=1, enable_fp_fusion=False)
        if packed:
            stripe_group = tiling.stripes
            ends = [0]
            for begin, stop, _, _, _, n, _ in groups:
                ends.append(ends[-1] + (stop - begin) * triton.cdiv(n, bn * stripe_group))
            total, gaps = ends[-1], []
            if residual is not None:
                # A residual is supplied only for a single full-width projection.
                # Include the frame and unadapted/padded rows in the same launch.
                first = 0
                for start, stop in sorted((p[0], p[1]) for p in pairs) + [(len(y), len(y))]:
                    if first < start:
                        limit = total + triton.cdiv((start - first) * y.size(1), 1024)
                        gaps.append((total, limit, first, start, y.size(1)))
                        total = limit
                    first = stop
            _grouped_lora_add_packed[(total,)](
                x, tuple(aa), y, tuple(bb), partial, y if residual is None else residual, tuple(groups), tuple(ends),
                tuple(gaps), 0 if residual is None else residual.stride(0), y.stride(0), splits,
                triton.next_power_of_2(splits), bn, stripe_group, fused, x.size(1), x.stride(0), bk,
                num_stages=1, enable_fp_fusion=False)
            return
        stripes = triton.cdiv(max(b.size(-2) for b in bb), bn)
        _grouped_lora_add[(tiles, stripes)](y, tuple(bb), partial, tuple(groups),
                                            tuple(b.ndim == 3 for b in bb), y.stride(0),
                                            splits, triton.next_power_of_2(splits), bn,
                                            num_stages=1, enable_fp_fusion=False)


def supported(x, a, b):
    return (x.is_cuda and x.dtype == torch.bfloat16 and x.stride(-1) == 1
            and a.dtype == b.dtype == torch.float32 and len(a) == 16
            and a.is_contiguous() and b.is_contiguous())
