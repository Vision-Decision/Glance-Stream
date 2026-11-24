"""Indexed chronological prefix, causal suffix, and continuation primitives."""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from glance_stream.engine.state import OwnerConfig, SharedConfig, owner_tiles, prefix_partition, prefix_tiles
from glance_stream.model.attn import window_attention


@triton.jit
def _pack_step_embeddings(F, TOKENS, W, LATENT, OUT, FRAME: tl.constexpr, D: tl.constexpr,
                          FR: tl.constexpr, FC: tl.constexpr, WR: tl.constexpr, WC: tl.constexpr,
                          HAS_LATENT: tl.constexpr, LC: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    d = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    if row < FRAME:
        value = tl.load(F + row * FR + d * FC, d < D, 0)
        tl.store(OUT + row * D + d, value, d < D)
    else:
        token = tl.load(TOKENS + row - FRAME)
        if HAS_LATENT and token == -1:
            tl.store(OUT + row * D + d, tl.load(LATENT + d * LC, d < D, 0), d < D)
        else:
            tl.store(OUT + row * D + d, tl.load(W + token * WR + d * WC, d < D, 0), d < D)


def pack_step_embeddings(frame, tokens, weight, latent):
    rows, dim = frame.shape
    x = frame.new_empty((rows + len(tokens), dim))
    _pack_step_embeddings[(len(x), triton.cdiv(dim, 256))](
        frame, tokens, weight, latent, x, rows, dim, *frame.stride(), *weight.stride(),
        latent.numel() != 0, latent.stride(-1), 256)
    return x


@triton.jit
def _prefix(Q, K, V, DIR, OWNER, O, LSE, ROWS: tl.constexpr, H: tl.constexpr,
            W, CAP: tl.constexpr, D: tl.constexpr, SPLITS,
            SCALE: tl.constexpr, M: tl.constexpr, N: tl.constexpr, CHUNK: tl.constexpr):
    mi = tl.program_id(0) * M + tl.arange(0, M)
    branch_head = tl.program_id(1)
    split = tl.program_id(2)
    owner = tl.load(OWNER + branch_head // H)
    head = branch_head % H
    di = tl.arange(0, D)
    q = tl.load(Q + (branch_head * ROWS + mi[:, None]) * D + di[None, :],
                mi[:, None] < ROWS, other=0)
    m = tl.full((M,), -float("inf"), tl.float32)
    z = tl.full((M,), 0, tl.float32)
    acc = tl.full((M, D), 0, tl.float32)
    # All launched splits contain at least one key.
    for tile in range(tl.cdiv(CHUNK, N)):
        logical = split * CHUNK + tile * N + tl.arange(0, N)
        slot = tl.load(DIR + logical, logical < W, other=0)
        ptr = (((owner * H + head) * CAP + slot[:, None]) * D + di[None, :])
        k = tl.load(K + ptr, logical[:, None] < W, other=0)
        v = tl.load(V + ptr, logical[:, None] < W, other=0)
        scores = tl.dot(q, tl.trans(k)) * SCALE
        scores = tl.where(logical[None, :] < W, scores, -float("inf"))
        new_m = tl.maximum(m, tl.max(scores, axis=1))
        w = tl.exp(scores - new_m[:, None])
        correction = tl.exp(m - new_m)
        z = z * correction + tl.sum(w, axis=1)
        acc = acc * correction[:, None] + tl.dot(w.to(v.dtype), v)
        m = new_m
    row = (branch_head * SPLITS + split) * ROWS + mi
    tl.store(O + row[:, None] * D + di[None, :], acc / z[:, None], mi[:, None] < ROWS)
    tl.store(LSE + row, m + tl.log(z), mi < ROWS)


def indexed_prefix(q, keys, values, slots, owners, scale):
    b, h, rows, d = q.shape
    width, chunk = slots.numel(), 512
    splits = triton.cdiv(width, chunk)
    partial = torch.empty((b, h, splits, rows, d), dtype=torch.float32, device=q.device)
    lse = torch.empty(partial.shape[:-1], dtype=torch.float32, device=q.device)
    _prefix[(triton.cdiv(rows, 16), b * h, splits)](
        q.contiguous(), keys, values, slots, owners, partial, lse, rows, h, width,
        keys.size(2), d, splits, scale, 16, 32, chunk)
    total = lse.logsumexp(2)
    mass = (lse - total.unsqueeze(2)).exp()
    return (partial * mass.unsqueeze(-1)).sum(2).to(q.dtype), total


def gathered_prefix(q, keys, values, slots, groups, scratch, scale):
    """Gather once per owner and layer. Fold all of an owner's queries into flash rows."""
    n, h, rows, d = q.shape
    output, lse = torch.empty_like(q), q.new_empty((n, h, rows), dtype=torch.float32)
    k, v = scratch
    for owner, branches in groups:
        torch.index_select(keys[owner:owner + 1], 2, slots, out=k)
        torch.index_select(values[owner:owner + 1], 2, slots, out=v)
        folded = q.index_select(0, branches).transpose(0, 1).reshape(1, h, -1, d)
        o, l = window_attention(folded, k, v, scale)
        output.index_copy_(0, branches, o.reshape(h, -1, rows, d).transpose(0, 1))
        l = l[..., :folded.size(2)]
        lse.index_copy_(0, branches, l.reshape(h, -1, rows).transpose(0, 1))
    return output, lse


def attention(q, k, v, keys, values, slots, owners, groups, scratch, backend, scale):
    b, h, length, d = q.shape
    kh = k.size(1)
    repeat = h // kh
    if not slots.numel():
        return F.scaled_dot_product_attention(q, k.repeat_interleave(repeat, 1),
                                              v.repeat_interleave(repeat, 1),
                                              is_causal=True, scale=scale)
    folded = q.reshape(b, kh, repeat * length, d)
    if backend == "indexed":
        prefix, lw = indexed_prefix(folded, keys, values, slots, owners, scale)
    else:
        prefix, lw = gathered_prefix(folded, keys, values, slots, groups, scratch, scale)
    i = torch.arange(length, device=q.device)
    scores = (folded.float() @ k.float().transpose(-2, -1)) * scale
    scores.masked_fill_(i[None] > i.repeat(repeat)[:, None], -torch.inf)
    ll = scores.logsumexp(-1)
    local = (scores - ll.unsqueeze(-1)).exp() @ v.float()
    total = torch.logaddexp(lw, ll)
    out = prefix.float() * (lw - total).exp().unsqueeze(-1) + local * (ll - total).exp().unsqueeze(-1)
    return out.to(q.dtype).reshape_as(q)


@triton.jit
def _rotate(K, DIR, COS, SIN, W, CAP: tl.constexpr, D: tl.constexpr, HALF: tl.constexpr,
            BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < W * HALF
    slot = tl.load(DIR + i // HALF, valid, other=0)
    freq = i % HALF
    ptr = K + (tl.program_id(1) * CAP + slot) * D + freq
    a = tl.load(ptr, valid, other=0).to(tl.float32)
    b = tl.load(ptr + HALF, valid, other=0).to(tl.float32)
    c, s = tl.load(COS + freq), tl.load(SIN + freq)
    tl.store(ptr, a * c - b * s, valid)
    tl.store(ptr + HALF, b * c + a * s, valid)


def rotate_slots(keys, slots, inv_freq, shift, backend):
    angle = -shift * inv_freq
    if backend == "indexed":
        c, s = angle.cos(), angle.sin()
        if slots.numel():
            for key in keys:
                _rotate[(triton.cdiv(slots.numel()*c.numel(), 256), key.size(0)*key.size(1))](
                    key, slots, c, s, slots.numel(), key.size(2), key.size(3), c.numel(), 256,
                    enable_fp_fusion=False)
        return
    cos, sin = angle.cos().repeat(2), angle.sin().repeat(2)
    for key in keys:
        part = key[..., :cos.numel()]
        x = part.index_select(2, slots).float()
        a, b = x.chunk(2, -1)
        part.index_copy_(2, slots, (x * cos + torch.cat((-b, a), -1) * sin).to(key.dtype))


@triton.jit
def _pack_read_qkv(CONV, Z, Q, K, V, ZOUT, LENGTH: tl.constexpr, KH: tl.constexpr, VH: tl.constexpr,
                   KD: tl.constexpr, VD: tl.constexpr, SB: tl.constexpr, SC: tl.constexpr,
                   ST: tl.constexpr, ZB: tl.constexpr, ZT: tl.constexpr, ZC: tl.constexpr,
                   BT: tl.constexpr, BC: tl.constexpr):
    b = tl.program_id(2)
    t = tl.program_id(0) * BT + tl.arange(0, BT)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    base = CONV + b * SB + t[:, None] * ST
    row = b * LENGTH + t
    if tl.program_id(1) * BC < VH * KD:
        src = c // KD // (VH // KH) * KD + c % KD
        mask = (t[:, None] < LENGTH) & (c[None, :] < VH * KD)
        ptr = base + src[None, :] * SC
        dst = row[:, None] * (VH * KD) + c[None, :]
        tl.store(Q + dst, tl.load(ptr, mask, 0), mask)
        tl.store(K + dst, tl.load(ptr + KH * KD * SC, mask, 0), mask)
    if tl.program_id(1) * BC < VH * VD:
        mask = (t[:, None] < LENGTH) & (c[None, :] < VH * VD)
        dst = row[:, None] * (VH * VD) + c[None, :]
        tl.store(V + dst, tl.load(base + (2 * KH * KD + c[None, :]) * SC, mask, 0), mask)
        tl.store(ZOUT + dst, tl.load(Z + b * ZB + t[:, None] * ZT + c[None, :] * ZC, mask, 0), mask)


def pack_read_qkv(conv, z, kh, vh, kd, vd):
    """Pack FLA operands and its norm gate, replicating grouped key heads."""
    batch, _, length = conv.shape
    q, k, v = (conv.new_empty((batch, length, vh, dim)) for dim in (kd, kd, vd))
    gate = z.new_empty(z.shape)
    _pack_read_qkv[(triton.cdiv(length, 16), triton.cdiv(vh * max(kd, vd), 64), batch)](
        conv, z, q, k, v, gate, length, kh, vh, kd, vd, *conv.stride(), *z.stride(), 16, 64)
    return q, k, v, gate


def continuation(m, mixed, z, beta, g, seed, tail, write, pack_read=False):
    """One write batch or one bounded read tile. HF caches raw pre-convolution x."""
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    batch, length, _ = mixed.shape
    raw = mixed.transpose(1, 2)
    width = m.conv_kernel_size
    fused_conv = getattr(m, "causal_conv1d_fn", None)
    pack_read = pack_read and not write
    if pack_read and fused_conv is not None:
        # Directly build the contiguous argument the fast convolution consumes.
        x = torch.cat((tail[..., tail.size(-1) - width + 1:], raw), -1)
    else:
        history = torch.cat((tail, raw), -1)
        x = history[..., history.size(-1) - length - width + 1:]
    if fused_conv is not None:
        # The installed HF fast path fuses SiLU before the BF16 store.
        conv = fused_conv(x.contiguous(), m.conv1d.weight.squeeze(1), m.conv1d.bias,
                          activation=getattr(m, "activation", "silu"))[..., -length:]
    else:
        conv = F.silu(F.conv1d(x, m.conv1d.weight, m.conv1d.bias, groups=x.size(1)))
    if pack_read:
        q, k, v, z = pack_read_qkv(conv, z, m.key_dim // m.head_k_dim, m.value_dim // m.head_v_dim,
                                   m.head_k_dim, m.head_v_dim)
    else:
        q, k, v = conv.transpose(1, 2).split((m.key_dim, m.key_dim, m.value_dim), -1)
        q = q.reshape(batch, length, -1, m.head_k_dim)
        k = k.reshape_as(q)
        v = v.reshape(batch, length, -1, m.head_v_dim)
        repeat = v.size(2) // q.size(2)
        if repeat > 1:
            q, k = q.repeat_interleave(repeat, 2), k.repeat_interleave(repeat, 2)
    o, state = chunk_gated_delta_rule(q.contiguous(), k.contiguous(), v.contiguous(),
                                      g=g.contiguous(), beta=beta.contiguous(), initial_state=seed,
                                      output_final_state=write, use_qk_l2norm_in_kernel=True)
    o = m.norm(o.reshape(-1, m.head_v_dim), z.reshape(-1, m.head_v_dim))
    return o.reshape(batch, length, -1), state, history[..., -tail.size(-1):].contiguous() if write else None


def prefix_schedule(segments, repeat, tile, device, backend, fixed_spans=True):
    if backend == "shared":
        schedule = owner_tiles(segments, repeat, tile)
        return (), tuple(torch.tensor(v, device=device, dtype=torch.int32) for v in schedule)
    spec, schedule = prefix_tiles(segments, repeat, tile, backend, fixed_spans)
    dtype = torch.int32 if backend == "indexed" else torch.long
    return spec, tuple(torch.tensor(rows, dtype=dtype, device=device) for rows in schedule)


@triton.jit
def _dual_prefix(Q, K, V, OLD, POST, WIDTHS, TILES, OUT, LSE,
                 ROWS: tl.constexpr, H: tl.constexpr, G: tl.constexpr,
                 QROW: tl.constexpr, QHEAD: tl.constexpr, CAP: tl.constexpr, D: tl.constexpr,
                 SCALE: tl.constexpr, M: tl.constexpr, N: tl.constexpr, CHUNK: tl.constexpr):
    desc = TILES + tl.program_id(0) * 4
    r = tl.load(desc) + tl.arange(0, M)
    end = tl.load(desc + 1)
    owner = tl.load(desc + 2)
    view = tl.load(desc + 3)
    w = tl.load(WIDTHS + view)
    head, split = tl.program_id(1), tl.program_id(2)
    d = tl.arange(0, D)
    q = tl.load(Q + ((r // G) * QROW + (head * G + r % G) * QHEAD)[:, None] + d[None, :],
                r[:, None] < end, other=0)
    acc = tl.full((M, D), 0, tl.float32)
    mass = tl.full((M,), 0, tl.float32)
    peak = tl.full((M,), -float("inf"), tl.float32)
    directory = tl.where(view == 0, OLD, POST)
    for tile in range(tl.cdiv(tl.minimum(tl.maximum(w - split * CHUNK, 0), CHUNK), N)):
        j = split * CHUNK + tile * N + tl.arange(0, N)
        slot = tl.load(directory + j, j < w, other=0)
        addr = (((owner * H + head) * CAP + slot[:, None]) * D + d[None, :])
        k = tl.load(K + addr, j[:, None] < w, other=0)
        v = tl.load(V + addr, j[:, None] < w, other=0)
        logits = tl.where(j[None, :] < w, tl.dot(q, tl.trans(k)) * SCALE, -float("inf"))
        new = tl.maximum(peak, tl.max(logits, 1))
        weights = tl.exp(logits - new[:, None])
        corr = tl.exp(peak - new)
        mass = mass * corr + tl.sum(weights, 1)
        acc = acc * corr[:, None] + tl.dot(weights.to(v.dtype), v)
        peak = new
    idx = ((split * ROWS + r // G) * H + head) * G + r % G
    tl.store(OUT + idx[:, None] * D + d[None, :], acc / tl.maximum(mass, 1.)[:, None], r[:, None] < end)
    tl.store(LSE + idx, peak + tl.log(tl.maximum(mass, 1.)), r < end)


@triton.jit
def _reduce_prefix(P, L, O, LOG, ROWS: tl.constexpr, D: tl.constexpr,
                   S: tl.constexpr, PAD: tl.constexpr):
    row = tl.program_id(0)
    s = tl.arange(0, PAD)
    d = tl.arange(0, D)
    idx = s * ROWS + row
    l = tl.load(L + idx, s < S, other=-float("inf"))
    mx = tl.max(l, 0)
    weights = tl.exp(l - tl.where(mx == -float("inf"), 0., mx))
    z = tl.sum(weights, 0)
    p = tl.load(P + idx[:, None] * D + d[None, :], s[:, None] < S, other=0)
    tl.store(O + row * D + d, tl.sum(weights[:, None] * p, 0) / tl.maximum(z, 1.))
    tl.store(LOG + row, mx + tl.log(tl.maximum(z, 1.)))


@triton.jit
def _gather_view(K, V, DIR, W, KO, VO, OWNER: tl.constexpr, H: tl.constexpr,
                 CAP: tl.constexpr, D: tl.constexpr, TILE: tl.constexpr):
    j = tl.program_id(0) * TILE + tl.arange(0, TILE)
    h = tl.program_id(1)
    d = tl.arange(0, D)
    live = j < tl.load(W)
    slot = tl.load(DIR + j, live, other=0)
    src = ((OWNER * H + h) * CAP + slot[:, None]) * D + d[None, :]
    dst = (j[:, None] * H + h) * D + d[None, :]
    # Only the empty-prefix flash call consumes the dummy at j=0.
    store = (live | (j == 0))[:, None]
    tl.store(KO + dst, tl.load(K + src, live[:, None], other=0), store)
    tl.store(VO + dst, tl.load(V + src, live[:, None], other=0), store)


def prefix_scratch(q, keys, width, spec, indices, widths):
    shape = (width, keys.size(1), keys.size(-1))
    # Build flash offsets once for the decoder traversal, outside the layer loop.
    cu_k = [F.pad(w.clamp_min(1), (1, 0)) for w in widths.split(1)]
    ends = torch.arange(2, device=q.device, dtype=torch.int32)
    cu_q = {n: ends*n for n in {len(ix) for ix in indices}}
    sequences = [(cu_q[len(ix)], cu_k[view]) for (_, view, _), ix in zip(spec, indices)]
    return q.new_empty(shape), q.new_empty(shape), sequences


def packed_prefix(q, keys, values, old, post, widths, spec, indices, scratch, tuning, scale):
    if isinstance(tuning, OwnerConfig):
        from glance_stream.engine.owner_prefix import launch
        return launch(q, keys, values, old, widths, spec, indices, tuning, scale, scratch)
    if isinstance(tuning, SharedConfig):
        from glance_stream.engine.shared_prefix import launch
        return launch(q, keys, values, old, widths, indices, tuning, scale)
    rows, heads, dim = q.shape
    kh = keys.size(1)
    repeat = heads // kh
    single_span = len(spec) == 1 and spec[0][2] == 0
    if not single_span:
        out = torch.empty_like(q, memory_format=torch.contiguous_format)
        lse = q.new_empty((rows, heads), dtype=torch.float32)
    if scratch is None:
        chunk, m, n = tuning
        # Limit split workspace to ~64 MiB or a single full partial for large B/N.
        chunk, splits = prefix_partition(old.numel(), rows, heads, dim, chunk, n)
        part = q.new_empty((splits, rows, heads, dim), dtype=torch.float32)
        log = q.new_empty(part.shape[:-1], dtype=torch.float32)
        tiles, = indices
        _dual_prefix[(len(tiles), kh, splits)](
            q, keys, values, old, post, widths, tiles, part, log, rows, kh, repeat,
            q.stride(0), q.stride(1), keys.size(2), dim, scale, m, n, chunk)
        _reduce_prefix[(rows * heads,)](part, log, out, lse, rows * heads, dim,
                                         splits, triton.next_power_of_2(splits))
    else:
        k, v, sequences = scratch
        for (owner, view, start), ix, (cu_q, cu_k) in zip(spec, indices, sequences):
            slots = (old, post)[view]
            w = widths[view:view + 1]
            _gather_view[(triton.cdiv(slots.numel(), 16), kh)](
                keys, values, slots, w, k, v, owner, kh, keys.size(2), dim, 16)
            span = slice(start, start + len(ix))
            query = q[span] if start >= 0 else q.index_select(0, ix)
            # ATen varlen flash exposes both the output and natural-log LSE.
            result = torch.ops.aten._flash_attention_forward(
                query, k, v, cu_q, cu_k, len(ix), slots.numel(), 0., False, False, scale=scale)
            o, log = result[:2]
            log = log.reshape(heads, -1)[:, :len(ix)].T
            log = log.masked_fill(w == 0, -torch.inf)
            if single_span:
                return o, log
            if start >= 0:
                out[span].copy_(o)
                lse[span].copy_(log)
            else:
                out.index_copy_(0, ix, o)
                lse.index_copy_(0, ix, log)
    return out, lse


@triton.jit
def _merge_triangle(PREFIX, LW, LOCAL, LL, OUT, SIZE: tl.constexpr, HEADS: tl.constexpr,
                    LENGTH: tl.constexpr, D: tl.constexpr, M: tl.constexpr):
    # LOCAL is block/head/token-major. Write its merge directly in projection order.
    row = tl.program_id(0) * M + tl.arange(0, M)
    token = row // HEADS
    local_row = ((token // LENGTH * HEADS + row % HEADS) * LENGTH + token % LENGTH)
    valid = row < SIZE * HEADS
    lp = tl.load(LW + row, valid, 0)
    ll = tl.load(LL + local_row, valid, 0)
    total = tl.maximum(lp, ll) + tl.log(1. + tl.exp(-tl.abs(lp - ll)))
    d = tl.arange(0, D)
    p = tl.load(PREFIX + row[:, None] * D + d[None, :], valid[:, None], 0).to(tl.float32)
    loc = tl.load(LOCAL + local_row[:, None] * D + d[None, :], valid[:, None], 0)
    tl.store(OUT + row[:, None] * D + d[None, :],
             p * tl.exp(lp - total)[:, None] + loc * tl.exp(ll - total)[:, None], valid[:, None])


def triangle_masks(regions, repeat, device):
    masks = {}
    for length in {length for _, length in regions}:
        ar = torch.arange(length, device=device)
        masks[length] = ar[None] > ar.repeat(repeat)[:, None]
    return masks


def step_attention(q, k, v, slabs, old, post, widths, regions, spec, indices, scratch, tuning, scale, masks=None):
    """One prefix invocation for all blocks. Each local triangle has its own origin."""
    prefix, lw = packed_prefix(q, *slabs, old, post, widths, spec, indices, scratch, tuning, scale)
    start, outputs = 0, []
    repeat = q.size(1) // k.size(1)
    shared = isinstance(tuning, (SharedConfig, OwnerConfig))
    if shared:
        packed = torch.empty_like(prefix)
    if masks is None:
        masks = triangle_masks(regions, repeat, q.device)
    for count, length in regions:
        size = count * length
        sl = slice(start, start + size)
        start += size
        def block(t):
            return t[sl].unflatten(0, (count, length)).transpose(1, 2)
        qb, kb, vb = block(q), block(k), block(v)
        folded = qb.reshape(count, k.size(1), repeat * length, q.size(-1))
        score = folded.float() @ kb.float().transpose(-1, -2) * scale
        score.masked_fill_(masks[length], -torch.inf)
        ll = score.logsumexp(-1)
        local = (score - ll.unsqueeze(-1)).exp() @ vb.float()
        if shared:
            _merge_triangle[(triton.cdiv(size * q.size(1), 16),)](
                prefix[sl], lw[sl], local, ll, packed[sl], size, q.size(1), length, q.size(-1),
                16, enable_fp_fusion=False)
            continue
        lp = block(lw).reshape_as(ll)
        total = torch.logaddexp(lp, ll)
        o = block(prefix).reshape_as(local) * (lp - total).exp().unsqueeze(-1)
        o = o + local * (ll - total).exp().unsqueeze(-1)
        outputs.append(o.to(q.dtype).reshape_as(qb).transpose(1, 2).reshape(size, -1))
    if shared:
        return packed.flatten(1)
    return torch.cat(outputs) if len(outputs) > 1 else outputs[0]
