"""Owner-grouped BF16 split-key attention for a shared old/post KV slab."""
import torch
import triton
import triton.language as tl

from glance_stream.engine.state import checked_shared_config, prefix_partition


@triton.jit
def _shared_prefix(Q, K, V, DIR, WIDTHS, OWNERS, TOKENS, P, L,
                   ROWS: tl.constexpr, H: tl.constexpr, G: tl.constexpr, D: tl.constexpr,
                   CAP: tl.constexpr, QR: tl.constexpr, QH: tl.constexpr, SCALE: tl.constexpr,
                   DUAL: tl.constexpr, M: tl.constexpr, N: tl.constexpr, CHUNK: tl.constexpr):
    # Adjacent CTAs working on other query passes request the same key split in L2.
    tile, head, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    owner = tl.load(OWNERS + tile)
    lane = tl.arange(0, M)
    token = tl.load(TOKENS + tile * (M // G) + lane // G)
    row = token // 2
    qh = head * G + lane % G
    d = tl.arange(0, D)
    q = tl.load(Q + (row * QR + qh * QH)[:, None] + d[None, :], token[:, None] >= 0, 0)
    common = tl.load(WIDTHS)
    if DUAL:
        post = tl.load(WIDTHS + 1)
        width = tl.load(WIDTHS + 2)
    else:
        width = common
    peak = tl.full((M,), -float("inf"), tl.float32)
    mass = tl.full((M,), 0, tl.float32)
    acc = tl.full((M, D), 0, tl.float32)
    for block in range(tl.cdiv(tl.minimum(CHUNK, tl.maximum(width - split * CHUNK, 0)), N)):
        j = split * CHUNK + block * N + tl.arange(0, N)
        slot = tl.load(DIR + j, j < width, 0)
        address = ((owner * H + head) * CAP + slot[:, None]) * D + d[None, :]
        k = tl.load(K + address, j[:, None] < width, 0)
        v = tl.load(V + address, j[:, None] < width, 0)
        valid = j[None, :] < width
        if DUAL:
            # Union order is shared, post-only, old-only.
            in_old = (j < common) | (j >= post)
            valid = valid & tl.where(token[:, None] % 2 == 1, j[None, :] < post, in_old[None, :])
        score = tl.where(valid, tl.dot(q, tl.trans(k)) * SCALE, -float("inf"))
        new = tl.maximum(peak, tl.max(score, 1))
        finite = tl.where(new == -float("inf"), 0., new)
        p = tl.exp(score - finite[:, None])
        correction = tl.exp(peak - finite)
        mass = mass * correction + tl.sum(p, 1)
        acc = acc * correction[:, None] + tl.dot(p.to(v.dtype), v)
        peak = new
    out = (split * ROWS + row) * H * G + qh
    denom = tl.maximum(mass, 1.)
    tl.store(P + out[:, None] * D + d[None, :], acc / denom[:, None], token[:, None] >= 0)
    tl.store(L + out, peak + tl.log(denom), token >= 0)


def launch(q, keys, values, directory, widths, schedule, config, scale, *, resources=None):
    from glance_stream.engine.kernels import _reduce_prefix
    rows, heads, dim = q.shape
    kh = keys.size(1)
    owners, tokens = schedule
    chunk, m, n, warps = checked_shared_config(config)
    chunk, splits = prefix_partition(len(directory), rows, heads, dim, chunk, n)
    part = q.new_empty((splits, rows, heads, dim), dtype=torch.float32)
    logs = part.new_empty((splits, rows, heads))
    kernel = _shared_prefix[(len(owners), kh, splits)](
        q, keys, values, directory, widths, owners, tokens, part, logs, rows, kh, heads // kh,
        dim, keys.size(2), q.stride(0), q.stride(1), scale, widths.numel() == 3, m, n, chunk,
        num_warps=warps, num_stages=1)
    if resources is not None:
        resources.update(registers=kernel.n_regs, spills=kernel.n_spills,
                         shared_bytes=kernel.metadata.shared, effective_split=chunk, splits=splits)
    out = torch.empty_like(q, memory_format=torch.contiguous_format)
    lse = q.new_empty((rows, heads), dtype=torch.float32)
    _reduce_prefix[(rows * heads,)](part, logs, out, lse, rows * heads, dim,
                                    splits, triton.next_power_of_2(splits))
    return out, lse
