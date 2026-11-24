"""Read the old/post union from the slab once per owner. Flash shares the common keys
across frame, question and box rows. Its private suffixes have independent normalizers.
"""
from itertools import groupby

import torch
import triton
import triton.language as tl


@triton.jit
def _stage(K, V, DIR, END, KO, VO, OWNER: tl.constexpr, CAP: tl.constexpr,
           H: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
    head = tl.program_id(1)
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < tl.load(END)
    slot = tl.load(DIR + i, valid, -1)
    d = tl.arange(0, D)
    src = ((OWNER * H + head) * CAP + slot[:, None]) * D + d[None, :]
    dst = (i[:, None] * H + head) * D + d[None, :]
    # Write finite zeros for the flash placeholders without touching the slab.
    tl.store(KO + dst, tl.load(K + src, slot[:, None] >= 0, 0), valid[:, None])
    tl.store(VO + dst, tl.load(V + src, slot[:, None] >= 0, 0), valid[:, None])


@triton.jit
def _pack_q(Q, IX, PACKED, R: tl.constexpr, H: tl.constexpr, G: tl.constexpr, D: tl.constexpr,
            QR: tl.constexpr, QH: tl.constexpr):
    i, h = tl.program_id(0), tl.program_id(1)
    row = tl.load(IX + i % R)
    g = tl.arange(0, G)
    d = tl.arange(0, D)
    src = row * QR + ((h * G + g) * QH)[:, None] + d[None, :]
    dst = (((i * G + g) * H + h) * D)[:, None] + d[None, :]
    tl.store(PACKED + dst, tl.load(Q + src))


@triton.jit
def _join(P, L, LENGTHS, ROWS, OUT, LOG, R: tl.constexpr, OLD: tl.constexpr,
          H: tl.constexpr, G: tl.constexpr, D: tl.constexpr, S: tl.constexpr,
          COMMON: tl.constexpr, PAD: tl.constexpr, LS: tl.constexpr, LH: tl.constexpr,
          LR: tl.constexpr):
    r, head = tl.program_id(0), tl.program_id(1)
    row = tl.load(ROWS + r)
    s = tl.arange(0, PAD)
    qi = r * G + head % G
    kh = head // G
    valid = (s < S) & (tl.load(LENGTHS + s, s < S, 0) > 0)
    # The two tails in physical union order are post-only, then old-only.
    valid = valid & ((s < COMMON) | (s == tl.where(r < OLD, COMMON + 1, COMMON)))
    logs = tl.load(L + s * LS + kh * LH + qi * LR, valid, -float("inf"))
    peak = tl.max(logs, 0)
    weights = tl.exp(logs - tl.where(peak == -float("inf"), 0., peak))
    mass = tl.maximum(tl.sum(weights, 0), 1.)
    d = tl.arange(0, D)
    offset = ((s * R * G + qi) * H + kh) * D
    p = tl.load(P + offset[:, None] + d[None, :], valid[:, None], 0).to(tl.float32)
    out = row * H * G + head
    tl.store(OUT + out * D + d, tl.sum(weights[:, None] * p, 0) / mass)
    tl.store(LOG + out, peak + tl.log(mass))


def prepare(keys, directory, widths, spec, indices, config, repeat):
    device = directory.device
    seq = torch.arange(config.splits + 1, dtype=torch.int32, device=device)
    ends = (seq * widths[0]) // config.splits
    if widths.numel() == 3:
        ends = torch.cat((ends, widths[1:]))
    lengths = ends.diff()
    cu = torch.nn.functional.pad(lengths.clamp_min(1).cumsum(0, dtype=torch.int32), (1, 0))
    # Insert dummy slots at empty sections outside the layer loop. The full window
    # directory and sequence lengths stay device-resident across changing-width replays.
    capacity = len(directory) + len(lengths)
    i = torch.arange(capacity, device=device, dtype=torch.int32)
    segment = torch.searchsorted(cu[1:], i, right=True).clamp_max(len(lengths)-1).long()
    logical = (i + (ends - cu)[segment]).clamp(0, len(directory)-1).long()
    slots = torch.where(i - cu[segment] < lengths[segment], directory.index_select(0, logical), -1)
    buffers = tuple(keys.new_empty((capacity, keys.size(1), keys.size(-1))) for _ in range(2))
    owners = []
    for owner, chunks in groupby(zip(spec, indices), key=lambda v: v[0][0]):
        groups = list(chunks)
        rows = torch.cat([ix for _, ix in groups])
        old = sum(len(ix) for (_, view, _), ix in groups if view == 0)
        cu_q = torch.arange(len(lengths)+1, device=device, dtype=torch.int32) * (len(rows)*repeat)
        owners.append((owner, rows, old, cu_q))
    return slots, lengths, cu, buffers, owners


def launch(q, keys, values, directory, widths, spec, indices, config, scale, scratch=None):
    rows, heads, d = q.shape
    kh = keys.size(1)
    g = heads // kh
    if scratch is None:
        scratch = prepare(keys, directory, widths, spec, indices, config, g)
    slots, lengths, cu, (k, v), owners = scratch
    count = len(lengths)
    out = q.new_empty((rows, heads, d))
    lse = q.new_empty((rows, heads), dtype=torch.float32)
    for owner, ix, old, cu_q in owners:
        _stage[(triton.cdiv(len(slots), 16), kh)](
            keys, values, slots, cu[-1:], k, v, owner, keys.size(2), kh, d, 16)
        # Fold GQA into query rows and replicate across independent key sections.
        r = len(ix)
        query = q.new_empty((count*r*g, kh, d))
        _pack_q[(count*r, kh)](q, ix, query, r, kh, g, d, q.stride(0), q.stride(1))
        p, logs = torch.ops.aten._flash_attention_forward(
            query, k, v, cu_q, cu, r*g, len(directory), 0., False, False, scale=scale)[:2]
        # ATen builds expose either compact head-major or batch/head/padded-row LSE.
        if logs.ndim == 2:
            ls, lh, lr = r*g*logs.stride(1), logs.stride(0), logs.stride(1)
        else:
            ls, lh, lr = logs.stride()
        _join[(r, heads)](p, logs, lengths, ix, out, lse, r, old, kh, g, d, count,
                          config.splits, triton.next_power_of_2(count), ls, lh, lr)
    return out, lse
