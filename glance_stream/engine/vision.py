"""Static Qwen3.5 vision tower (CUDA graphs, cached rotary tables)."""
from functools import lru_cache

from glance_stream.engine.graphs import Graphs
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _rotary(Q, K, Cos, Sin, OutQ, OutK, N, H: tl.constexpr, D: tl.constexpr,
            TOKEN_STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    # Each lane owns a pair across the two head halves, so each source value is read once.
    pair = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    token = pair // (H * (D // 2))
    head = pair // (D // 2) % H
    col = pair % (D // 2)
    mask = token < N
    offs = token * TOKEN_STRIDE + head * D + col
    table = token * D + col
    c0 = tl.load(Cos + table, mask, 0).to(tl.float32)
    c1 = tl.load(Cos + table + D // 2, mask, 0).to(tl.float32)
    s0 = tl.load(Sin + table, mask, 0).to(tl.float32)
    s1 = tl.load(Sin + table + D // 2, mask, 0).to(tl.float32)
    q0 = tl.load(Q + offs, mask, 0).to(tl.float32)
    q1 = tl.load(Q + offs + D // 2, mask, 0).to(tl.float32)
    k0 = tl.load(K + offs, mask, 0).to(tl.float32)
    k1 = tl.load(K + offs + D // 2, mask, 0).to(tl.float32)
    out = (token * H + head) * D + col
    tl.store(OutQ + out, q0 * c0 - q1 * s0, mask)
    tl.store(OutQ + out + D // 2, q1 * c1 + q0 * s1, mask)
    tl.store(OutK + out, k0 * c0 - k1 * s0, mask)
    tl.store(OutK + out + D // 2, k1 * c1 + k0 * s1, mask)


def rotary(q, k, cos, sin):
    n, h, d = q.shape
    oq, ok = (torch.empty((n, h, d), dtype=q.dtype, device=q.device) for _ in range(2))
    block = 512
    _rotary[(triton.cdiv(n * h * (d // 2), block),)](
        q, k, cos.contiguous(), sin.contiguous(), oq, ok, n, h, d, q.stride(0), block,
        enable_fp_fusion=False)
    return oq, ok


class FastVision:
    """Uniform grids of independent two-frame tubelets, matching Stream.encode."""
    def __init__(self, visual, graph_limit=8, graph_memory_mb=8192):
        self.visual = visual
        self.graphs = Graphs(graph_limit, graph_memory_mb * 1024**2)
        self.tables = lru_cache(maxsize=max(1, graph_limit))(self._tables)

    def _tables(self, groups, height, width):
        from transformers.models.qwen3_5 import modeling_qwen3_5 as Q
        v = self.visual
        grid = torch.tensor([[1, height, width]] * groups, device=v.pos_embed.weight.device)
        idx, weights = Q.get_vision_interpolation_indices_and_weights(
            grid, num_grid_per_side=v.num_grid_per_side, mode=v.interpolation_mode,
            align_corners=v.interpolation_align_corners, spatial_merge_size=v.spatial_merge_size, kwargs={})
        pos = (v.pos_embed(idx) * weights[:, :, None]).sum(1)
        pid = Q.get_vision_position_ids(grid, v.spatial_merge_size, kwargs={})
        cos, sin = v.rotary_pos_emb(pos, pid)
        return pos, cos.contiguous(), sin.contiguous()

    @torch.no_grad()
    def __call__(self, pixels, groups, height, width):
        key = groups, height, width
        length = height * width
        if min(key) <= 0 or pixels.ndim != 2 or len(pixels) != groups * length:
            raise ValueError("patch rows must match the supplied tubelet grid")
        pos, cos, sin = self.tables(*key)
        v = self.visual
        def tower(px):
            x = v.patch_embed(px) + pos.to(px.dtype)
            for block in v.blocks:
                attn = block.attn
                # HF stores all three projections in one fused QKV weight.
                qkv = attn.qkv(block.norm1(x)).reshape(len(x), 3, attn.num_heads, -1)
                q, k, value = qkv.unbind(1)
                q, k = rotary(q, k, cos, sin)
                # Match HF's independent SDPA reduction for each image.
                attended = []
                for i in range(groups):
                    sl = slice(i * length, (i+1) * length)
                    qi, ki, vi = (t[sl].transpose(0, 1)[None] for t in (q, k, value))
                    o = F.scaled_dot_product_attention(qi, ki, vi, scale=attn.scaling)
                    attended.append(o[0].transpose(0, 1).reshape(length, -1))
                x = x + attn.proj(attended[0] if groups == 1 else torch.cat(attended))
                x = x + block.mlp(block.norm2(x))
            return v.merger(x)
        if len(pixels) > 16384:
            return tower(pixels)
        return self.graphs.run(key, tower, (pixels,), own=True)


def bf16_errors(actual, oracle):
    if actual.shape != oracle.shape or actual.dtype != oracle.dtype or actual.dtype != torch.bfloat16:
        raise AssertionError("merged tower output shape or dtype differs")
    ref = oracle.float()
    delta = actual.float() - ref
    maximum = delta.abs().max().item()
    relative = (delta.norm() / ref.norm().clamp_min(1e-12)).item()
    # One BF16 relative spacing in L2, two spacings at the largest activation.
    eps = torch.finfo(torch.bfloat16).eps
    if not (maximum <= 2*eps*ref.abs().max().item() and relative <= eps):
        raise AssertionError(f"tower exceeds BF16 envelope: max abs {maximum}, relative L2 {relative}")
    return dict(max_abs=maximum, relative_l2=relative)
