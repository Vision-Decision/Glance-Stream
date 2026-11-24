"""Inference-only decoder over addressed KV and fixed GDN state tensors."""
from dataclasses import dataclass
import math
from types import MappingProxyType

import torch
import torch.nn.functional as F
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

from glance_stream.engine import bounds, grouped_lora, kernels
from glance_stream.engine.state import DEFAULT_LORA_TILE, OwnerConfig
from glance_stream.model.heads import LoRALinear, TARGETS


def base(module):
    return module.base if isinstance(module, LoRALinear) else module


def base_view(module):
    """Cold HF sink forward, sharing parameters without touching the active adapter bank."""
    import copy
    if isinstance(module, LoRALinear):
        return module.base
    children = {name: base_view(child) if child is not None else None
                for name, child in module._modules.items()}
    if all(child is module._modules[name] for name, child in children.items()):
        return module
    view = copy.copy(module)
    view._modules = children
    return view


@dataclass(frozen=True, eq=False)
class Bank:
    name: str
    generation: int


def apply_routes(y, x, modules, routes, lora, lora_tiles=None, residual=None):
    if not routes:
        return y if residual is None else residual + y
    tile = None
    if lora_tiles:
        longest = max(len(range(*region.indices(len(x)))) for region, _ in routes)
        tile = lora_tiles.get((x.size(1), y.size(1), longest))
    pairs = []
    offset = 0
    with torch.autograd.profiler.record_function("adapter"):
        for module in modules:
            for region, bank in routes:
                if module not in bank:
                    continue
                a, b, scale, packed = bank[module]
                if lora == "grouped" and grouped_lora.supported(x, a, b):
                    first, end, _ = region.indices(len(x))
                    pa, pb = (a, b) if packed is None else packed
                    pairs.append((first, end, offset, pa, pb if tile is not None else b, scale))
                else:
                    delta = (x[region].to(a.dtype) @ a.t() @ b.t()).to(y.dtype)
                    y[region, offset:offset + len(b)].add_(delta * scale)
            offset += module.out_features
        if (residual is not None and len(modules) == 1 and pairs and tile is not None
                and y.dtype == residual.dtype == torch.bfloat16 and residual.stride(-1) == 1
                and all(b.ndim == 3 for _, _, _, _, b, _ in pairs)):
            grouped_lora.update(x, y, pairs, residual, tile)
            return y
        grouped_lora.update(x, y, pairs, tiling=tile or DEFAULT_LORA_TILE)
    return y if residual is None else residual + y


def project(module, x, routes, label, lora, lora_tiles=None, residual=None):
    linear = base(module)
    with torch.autograd.profiler.record_function(label):
        y = linear(x)
    return apply_routes(y, x, (linear,), routes, lora, lora_tiles, residual)


class Packed:
    """Read-only copies. The oracle and checkpoint wrappers retain their own storage."""
    def __init__(self, *modules):
        bases = [base(m) for m in modules]
        self.modules = bases
        self.sizes = [m.out_features for m in bases]
        self.weight = torch.cat([m.weight.detach() for m in bases])
        self.bias = None
        if any(m.bias is not None for m in bases):
            self.bias = torch.cat([m.bias if m.bias is not None else m.weight.new_zeros(m.out_features)
                                   for m in bases])

    def __call__(self, x, routes, label, lora, lora_tiles=None):
        with torch.autograd.profiler.record_function(label):
            y = F.linear(x, self.weight, self.bias)
        return apply_routes(y, x, self.modules, routes, lora, lora_tiles).split(self.sizes, dim=-1)


class Decoder:
    def __init__(self, lm, read_tile=4, lora="reference", lora_tiles=None):
        if lora not in ("reference", "grouped"):
            raise ValueError("expected reference or grouped LoRA")
        self.lora = lora
        self.lora_tiles = dict(lora_tiles or {})
        self.lm, self.read_tile = lm, read_tile
        # Keep the projection locations: callers can install wrappers after opening the stream.
        self.targets = [(m, name) for m in lm.modules() for name in TARGETS
                        if isinstance(getattr(m, name, None), (LoRALinear, torch.nn.Linear))]
        self.input, self.mlp = [], []
        for kind, layer in zip(lm.config.layer_types, lm.layers):
            if kind == "full_attention":
                a = layer.self_attn
                self.input.append(Packed(a.q_proj, a.k_proj, a.v_proj))
            else:
                m = layer.linear_attn
                self.input.append(Packed(m.in_proj_qkv, m.in_proj_z, m.in_proj_b, m.in_proj_a))
            self.mlp.append(Packed(layer.mlp.gate_proj, layer.mlp.up_proj))

    def adapter(self, enabled):
        if not enabled:
            return {}
        modules = (getattr(m, name) for m, name in self.targets)
        return {m.base: (m.A, m.B, m.s, None) for m in modules if isinstance(m, LoRALinear)}

    def bank_weights(self, params, scale=None):
        params = tuple(params)
        if len(params) != 2 * len(self.targets):
            raise ValueError("expected one A/B pair per LoRA target in checkpoint order")
        weights = {}
        for i, (parent, name) in enumerate(self.targets):
            wrapper = getattr(parent, name)
            linear = base(wrapper)
            a, b = params[2*i:2*i+2]
            s = getattr(wrapper, "s", None) if scale is None else scale
            if (a.ndim != 2 or b.ndim != 2 or a.size(0) == 0
                    or (a.size(1), b.size(0), b.size(1)) != (linear.in_features, linear.out_features, a.size(0))
                    or a.dtype != b.dtype or a.dtype not in (torch.float32, torch.float16, torch.bfloat16)
                    or (scale is None and a.size(0) != getattr(wrapper, "r", None))
                    or s is None or not math.isfinite(s)):
                raise ValueError(f"incompatible LoRA pair or scale at {name}")
            a, b = (v.detach().to(device=linear.weight.device, copy=True, memory_format=torch.contiguous_format)
                    for v in (a, b))
            packed = None
            if self.lora == "grouped" and grouped_lora.supported(linear.weight, a, b):
                pb = grouped_lora.pack_b(b) if getattr(self, "lora_tiles", None) else b
                packed = grouped_lora.pack_a(a), pb
            weights[linear] = (a, b, float(s), packed)
        return MappingProxyType(weights)

    def packed(self, x, pos, old, post, widths, append, owners, regions, frame, spec, indices,
               kv, states, tap, routes, backend, tuning, read_counts=()):
        """regions contains (block count, padded stride). Only the first can write."""
        if bounds.ENABLED:
            cfg = self.lm.config
            bounds.packed(x, old, post, widths, append, owners, regions, frame, spec, indices, kv, states, routes,
                          cfg.num_attention_heads // cfg.num_key_value_heads)
        rope = self.lm.rotary_emb(x[None], pos)
        scratch = (kernels.prefix_scratch(x, next(iter(kv.values()))[0], old.numel(), spec, indices, widths)
                   if backend == "gather" else None)
        if isinstance(tuning, OwnerConfig):
            from glance_stream.engine.owner_prefix import prepare
            cfg = self.lm.config
            scratch = prepare(next(iter(kv.values()))[0], old, widths, spec, indices, tuning,
                              cfg.num_attention_heads // cfg.num_key_value_heads)
        masks = kernels.triangle_masks(regions, self.lm.config.num_attention_heads //
                                       self.lm.config.num_key_value_heads, x.device)
        pending = []
        lora_tiles = self.lora_tiles if read_counts else None
        for li, layer in enumerate(self.lm.layers):
            tap(li, x)
            h = layer.input_layernorm(x)
            parts = self.input[li](h, routes, "base.input", self.lora, lora_tiles)
            if li in kv:
                a = layer.self_attn
                qg, k, v = parts
                q, gate = qg.view(len(x), -1, 2 * a.head_dim).chunk(2, -1)
                q = a.q_norm(q).transpose(0, 1)[None]
                k = a.k_norm(k.reshape(len(x), -1, a.head_dim)).transpose(0, 1)[None]
                q, k = apply_rotary_pos_emb(q, k, *rope)
                q, k = q[0].transpose(0, 1), k[0].transpose(0, 1)
                v = v.reshape_as(k)
                if frame:
                    b, length = regions[0]
                    for slab, local in zip(kv[li], (k, v)):
                        slab.index_copy_(2, append, local[:b * length].unflatten(0, (b, length)).transpose(1, 2))
                with torch.autograd.profiler.record_function("step.attention"):
                    o = kernels.step_attention(q, k, v, kv[li], old, post, widths, regions, spec, indices,
                                               scratch, tuning, a.scaling, masks)
                o = o * gate.reshape_as(o).sigmoid()
                output = a.o_proj
            else:
                m = layer.linear_attn
                mixed, z, beta, decay = parts
                decay = -m.A_log.float().exp() * F.softplus(decay.float() + m.dt_bias)
                with torch.autograd.profiler.record_function("gdn"):
                    o, terminal = self.continuations(m, (mixed, z, beta.sigmoid(), decay), states[li],
                                                     regions, frame, owners, read_counts)
                pending.extend(terminal)
                output = m.out_proj
            x = project(output, o, routes, "base.output", self.lora, lora_tiles, x)
            h = layer.post_attention_layernorm(x)
            gate, up = self.mlp[li](h, routes, "base.gate_up", self.lora, lora_tiles)
            x = project(layer.mlp.down_proj, layer.mlp.act_fn(gate) * up, routes, "base.down",
                        self.lora, lora_tiles, x)
        return x, tuple(pending)

    def continuations(self, m, projections, state_pair, regions, frame, owners, read_counts=()):
        seed, tail = state_pair
        terminal = ()
        tiles, start, read = [], 0, 0
        repeated_seed = None
        for region, (count, length) in enumerate(regions):
            end = start + count * length
            block = [t[start:end].unflatten(0, (count, length)) for t in projections]
            start = end
            if frame and region == 0:
                o, seed, tail = kernels.continuation(m, *block, seed, tail, True)
                terminal = seed, tail
                tiles.append(o.flatten(0, 1))
            else:
                live = read_counts[region-1] if read_counts else count
                for i in range(0, live, self.read_tile):
                    # Retain the original batch shape even for a partly live tile.
                    n = min(count - i, self.read_tile)
                    if read_counts and len(seed) == 1:
                        if n == 1:
                            state = seed
                        else:
                            if repeated_seed is None or len(repeated_seed) < n:
                                repeated_seed = seed.expand(n, -1, -1, -1).contiguous()
                            state = repeated_seed[:n]
                        hist = tail.expand(n, -1, -1)
                    elif len(seed) == n == 1:
                        state, hist = seed, tail
                    else:
                        src = owners[read+i:read+i+n]
                        state, hist = seed.index_select(0, src), tail.index_select(0, src)
                    with torch.autograd.profiler.record_function("gdn.read"):
                        o, _, _ = kernels.continuation(m, *(t[i:i+n] for t in block), state, hist, False,
                                                       pack_read=bool(read_counts))
                    tiles.append(o.flatten(0, 1))
                # Base GEMMs and full attention keep their original shapes. Entirely
                # unrequested GDN branches are independent of every published row.
                consumed = min(count, ((live + self.read_tile - 1) // self.read_tile) * self.read_tile)
                if consumed < count:
                    tiles.append(o.new_zeros(((count-consumed)*length, o.size(-1))))
                read += count
        return torch.cat(tiles) if len(tiles) > 1 else tiles[0], terminal
