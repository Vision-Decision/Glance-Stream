"""Inference-only Qwen3.5 video stream. Debug snapshots use ordinary HF cache layers."""
import copy
from contextlib import contextmanager
import math
from functools import lru_cache, wraps
from operator import index
import threading
import weakref

import torch
import torch.nn.functional as F

from glance_stream.engine import bounds
from glance_stream.engine.graphs import Graphs
from glance_stream.engine.layers import Bank, Decoder, base_view
from glance_stream.engine.kernels import pack_step_embeddings, prefix_schedule, rotate_slots
from glance_stream.engine.state import OwnerConfig, Region, Snapshot, Window, plan_write, shared_config, shared_window, window_bucket
from glance_stream.model import heads
from glance_stream.model.box import BoxHeads, BoxReadout, patch_centers
from glance_stream.model.stream import Stream, patch_rows, yes_no_ids, p_yes


def ordered(method):
    """Serialize publication and establish dependencies with the caller's CUDA stream."""
    @wraps(method)
    @torch.no_grad()
    def run(self, *args, **kwargs):
        with self.lock:
            producer = torch.cuda.current_stream(self.dev)
            same = producer == self.service
            if not same:
                self.service.wait_stream(producer)
            try:
                with torch.cuda.stream(self.service):
                    result = method(self, *args, **kwargs)
                    if not same:
                        from torch.utils._pytree import tree_map_only
                        tree_map_only(torch.Tensor, lambda t: t.record_stream(producer), result)
                    return result
            finally:
                if not same:
                    producer.wait_stream(self.service)
    return run


def directory(runs, device):
    # Expand compressed runs on device. One index is shared by all layers.
    count = sum(len(r.positions) for r in runs)
    if not count:
        return torch.empty(0, dtype=torch.long, device=device)
    lengths, starts = [], []
    for run in runs:
        if starts and starts[-1] + lengths[-1] == run.start:
            lengths[-1] += len(run.positions)
        elif run.positions:
            starts.append(run.start)
            lengths.append(len(run.positions))
    if len(starts) == 1:
        return torch.arange(starts[0], starts[0] + count, device=device)
    offsets, logical = [], 0
    for start, length in zip(starts, lengths):
        offsets.append(start - logical)
        logical += length
    desc = torch.tensor([offsets, lengths], device=device)
    return torch.arange(count, device=device) + desc[0].repeat_interleave(desc[1], output_size=count)


class StaticStream:
    question_ids = Stream.question_ids
    @ordered
    def encode(self, pairs):
        if self.fast_vision is None:
            return Stream.encode(self, pairs)
        px, (ph, pw) = patch_rows(pairs.to(self.dev, torch.float32))
        emb = self.fast_vision(px.to(self.dtype), len(pairs), ph, pw)
        return emb.reshape(len(pairs), ph*pw//4, -1), (ph//2, pw//2)

    def __init__(self, model, tok, *args, attention="gather", graph_limit=8, read_tile=4,
                 lora="reference", vision="hf", shared_table=None, prefix_split=512, prefix_m=16, prefix_n=32,
                 graph_memory_mb=8192, **kwargs):
        if attention not in ("gather", "indexed", "shared") or read_tile < 1:
            raise ValueError("expected gather/indexed/shared attention and a positive read tile")
        if prefix_m not in (16, 32, 64) or prefix_n not in (16, 32, 64, 128) or prefix_split <= 0 or prefix_split % prefix_n:
            raise ValueError("prefix split must be a positive multiple of N; unsupported Triton tile")
        if vision not in ("hf", "fast"):
            raise ValueError("expected hf or fast vision")
        self.tuning = (prefix_split, prefix_m, prefix_n)
        self.shared_table = dict(shared_table or {})
        self._cached_schedule = lru_cache(maxsize=32)(prefix_schedule)
        self.banks = {}
        self.head_storage = weakref.WeakKeyDictionary()
        self.bank_generation = 0
        self.attention = attention
        self.optimize_readout = True
        self._box_centers = None
        self.graphs = Graphs(graph_limit, graph_memory_mb * 1024**2)
        self.lock = threading.RLock()
        self.service = torch.cuda.current_stream(next(model.parameters()).device)
        # Bind the same model, tokenizer, policy, and header configuration as Stream.
        Stream.__init__(self, model, tok, *args, **kwargs)
        self.fast_vision = None
        if vision == "fast":
            from glance_stream.engine.vision import FastVision
            self.fast_vision = FastVision(self.visual, graph_limit, graph_memory_mb)
        self.decoder = Decoder(self.lm, read_tile, lora=lora, lora_tiles=self.shared_table.get("lora_tiles"))
        y, n = yes_no_ids(tok)
        ids = sorted(set(y + n))
        self.yn_map = tuple(torch.tensor([ids.index(i) for i in group], device=self.dev) for group in (y, n))
        self.yn_weight = self.m.lm_head.weight[ids].detach()
        bias = getattr(self.m.lm_head, "bias", None)
        self.yn_bias = None if bias is None else bias[ids].detach()

    @ordered
    def reset(self, zero_state=False, batch=1):
        if batch < 1:
            raise ValueError("reset requires at least one row")
        seed = Stream.__new__(Stream)
        seed.__dict__.update(vars(self))
        seed.lm = base_view(self.lm)
        Stream.reset(seed, zero_state, batch)
        # A sink-only snapshot has no visual rows from the previous batch.
        self.import_hf(Snapshot(seed.cache, Window.sink(seed.pos), batch, None, None))

    def _publish(self, window, live):
        self.window, self.live = window, live
        for name in ("pos", "now", "last_ts", "n_written", "rebases"):
            setattr(self, name, getattr(window, name))

    def kv_len(self):
        return self.live.numel()

    @ordered
    def import_hf(self, snapshot):
        """Install an independent debug/fixture snapshot; inputs are in chronological order."""
        w = snapshot.window
        # Repack holes on import; preserve partial runs, timestamps and logical positions.
        from dataclasses import replace
        start, runs = 0, []
        for r in w.runs:
            runs.append(replace(r, start=start))
            start += len(r.positions)
        w = replace(w, runs=tuple(runs))
        cap = 1 << max(7, (start - 1).bit_length())
        kv, states = {}, {}
        template = copy.copy(snapshot.cache)
        template.layers = []
        if len(snapshot.cache.layers) != len(self.lm.layers):
            raise ValueError("HF snapshot has a different decoder depth")
        for li, src in enumerate(snapshot.cache.layers):
            layer = copy.copy(src)
            template.layers.append(layer)
            if li in self.full_layers:
                expected = (snapshot.batch, self.lm.config.num_key_value_heads, start,
                            self.lm.layers[li].self_attn.head_dim)
                if src.keys.shape != expected or src.values.shape != expected:
                    raise ValueError(f"HF KV dimensions disagree with layer {li} or its window")
                k, v = [x.new_empty((x.size(0), x.size(1), cap, x.size(3))) for x in (src.keys, src.values)]
                k[:, :, :start].copy_(src.keys)
                v[:, :, :start].copy_(src.values)
                kv[li] = k, v
                layer.keys = layer.values = None
            else:
                m = self.lm.layers[li].linear_attn
                seed, tail = src.recurrent_states[0], src.conv_states[0]
                if (seed.shape != (snapshot.batch, m.num_v_heads, m.head_k_dim, m.head_v_dim)
                        or seed.dtype != torch.float32 or tail.shape[:2] != (snapshot.batch, m.conv1d.in_channels)
                        or tail.size(-1) not in (m.conv_kernel_size-1, m.conv_kernel_size)):
                    raise ValueError(f"unexpected HF recurrent or convolution state layout at layer {li}")
                states[li] = seed.contiguous().clone(), tail.contiguous().clone()
                for name in ("conv_states", "recurrent_states", "is_conv_states_initialized",
                             "is_recurrent_states_initialized", "has_previous_state", "conv_kernel_size"):
                    if hasattr(src, name):
                        setattr(layer, name, dict(getattr(src, name)))
                layer.conv_states[0] = layer.recurrent_states[0] = None
        live = torch.arange(start, device=self.dev)
        vis = None if snapshot.last_vis is None else snapshot.last_vis.clone()
        self.graphs.clear()
        self.kv, self.states, self.template, self.capacity = kv, states, template, cap
        self.last_vis = vis
        self.last_grid, self.B = snapshot.last_grid, snapshot.batch
        self._publish(w, live)

    @ordered
    def export_hf(self):
        cache = copy.copy(self.template)
        cache.layers = []
        for li, src in enumerate(self.template.layers):
            layer = copy.copy(src)
            cache.layers.append(layer)
            if li in self.kv:
                k, v = self.kv[li]
                layer.keys, layer.values = k.index_select(2, self.live), v.index_select(2, self.live)
            else:
                for name in ("conv_states", "recurrent_states", "is_conv_states_initialized",
                             "is_recurrent_states_initialized", "has_previous_state", "conv_kernel_size"):
                    if hasattr(src, name):
                        setattr(layer, name, dict(getattr(src, name)))
                state, tail = self.states[li]
                layer.recurrent_states[0], layer.conv_states[0] = state.clone(), tail.clone()
        return Snapshot(cache, self.window, self.B, None if self.last_vis is None else self.last_vis.clone(),
                        self.last_grid)

    def _grow(self, cap):
        if cap <= self.capacity:
            return
        kv = {}
        for li, pair in self.kv.items():
            k, v = (x.new_empty((x.size(0), x.size(1), cap, x.size(3))) for x in pair)
            for dst, src in zip((k, v), pair):
                dst[:, :, :self.capacity].copy_(src)
            kv[li] = k, v
        self.graphs.clear()
        self.kv, self.capacity = kv, cap

    @ordered
    def drop(self, keep):
        rows = torch.as_tensor(keep, device=self.dev, dtype=torch.long)
        # Cold path; physical selection also gives duplicated rows independent storage.
        kv = {i: tuple(t.index_select(0, rows) for t in pair) for i, pair in self.kv.items()}
        states = {i: tuple(t.index_select(0, rows) for t in pair) for i, pair in self.states.items()}
        vis = None if self.last_vis is None else self.last_vis.index_select(0, rows)
        self.graphs.clear()
        self.kv, self.states, self.last_vis, self.B = kv, states, vis, len(rows)

    @ordered
    def register_bank(self, name, params, *, scale=None):
        """Own a checkpoint-ordered bank. Replacement invalidates its previous handle."""
        weights = self.decoder.bank_weights(params, scale)
        handle = Bank(name, self.bank_generation + 1)
        self.graphs.clear()
        self.banks[name] = handle, weights
        self.bank_generation = handle.generation
        return handle

    def _weights(self, handle):
        if handle is None:
            return {}
        if not isinstance(handle, Bank) or self.banks.get(handle.name, (None,))[0] is not handle:
            raise ValueError("foreign or replaced adapter bank")
        return self.banks[handle.name][1]

    def _schedule(self, segments, backend, width, fixed_spans=True):
        cfg = self.lm.config
        repeat = cfg.num_attention_heads // cfg.num_key_value_heads
        tuning = self.tuning
        if backend == "shared":
            counts = {}
            for start, end, owner, _ in segments:
                counts[owner] = counts.get(owner, 0) + (end - start) * repeat
            tuning = shared_config(max(counts.values()), width, self.shared_table, dual=any(view for _, _, _, view in segments))
        owner = isinstance(tuning, OwnerConfig)
        spec, indices = self._cached_schedule(tuple(segments), repeat, 0 if owner else tuning[1],
                                              self.dev, "owner" if owner else backend, fixed_spans)
        return spec, indices, tuning

    def _union_directory(self, plan, live):
        expired, fresh = shared_window(self.window, plan)
        union = torch.cat((live, directory(expired, self.dev))) if expired else live
        cuts = torch.tensor([len(live)-fresh, len(live), len(union)], device=self.dev, dtype=torch.int32)
        return F.pad(union, (0, window_bucket(len(union)) - len(union))), cuts

    def _directories(self, post=None):
        width = window_bucket(len(self.live), len(post) if post is not None else 0)
        old = F.pad(self.live, (0, width - len(self.live)))
        if post is None:
            return old, old, torch.tensor([len(self.live)], device=self.dev, dtype=torch.int32)
        return old, F.pad(post, (0, width - len(post))), torch.tensor(
            [len(self.live), len(post)], device=self.dev, dtype=torch.int32)

    def _run(self, x, pos, rows, layers, op, attention, append=None, *,
             finish=None, head_args=(), signature=(), eager=False, adapter="active"):
        backend = self.attention if attention is None else attention
        if backend not in ("gather", "indexed", "shared"):
            raise ValueError("unknown attention backend")
        n, length = x.shape[:2]
        rows = [0] * n if rows is None else list(map(index, rows))
        if len(rows) != n or any(row < 0 or row >= self.B for row in rows):
            raise ValueError("one valid stream owner is required per block")
        write = append is not None
        owners = torch.tensor(rows, dtype=torch.long, device=self.dev)
        depth = len(self.lm.layers)
        if any(i < -depth - 1 or i > depth for i in layers):
            raise IndexError("hidden layer outside HF tuple")
        bank = (self.decoder.adapter(not write and heads.LORA["on"]) if adapter == "active"
                else self._weights(adapter))
        old, _, widths = self._directories()
        spec, indices, tuning = self._schedule([(i*length, (i+1)*length, row, 0)
                                        for i, row in enumerate(rows)], backend, len(old), fixed_spans=write)
        key = (op, backend, tuning, self.decoder.lora, tuple(x.shape), len(old), tuple(sorted(layers)), spec,
               tuple(t.shape for t in indices), tuple((id(a), id(b), s) for a, b, s, _ in bank.values()), signature)
        staged_indices = () if write else indices
        nindices = len(staged_indices)
        def forward(e, p, slots, w, dst, src, *extra):
            hidden = {}
            def tap(li, h):
                for layer in layers:
                    if layer % (depth + 1) == li:
                        hidden[layer] = h.view(n, length, -1)
            h, pending = self.decoder.packed(e.flatten(0, 1), p.reshape(3, 1, -1), slots, slots, w,
                    dst, src, ((n, length),), write, spec, indices if write else extra[:nindices], self.kv, self.states,
                    tap, ((slice(None), bank),) if bank else (), backend, tuning)
            if any(layer % (depth + 1) == depth for layer in layers):
                tap(depth, self.lm.norm(h))
            return (hidden, pending) if finish is None else finish(hidden, *extra[nindices:])
        args = (x, pos, old, widths, append if write else owners[:0], owners, *staged_indices, *head_args)
        return forward(*args) if eager else self.graphs.run(key, forward, args, own=finish is not None)

    def _frame(self, vis, times, grid):
        if len(grid) != 2 or min(map(index, grid)) <= 0:
            raise ValueError("expected a positive two-dimensional visual grid")
        times = tuple(map(float, times))
        batched = vis.ndim == 4
        v = vis if batched else vis.unsqueeze(0)
        if (v.ndim != 4 or v.size(0) not in (1, self.B) or v.size(1) != len(times)
                or v.size(2) != grid[0] * grid[1] or v.size(3) != self.emb.weight.size(1)):
            raise ValueError("visual embedding shape does not match the batch, times and grid")
        if not all(math.isfinite(t) for t in times):
            raise ValueError("expected finite video times")
        with torch.autograd.profiler.record_function("write.plan"):
            plan = plan_write(self.window, times, grid, lambda s: self.tok.encode(s, add_special_tokens=False),
                              full_s=self.full_s, sub_s=self.sub_s, sub_fps=self.sub_fps,
                              ts_every=self.ts_every, max_pos=self.max_pos, keep_all=self.keep_all,
                              capacity=self.capacity)
        x = self.emb.weight.new_empty((self.B, len(plan.axes[0]), v.size(-1)))
        for i, ((a, b), stamp) in enumerate(zip(plan.spans, plan.stamps)):
            if stamp:
                x[:, a-1-len(stamp):a-1] = self.emb(torch.tensor(stamp, device=self.dev))
            x[:, a-1] = self.emb.weight[self.ids["<|vision_start|>"]]
            x[:, a:b] = v[:, i]
            x[:, b] = self.emb.weight[self.ids["<|vision_end|>"]]
        p = torch.tensor(plan.axes, device=self.dev)[:, None].expand(-1, self.B, -1)
        with torch.autograd.profiler.record_function("write.directory"):
            dst = directory(plan.append, self.dev)
            if len(plan.window.runs) == len(self.window.runs) + len(plan.append):
                live = torch.cat((self.live, dst))
            else:
                live = directory(plan.window.runs, self.dev)
        return plan, x, p, dst, live

    @contextmanager
    def _transaction(self, plan, attention):
        self._grow(plan.capacity)
        # A failed forward leaves the old published state usable, including rounded keys.
        undo = []
        if plan.shift:
            sel = directory([r for r in self.window.runs if r.time >= 0], self.dev)
            keys = [pair[0] for pair in self.kv.values()]
            if bounds.ENABLED:
                bounds.indices(sel, min(k.size(2) for k in keys), "rebase slab slot")
            rd = 2 * self.inv_freq.numel()
            undo = [(k[..., :rd], k[..., :rd].index_select(2, sel)) for k in keys]
        try:
            if plan.shift:
                rotate_slots(keys, sel, self.inv_freq, plan.shift, attention or self.attention)
            yield
        except Exception:
            for key, saved in undo:
                key.index_copy_(2, sel, saved)
            raise

    def _commit(self, plan, live, pending, visual, grid):
        # Group same-dtype recurrent states and convolution tails. All captured
        # graphs retain their original state addresses, including on head failure.
        if len(pending) != 2 * len(self.states):
            raise ValueError("incomplete decoder state at commit")
        for i in (0, 1):
            torch._foreach_copy_([pair[i] for pair in self.states.values()], pending[i::2])
        if visual is not None:
            self.last_vis, self.last_grid = visual, tuple(grid)
        self._publish(plan.window, live)

    @ordered
    def write(self, vis, times, grid, return_feats=False, *, attention=None):
        if attention is not None and attention not in ("gather", "indexed", "shared"):
            raise ValueError("unknown attention backend")
        plan, x, p, dst, live = self._frame(vis, times, grid)
        batched = vis.ndim == 4
        layers = self.readout_layers or (len(self.lm.layers),)
        taps = set(layers) if return_feats else set()
        if self.vis_layer is not None:
            taps.add(self.vis_layer)
        visual = None
        with self._transaction(plan, attention):
            hs, pending = self._run(x, p, list(range(self.B)), taps, "write", attention, dst)
            features = None
            if return_feats:
                features = {l: torch.stack([hs[l][:, a:b].float().mean(1) for a, b in plan.spans]) for l in layers}
                if not batched:
                    features = {l: h[:, 0] for l, h in features.items()}
            if self.vis_layer is not None:
                a, b = plan.spans[-1]
                visual = hs[self.vis_layer][:, a:b].clone()

            with torch.autograd.profiler.record_function("step.commit"):
                self._commit(plan, live, pending, visual, grid)
        return features

    def _block(self, questions, latent=None):
        ids = [self.question_ids(q) if isinstance(q, str) else list(q) for q in questions]
        if not ids or not all(ids):
            raise ValueError("expected nonempty question blocks")
        lengths = [len(q) for q in ids]
        extra = len(latent) if latent is not None else 0
        length = max(lengths) + extra
        pad = torch.tensor([q + [0] * (length - len(q)) for q in ids], device=self.dev)
        li = torch.tensor(lengths, device=self.dev)
        ar = torch.arange(len(ids), device=self.dev)
        offsets = torch.arange(length, device=self.dev)
        e = self.emb(pad).masked_fill((offsets >= li[:, None]).unsqueeze(-1), 0)
        idx = li[:, None] + offsets[:extra]
        if extra:
            e[ar[:, None], idx] = latent.to(e.dtype)
        p = (self.pos + offsets).view(1, 1, -1).expand(3, len(ids), -1)
        return e, p, ar, li, idx

    def _first(self, questions, rows, attention, score, bank):
        e, p, _, li, _ = self._block(questions)
        def finish(hs, last):
            h = hs[-1][torch.arange(len(last), device=last.device), last]
            if score:
                return p_yes(F.linear(h, self.yn_weight, self.yn_bias).float(), *self.yn_map)
            return self.m.lm_head(h).float()
        return self._run(e, p, rows, (-1,), "yesno" if score else "logits", attention,
                         finish=finish, head_args=(li - 1,), adapter=bank)

    @ordered
    def first_logits(self, questions, rows=None, *, attention=None, bank="active"):
        return self._first(questions, rows, attention, False, bank)

    @ordered
    def score_yes(self, questions, rows=None, *, attention=None, bank="active"):
        return self._first(questions, rows, attention, True, bank)

    @ordered
    def read_s1(self, head, qids, opt_ends=None, rows=None, *, attention=None):
        e, p, ar, _, idx = self._block(qids, head.latent)
        ends = opt_ends if opt_ends is not None else ()
        sizes = tuple(len(v) if v is not None else 0 for v in ends)
        flat = torch.tensor([i for row in ends if row is not None for i in row], device=self.dev,
                            dtype=torch.long)
        def finish(hs, rows_idx, latents, indices):
            options = indices.split(sizes) if any(sizes) else None
            return head.apply_heads(hs, rows_idx, latents, options)
        return self._run(e, p, rows, head.layers, "s1", attention, finish=finish,
                         head_args=(ar, idx, flat), signature=(id(head), sizes))

    @ordered
    def read_box(self, head, qids, rows=None, sel_idx=None, hires=None, *, attention=None, bank="active"):
        selected = () if sel_idx is None else (torch.as_tensor(sel_idx, device=self.dev, dtype=torch.long),)
        layer = head.layer
        if hires is None:
            e, p, ar, li, _ = self._block(qids, head.latent)
            owners = [0] * len(qids) if rows is None else rows
            vis = self.last_vis.index_select(0, torch.as_tensor(owners, device=self.dev, dtype=torch.long))
            grid = self.last_grid
            def finish(hs, row, last, visual, *selection):
                return head.apply_heads(hs[layer][row, last], visual, grid, selection[0] if selection else None)
            args = (ar, li, vis, *selected)
            op = "b1"
        else:
            e, p, grid = head.hires_block(self, qids, hires)
            def finish(hs, *selection):
                h = hs[layer][0]
                return head.apply_heads(h[-1:], h[None, 2:2 + grid[0] * grid[1]], grid,
                                        selection[0] if selection else None)
            args, op = selected, "b2"
        return self._run(e, p, rows, (layer,), op, attention, finish=finish, head_args=args,
                         signature=(id(head), grid, bool(selected)), eager=bool(head.box._forward_pre_hooks), adapter=bank)

    def _question_region(self, questions, owners, latent, stage_ids=False):
        ids = [self.question_ids(q) if isinstance(q, str) else list(map(index, q)) for q in questions]
        if any(t < 0 or t >= self.emb.num_embeddings for row in ids for t in row):
            raise ValueError("question token outside vocabulary")
        layout = Region.questions(self.B, list(map(len, ids)), owners, box=latent is not None)
        if not layout.rows:
            return layout, None
        padded, latent_rows = [], []
        for i in layout.order:
            seq = ids[i] if i >= 0 else []
            latent_rows.append(len(seq))
            padded.append(seq + [0] * (layout.stride - len(seq)))
            if stage_ids and latent is not None:
                padded[-1][len(seq)] = -1
        if stage_ids:
            return layout, [token for row in padded for token in row]
        ids = torch.tensor(padded, device=self.dev)
        if bounds.ENABLED:
            bounds.indices(ids, self.emb.num_embeddings, "region embedding index")
        e = self.emb(ids)
        if latent is not None:
            offsets = torch.tensor(latent_rows, device=self.dev)
            if bounds.ENABLED:
                bounds.indices(offsets, layout.stride, "box latent offset")
            e[torch.arange(len(padded), device=self.dev), offsets] = latent.to(e.dtype)
        return layout, e.flatten(0, 1)

    @ordered
    def step(self, vis, times, grid, *, judgements=(), judgement_rows=None, boxes=(), box_rows=None,
             box_head=None, judgement_bank=None, box_bank=None, full_rows=(), sel_idx=None):
        """Publish one frame and return its post-write first-token and B1 results in caller order."""
        times, grid = tuple(times), tuple(grid)
        if len(times) != 1:
            raise ValueError("step requires exactly one frame group")
        depth = len(self.lm.layers)
        layer = self.vis_layer
        if layer is not None and not -depth-1 <= layer <= depth:
            raise IndexError("visual tap outside HF hidden tuple")
        if len(boxes) and (box_head is None or layer != box_head.layer
                           or box_head.latent.shape != (1, self.emb.embedding_dim)
                           or box_head.latent.device != self.dev):
            raise ValueError("B1 requires a matching box head and configured visual tap")
        full_rows = tuple(map(index, full_rows))
        if any(i < 0 or i >= len(judgements) for i in full_rows):
            raise IndexError("full_rows outside judgement request list")
        selected = None if sel_idx is None else list(map(index, sel_idx))
        if selected is not None and (len(selected) != len(boxes) or
                                     any(i < 0 or i >= math.prod(grid) for i in selected)):
            raise ValueError("one valid selected patch index is required per box")
        weights = self._weights(judgement_bank), self._weights(box_bank)
        trim = (self.optimize_readout and self.emb.embedding_dim == 2560 and self.B == 1
                and len(judgements) == 16 and len(boxes) == 1 and judgement_bank is not None and box_bank is not None
                and judgement_bank is not box_bank and self.decoder.lora == "grouped" and self.attention == "shared")
        latent_value = box_head.latent if len(boxes) else None
        stage_ids = (trim and type(self.emb) is torch.nn.Embedding and self.emb.max_norm is None
                     and not self.emb._forward_hooks and not self.emb._forward_pre_hooks
                     and (latent_value is None or latent_value.dtype in (self.emb.weight.dtype, torch.float32)))
        with torch.autograd.profiler.record_function("step.readout_inputs"):
            j, je = self._question_region(judgements, judgement_rows, None, stage_ids)
            b, be = self._question_region(boxes, box_rows, latent_value, stage_ids)
        if not j.rows and not b.rows:
            self.write(vis, times, grid)
            return dict(p_yes=self.emb.weight.new_empty(0, dtype=torch.float32),
                        logits=self.emb.weight.new_empty((0, self.m.lm_head.out_features), dtype=torch.float32),
                        greedy=self.emb.weight.new_empty(0, dtype=torch.long), boxes={})
        plan, f, pos, dst, live = self._frame(vis, times, grid)
        batch, length, dim = f.shape
        frame_rows = batch * length
        blocks = [(batch, length)]
        segments = [(i*length, (i+1)*length, i, 0) for i in range(batch)]
        embeddings, axes = [f.flatten(0, 1)], [pos.reshape(3, -1)]
        routes, owners, tokens = [], [], []
        # On a single stream the padded slots form a suffix in each region.
        # Keep all base GEMM shapes, but omit updates private to those slots.
        read_counts = tuple(len(layout.ends) for layout in (j, b) if layout.rows) if trim else ()
        offset = frame_rows
        for layout, embed, bank in zip((j, b), (je, be), weights):
            if embed is None:
                continue
            if stage_ids:
                tokens.extend(embed)
            else:
                embeddings.append(embed)
            count = len(layout.order)
            blocks.append((count, layout.stride))
            axes.append((torch.arange(layout.stride, device=self.dev) + plan.window.pos).repeat(count).expand(3, -1))
            width = layout.per_owner * layout.stride
            for row in range(batch):
                segments.append((offset+row*width, offset+(row+1)*width, row, 1))
                owners.extend([row] * layout.per_owner)
            if bank:
                first = offset
                if routes and routes[-1][1] is bank and routes[-1][0].stop == offset:
                    first = routes.pop()[0].start
                used = len(layout.ends) * layout.stride if trim else layout.rows
                routes.append((slice(first, offset+used), bank))
            offset += layout.rows
        def padded(values, count):
            return [*values, *([0] * (count - len(values)))]
        jcap, bcap = len(j.order), len(b.order)
        box_src = [0] * len(boxes) if box_rows is None else list(map(index, box_rows))
        full_cap = len(full_rows)
        backend = self.attention
        if backend == "shared":
            old, widths = self._union_directory(plan, live)
            post = old[:0]
        else:
            old, post, widths = self._directories(live)
        spec, indices, tuning = self._schedule(segments, backend, len(old))
        metadata = (tokens, owners, padded(j.ends, jcap), padded(b.ends, bcap), padded(box_src, bcap),
                    full_rows, padded(selected or [], bcap))
        sizes = tuple(map(len, metadata))
        packed_ids = torch.tensor([i for field in metadata for i in field], device=self.dev, dtype=torch.long)
        args = (embeddings[0] if stage_ids else torch.cat(embeddings), torch.cat(axes, 1)[:, None],
                old, post, widths, dst, packed_ids,
                latent_value if stage_ids and bcap else f.new_empty(0))
        tap_layer = None if layer is None else layer % (depth+1)
        start, end = plan.spans[0]
        box_start = frame_rows + j.rows
        has_selection = selected is not None
        head_storage = (tuple(p.data_ptr() for p in (*box_head.box.parameters(), *box_head.box.buffers()))
                        if bcap else ())
        if bcap:
            if box_head in self.head_storage and self.head_storage[box_head] != head_storage:
                self.graphs.clear()
            self.head_storage[box_head] = head_storage
        centers = None
        if (trim and type(box_head) is BoxReadout and type(box_head.box) is BoxHeads
                and not any(m._forward_pre_hooks or m._forward_hooks for m in box_head.box.modules())):
            centers_key = grid, self.dev, torch.get_default_dtype()
            if self._box_centers is None or self._box_centers[0] != centers_key:
                self._box_centers = centers_key, patch_centers(*grid, self.dev)
            centers = self._box_centers[1]
        embedding = self.emb.weight if stage_ids else None
        embedding_key = (embedding.data_ptr(), embedding.stride(), args[-1].dtype) if stage_ids else None
        key = ("step", backend, grid, tuple(blocks), len(old), tuning, self.decoder.lora, self.decoder.read_tile,
               tap_layer, judgement_bank, box_bank, id(box_head), head_storage, has_selection, full_cap,
               read_counts, embedding_key, None if centers is None else centers.dtype)
        def forward(x, p, old, post, w, dst, metadata, latent_value):
            # The graph key fixes the owner-major layout. Its closure pins the schedule.
            tokens, owners, last, latent, box_src, full, sel = metadata.split(sizes)
            if stage_ids:
                if bounds.ENABLED:
                    torch._assert_async(((tokens >= -int(bool(bcap))) & (tokens < len(embedding))).all(),
                                        "staged readout token or box latent")
                x = pack_step_embeddings(x, tokens, embedding, latent_value)
            if bounds.ENABLED:
                bounds.indices(last, j.rows, "judgement readout offset")
                bounds.indices(latent, b.rows, "box readout offset")
                bounds.indices(box_src, batch, "visual owner")
                bounds.indices(full, jcap, "full-logit gather index")
                bounds.indices(sel, end - start, "selected box patch")
                valid = 0 <= start < end <= length and box_start + b.rows == len(x)
                torch._assert_async(w.new_full((), valid, dtype=torch.bool), "visual or box region")
            visual = hq = None
            def tap(li, h):
                nonlocal visual, hq
                if li == tap_layer:
                    visual = h[:frame_rows].view(batch, length, dim)[:, start:end].clone(memory_format=torch.contiguous_format)
                    if bcap:
                        hq = h[box_start:].index_select(0, latent)
                    if li == depth:
                        visual = self.lm.norm(visual)
                        if bcap:
                            hq = self.lm.norm(hq)
            h, states = self.decoder.packed(x, p, old, post, w, dst, owners, blocks, True,
                    spec, indices, self.kv, self.states, tap, routes, backend, tuning, read_counts)
            tap(depth, h)
            with torch.autograd.profiler.record_function("step.heads"):
                prob = h.new_empty(0, dtype=torch.float32)
                logits = h.new_empty((0, self.m.lm_head.out_features), dtype=torch.float32)
                if jcap:
                    hj = self.lm.norm(h[frame_rows:].index_select(0, last))
                    prob = p_yes(F.linear(hj, self.yn_weight, self.yn_bias).float(), *self.yn_map)
                    if full_cap:
                        logits = self.m.lm_head(hj.index_select(0, full)).float()
                greedy = logits.argmax(-1) if full_cap else full
                boxes_out = {}
                selection = sel if has_selection else None
                if centers is not None:
                    # This is the single-owner, single-box stock head. Its owner index
                    # is always zero and both MLPs consume the identical predicted features.
                    boxes_out = box_head.box(hq, visual, centers, selection, reuse_features=True)
                elif bcap:
                    boxes_out = box_head.apply_heads(hq, visual.index_select(0, box_src), grid, selection)
            return (prob, logits, greedy, boxes_out, visual), states

        def own(result):
            from torch.utils._pytree import tree_map_only
            public, states = result
            return tree_map_only(torch.Tensor, torch.clone, public), states

        with self._transaction(plan, backend):
            hooked = bcap and (box_head.box._forward_pre_hooks or box_head.box._forward_hooks)
            result, states = forward(*args) if hooked else self.graphs.run(key, forward, args, own=own)
            prob, logits, greedy, boxes_out, visual = result
            output = dict(p_yes=prob[:len(judgements)], logits=logits[:len(full_rows)], greedy=greedy[:len(full_rows)],
                          boxes={k: t[:len(boxes)] for k, t in boxes_out.items()})
            with torch.autograd.profiler.record_function("step.commit"):
                self._commit(plan, live, states, visual, grid)
        return output
