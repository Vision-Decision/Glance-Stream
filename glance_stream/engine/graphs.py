"""Bounded, per-stream CUDA graph registry for the decoder. All arguments are tensors.

The decoder may fill reserved KV, but publication and recurrent-state commit happen
in the caller. An eager warm-up or a capture therefore never advances a stream.
"""
from collections import Counter
import torch
from torch.utils._pytree import tree_map_only


class Graphs:
    def __init__(self, limit=8, byte_limit=8 << 30):
        self.limit, self.byte_limit = limit, byte_limit
        self.sizes = {}
        self.entries = {}
        self.warmed = set()
        self.failed = {}
        self.counts = Counter()

    def clear(self):
        self.entries.clear()
        self.sizes.clear()
        self.warmed.clear()
        self.failed.clear()

    def _evict(self):
        oldest = next(iter(self.entries))
        del self.entries[oldest], self.sizes[oldest]
        self.counts["evict"] += 1

    def run(self, key, fn, args, *, own=False):
        def owned(result):
            if callable(own):
                return own(result)
            return tree_map_only(torch.Tensor, torch.clone, result) if own else result
        entry = self.entries.get(key)
        if entry is not None:
            self.entries[key] = self.entries.pop(key)  # LRU across moving window buckets
            graph, stage, result, _ = entry
            with torch.autograd.profiler.record_function("graph.stage"):
                for buf, arg in zip(stage, args):
                    buf.copy_(arg)
            with torch.autograd.profiler.record_function("graph.replay"):
                graph.replay()
            self.counts["replay"] += 1
            return owned(result)
        if self.limit <= 0 or self.byte_limit <= 0 or key in self.failed:
            self.counts["eager"] += 1
            return fn(*args)
        if key not in self.warmed:
            # Bound admission bookkeeping as well as graph-private memory.
            if len(self.warmed) >= self.limit * 8:
                self.warmed.clear()
            self.warmed.add(key)
            self.counts["warm"] += 1
            return fn(*args)
        if len(self.entries) >= self.limit:
            self._evict()
        error = None
        try:
            stage = tuple(a.clone(memory_format=torch.contiguous_format) for a in args)
            reserved = torch.cuda.memory_reserved()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                result = fn(*stage)
            pool = graph.pool()
            private = sum(seg["total_size"] for seg in torch.cuda.memory_snapshot()
                          if tuple(seg.get("segment_pool_id") or ()) == pool)
            size = sum(a.nbytes for a in stage) + (private or max(0, torch.cuda.memory_reserved()-reserved))
            if size > self.byte_limit:
                error = f"graph requires {size} bytes, budget {self.byte_limit}"
            else:
                while self.entries and size + sum(self.sizes.values()) > self.byte_limit:
                    self._evict()
        except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
            error = str(exc)
        if error is not None:
            self.counts["capture_failed"] += 1
            if len(self.failed) < self.limit * 8:
                self.failed[key] = error
            # Release a partial capture before allocating fallback scratch.
            graph = stage = result = None
            return fn(*args)
        self.sizes[key] = size
        # Retain fn: it owns the exact adapter parameters referenced in this graph.
        self.entries[key] = graph, stage, result, fn
        self.counts["capture"] += 1
        graph.replay()
        return owned(result)
