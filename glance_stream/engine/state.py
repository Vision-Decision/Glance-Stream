"""Host-owned window planning. Positions, physical addresses and times are independent."""
from dataclasses import dataclass, replace
from itertools import groupby
from math import floor
from struct import Struct
from typing import NamedTuple

_float = Struct("f")


def f32(x):
    return _float.unpack(_float.pack(x))[0]


@dataclass(frozen=True)
class Run:
    start: int
    time: float
    positions: tuple
    offset: int = 0

    @property
    def end(self):
        return self.start + len(self.positions)

    def temporal(self):
        return (p - self.offset for p in self.positions)


@dataclass(frozen=True)
class Window:
    runs: tuple
    pos: int
    now: float = 0.
    last_ts: float = -1e9
    n_written: int = 0
    rebases: int = 0

    @classmethod
    def sink(cls, length):
        return cls((Run(0, -1., tuple(range(length))),), length)

    @classmethod
    def from_metadata(cls, times, positions, **counters):
        if len(times) != len(positions):
            raise ValueError("metadata length mismatch")
        runs = []
        for t, items in groupby(enumerate(times), key=lambda item: item[1]):
            idx = [i for i, _ in items]
            runs.append(Run(idx[0], t, tuple(positions[idx[0]:idx[-1] + 1])))
        return cls(tuple(runs), **counters)

    def shift(self):
        sink_end = max((max(r.positions) - r.offset + 1 for r in self.runs if r.time < 0), default=0)
        oldest = min((min(r.positions) - r.offset for r in self.runs if r.time >= 0), default=self.pos)
        return max(0, oldest - sink_end)

    def addresses(self):
        return [i for run in self.runs for i in range(run.start, run.end)]

    def retain(self, full_s, sub_s, sub_fps):
        def bin_id(r):
            return floor(f32(f32(r.time * f32(sub_fps)) + f32(1e-6)))
        recent, oldest = f32(self.now - full_s), f32(self.now - sub_s)
        if all(r.time < 0 or r.time >= recent for r in self.runs):
            return self
        bins = [bin_id(r) for r in self.runs]
        # HF subtracts the minimum in float32 before converting to integer.
        base = min(bins, default=0)
        bins = [int(f32(b - base)) for b in bins]
        minima = {}
        for b, r in zip(bins, self.runs):
            minima[b] = min(r.time, minima.get(b, r.time))
        return replace(self, runs=tuple(r for b, r in zip(bins, self.runs)
                                        if r.time < 0 or r.time >= recent
                                        or (r.time >= oldest and r.time == minima[b])))


@dataclass(frozen=True)
class WritePlan:
    window: Window
    append: tuple
    shift: int
    stamps: tuple
    axes: tuple
    spans: tuple
    capacity: int


def plan_write(old, times, grid, tokenize, *, full_s, sub_s, sub_fps, ts_every,
               max_pos, keep_all, capacity):
    gh, gw = grid
    if not times or min(gh, gw) <= 0:
        raise ValueError("a write requires at least one group with a positive grid")
    shift = old.shift() if not keep_all and old.pos >= max_pos else 0
    runs = [replace(r, offset=r.offset + shift) if shift and r.time >= 0 else r for r in old.runs]
    # First-fit in the complement of the OLD window. Expiring slots are still readable.
    gaps, end = [], 0
    for run in sorted(runs, key=lambda r: r.start):
        if run.start > end:
            gaps.append((end, run.start))
        end = run.end
    p, last_ts = old.pos - shift, old.last_ts
    axes, stamps, spans, append = [[], [], []], [], [], []
    for t in times:
        ids = ()
        if t - last_ts >= ts_every - 1e-6:
            ids = tuple(tokenize(f"<{t:.1f} seconds>"))
            last_ts = t
        stamps.append(ids)
        v = p + len(ids) + 1
        stop = v + max(grid)
        length = len(ids) + gh * gw + 2
        for j, (a, b) in enumerate(gaps):
            if b - a >= length:
                start = a
                gaps[j] = (a + length, b)
                break
        else:
            start = end
            end += length
        begin = len(axes[0])
        for axis, values in enumerate(axes):
            values.extend(range(p, v))
            values.extend(v + (0 if axis == 0 else (i // gw if axis == 1 else i % gw))
                          for i in range(gh * gw))
            values.append(stop)
        spans.append((begin + len(ids) + 1, len(axes[0]) - 1))
        group = Run(start, f32(t), tuple(axes[0][begin:]))
        append.append(group)
        runs.append(group)
        p = stop + 1
    window = Window(tuple(runs), p, times[-1], last_ts, old.n_written + len(times),
                    old.rebases + int(shift > 0))
    if not keep_all:
        window = window.retain(full_s, sub_s, sub_fps)
    capacity = capacity if capacity >= end else 1 << (end - 1).bit_length()
    return WritePlan(window, tuple(append), shift, tuple(stamps), tuple(axes), tuple(spans), capacity)


@dataclass
class Snapshot:
    """Independent chronological HF cache and matching video state, for debug/fixtures."""
    cache: object
    window: Window
    batch: int
    last_vis: object
    last_grid: object

    @classmethod
    def from_stream(cls, st):
        window = Window.from_metadata(st.meta_t.tolist(), st.meta_pos.tolist(), pos=st.pos,
                                      now=st.now, last_ts=st.last_ts, n_written=st.n_written,
                                      rebases=st.rebases)
        cache = st.fork(st.B, list(range(st.B)))
        for layer in cache.layers:
            vars(layer).pop("update_recurrent_state", None)
        return cls(cache, window, st.B,
                   None if st.last_vis is None else st.last_vis.clone(), st.last_grid)

    def metadata(self, device):
        import torch
        return (torch.tensor([r.time for r in self.window.runs for _ in r.positions],
                             dtype=torch.float32, device=device),
                torch.tensor([p for r in self.window.runs for p in r.temporal()],
                             dtype=torch.long, device=device))

    def restore_stream(self, st):
        """Copy a debug snapshot into a reference Stream; the fixture can be reused."""
        from glance_stream.model.stream import Stream
        import copy
        source = copy.copy(st)
        source.cache = self.cache
        st.cache = Stream.fork(source, self.batch, list(range(self.batch)))
        for layer in st.cache.layers:
            vars(layer).pop("update_recurrent_state", None)
        st.B = self.batch
        for name in ("pos", "now", "last_ts", "n_written", "rebases"):
            setattr(st, name, getattr(self.window, name))
        st.meta_t, st.meta_pos = self.metadata(st.dev)
        st.last_grid = self.last_grid
        st.last_vis = None if self.last_vis is None else self.last_vis.clone()


def window_bucket(*widths):
    return max(1, (max(widths) + 2047) // 2048) * 2048


def prefix_tiles(segments, repeat, tile, backend, fixed_spans=True):
    """Segments are contiguous token ranges (start, end, owner, view) in packed Q."""
    tiles, groups = [], {}
    for start, end, owner, view in segments:
        if backend == "indexed":
            for offset in range(start * repeat, end * repeat, tile):
                tiles.append((offset, end * repeat, owner, view))
        else:
            groups.setdefault((owner, view), []).extend(range(start, end))
    if backend == "indexed":
        return (), (tiles,)
    spec, indices = [], []
    for (owner, view), rows in sorted(groups.items()):
        contiguous = (fixed_spans or len(groups) == 1) and rows[-1] - rows[0] + 1 == len(rows)
        spec.append((owner, view, rows[0] if contiguous else -1))
        indices.append(rows)
    return tuple(spec), tuple(indices)


def prefix_partition(width, rows, heads, dim, requested, tile):
    # Allow at least one split. Bound the other FP32 partials to 64 MiB total.
    max_splits = max(1, min(32, (64 << 20) // (rows * heads * dim * 4)))
    min_chunk = (width + max_splits - 1) // max_splits
    chunk = max(requested, (min_chunk + tile - 1) // tile * tile)
    return chunk, (width + chunk - 1) // chunk


def bucket(value, sizes):
    return next((size for size in sizes if value <= size), value)


@dataclass(frozen=True)
class Region:
    """Equal-stride blocks, owner-major. -1 slots are finite dummy blocks."""
    batch: int
    per_owner: int
    stride: int
    order: tuple
    ends: tuple

    @property
    def rows(self):
        return self.batch * self.per_owner * self.stride

    @classmethod
    def questions(cls, batch, lengths, owners, box=False):
        from operator import index
        owners = [0] * len(lengths) if owners is None else list(map(index, owners))
        if len(owners) != len(lengths) or any(b < 0 or b >= batch for b in owners):
            raise ValueError("one valid owner is required per question")
        if any(n < 1 for n in lengths):
            raise ValueError("empty question")
        grouped = [[] for _ in range(batch)]
        for i, owner in enumerate(owners):
            grouped[owner].append(i)
        count = bucket(max(map(len, grouped)), (0, 1, 4) if box else (0, 1, 8, 21, 32))
        stride = bucket(max(lengths, default=0), (0, 8, 16, 20, 24, 32, 48, 64)) + int(box)
        order = tuple(i for group in grouped for i in group + [-1] * (count - len(group)))
        ends = [0] * len(owners)
        for block, i in enumerate(order):
            if i >= 0:
                ends[i] = block * stride + lengths[i] - 1 + int(box)
        return cls(batch, count, stride, order, tuple(ends))


def fields(snapshot):
    import torch
    for li, layer in enumerate(snapshot.cache.layers):
        for field in ("keys", "values"):
            val = getattr(layer, field, None)
            if isinstance(val, torch.Tensor) and val.numel():
                yield f"{field}.{li}", val
        for attr in ("recurrent_states", "conv_states"):
            for idx, t in getattr(layer, attr, {}).items():
                if t is not None:
                    yield f"{attr}.{li}.{idx}", t
    if snapshot.last_vis is not None:
        yield "last_vis", snapshot.last_vis


def same_window(a, b):
    def unpack(w):
        return (w.pos, w.now, w.last_ts, w.n_written, w.rebases,
                [(r.time, p) for r in w.runs for p in r.temporal()])
    return unpack(a) == unpack(b)


def shared_window(old, plan):
    """The published directory followed by expired slots is the union of both views."""
    retained = {run.start for run in plan.window.runs}
    expired = tuple(run for run in old.runs if run.start not in retained)
    fresh = sum(len(run.positions) for run in plan.append if run.start in retained)
    return expired, fresh


def owner_tiles(segments, repeat, tile):
    """Map owner-local tiles to packed tokens. Store the view in the low bit."""
    by_owner = {}
    for start, end, owner, view in segments:
        by_owner.setdefault(owner, ([], []))[view].extend(2 * i + view for i in range(start, end))
    per_tile = tile // repeat
    owners, tokens = [], []
    for owner, (old, post) in sorted(by_owner.items()):
        count = (len(old) + len(post) + per_tile - 1) // per_tile
        # Distribute frame rows across the passes so a CTA can service both views.
        used = 0
        for i in range(count):
            rows = old[i * len(old) // count:(i+1) * len(old) // count]
            take = min(len(post) - used, per_tile - len(rows))
            rows += post[used:used+take]
            used += take
            tokens.append(rows + [-1] * (per_tile - len(rows)))
            owners.append(owner)
    return owners, tokens


# Initial SM120 candidates. Measured selections can be supplied by the offline tuner.
# The row key includes GQA folding. The column key is the union directory capacity.
def shared_bucket(rows, width):
    # Timestamp tokens can push the padded N16 readout just over 2048 folded rows.
    return (bucket(rows, (64, 128, 256, 512, 1024, 2048, 2112)), window_bucket(width))


class OwnerConfig(NamedTuple):
    splits: int


class SharedConfig(NamedTuple):
    chunk: int
    rows: int
    keys: int
    warps: int


# D=256, one pipeline stage. The two SM120 specializations in the RTX PRO 6000 log
# use 53,248 and 90,112 bytes. A 128 x 64 tile required 114,688 bytes.
SM120_SHARED_LIMIT = 99 * 1024
SHARED_TILES = {(64, 32, 4): 53248, (128, 32, 8): 90112}


class LoraTile(NamedTuple):
    rank_rows: int
    stripes: int
    fused: bool = False


DEFAULT_LORA_TILE = LoraTile(32, 2)
LORA_READOUT_TILES = (LoraTile(16, 1), LoraTile(32, 1), DEFAULT_LORA_TILE, LoraTile(32, 4),
                      LoraTile(16, 8, True))


def checked_shared_config(config):
    if isinstance(config, OwnerConfig):
        if config.splits not in (1, 4, 8):
            raise ValueError("unsupported owner flash split count")
        return config
    cfg = SharedConfig(*config)
    if (cfg[1:] not in SHARED_TILES or cfg.chunk < cfg.keys or cfg.chunk % cfg.keys):
        raise ValueError(f"unsupported SM120 shared attention configuration {cfg}")
    return cfg


def shared_config(rows, width, table, dual=False):
    r, w = shared_bucket(rows, width)
    fallback = table.get("mixed_fallback") if dual and r <= 1024 else None
    # Older externally supplied tables used exact row counts above 2048.
    if 2048 < rows < r and (rows, w) in table:
        return checked_shared_config(table[rows, w])
    return checked_shared_config(table.get((r, w), fallback or (512, 64, 32, 4)))


def load_shared_table(path):
    if path is None:
        return {}
    import json
    from pathlib import Path
    data = json.loads(Path(path).read_text())
    table = {}
    for row in data["selections"]:
        cfg = checked_shared_config(OwnerConfig(row["owner_splits"]) if "owner_splits" in row else row["config"])
        if row.get("shared_bytes", 0) > SM120_SHARED_LIMIT:
            raise ValueError(f"compiled tile exceeds SM120 shared memory: {row}")
        table[row["folded_rows_bucket"], row["width_bucket"]] = cfg
        if data.get("use_mixed_fallback") and row["width_bucket"] == 32768 and row["folded_rows_bucket"] == 1024:
            table["mixed_fallback"] = cfg
    if "lora_selections" in data:
        lora = {}
        for row in data["lora_selections"]:
            if row.get("implementation") == "original" and row["tile"] is None:
                continue
            tile = LoraTile(*row["tile"])
            if tile not in LORA_READOUT_TILES or row["bitwise"] is not True:
                raise ValueError(f"unqualified grouped LoRA tile {tile}")
            lora[row["in_features"], row["out_features"], row["route_rows"]] = tile
        table["lora_tiles"] = lora
    return table
