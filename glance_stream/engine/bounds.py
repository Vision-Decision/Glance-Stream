"""Optional staged-address assertions. Call inside the captured forward so replay checks new values."""
import os

import torch

ENABLED = os.environ.get("GLANCE_ENGINE_BOUNDS_CHECK") == "1"


def indices(t, size, label):
    torch._assert_async(((t >= 0) & (t < size)).all(), label)


def packed(x, old, post, widths, append, owners, regions, frame, spec, schedule, kv, states, routes, repeat):
    slabs = [t for pair in kv.values() for t in pair]
    capacity = min(t.size(2) for t in slabs)
    batch = min(t.size(0) for pair in (*kv.values(), *states.values()) for t in pair)
    indices(old, capacity, "old slab directory including padding")
    indices(post, capacity, "post slab directory including padding")
    indices(append, capacity, "slab append slot")
    indices(owners, batch, "recurrent state owner")
    indices(widths, len(old) + 1, "live directory width")

    valid = (sum(n * stride for n, stride in regions) == len(x)
             and all(n > 0 and stride > 0 for n, stride in regions)
             and len(owners) >= sum(n for n, _ in regions[int(frame):])
             and (not frame or (regions[0][0] == batch and regions[0][1] == len(append))))
    for region, _ in routes:
        start = 0 if region.start is None else region.start
        end = len(x) if region.stop is None else region.stop
        valid &= 0 <= start <= end <= len(x)
    if spec:  # gather
        for (owner, view, start), ix in zip(spec, schedule, strict=True):
            valid &= (0 <= owner < batch and 0 <= view < len(widths)
                      and (start == -1 or 0 <= start <= len(x) - len(ix)))
            indices(ix, len(x), "gather query/output index")
    elif len(schedule) == 2:  # shared old/post union, or a single static-write view
        tile_owners, tokens = schedule
        indices(tile_owners, batch, "shared slab owner")
        torch._assert_async(((tokens >= -1) & (tokens < 2 * len(x))).all(), "shared query row")
        torch._assert_async((widths[:-1] <= widths[1:]).all(), "shared key membership cuts")
        valid &= len(tokens) == len(tile_owners)
    else:
        tiles, = schedule
        start, end, owner, view = tiles.unbind(1)
        torch._assert_async(((start >= 0) & (start < end) & (end <= len(x) * repeat)).all(),
                            "indexed query/output region")
        indices(owner, batch, "indexed slab owner")
        indices(view, len(widths), "indexed directory view")
    torch._assert_async(widths.new_full((), valid, dtype=torch.bool), "packed region or gather span")
