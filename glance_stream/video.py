"""Video input: decode at a target fps (for each time k/fps, the latest source frame shown at that time), resize to
SIZE x SIZE, and build the sliding tubelet pairs (frame at t - 1/fps, frame at t) that the stream writes as one group."""
import math

import numpy as np
import torch


def probe(path):
    """-> (fps, duration_s, width, height) of the first video stream."""
    import av
    with av.open(path) as c:
        s = c.streams.video[0]
        fps = float(s.average_rate or s.guessed_rate or 30)
        dur = float(s.duration * s.time_base) if s.duration else float(c.duration / 1e6)
        return fps, dur, s.codec_context.width, s.codec_context.height


def decode(path, fps, size=256, t0=0.0, t1=None, threads=2):
    """-> (frames uint8 (T, 3, size, size), times np.array (T,)) for the grid times k/fps in [t0, t1).
    Frames repeat when the source is slower than `fps`; swscale resizes on decode (no aspect-ratio padding)."""
    import av
    ts, imgs = [], []
    with av.open(path) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        s.thread_count = threads
        tb = float(s.time_base)
        st = float(s.start_time * tb) if s.start_time is not None else 0.0
        if t0 > 1.0:
            c.seek(int((t0 - 1.0 + st) / tb), stream=s, backward=True)
        for fr in c.decode(s):
            if fr.pts is None:
                continue
            t = fr.pts * tb - st
            if t1 is not None and t >= t1:
                break
            if t < t0 - 1.0:
                continue
            ts.append(t)
            imgs.append(fr.to_ndarray(format="rgb24", width=size, height=size))
    if not imgs:
        return torch.zeros(0, 3, size, size, dtype=torch.uint8), np.zeros(0)
    ts = np.array(ts)
    end = ts[-1] + (ts[-1] - ts[-2] if len(ts) > 1 else 1 / fps)
    hi = end if t1 is None else min(end, t1)
    grid = np.arange(math.ceil(t0 * fps - 1e-6), math.ceil(hi * fps - 1e-6)) / fps
    if len(grid) == 0:
        return torch.zeros(0, 3, size, size, dtype=torch.uint8), grid
    idx = np.clip(np.searchsorted(ts, grid + 1e-4, side="right") - 1, 0, len(ts) - 1)
    x = torch.from_numpy(np.stack([imgs[i] for i in idx])).permute(0, 3, 1, 2).contiguous()
    return x, grid


def pairs(frames, prev=None, device=None):
    """uint8 (T, 3, S, S) -> float (T, 2, 3, S, S) in [0, 1]: group k = (frame k-1, frame k); the first uses `prev`
    (the last frame of the previous chunk) or itself."""
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    f = frames.to(dev, non_blocking=True).float() / 255
    p = torch.cat([(prev.to(dev).float()[None] / 255) if prev is not None else f[:1], f[:-1]], 0)
    return torch.stack([p, f], 1)
