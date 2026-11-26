"""Render rule states, alerts and boxes over a video (assets/demo.gif and assets/demo.mp4 come from this script).

    # 1. run the model (CUDA GPU) and save the raw readouts next to the video
    python examples/run_rules.py kitchen.mp4 --rules examples/kitchen_rules.json --out assets/demo_predictions.json
    # 2. render (CPU only; needs ffmpeg on PATH)
    python examples/make_demo.py kitchen.mp4 assets/demo_predictions.json --out assets/demo

Every probability, alert and box in the video is read from the predictions file; nothing is edited by hand. A frame
shows the newest readout at or before its time (readouts come every 0.5 s by default).
"""
import argparse
import json
import os
import shutil
import subprocess
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

COLORS = [(64, 196, 170), (150, 134, 245), (245, 166, 64), (236, 104, 150), (96, 170, 245), (190, 210, 90)]
ALERT = (255, 82, 82)
BG, PANEL, TRACK, TEXT, MUTED = (14, 15, 18), (24, 26, 31), (52, 55, 63), (236, 238, 242), (150, 155, 166)


def font(size, bold=False):
    for name in (("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"), ("Arial Bold.ttf" if bold else "Arial.ttf"),
                 "/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def frames(path, width, start, end):
    """yield (t, PIL frame) at the source frame rate, scaled to `width`."""
    import av
    with av.open(path) as c:
        s = c.streams.video[0]
        tb = float(s.time_base)
        st = float(s.start_time * tb) if s.start_time is not None else 0.0
        h = None
        for fr in c.decode(s):
            if fr.pts is None:
                continue
            t = fr.pts * tb - st
            if t < start - 1e-6:
                continue
            if end is not None and t >= end:
                break
            if h is None:
                h = round(fr.height * width / fr.width / 2) * 2
            yield t - start, fr.to_image().convert("RGB").resize((width, h), Image.BILINEAR)


def render(video, pred, out, width=720, start=None, end=None, gif_fps=8, gif_width=540, title=None, credit=None):
    P = json.load(open(pred))
    rules = P["rules"]
    names = [r["name"] for r in rules]
    R = P["readouts"]
    ts = np.array([r["t"] for r in R])
    start = P.get("start", 0) if start is None else start
    end = (start + ts[-1] + 0.5) if end is None else end
    f_title, f_name, f_small, f_big = font(17, True), font(15, True), font(13), font(15, True)
    row_h, head_h = 46, 34
    tmp = tempfile.mkdtemp()
    n = 0
    for tf, im in frames(video, width, start, end):
        k = int(np.searchsorted(ts, tf + 1e-6, side="right")) - 1
        r = R[k] if k >= 0 else None
        vh = im.height
        canvas = Image.new("RGB", (width, vh + head_h + row_h * len(rules) + 10), BG)
        canvas.paste(im, (0, 0))
        d = ImageDraw.Draw(canvas)
        # boxes: drawn while the rule is on and the box head says the person is present
        if r is not None:
            for i, name in enumerate(names):
                b = r.get("boxes", {}).get(name)
                if b and r["on"][name] and b["presence"] >= 0.5:
                    x1, y1, x2, y2 = (b["xyxy"][0] * width, b["xyxy"][1] * vh, b["xyxy"][2] * width, b["xyxy"][3] * vh)
                    col = COLORS[i % len(COLORS)]
                    d.rectangle([x1, y1, x2, y2], outline=col, width=3)
                    lab = f"{rules[i].get('box')}  {b['presence']:.2f}"
                    tw = d.textlength(lab, font=f_small)
                    d.rectangle([x1, max(0, y1 - 20), x1 + tw + 10, max(0, y1 - 20) + 20], fill=col)
                    d.text((x1 + 5, max(0, y1 - 20) + 3), lab, font=f_small, fill=BG)
        # header
        y0 = vh
        d.rectangle([0, y0, width, y0 + head_h], fill=PANEL)
        d.text((12, y0 + 8), title or f"{P.get('model', 'Glance-Stream').split('/')[-1]} · real model outputs", font=f_title, fill=TEXT)
        clock = f"t = {tf:5.1f} s"
        d.text((width - 12 - d.textlength(clock, font=f_big), y0 + 9), clock, font=f_big, fill=TEXT)
        # one row per rule: name, p(yes) bar with threshold tick, history sparkline with alert marks
        bar_x, bar_w = 12, 150
        sp_x = bar_x + bar_w + 70
        sp_w = width - sp_x - 12
        T = max(end - start, 1e-6)
        for i, name in enumerate(names):
            col = COLORS[i % len(COLORS)]
            y = y0 + head_h + 6 + i * row_h
            p = r["p"][name] if r else 0.0
            on = bool(r and r["on"][name])
            recent = [x["t"] for x in R[:k + 1] if name in x["alerts"]]
            flash = bool(recent) and tf - recent[-1] < 1.5
            if flash:
                d.rectangle([4, y - 3, width - 4, y + row_h - 6], outline=ALERT, width=2)
            d.text((bar_x, y), name, font=f_name, fill=col if on else TEXT)
            if rules[i].get("type") not in (None, "now", "custom"):
                tag = rules[i]["type"].replace("_", " ")
                d.text((bar_x + d.textlength(name, font=f_name) + 8, y + 2), tag, font=f_small, fill=MUTED)
            by = y + 22
            d.rounded_rectangle([bar_x, by, bar_x + bar_w, by + 12], 6, fill=TRACK)
            if p > 0.01:
                d.rounded_rectangle([bar_x, by, bar_x + max(12, bar_w * p), by + 12], 6, fill=col if on else MUTED)
            thr = rules[i].get("threshold", 0.5)
            d.line([bar_x + bar_w * thr, by - 3, bar_x + bar_w * thr, by + 15], fill=TEXT, width=1)
            d.text((bar_x + bar_w + 8, by - 2), f"{p:.2f}", font=f_small, fill=TEXT)
            if flash:
                d.text((width - 12 - d.textlength("ALERT", font=f_name), y), "ALERT", font=f_name, fill=ALERT)
            # sparkline of p over the clip so far
            d.rectangle([sp_x, by - 4, sp_x + sp_w, by + 16], fill=PANEL)
            d.line([sp_x, by + 16 - 20 * thr, sp_x + sp_w, by + 16 - 20 * thr], fill=TRACK, width=1)
            pts = [(sp_x + sp_w * x["t"] / T, by + 16 - 20 * x["p"][name]) for x in R[:k + 1]]
            if len(pts) > 1:
                d.line(pts, fill=col, width=2)
            for ta in recent:
                xa = sp_x + sp_w * ta / T
                d.polygon([(xa - 5, by - 6), (xa + 5, by - 6), (xa, by + 2)], fill=ALERT)
        if credit:
            d.text((8, vh - 18), credit, font=f_small, fill=(255, 255, 255))
        canvas.save(os.path.join(tmp, f"f{n:05d}.png"))
        n += 1
    if not n:
        raise SystemExit("no frames rendered")
    fps = n / max(end - start, 1e-6)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    seq = os.path.join(tmp, "f%05d.png")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", f"{fps:.4f}", "-i", seq, "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "24", "-preset", "slow", "-movflags", "+faststart", out + ".mp4"],
                   check=True)
    pal = os.path.join(tmp, "pal.png")
    vf = f"fps={gif_fps},scale={gif_width}:-1:flags=lanczos"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", f"{fps:.4f}", "-i", seq, "-vf",
                    vf + ",palettegen=max_colors=96:stats_mode=diff", pal], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", f"{fps:.4f}", "-i", seq, "-i", pal, "-lavfi",
                    vf + " [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle", out + ".gif"],
                   check=True)
    shutil.rmtree(tmp)
    print(f"{n} frames -> {out}.mp4, {out}.gif")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("predictions", help="JSON written by examples/run_rules.py --out")
    ap.add_argument("--out", default="assets/demo", help="output path without extension (.mp4 and .gif)")
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--start", type=float, help="video time of readout t = 0 (default: the predictions' start)")
    ap.add_argument("--end", type=float)
    ap.add_argument("--gif_fps", type=int, default=8)
    ap.add_argument("--gif_width", type=int, default=540)
    ap.add_argument("--title")
    ap.add_argument("--credit", help="footage credit drawn in the corner")
    a = ap.parse_args()
    render(a.video, a.predictions, a.out, a.width, a.start, a.end, a.gif_fps, a.gif_width, a.title, a.credit)


if __name__ == "__main__":
    main()
