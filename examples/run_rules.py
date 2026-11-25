"""Run standing rules over your own video and print alerts as they happen.

    python examples/run_rules.py my_video.mp4                                  # the kitchen rules below
    python examples/run_rules.py my_video.mp4 --rules examples/kitchen_rules.json --out readouts.json
    python examples/run_rules.py my_video.mp4 --rule "Is someone on the phone right now?" --every 1

Rules file: a JSON list of objects with "type" (now, after_then, b_without_a, a_not_followed, count_in_window,
while_state, custom) and its fields (X / A / B / C / T / N / text), plus optional "name", "box", "threshold", "k",
"cooldown_s"; see glance_stream/rules.py. For count_in_window, N is the start that violates the rule: "N": 3 means
"at most 2 times" (Rule.at_most(x, times=2, within=T) in Python). --out writes every readout (time, p(yes) per rule, alerts, boxes) as JSON;
examples/make_demo.py renders such a file over the video.
"""
import argparse
import json
import time

from glance_stream import GlanceStream, Rule

DEFAULT_RULES = [
    dict(type="now", X="someone is cutting vegetables", name="cutting vegetables"),
    dict(type="a_not_followed", A="someone opens the fridge", B="someone closes the fridge", T=30,
         name="fridge left open > 30 s"),
    dict(type="now", X="someone is holding a knife", box="the woman holding a knife", name="holding a knife"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--rules", help="JSON file with a list of rules (default: three kitchen rules)")
    ap.add_argument("--rule", action="append", default=[], help="a custom yes/no question; repeatable")
    ap.add_argument("--model", default="Vision-Decision/Glance-Stream-0.8B")
    ap.add_argument("--base_model", help="local path or id of Qwen3.5-0.8B (default: from the model's config)")
    ap.add_argument("--fps", type=float, default=30)
    ap.add_argument("--every", type=float, default=0.5, help="readout period in seconds")
    ap.add_argument("--start", type=float, default=0)
    ap.add_argument("--end", type=float)
    ap.add_argument("--threshold", type=float, help="override every rule's threshold")
    ap.add_argument("--out", help="write all readouts to this JSON file")
    a = ap.parse_args()

    specs = json.load(open(a.rules)) if a.rules else ([] if a.rule else DEFAULT_RULES)
    rules = [Rule(**s) for s in specs] + [Rule.custom(q) for q in a.rule]
    if a.threshold is not None:
        for r in rules:
            r.threshold = a.threshold
    t = time.perf_counter()
    gs = GlanceStream.from_pretrained(a.model, base_model=a.base_model)
    print(f"loaded {a.model} on {gs.device} in {time.perf_counter() - t:.1f} s")
    for r in rules:
        print(f"  rule: {r.name}" + (f"  [box: {r.box}]" if r.box else ""))

    def show(r):
        for name in r.alerts:
            box = r.boxes.get(name)
            where = f"  box {[round(v, 3) for v in box.xyxy]} p(present) {box.presence:.2f}" if box else ""
            print(f"{r.t:8.2f} s  ALERT  {name}  p={r.p[name]:.2f}{where}")

    t = time.perf_counter()
    mon = gs.monitor(rules, every=a.every)
    readouts = mon.run(a.video, fps=a.fps, start=a.start, end=a.end, callback=show)
    wall = time.perf_counter() - t
    span = readouts[-1].t if readouts else 0
    print(f"{len(readouts)} readouts over {span:.1f} s of video in {wall:.1f} s wall time")
    if a.out:
        json.dump(dict(model=a.model, video=a.video, fps=a.fps, every=a.every, start=a.start,
                       rules=[r.as_dict() for r in rules], questions=[r.question() for r in rules],
                       readouts=[r.as_dict() for r in readouts]), open(a.out, "w"), indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
