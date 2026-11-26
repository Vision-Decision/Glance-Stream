# Changelog

## Glance-Stream-4B (2026-11-26, package 0.1.0)

- **Glance-Stream-4B** ([Vision-Decision/Glance-Stream-4B](https://huggingface.co/Vision-Decision/Glance-Stream-4B)):
  the same recipe on Qwen3.5-4B, a rule adapter (1,200 steps), a box adapter and box head (1,500 steps). Same package
  and API; load it with `GlanceStream.from_pretrained("Vision-Decision/Glance-Stream-4B")`. Cross-time false alarms
  per rule-hour against the 4B baseline: 152.8 → 76.5 on the pre-registered 8-stream primary set (p = 0.0625, **primary
  not met**), 162.8 → 93.3 on 22 streams (p = 0.004); box Acc@0.5 45.7. Rules bit-identical to the rule checkpoint
  alone and boxes to the box checkpoint alone (`checks/independence.json` in the model repository).
- 4B demo on the same kitchen clip (`assets/demo_4B.*`, `assets/demo_predictions_4B.json`); the 0.8B demo stays the
  main one.
- `examples/make_demo.py` titles the video with the model in the predictions file; `examples/run_rules.py --model`
  takes either repository.
- No change to the 0.8B weights or the package code paths.

## 0.1.0

- Box weights replaced by the 1,500-step box adapter and head (HC-STVG v2, 267 test clips: Acc@0.5 48.5, was 43.7 with
  the 600-step version). The rule adapter is unchanged (byte-identical). The demo is re-rendered with the released
  weights.

- First release of Glance-Stream-0.8B: Qwen3.5-0.8B with a rule-question adapter, a box-question adapter and a box
  head ([Vision-Decision/Glance-Stream-0.8B](https://huggingface.co/Vision-Decision/Glance-Stream-0.8B)).
- `glance_stream` package: `GlanceStream.from_pretrained`, `Monitor.run` / `Monitor.feed` for files and live frames,
  seven rule types with the trained wording, the alert trigger used in evaluation, and the reference rule engine
  (`violations`) for scoring your own annotated videos.
- One streaming engine (all numbers). The fused single-pass engine and its latency number were removed from this
  release while the engine is being verified; it is planned for a later release.
- Examples: `run_rules.py` for your own video, `make_demo.py` to render rule states, alerts and boxes over a video.
