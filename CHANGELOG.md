# Changelog

## 0.1.0 (unreleased, private preview)

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
