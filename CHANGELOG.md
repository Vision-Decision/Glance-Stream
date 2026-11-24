# Changelog

## 0.1.0 (unreleased, private preview)

- First release of Glance-Stream-0.8B: Qwen3.5-0.8B with a rule-question adapter, a box-question adapter and a box
  head ([Vision-Decision/Glance-Stream-0.8B](https://huggingface.co/Vision-Decision/Glance-Stream-0.8B)).
- `glance_stream` package: `GlanceStream.from_pretrained`, `Monitor.run` / `Monitor.feed` for files and live frames,
  seven rule types with the trained wording, the alert trigger used in evaluation, and the reference rule engine
  (`violations`) for scoring your own annotated videos.
- Two engines: the reference streaming engine (default; all accuracy numbers) and the fused single-pass engine
  (CUDA, Triton 3.8; latency number; on an earlier 0.8B checkpoint it did not pass its fixed equivalence criterion; re-test pending).
- Examples: `run_rules.py` for your own video, `make_demo.py` to render rule states, alerts and boxes over a video.
