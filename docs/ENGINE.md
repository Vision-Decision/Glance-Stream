# Engines

Both engines run the same model and the same stream: a frame group (two consecutive frames through the vision tower's
2-frame tubelet, 64 tokens at 256×256) is written into the model's state, and the rule and box questions are short
blocks forked from that state, answered, and thrown away. They differ in how the GPU work is scheduled.

| | `engine="reference"` (default) | `engine="fused"` |
|---|---|---|
| code | `glance_stream/model/stream.py` | `glance_stream/engine/` (`StaticStream`) |
| how | Hugging Face modules; writes `fps × every` frame groups per forward (15 at 30 fps and 0.5 s readouts), then one batched forward for all rule questions and one for the boxes | one fused forward per frame: write + every rule question + every box, static KV slab, shared-window attention, both adapters applied as grouped LoRA, CUDA graphs and Triton kernels |
| devices | CUDA (bf16); CPU in fp32 works but is far from real time | CUDA only |
| measured | every accuracy number in this repository | the latency number |

## Latency

Fused engine, Qwen3.5-0.8B, one stream on an RTX PRO 6000 Blackwell, frames arriving at 30 fps, every frame written
**and** read (8 rule questions + 1 box), 600 s memory window after 600 s of warm-up (about 30k cached tokens): **P50 23.0 ms,
P99 24.3 ms** per frame from arrival to all results ready, including decoding, the vision tower and the step; no frame
dropped. Environment: torch 2.14.1 + CUDA 13.0, transformers 5.23.0, flash-linear-attention 0.6.0, Triton 3.8.0.

## Equivalence of the fused engine

The fused engine is not bit-identical to the reference: it packs frame rows and question rows into the same matrix
products, so bf16 rounding happens in a different order. It was therefore checked at the level that matters for a
deployment, which alerts fire, on held-out 10-minute streams with 16 rules each, against a rule fixed before the
test (the change in false alarms must stay within ±5% and be no larger than the change the reference engine itself
shows when only its batch layout changes).

On an earlier checkpoint of this 0.8B model, the fused engine did **not** pass that rule: its change in false alarms
was larger than the reference engine's own batch-layout change. A re-test against several batch layouts instead of one
is pending. The released weights (rule adapter, box adapter and box head) have not been compared across engines.

Until a re-test passes, the reference engine is the default for this model. Use `engine="fused"` only if you need
per-frame readouts at 30 fps and accept that its accuracy on the published benchmarks is unverified.

## Requirements of the fused engine

It was validated only with the environment above (Triton 3.8.0, which PyTorch 2.14 ships). It does **not** compile with
Triton 3.4 (a `tl.static_range` over tuple constants fails in the grouped-LoRA kernel); versions in between are
untested. The short generative decoding path for timestamps and counts is not part of this release.
