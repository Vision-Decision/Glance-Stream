# Engine

This release ships one streaming engine, `glance_stream/model/stream.py`. Every number in this repository was measured
with it.

- **Writing.** A frame group (two consecutive frames through the Qwen3.5 vision tower's 2-frame tubelet, 64 tokens at
  256×256) is written into the model's state. Groups are written in chunks of `fps × every` per forward (15 at 30 fps
  with 0.5 s readouts), with a text timestamp every 0.5 s.
- **Memory.** The Gated-DeltaNet layers carry their recurrent state through the whole stream. The full-attention layers
  keep every group of the last 4 s and older groups up to 600 s on a 2 s grid (about 30k tokens); positions are rotated
  back into the trained range for hour-long streams.
- **Reading.** At each readout, all rule questions are forked from the current state as one batched forward, and the
  box questions as a second one with the box adapter; nothing is written back.
- **Attention.** On CUDA, attention over the cached keys uses an exact split kernel without boolean masks
  (`glance_stream/model/attn.py`); elsewhere standard SDPA.
- **Devices.** CUDA in bf16 (install `flash-linear-attention` for the fast recurrent kernel). CPU in fp32 works but is
  far from real time.

An optimised engine (one fused forward per frame for the write, all rule questions and the boxes) is being verified and
is planned for a later release.
