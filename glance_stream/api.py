"""High-level API: load the model, attach standing rules to a video stream, get per-readout probabilities, alerts and
boxes.

    from glance_stream import GlanceStream, Rule
    gs = GlanceStream.from_pretrained("Vision-Decision/Glance-Stream-0.8B")
    mon = gs.monitor([Rule.now("someone is cutting vegetables"),
                      Rule.must_follow("someone opens the fridge", "someone closes the fridge", within=30),
                      Rule.now("someone is holding a knife", box="the woman holding a knife")])
    for r in mon.run("kitchen.mp4"):
        print(r.t, r.p, r.alerts, r.boxes)
"""
import json
import math
import os
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from glance_stream import video
from glance_stream.model import heads
from glance_stream.model.box import BoxReadout
from glance_stream.model.stream import Stream, load as load_base, p_yes, yes_no_ids
from glance_stream.rules import Rule, Trigger

REPO = "Vision-Decision/Glance-Stream-0.8B"
ASSIST = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
ENGINES = ("reference", "fused")


def box_question_ids(tok, description):
    """The box question the box head was trained with (HC-STVG person descriptions)."""
    return tok.encode(f"<|vision_end|>Where is the person described as \"{description}\" right now? Answer with a box."
                      f"{ASSIST}", add_special_tokens=False)


def _torch_gdn_kernels():
    """Off CUDA, use transformers' PyTorch Gated-DeltaNet path even when flash-linear-attention is installed (its Triton
    kernels need a GPU). Process-wide: call it only when the model runs on CPU."""
    import inspect
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq
    for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule", "causal_conv1d_fn",
                 "causal_conv1d_update"):
        if hasattr(mq, name):
            setattr(mq, name, inspect.unwrap(getattr(mq, name)))


def _lora_names(lm):
    names = []
    for name, mod in lm.named_modules():
        for t in heads.TARGETS:
            if isinstance(getattr(mod, t, None), heads.LoRALinear):
                names.append(f"{name}.{t}" if name else t)
    return names


@dataclass
class Box:
    xyxy: tuple          # normalised [0, 1] corners (x1, y1, x2, y2) of the current frame
    presence: float      # P(the described person is in the current frame)
    confidence: float    # predicted P(IoU >= 0.5)

    def pixels(self, width, height):
        x1, y1, x2, y2 = self.xyxy
        return (round(x1 * width), round(y1 * height), round(x2 * width), round(y2 * height))


@dataclass
class Readout:
    t: float                                    # stream time (s) of the newest frame written before this readout
    p: dict                                     # rule name -> p(yes, the rule holds / is violated right now)
    on: dict                                    # rule name -> p >= the rule's threshold
    alerts: list                                # rule names whose trigger fired at this readout
    boxes: dict = field(default_factory=dict)   # rule name -> Box (rules with `box`, every readout)
    ms: float = 0.0                             # wall time of the readout (questions + boxes; fused: write + read)

    def as_dict(self):
        d = dict(t=round(self.t, 4), p={k: round(v, 5) for k, v in self.p.items()}, on=self.on, alerts=self.alerts,
                 ms=round(self.ms, 2))
        d["boxes"] = {k: dict(xyxy=[round(x, 5) for x in b.xyxy], presence=round(b.presence, 5),
                              confidence=round(b.confidence, 5)) for k, b in self.boxes.items()}
        return d


class GlanceStream:
    """Qwen3.5-0.8B base + a rule-question LoRA + a box LoRA and box head. The video path always runs the base weights."""

    def __init__(self, model, tok, config, text_params, box_params, box_head, engine="reference", vision="hf"):
        if engine not in ENGINES:
            raise ValueError(f"engine must be one of {ENGINES}")
        if engine == "fused" and not torch.cuda.is_available():
            raise RuntimeError("the fused engine needs a CUDA GPU (Triton kernels); use engine='reference'")
        self.model, self.tok, self.config = model, tok, config
        self.lm = model.model.language_model
        self.text_params, self.box_params, self.box_head = text_params, box_params, box_head
        self.mods = heads.lora_modules(self.lm)
        self.engine, self.vision = engine, vision
        self.device = next(model.parameters()).device
        self.yn = yes_no_ids(tok)

    @classmethod
    def from_pretrained(cls, path=REPO, *, revision=None, device=None, dtype=None, engine="reference", vision="hf",
                        base_model=None, token=None):
        """path: a Hugging Face repo id or a local directory with config.json and the safetensors files.
        engine: 'reference' (the engine every published number was measured with; CUDA, or CPU in fp32) or 'fused'
        (the single-pass static engine, CUDA only; see docs/ENGINE.md for its equivalence status)."""
        from safetensors.torch import load_file
        if os.path.isdir(path):
            local = path
        else:
            from huggingface_hub import snapshot_download
            local = snapshot_download(path, revision=revision, token=token,
                                      allow_patterns=["config.json", "*.safetensors"])
        cfg = json.load(open(os.path.join(local, "config.json")))
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = dtype or (torch.bfloat16 if str(device).startswith("cuda") else torch.float32)
        if not str(device).startswith("cuda"):
            _torch_gdn_kernels()
        model, tok = load_base(base_model or cfg["base_model"], device=device, dtype=dtype)
        lm = model.model.language_model
        lc = cfg["lora"]
        text_params = heads.add_lora(lm, r=lc["rank"], alpha=lc["alpha"], targets=tuple(lc["targets"]))
        names = _lora_names(lm)
        if 2 * len(names) != len(text_params):
            raise RuntimeError("LoRA module list does not match the base model")

        def ordered(file):
            w = load_file(os.path.join(local, file))
            out = []
            for i, n in enumerate(names):
                for j, k in enumerate(("lora_A", "lora_B")):
                    t = w.pop(f"{n}.{k}")
                    if t.shape != text_params[2 * i + j].shape:
                        raise ValueError(f"{file}: {n}.{k} has shape {tuple(t.shape)}")
                    out.append(t)
            if w:
                raise ValueError(f"{file}: unexpected tensors {sorted(w)[:3]}")
            return out

        for p, t in zip(text_params, ordered(cfg["adapters"]["rules"])):
            p.data.copy_(t)
        box_params = [torch.nn.Parameter(t.to(device, text_params[0].dtype), requires_grad=False)
                      for t in ordered(cfg["adapters"]["box"])] if cfg["adapters"].get("box") else None
        bh = cfg["box_head"]
        box_head = BoxReadout(lm.embed_tokens.weight, D=lm.embed_tokens.embedding_dim, layer=bh["layer"]).to(device).eval()
        box_head.load_state_dict(load_file(os.path.join(local, bh["file"])))
        return cls(model, tok, cfg, text_params, box_params, box_head, engine=engine, vision=vision)

    def monitor(self, rules, every=None, window=None):
        """rules: Rule objects (or plain strings = custom yes/no questions). every: readout period in seconds
        (default 0.5). window: dict(full_s, sub_s, sub_fps) to override the bounded-memory window (default: 4 s full rate + up to 600 s at 0.5 fps)."""
        return Monitor(self, [r if isinstance(r, Rule) else Rule.custom(r) for r in rules], every, window)


class Monitor:
    def __init__(self, gs, rules, every=None, window=None):
        if not rules:
            raise ValueError("no rules")
        names = [r.name for r in rules]
        if len(set(names)) != len(names):
            raise ValueError("rule names must be unique")
        self.gs, self.rules = gs, rules
        sc = gs.config["stream"]
        self.every = float(every or sc["readout_every_s"])
        self.size = sc["size"]
        w = dict(full_s=sc["full_s"], sub_s=sc["sub_s"], sub_fps=sc["sub_fps"])
        w.update(window or {})
        self.box_rules = [i for i, r in enumerate(rules) if r.box]
        if gs.engine == "fused":
            from glance_stream.engine.static import StaticStream
            self.st = StaticStream(gs.model, gs.tok, attention="shared", lora="grouped", vision=gs.vision, **w)
            self.text_bank = self.st.register_bank("rules", gs.text_params)
            self.box_bank = self.st.register_bank("box", gs.box_params or gs.text_params) if self.box_rules else None
        else:
            self.st = Stream(gs.model, gs.tok, **w)
        self.st.vis_layer = gs.box_head.layer if self.box_rules else None
        self.questions = [r.question() for r in rules]
        self.qids = [self.st.question_ids(q) for q in self.questions]
        self.box_qids = [box_question_ids(gs.tok, rules[i].box) for i in self.box_rules]
        self.reset()

    def reset(self):
        """Start a new stream (empty memory, alert state cleared)."""
        self.st.reset()
        self.triggers = [Trigger(r.threshold, r.k, r.cooldown_s) for r in self.rules]
        self.prev = None
        self.next_read = None

    # -- reading -------------------------------------------------------------------------------------------------
    def _sync(self):
        if self.gs.device.type == "cuda":
            torch.cuda.synchronize()

    @torch.no_grad()
    def _read_reference(self):
        gs = self.gs
        heads.LORA["on"] = True
        try:
            p = p_yes(self.st.ask(self.questions)["logits"], *gs.yn)
            boxes = None
            if self.box_qids:
                if gs.box_params is not None:
                    heads.set_lora(gs.mods, gs.box_params)
                try:
                    boxes = gs.box_head(self.st, self.box_qids)
                finally:
                    if gs.box_params is not None:
                        heads.set_lora(gs.mods, gs.text_params)
            return p, boxes
        finally:
            heads.LORA["on"] = False

    def _readout(self, t, p, boxes, ms):
        p = p.float().cpu().numpy()
        names = [r.name for r in self.rules]
        out = Readout(t=float(t), p={n: float(x) for n, x in zip(names, p)},
                      on={n: bool(x >= r.threshold) for n, x, r in zip(names, p, self.rules)},
                      alerts=[n for n, x, trig in zip(names, p, self.triggers) if trig.update(float(t), float(x))],
                      ms=ms)
        if boxes is not None and self.box_rules:
            pres = boxes["presence_logit"].float().sigmoid().cpu().numpy()
            conf = boxes["iou_logit"].float().sigmoid().cpu().numpy()
            xyxy = boxes["box_pred"].float().clamp(0, 1).cpu().numpy()
            for j, i in enumerate(self.box_rules):
                out.boxes[names[i]] = Box(tuple(float(v) for v in xyxy[j]), float(pres[j]), float(conf[j]))
        return out

    # -- feeding -------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def feed(self, frames, times, fps):
        """frames uint8 (T, 3, S, S) at S = config size (256), times (T,) stream seconds on a k/fps grid, continuing the
        previous call. Returns the readouts that fell inside this chunk."""
        out = []
        if len(times) == 0:
            return out
        if self.next_read is None:
            self.next_read = (math.floor(float(times[0]) / self.every) + 1) * self.every
        chunk = 1 if self.gs.engine == "fused" else max(1, round(fps * self.every))
        for a in range(0, len(times), 256):
            fr = frames[a:a + 256]
            vis, grid = self.st.encode(video.pairs(fr, self.prev if a == 0 else frames[a - 1], device=self.gs.device))
            for b in range(0, len(fr), chunk):
                ts = [float(x) for x in times[a + b:a + min(b + chunk, len(fr))]]
                due = ts[-1] + 1e-6 >= self.next_read
                if due:
                    self.next_read = (math.floor(ts[-1] / self.every + 1e-6) + 1) * self.every
                if due and self.gs.engine == "fused":
                    self._sync()
                    t0 = time.perf_counter()
                    r = self.st.step(vis[b:b + 1], ts, grid, judgements=self.qids, judgement_bank=self.text_bank,
                                     boxes=self.box_qids, box_head=self.gs.box_head if self.box_qids else None,
                                     box_bank=self.box_bank)
                    self._sync()
                    out.append(self._readout(ts[-1], r["p_yes"], r["boxes"] if self.box_qids else None,
                                             (time.perf_counter() - t0) * 1e3))
                    continue
                self.st.write(vis[b:b + len(ts)], ts, grid)
                if due:
                    self._sync()
                    t0 = time.perf_counter()
                    p, boxes = self._read_reference()
                    self._sync()
                    out.append(self._readout(ts[-1], p, boxes, (time.perf_counter() - t0) * 1e3))
        self.prev = frames[-1]
        return out

    def run(self, path, fps=30, start=0.0, end=None, chunk_s=60.0, callback=None):
        """Stream a video file from `start` to `end` seconds (default: whole file) at `fps`, from an empty memory.
        Returns all readouts; `callback(readout)` is called as each one is produced."""
        self.reset()
        dur = video.probe(path)[1]
        end = dur if end is None else min(end, dur)
        out, t0 = [], float(start)
        while t0 < end - 1e-6:
            frames, times = video.decode(path, fps, self.size, t0, min(end, t0 + chunk_s))
            if len(frames) == 0:
                break
            for r in self.feed(frames, times - float(start), fps):
                out.append(r)
                if callback is not None:
                    callback(r)
            t0 += chunk_s
        return out
