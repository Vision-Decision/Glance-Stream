"""Question-block LoRA: active ONLY in the question-block forward (LORA["on"]), so the video path always runs the original
weights and the stream state is the frozen base model's. Rule questions and box questions use separate adapters of the
same shape; `set_lora` swaps them without copying."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

LORA = {"on": False}


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=16, alpha=32):
        super().__init__()
        self.base = base; self.r = r; self.s = alpha / r
        self.A = nn.Parameter(torch.randn(r, base.in_features, device=base.weight.device) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))

    def forward(self, x):
        y = self.base(x)
        if LORA["on"]:
            y = y + (x.to(self.A.dtype) @ self.A.t() @ self.B.t()).to(y.dtype) * self.s
        return y


TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj", "gate_proj", "up_proj", "down_proj")


def lora_modules(lm, targets=TARGETS):
    """LoRALinear modules in exactly the order add_lora created them (parent by parent, then the `targets` order) -
    the order of the checkpoint's `lora` list; lm.modules() order differs (that mismatch broke the first 3-way eval)"""
    out = []
    for name, mod in lm.named_modules():
        for t in targets:
            child = getattr(mod, t, None)
            if isinstance(child, LoRALinear):
                out.append(child)
    return out


def set_lora(mods, params):
    """point every LoRALinear at another adapter (params = [A0, B0, A1, B1, ...] in add_lora order): a free swap"""
    assert len(params) == 2 * len(mods)
    for i, m in enumerate(mods):
        assert params[2 * i].shape == m.A.shape and params[2 * i + 1].shape == m.B.shape, (i, params[2 * i].shape, m.A.shape)
        m.A, m.B = params[2 * i], params[2 * i + 1]


def add_lora(lm, r=16, alpha=32, targets=("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj",
                                          "gate_proj", "up_proj", "down_proj")):
    params = []
    for name, mod in list(lm.named_modules()):
        for t in targets:
            child = getattr(mod, t, None)
            if isinstance(child, nn.Linear):
                w = LoRALinear(child, r, alpha); setattr(mod, t, w); params += [w.A, w.B]
    return params
