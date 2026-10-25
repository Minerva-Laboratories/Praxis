"""Parameter-efficient weak-task adaptation. Freeze the base policy and train only a low-rank adapter, so the
base competence on the other tasks is preserved BY CONSTRUCTION (no need to replay all-task demos to avoid
forgetting) and the weak-task fit costs a tiny fraction of the parameters.

Variants:
  singlora : SingLoRA (Bensaid+ 2025) -- a SINGLE matrix A. dW = (a/r) u(t) A[:out] A[:in]^T. Symmetric factor,
             no B/A scale mismatch. A is nonzero-init, so a ramp u(t)=min(1,t/T) keeps dW~0 at the start.
  rslora   : rank-stabilised LoRA (Kalajdzievski 2023) -- two matrices, dW = (a/sqrt(r)) B A, B=0 init (dW=0 start).
  lora     : classic LoRA, dW = (a/r) B A.

Merge at save (merged_plain_state_dict) folds dW into the base weight and strips the adapter, so the exported
checkpoint is an ordinary model the unchanged inference path loads with no LoRA code.
"""

from __future__ import annotations
import math, torch, torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16, variant="singlora", ramp_steps=200):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)  # freeze the pretrained weight
        self.out, self.inp = base.out_features, base.in_features
        self.r, self.alpha, self.variant, self.ramp_steps = r, alpha, variant, ramp_steps
        self.register_buffer("step", torch.zeros((), dtype=torch.long), persistent=False)
        dev, dt = base.weight.device, base.weight.dtype
        if variant == "singlora":
            d = max(self.out, self.inp)
            self.A = nn.Parameter(torch.empty(d, r, device=dev, dtype=dt))
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            self.scale = alpha / r
        else:
            self.A = nn.Parameter(torch.empty(r, self.inp, device=dev, dtype=dt))
            self.B = nn.Parameter(torch.zeros(self.out, r, device=dev, dtype=dt))
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))  # B=0 -> dW=0 at init
            self.scale = alpha / (math.sqrt(r) if variant == "rslora" else r)

    def _u(self):
        if self.variant != "singlora" or self.ramp_steps <= 0:
            return 1.0
        if not self.training:  # eval / rollout / merge: adapter fully applied
            return 1.0
        return min(1.0, float(self.step.item()) / self.ramp_steps)

    def _delta_apply(self, x, u):
        if self.variant == "singlora":
            return (x @ self.A[: self.inp]) @ self.A[: self.out].t()
        return (x @ self.A.t()) @ self.B.t()

    def forward(self, x):
        y = self.base(x)
        u = self._u()
        y = y + (self.scale * u) * self._delta_apply(x, u)
        if self.training and self.variant == "singlora" and self.ramp_steps > 0:
            self.step += 1
        return y

    @torch.no_grad()
    def merged_weight(self):
        u = self._u()
        if self.variant == "singlora":
            dW = self.A[: self.out] @ self.A[: self.inp].t()
        else:
            dW = self.B @ self.A
        return self.base.weight + (self.scale * u) * dW, self.base.bias


def apply_lora(model, r=8, alpha=16, variant="singlora", ramp_steps=200, skip=("recog",)):
    """Replace every nn.Linear (except names containing any `skip` substring) with a LoRALinear wrapper."""
    n = 0
    for mod_name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f"{mod_name}.{child_name}" if mod_name else child_name
            if isinstance(child, nn.Linear) and not any(s in full for s in skip):
                setattr(module, child_name, LoRALinear(child, r, alpha, variant, ramp_steps))
                n += 1
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"LoRA[{variant}] r={r} a={alpha}: wrapped {n} Linear; trainable {trainable / 1e6:.3f}M / {total / 1e6:.2f}M "
        f"({100 * trainable / total:.2f}%)",
        flush=True,
    )
    return model


def merged_plain_state_dict(lora_model):
    """Fold every adapter into its base weight and strip the LoRA wrapper, yielding a plain state_dict whose keys
    match the original (un-wrapped) model, so the standard inference path can load it."""
    lmods = {name: m for name, m in lora_model.named_modules() if isinstance(m, LoRALinear)}
    merged = {}
    for name, m in lmods.items():
        W, b = m.merged_weight()
        merged[f"{name}.weight"] = W.detach().clone()
        if b is not None:
            merged[f"{name}.bias"] = b.detach().clone()
    out = {}
    for k, v in lora_model.state_dict().items():
        if any(
            k.startswith(name + ".") for name in lmods
        ):  # base.weight / A / B / step -> handled by merge
            continue
        out[k] = v
    out.update(merged)
    return out
