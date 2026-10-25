"""MuSGD: Muon (Jordan 2024) for >=2D weight matrices + SGD-momentum for everything else.

Muon takes the SGD-momentum update on a weight MATRIX and replaces it with the nearest semi-orthogonal
matrix (via a 5-step Newton-Schulz iteration approximating U V^T of the momentum's SVD), which
equalizes the update's singular values -> better-conditioned steps. 1D params (biases, norm scales)
and embeddings/lookup tables get plain SGD-momentum, since orthogonalization is meaningless there.
Conv/linear weights of ndim>=2 are flattened to (out, -1) for the iteration.
"""

import torch
from torch.optim.optimizer import Optimizer


def _zeropower_ns5(G, steps=5):
    """Newton-Schulz quintic iteration -> nearest semi-orthogonal matrix of G (2D)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.to(torch.bfloat16)
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class MuSGD(Optimizer):
    def __init__(
        self,
        params,
        lr=2e-2,
        momentum=0.95,
        weight_decay=0.0,
        nesterov=True,
        ns_steps=5,
        sgd_lr=None,
        sgd_momentum=0.9,
    ):
        # `params` groups may set use_muon=True/False; default routes >=2D non-embedding -> Muon.
        sgd_lr = sgd_lr if sgd_lr is not None else lr
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            ns_steps=ns_steps,
            sgd_lr=sgd_lr,
            sgd_momentum=sgd_momentum,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                use_muon = group.get("use_muon", p.ndim >= 2)
                if group["weight_decay"]:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                if use_muon:
                    buf = st.setdefault("mom", torch.zeros_like(p))
                    buf.mul_(group["momentum"]).add_(g)
                    upd = g.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf
                    o = _zeropower_ns5(upd.reshape(upd.shape[0], -1), group["ns_steps"]).view_as(
                        upd
                    )
                    scale = max(1.0, upd.shape[0] / upd.reshape(upd.shape[0], -1).shape[1]) ** 0.5
                    p.add_(o, alpha=-group["lr"] * scale)
                else:  # SGD-momentum for 1D / embeddings
                    buf = st.setdefault("mom", torch.zeros_like(p))
                    buf.mul_(group["sgd_momentum"]).add_(g)
                    p.add_(buf, alpha=-group["sgd_lr"])
        return loss


def musgd_param_groups(model, lr=2e-2, sgd_lr=None, weight_decay=0.05):
    """Muon for hidden weight matrices (Linear/Conv, ndim>=2, non-embedding); SGD for 1D + embeddings."""
    muon, sgd = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2 and "emb" not in n and "task_emb" not in n:
            muon.append(p)
        else:
            sgd.append(p)
    sgd_lr = sgd_lr if sgd_lr is not None else lr
    return [
        dict(params=muon, use_muon=True, lr=lr, weight_decay=weight_decay),
        dict(params=sgd, use_muon=False, sgd_lr=sgd_lr, weight_decay=0.0),
    ]
