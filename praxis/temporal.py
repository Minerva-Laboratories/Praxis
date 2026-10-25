"""Temporal gated delta-rule linear attention (KDA / Gated-DeltaNet family), applied over the TIME axis of
a per-position feature stack. Causal and recurrent: it maintains a per-head matrix state S (key-dim x
value-dim) updated by the delta rule with a forget gate, so it is O(1)/step at inference (streaming control)
and linear in T at training. 1-D RoPE encodes the frame index on q,k.

Interface: forward(x) with x = (B, P, T, C)  [P spatial positions, T frames] -> (B, P, C), the CURRENT
(last) frame's temporally-integrated feature, added residually. Compresses history through time only; no
spatial mixing here (that is left to the spatial block downstream -> factorized temporal-then-spatial).
"""

import torch, torch.nn as nn


def _rope_1d(x, pos):  # x (...,T,d), pos (T,) -> RoPE-rotated x
    d = x.shape[-1]
    inv = 1.0 / (
        10000.0 ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d)
    )  # (d/2,)
    ang = pos[:, None].float() * inv[None, :]  # (T, d/2)
    cos = ang.cos().repeat_interleave(2, -1).to(x.dtype)  # (T, d)
    sin = ang.sin().repeat_interleave(2, -1).to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    xr = torch.stack([-x2, x1], -1).reshape_as(x)  # rotate_half (interleaved)
    return x * cos + xr * sin


class TemporalDelta(nn.Module):
    def __init__(self, dim, heads=4, dh=None):
        super().__init__()
        dh = dh or dim
        assert dh % heads == 0
        self.heads, self.dk, self.dh = heads, dh // heads, dh
        self.norm = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, dh, bias=False)
        self.to_k = nn.Linear(dim, dh, bias=False)
        self.to_v = nn.Linear(dim, dh, bias=False)
        self.to_beta = nn.Linear(dim, heads)  # write strength (0,1) per head
        self.to_alpha = nn.Linear(dim, heads)  # forget gate (0,1) per head
        nn.init.constant_(self.to_alpha.bias, 2.0)  # start near alpha~0.88 (retain history)
        self.out = nn.Linear(dh, dim, bias=False)
        nn.init.normal_(
            self.out.weight, std=1e-3
        )  # near-identity start (~1% temporal contribution) so the
        #   temporal internals get gradient from step 1, while barely perturbing a warm-started model

    def _split(self, t, B, P, T):  # (B,P,T,dh) -> (B,P,T,H,dk)
        return t.view(B, P, T, self.heads, self.dk)

    def forward(self, x, return_all=False):  # x (B,P,T,C)
        B, P, T, C = x.shape
        h = self.norm(x)
        pos = torch.arange(T, device=x.device)
        q = _rope_1d(self._split(self.to_q(h), B, P, T).transpose(2, 3), pos).transpose(
            2, 3
        )  # RoPE on time
        k = _rope_1d(self._split(self.to_k(h), B, P, T).transpose(2, 3), pos).transpose(2, 3)
        v = self._split(self.to_v(h), B, P, T)  # (B,P,T,H,dk)
        beta = torch.sigmoid(self.to_beta(h))  # (B,P,T,H)
        alpha = torch.sigmoid(self.to_alpha(h))  # (B,P,T,H)
        S = x.new_zeros(B, P, self.heads, self.dk, self.dk)  # state: key-dim x value-dim
        outs = []
        for t in range(T):
            kt, vt, qt = k[:, :, t], v[:, :, t], q[:, :, t]  # (B,P,H,dk)
            pred = torch.einsum("bphi,bphij->bphj", kt, S)  # current value estimate for kt
            delta = vt - pred
            a = alpha[:, :, t, :, None, None]
            b = beta[:, :, t, :, None, None]
            S = a * S + b * torch.einsum("bphi,bphj->bphij", kt, delta)  # gated delta update
            ot = torch.einsum("bphi,bphij->bphj", qt, S)  # read with query
            if return_all:
                outs.append(ot.reshape(B, P, self.dh))
        o_last = ot.reshape(B, P, self.dh)
        y = x[:, :, -1] + self.out(o_last)  # residual on the current frame
        if return_all:
            return y, torch.stack(outs, 2)  # (B,P,C), (B,P,T,dh)
        return y
