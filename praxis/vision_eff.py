"""Efficient vision encoder V2 — extends praxis/vision.py's MobileNetV3-style backbone with:
  * xATLU activation option (Huang 2024) in place of h-swish/ReLU.
  * A lightweight SDPA self-attention block over the stride-16 tokens (global context / relational
    reasoning across the ~64 tokens), run at the conv channel dim (cheap, MobileViT-style).
  * Multi-level feature CONCAT (C2 s4 + C3 s8 + C4 s16), projected + resized to stride-16 and
    concatenated (not added) then 1x1-mixed -> each token carries detail + mid + semantic context.

Inverted bottlenecks & depthwise-separable convs are already the backbone (MBConv). Output is the
same dense stride-16 token grid (B,N,d) + (row,col) coords for the policy's 2D-RoPE cross-attention.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vision import SE
from .policy_eff import XATLU


def _act(kind):
    return {"hswish": nn.Hardswish(inplace=True), "relu": nn.ReLU(inplace=True), "xatlu": XATLU()}[
        kind
    ]


class MBConv(nn.Module):
    """inverted residual: expand 1x1 -> depthwise kxk (stride) -> optional SE -> project 1x1 (linear)."""

    def __init__(self, cin, cout, stride=1, expand=4, k=3, se=False, act="hswish"):
        super().__init__()
        mid = cin * expand
        self.use_res = stride == 1 and cin == cout
        layers = []
        if expand != 1:
            layers += [nn.Conv2d(cin, mid, 1, bias=False), nn.BatchNorm2d(mid), _act(act)]
        layers += [
            nn.Conv2d(mid, mid, k, stride, k // 2, groups=mid, bias=False),
            nn.BatchNorm2d(mid),
            _act(act),
        ]
        if se:
            layers += [SE(mid)]
        layers += [nn.Conv2d(mid, cout, 1, bias=False), nn.BatchNorm2d(cout)]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        y = self.block(x)
        return x + y if self.use_res else y


class TokenAttn(nn.Module):
    """One prenorm SDPA self-attention block over the token grid, at channel dim `ch` (cheap).
    Learned position embedding gives the tokens global spatial context before the policy sees them."""

    def __init__(self, ch, n_tokens, heads=4, mlp=2):
        super().__init__()
        self.h = heads
        self.pos = nn.Parameter(0.02 * torch.randn(1, n_tokens, ch))
        self.n1 = nn.LayerNorm(ch)
        self.qkv = nn.Linear(ch, 3 * ch, bias=False)
        self.o = nn.Linear(ch, ch, bias=False)
        self.n2 = nn.LayerNorm(ch)
        self.ff = nn.Sequential(nn.Linear(ch, mlp * ch), XATLU(), nn.Linear(mlp * ch, ch))

    def _posN(self, N):  # interpolate the learned pos grid to N tokens
        if N == self.pos.shape[1]:
            return self.pos
        s0 = int(round(self.pos.shape[1] ** 0.5))
        s1 = int(round(N**0.5))
        p = self.pos.reshape(1, s0, s0, -1).permute(0, 3, 1, 2)  # (1,ch,s0,s0)
        p = F.interpolate(p, size=(s1, s1), mode="bicubic", align_corners=False)
        return p.permute(0, 2, 3, 1).reshape(1, s1 * s1, -1)

    def forward(self, x):  # (B,N,ch)
        B, N, C = x.shape
        h = self.n1(x + self._posN(N))
        q, k, v = self.qkv(h).reshape(B, N, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v)  # SDPA
        x = x + self.o(a.transpose(1, 2).reshape(B, N, C))
        return x + self.ff(self.n2(x))


class EfficientEncoderV2(nn.Module):
    def __init__(self, d_model=256, w=1.0, in_ch=3, act="hswish", attn=True, multicat=True):
        super().__init__()
        c = lambda x: int(round(x * w / 8) * 8)
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, c(16), 3, 2, 1, bias=False), nn.BatchNorm2d(c(16)), _act(act)
        )
        cfg = [
            (16, 24, 2, 4, 3, False),
            (24, 24, 1, 3, 3, False),
            (24, 40, 2, 4, 5, True),
            (40, 40, 1, 4, 5, True),
            (40, 40, 1, 4, 5, True),
            (40, 96, 2, 5, 5, True),
            (96, 96, 1, 4, 5, True),
            (96, 96, 1, 4, 5, True),
        ]
        self.blocks = nn.ModuleList(
            [MBConv(c(a), c(b), s, e, k, se, act) for a, b, s, e, k, se in cfg]
        )
        self.s4_idx, self.s8_idx = 1, 4  # last blocks at stride 4 / 8
        self.attn = TokenAttn(c(96), n_tokens=64, heads=4) if attn else None  # 8x8 @128 = 64 tokens
        self.multicat = multicat
        if multicat:  # concat C2+C3+C4 -> mix
            self.p2 = nn.Conv2d(c(24), d_model, 1, bias=False)
            self.p3 = nn.Conv2d(c(40), d_model, 1, bias=False)
            self.p4 = nn.Conv2d(c(96), d_model, 1, bias=False)
            self.mix = nn.Conv2d(3 * d_model, d_model, 1, bias=False)
        else:
            self.p3 = nn.Conv2d(c(40), d_model, 1, bias=False)
            self.p4 = nn.Conv2d(c(96), d_model, 1, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model

    def _backbone(self, x):  # (Bf,3,H,W) -> C2,C3,C4 feature maps
        x = self.stem(x)
        c2 = c3 = None
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i == self.s4_idx:
                c2 = x
            if i == self.s8_idx:
                c3 = x
        return c2, c3, x  # c4 at stride 16

    def _head(self, c4, c3, c2):  # (B,*,8,8) -> tokens (B,64,d), coords (B,64,2)
        hw = c4.shape[-2:]
        if self.attn is not None:  # SDPA global context over s16 tokens
            B, C, H, W = c4.shape
            t = self.attn(c4.flatten(2).transpose(1, 2)).transpose(1, 2).reshape(B, C, H, W)
            c4 = c4 + t
        if self.multicat:  # concat all feature levels
            f = self.mix(
                torch.cat(
                    [
                        self.p4(c4),
                        F.adaptive_avg_pool2d(self.p3(c3), hw),
                        F.adaptive_avg_pool2d(self.p2(c2), hw),
                    ],
                    1,
                )
            )
        else:
            f = self.p4(c4) + F.adaptive_avg_pool2d(self.p3(c3), hw)
        B, C, H, W = f.shape
        tokens = self.norm(f.flatten(2).transpose(1, 2))
        ys = torch.linspace(0, 1, H, device=c4.device)
        xs = torch.linspace(0, 1, W, device=c4.device)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([gy.reshape(-1), gx.reshape(-1)], -1)[None].expand(B, -1, -1)
        return tokens, coords

    def forward(self, x):
        c2, c3, c4 = self._backbone(x)
        return self._head(c4, c3, c2)


class EfficientEncoderV3(EfficientEncoderV2):
    """V2 + TEMPORAL gated-delta attention at stride-16, BEFORE the spatial block (factorized
    temporal-then-spatial). Input (B,T,3,H,W) [or (B,3,H,W) for T=1]: the backbone runs per frame; the C4
    (s16) tokens are delta-integrated over time per position (causal, 1-D temporal RoPE, recurrent -> O(1)/step
    at inference); C2/C3 detail is taken from the CURRENT (last) frame. Output interface is unchanged (64
    tokens + coords). Temporal path is zero-init -> at start it equals the single-frame V2 output (clean warm-start)."""

    def __init__(
        self, d_model=256, w=1.0, in_ch=3, act="hswish", attn=True, multicat=True, t_heads=4
    ):
        super().__init__(d_model, w, in_ch, act, attn, multicat)
        from .temporal import TemporalDelta

        c = lambda x: int(round(x * w / 8) * 8)
        self.temporal = TemporalDelta(
            c(96), heads=t_heads
        )  # at the C4 conv-channel dim (cheap, like TokenAttn)

    def forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(1)  # (B,3,H,W) -> (B,1,3,H,W)
        B, T = x.shape[:2]
        c2, c3, c4 = self._backbone(x.reshape(B * T, *x.shape[2:]))
        Cc, hw = c4.shape[1], c4.shape[-2:]
        c4t = c4.view(B, T, Cc, -1).permute(0, 3, 1, 2)  # (B, P=64, T, Cc)
        c4 = (
            self.temporal(c4t).permute(0, 2, 1).reshape(B, Cc, *hw)
        )  # temporally-integrated current frame
        c2 = c2.view(B, T, *c2.shape[1:])[:, -1]  # current-frame detail
        c3 = c3.view(B, T, *c3.shape[1:])[:, -1]
        return self._head(c4, c3, c2)
