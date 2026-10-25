"""Compact dense vision encoder for the action policy.

Design:
  * Depthwise-separable convolutions everywhere (Howard et al., MobileNets, 2017): a
    3x3 depthwise conv + 1x1 pointwise replaces a dense 3x3 at ~1/9 the FLOPs.
  * Inverted residual / linear bottleneck blocks (Sandler et al., MobileNetV2, 2018):
    expand (1x1) -> depthwise (kxk) -> project (1x1, linear), residual on stride-1 same-ch.
    Expansion in the high-D space keeps depthwise cheap; the linear projection avoids ReLU
    destroying low-D info.
  * Squeeze-and-excitation channel gating (Hu et al., 2018) + h-swish (Howard et al.,
    MobileNetV3, 2019) in the deeper stages, where they buy accuracy cheaply.
  * Lite top-down multi-scale aggregation (FPN idea, Lin et al., 2017, but a single add):
    fold the fine stride-8 map into the semantic stride-16 map so each output token carries
    both local detail and context -- one 1x1 + resize + add, not a full pyramid.

Output: a DENSE stride-16 token grid (B, N, d_model) plus normalized (row,col) coords in
[0,1] for the cross-attention's 2D RoPE. This is fed as keys/values to the policy; the
backbone never runs self-attention over the pixels, so it stays cheap.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def _act(hswish: bool):
    return nn.Hardswish(inplace=True) if hswish else nn.ReLU(inplace=True)


class SE(nn.Module):
    """squeeze-excite: global-pool -> bottleneck -> gate channels in [0,1]."""

    def __init__(self, ch, r=4):
        super().__init__()
        m = max(8, ch // r)
        self.fc1 = nn.Conv2d(ch, m, 1)
        self.fc2 = nn.Conv2d(m, ch, 1)

    def forward(self, x):
        s = x.mean((2, 3), keepdim=True)
        s = F.relu(self.fc1(s), inplace=True)
        return x * torch.sigmoid(self.fc2(s))


class MBConv(nn.Module):
    """inverted residual: expand 1x1 -> depthwise kxk (stride) -> optional SE -> project 1x1."""

    def __init__(self, cin, cout, stride=1, expand=4, k=3, se=False, hswish=False):
        super().__init__()
        mid = cin * expand
        self.use_res = stride == 1 and cin == cout
        layers = []
        if expand != 1:
            layers += [nn.Conv2d(cin, mid, 1, bias=False), nn.BatchNorm2d(mid), _act(hswish)]
        layers += [
            nn.Conv2d(mid, mid, k, stride, k // 2, groups=mid, bias=False),
            nn.BatchNorm2d(mid),
            _act(hswish),
        ]
        if se:
            layers += [SE(mid)]
        layers += [nn.Conv2d(mid, cout, 1, bias=False), nn.BatchNorm2d(cout)]  # linear project
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        y = self.block(x)
        return x + y if self.use_res else y


class EfficientEncoder(nn.Module):
    """MobileNetV3-small-scale backbone -> dense stride-16 tokens with a lite FPN fuse.

    width `w` scales all channels. Default ~1.6M params, ~0.3 GFLOP at 224. Emits tokens at
    stride 16 (14x14 @224, 8x8 @128) with fine stride-8 detail folded in.
    """

    def __init__(self, d_model=256, w=1.0, in_ch=3, hswish=True):
        super().__init__()
        c = lambda x: int(round(x * w / 8) * 8)  # channel rounder
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, c(16), 3, 2, 1, bias=False), nn.BatchNorm2d(c(16)), _act(hswish)
        )  # s2
        # (cin, cout, stride, expand, k, se) -- stride-4,8,16 stages
        cfg = [
            (16, 24, 2, 4, 3, False),  # -> s4
            (24, 24, 1, 3, 3, False),
            (24, 40, 2, 4, 5, True),  # -> s8   (C3, fine)
            (40, 40, 1, 4, 5, True),
            (40, 40, 1, 4, 5, True),
            (40, 96, 2, 5, 5, True),  # -> s16  (C4, semantic)
            (96, 96, 1, 4, 5, True),
            (96, 96, 1, 4, 5, True),
        ]
        blocks = []
        for cin, cout, s, e, k, se in cfg:
            blocks.append(MBConv(c(cin), c(cout), s, e, k, se, hswish))
        self.blocks = nn.ModuleList(blocks)
        self.s8_idx = 4  # last block at stride 8
        self.proj3 = nn.Conv2d(c(40), d_model, 1, bias=False)  # fine detail
        self.proj4 = nn.Conv2d(c(96), d_model, 1, bias=False)  # semantic
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(self, x):
        """x: (B,3,H,W) in [0,1]. Returns tokens (B,N,d), coords (B,N,2) in [0,1]."""
        x = self.stem(x)
        c3 = None
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i == self.s8_idx:
                c3 = x  # stride 8
        c4 = x  # stride 16
        f = self.proj4(c4) + F.adaptive_avg_pool2d(self.proj3(c3), c4.shape[-2:])  # lite fuse
        B, C, H, W = f.shape
        tokens = self.norm(f.flatten(2).transpose(1, 2))  # (B, H*W, d)
        ys = torch.linspace(0, 1, H, device=x.device)
        xs = torch.linspace(0, 1, W, device=x.device)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([gy.reshape(-1), gx.reshape(-1)], -1)[None].expand(B, -1, -1)
        return tokens, coords
