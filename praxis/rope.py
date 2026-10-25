"""Rotary position embeddings: 1D for the token stream, axial 2D for dense vision.

1D RoPE (Su et al., RoFormer, 2021): rotate pairs of channels by an angle that grows
linearly with position, so attention depends on RELATIVE offset q_pos - k_pos. We use it
on the short self-attention sequence [prefix, state, action-queries]; the state token
prepended to that sequence shares the same 1D schedule (position 0..S-1).

2D RoPE (axial): the vision feature map has (row, col) coordinates. We split each head's
channels into two halves and apply an independent 1D rotation to each -- one driven by the
row index, one by the column -- so a query token cross-attending to vision sees RELATIVE 2D
offsets (Heo et al., "Rotary Position Embedding for Vision", 2024). This is what lets the
action/state queries localize on the image through cross-attention without learned position
tables, and it generalizes across resolutions.
"""

from __future__ import annotations
import torch


def _inv_freq(dim: int, base: float, device, dtype):
    # dim must be even; theta_i = base^{-2i/dim}
    i = torch.arange(0, dim, 2, device=device, dtype=torch.float32)
    return (base ** (-i / dim)).to(dtype)


def _rotate_half(x):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _apply(x, cos, sin):
    # x: (..., D) ; cos/sin: (..., D) with each freq duplicated to interleave with rotate_half
    return x * cos + _rotate_half(x) * sin


def build_1d(positions: torch.Tensor, dim: int, base: float = 10_000.0):
    """positions: (S,) or (B,S) integer/float. Returns (cos, sin) each (..., dim)."""
    inv = _inv_freq(dim, base, positions.device, torch.float32)  # (dim/2,)
    ang = positions[..., None].float() * inv  # (...,dim/2)
    ang = ang.repeat_interleave(2, dim=-1)  # (...,dim) interleaved
    return ang.cos(), ang.sin()


def build_2d(coords: torch.Tensor, dim: int, base: float = 100.0):
    """coords: (...,2) as (row, col). Splits `dim` in half; row drives the first half, col
    the second. Returns (cos, sin) each (...,dim). base is smaller than 1D (image grids are
    short) so the highest frequency still resolves adjacent pixels."""
    assert dim % 2 == 0
    half = dim // 2
    r = build_1d(coords[..., 0], half, base)
    c = build_1d(coords[..., 1], half, base)
    cos = torch.cat([r[0], c[0]], dim=-1)
    sin = torch.cat([r[1], c[1]], dim=-1)
    return cos, sin


def apply_rope(q, k, cos, sin):
    """q,k: (B, n_heads, S, Dh). cos,sin: (B, S, Dh) or (S, Dh) -> broadcast over heads."""
    if cos.dim() == q.dim() - 1:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)  # add head axis
    return _apply(q, cos, sin), _apply(k, cos, sin)


def apply_rope_q(q, cos, sin):
    """Rotate only queries (for cross-attention where K carries its own 2D rope)."""
    if cos.dim() == q.dim() - 1:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    return _apply(q, cos, sin)
