"""The VLA policy: a compact self-attention stream that cross-attends to dense vision.

Sequence design (per the spec):
  self-attn tokens = [ prefix task tokens | state token | action-query tokens ]
  * 1D RoPE over this whole sequence -- the STATE token is PREPENDED into the stream (not
    cross-attended) so it shares the same 1D positional schedule and can attend/be-attended
    cheaply.
  * PREFIX task tokens replace a text encoder: LIBERO tasks are a fixed finite set, so a
    learned per-task embedding -> P prefix tokens carries the instruction (no tokenizer, no
    language model).
  * Each block also CROSS-attends from the stream into the dense vision tokens (base + wrist
    cameras) with 2D RoPE on both sides: vision keys use their (row,col) grid; queries use a
    small learned per-slot 2D reference point, so cross-attention is relative-2D aware.
Only the short stream (P + 1 + chunk tokens, ~13) runs self-attention; the ~hundreds of
vision tokens are keys/values only. Cost is dominated by cheap cross-attention, not O(pixels^2).

Head: the action-query outputs regress the H-step action chunk directly (continuous ceiling;
the VQ/SEAL token head plugs in here later).
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import build_1d, build_2d, apply_rope, apply_rope_q
from .vision import EfficientEncoder


class Attention(nn.Module):
    def __init__(self, d, heads, cross=False, p_drop=0.0):
        super().__init__()
        self.h, self.dh, self.cross = heads, d // heads, cross
        self.p_drop = p_drop
        self.q = nn.Linear(d, d, bias=False)
        self.kv = nn.Linear(d, 2 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)

    def _split(self, x):
        B, S, _ = x.shape
        return x.view(B, S, self.h, self.dh).transpose(1, 2)  # (B,h,S,dh)

    def forward(self, x, mem, rope_q, rope_kv):
        q = self._split(self.q(x))
        k, v = self.kv(mem).chunk(2, -1)
        k = self._split(k)
        v = self._split(v)
        if self.cross:  # different coords for q,k
            q = apply_rope_q(q, *rope_q)
            k = apply_rope_q(k, *rope_kv)
        else:  # self: same schedule
            q, k = apply_rope(q, k, *rope_kv)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.p_drop if self.training else 0.0)
        return self.o(y.transpose(1, 2).flatten(2))


class Block(nn.Module):
    def __init__(self, d, heads, mlp=4, p_drop=0.0):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.sa = Attention(d, heads, cross=False, p_drop=p_drop)
        self.n2 = nn.LayerNorm(d)
        self.ca = Attention(d, heads, cross=True, p_drop=p_drop)
        self.n3 = nn.LayerNorm(d)
        # keep the FFN structure IDENTICAL to the pre-dropout model so old checkpoints load;
        # regularization comes from the residual dropout below (which covers the FFN output).
        self.ff = nn.Sequential(
            nn.Linear(d, mlp * d), nn.GELU(), nn.Dropout(p_drop), nn.Linear(mlp * d, d)
        )
        self.drop = nn.Dropout(p_drop)  # residual dropout (val bottomed at ep2 without it)

    def forward(self, x, vis, rope_self, rope_qref, rope_vis):
        x = x + self.drop(self.sa(self.n1(x), self.n1(x), None, rope_self))
        x = x + self.drop(self.ca(self.n2(x), vis, rope_qref, rope_vis))
        x = x + self.drop(self.ff(self.n3(x)))
        return x


class VLAPolicy(nn.Module):
    def __init__(
        self,
        n_tasks,
        action_dim=7,
        state_dim=8,
        chunk=8,
        d=256,
        heads=4,
        depth=5,
        n_prefix=4,
        w=2.15,
        img_res=128,
        p_drop=0.0,
        hist=1,
        vhist=1,
    ):
        # budgeted config: vision ~2.1M (w=2.15), policy transformer ~5.3M (d=256, depth=5).
        # hist>1: the state token encodes the last `hist` proprio states (flattened) -> temporal
        # memory for long-horizon tasks, WITHOUT changing the sequence length / q_ref shapes, so a
        # single-frame checkpoint still warm-starts (only state_mlp[0] is reinitialized).
        super().__init__()
        self.d, self.chunk, self.action_dim, self.hist = d, chunk, action_dim, hist
        self.vhist = vhist
        self.n_prefix = n_prefix
        self.vision = EfficientEncoder(d_model=d, w=w)
        self.cam_emb = nn.Parameter(0.02 * torch.randn(2, d))  # base / wrist tag
        if vhist > 1:  # per-frame temporal tag (visual memory)
            self.frame_emb = nn.Parameter(0.02 * torch.randn(vhist, d))
        self.task_emb = nn.Embedding(n_tasks, n_prefix * d)  # prefix tokens per task
        self.state_mlp = nn.Sequential(nn.Linear(state_dim * hist, d), nn.GELU(), nn.Linear(d, d))
        self.act_query = nn.Parameter(0.02 * torch.randn(chunk, d))  # learned action slots
        # learned 2D reference point per stream slot (for cross-attn query rope)
        self.q_ref = nn.Parameter(torch.rand(n_prefix + 1 + chunk, 2))
        self.blocks = nn.ModuleList([Block(d, heads, p_drop=p_drop) for _ in range(depth)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, action_dim)  # per action-query -> action
        self.S = n_prefix + 1 + chunk

    def encode_vision(self, imgs):
        """imgs: (B,2,3,H,W) single-frame OR (B,F,2,3,H,W) frame-stack (visual memory).
        Returns memory (B,F*2*Nvis,d), coords (B,F*2*Nvis,2). Each frame's tokens carry a temporal
        embedding; the SAME encoder + cross-attn params serve every frame (so a single-frame
        checkpoint warm-starts)."""
        if imgs.dim() == 5:  # single frame -> add F=1 axis
            imgs = imgs.unsqueeze(1)
        F = imgs.shape[1]
        toks, crds = [], []
        for f in range(F):
            fe = self.frame_emb[f] if hasattr(self, "frame_emb") else 0.0
            for cam in range(2):
                t, c = self.vision(imgs[:, f, cam])
                toks.append(t + self.cam_emb[cam] + fe)
                crds.append(c)
        return torch.cat(toks, 1), torch.cat(crds, 1)

    def features(self, imgs, state, task_id, cond=None, vis=None, vcoords=None):
        """Backbone up to (and including) the final norm; returns the per-action-query
        features act_tok (B,chunk,d) that self.head decodes. Exposed so alternative heads
        (e.g. a CVAE style-latent head) can tap the same representation.

        cond: optional (B,d) global conditioning added to every action query (e.g. a CVAE
        latent z projected to d). cond=None (or z=0 -> cond=0) leaves the backbone unchanged,
        so a plain-regression checkpoint warm-starts and z=0 inference == the base policy.
        vis/vcoords: optionally pass precomputed vision tokens to avoid re-encoding (e.g. when a
        conditional-prior network already encoded the observation for p(z|x))."""
        B = imgs.shape[0]
        if vis is None:
            vis, vcoords = self.encode_vision(imgs)  # (B,Nvis,d),(B,Nvis,2)
        prefix = self.task_emb(task_id).view(B, self.n_prefix, self.d)
        st = self.state_mlp(state).unsqueeze(1)  # (B,1,d)
        aq = self.act_query.expand(B, -1, -1)  # (B,chunk,d)
        if cond is not None:  # condition action queries on latent z
            aq = aq + (
                cond.unsqueeze(1) if cond.dim() == 2 else cond
            )  # (B,d) global OR (B,chunk,d) per-token
        x = torch.cat([prefix, st, aq], 1)  # (B,S,d)
        # 1D rope for the self-attn stream
        pos = torch.arange(self.S, device=imgs.device)
        rope_self = build_1d(pos, self.d // self.blocks[0].sa.h)
        # 2D rope: query reference points (stream) and vision grid coords
        rope_qref = build_2d(self.q_ref.to(imgs.device), self.d // self.blocks[0].ca.h)
        rope_vis = build_2d(vcoords, self.d // self.blocks[0].ca.h)
        for blk in self.blocks:
            x = blk(x, vis, rope_self, rope_qref, rope_vis)
        x = self.norm(x)
        return x[:, self.n_prefix + 1 :]  # (B,chunk,d)

    def forward(self, imgs, state, task_id, cond=None, vis=None, vcoords=None):
        return self.head(
            self.features(imgs, state, task_id, cond, vis, vcoords)
        )  # (B,chunk,action_dim)
