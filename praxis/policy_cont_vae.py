"""Fixed-variance CVAE regression head for the VLA (ACT-style, variance pinned to I).

This is the conditional-VAE action head of ACT (Zhao et al. 2023, arXiv:2304.13705) with the
posterior variance FIXED to the identity (learn only the mean mu). Structure:

Latent regularizer (reg=):
  * "kl"  -- KL(N(mu,I)||N(0,I)) = 0.5*||mu||^2. Only matches the FIRST moment (pulls each mu->0);
            cannot shape the aggregate code's covariance/higher moments, and over-squeezing collapses
            the latent back to the plain regressor.
  * "mmd" -- InfoVAE / WAE-MMD (Zhao-Song-Ermon 2017; Tolstikhin 2018). MMD(q(mu), N(0,I)) with a
            multi-scale IMQ kernel matches the aggregate code distribution to the prior across ALL
            moments (cov, skew, kurtosis) -- the continuous-latent analog of gaussianizing a latent
            with a random rotation so a fixed Gaussian codebook fits (BBGr gaussian-VQ), but here the
            decoder is learned so a fixed rotation would be absorbed; MMD is a penalty on the AGGREGATE
            and is not absorbable. It also PREVENTS collapse (it wants {mu} to fill N(0,I), not vanish).
            This is the "MMD-CVAE" flagged in FINDINGS_VLA.md 5.

  recognition encoder  q(z | a, state)  -- TRAIN ONLY. A small transformer that SEES the ground-
      truth action chunk (+ proprio) and outputs a global style latent mean mu in R^latent_dim.
      Fixed variance I, so z = mu + eps, eps~N(0,I). Seeing the action is what lets z carry the
      *mode identity* -> the decoder can commit to a mode instead of mode-averaging (the reason a
      plain L1 regressor fails on multimodal data). The encoder is DISCARDED at test.
  decoder  p(a | obs, z)               -- the VLAPolicy backbone, conditioned on z (projected to d
      and added to the action queries), regresses the action chunk.

Because both variances are fixed to I, the ELBO collapses to a regularized regression loss:
    L = L1(dec(obs, mu+eps), a)  +  beta * KL(N(mu,I) || N(0,I))
      = L1(...)                   +  beta * 0.5 * ||mu||^2            # KL reduces to L2 on mu
Inference: z = 0 (the prior mean) -> a single deterministic forward pass; with z=0 the decoder
reduces EXACTLY to the base regression policy (cond=0), so a plain-regression checkpoint warm-starts.

Drop-in with ContPolicy: .predict/.sample are deterministic (z=0) for eval/SEAL rollout;
.train_forward(imgs, state, task, action) -> (pred, kl) for training.
"""

import torch, torch.nn as nn
from .policy import VLAPolicy, Block
from .rope import build_1d, build_2d


def imq_mmd(z, zp, scales=(0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)):
    """Unbiased MMD^2 between samples z (aggregate code) and prior samples zp, summed over a bank of
    inverse-multiquadratic kernels k(x,y)=C/(C+||x-y||^2), C=2*d*s (WAE-MMD, Tolstikhin 2018). The IMQ
    kernel has heavy tails -> matches Gaussians well in moderate dim. z,zp: (n,d)."""
    n, d = z.shape
    d2_zz = torch.cdist(z, z).pow(2)
    d2_pp = torch.cdist(zp, zp).pow(2)
    d2_zp = torch.cdist(z, zp).pow(2)
    eye = torch.eye(n, device=z.device, dtype=torch.bool)
    mmd = z.new_zeros(())
    for s in scales:
        C = 2 * d * s
        k_zz = (C / (C + d2_zz)).masked_fill(eye, 0.0).sum() / (n * (n - 1))
        k_pp = (C / (C + d2_pp)).masked_fill(eye, 0.0).sum() / (n * (n - 1))
        k_zp = (C / (C + d2_zp)).mean()
        mmd = mmd + k_zz + k_pp - 2 * k_zp
    return mmd


class ActionEncoder(nn.Module):
    """q(z|a,state): transformer over [state, action_tokens] -> PER-TOKEN latent mean mu_q
    (B,chunk,latent_dim), fixed var I. Train-only; sees the target action chunk so each token's
    latent encodes that step's action structure."""

    def __init__(self, action_dim, chunk, state_dim, d=128, heads=4, depth=2, latent_dim=32):
        super().__init__()
        self.act_in = nn.Linear(action_dim, d)
        self.state_in = nn.Linear(state_dim, d)
        self.pos = nn.Parameter(0.02 * torch.randn(1, chunk + 1, d))  # state + chunk actions
        layer = nn.TransformerEncoderLayer(
            d, heads, 4 * d, dropout=0.0, batch_first=True, activation="gelu", norm_first=True
        )
        self.enc = nn.TransformerEncoder(layer, depth)
        self.to_mu = nn.Linear(d, latent_dim)

    def forward(self, action, state):  # action (B,chunk,A), state (B,S)
        s = self.state_in(state).unsqueeze(1)  # (B,1,d)
        x = torch.cat([s, self.act_in(action)], 1) + self.pos
        return self.to_mu(self.enc(x)[:, 1:])  # action positions -> (B,chunk,latent_dim)


class PriorTransformer(nn.Module):
    """Conditional prior p(z|x) = N(mu_p(x), I), PER chunk token. Learned prior-query tokens self-
    attend and CROSS-ATTEND to the observation (vision tokens + state + task prefix) with the same
    1D/2D RoPE as the decoder -- no vision pooling -- producing mu_p (B,chunk,latent_dim). Structurally
    a small copy of the decoder stream with a latent head, so mu_p(x) is a context-aware latent the
    deterministic inference reads (vs a fixed N(0,I) prior at z=0)."""

    def __init__(self, d, heads, state_dim, n_tasks, chunk, latent_dim, n_prefix=4, depth=2):
        super().__init__()
        self.d, self.chunk, self.n_prefix, self.heads = d, chunk, n_prefix, heads
        self.task_emb = nn.Embedding(n_tasks, n_prefix * d)
        self.state_mlp = nn.Sequential(nn.Linear(state_dim, d), nn.GELU(), nn.Linear(d, d))
        self.prior_q = nn.Parameter(0.02 * torch.randn(chunk, d))
        self.q_ref = nn.Parameter(torch.rand(n_prefix + 1 + chunk, 2))
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(d)
        self.to_mu = nn.Linear(d, latent_dim)
        self.S = n_prefix + 1 + chunk

    def forward(self, vis, vcoords, state, task_id):  # precomputed vision tokens
        B = vis.shape[0]
        prefix = self.task_emb(task_id).view(B, self.n_prefix, self.d)
        st = self.state_mlp(state).unsqueeze(1)
        pq = self.prior_q.expand(B, -1, -1)
        x = torch.cat([prefix, st, pq], 1)  # (B,S,d)
        dh = self.d // self.heads
        rope_self = build_1d(torch.arange(self.S, device=vis.device), dh)
        rope_qref = build_2d(self.q_ref.to(vis.device), dh)
        rope_vis = build_2d(vcoords, dh)
        for blk in self.blocks:
            x = blk(x, vis, rope_self, rope_qref, rope_vis)
        return self.to_mu(self.norm(x)[:, self.n_prefix + 1 :])  # (B,chunk,latent_dim)


class ContCVAEPolicy(nn.Module):
    def __init__(
        self,
        n_tasks,
        action_dim=7,
        chunk=8,
        p_drop=0.1,
        hist=1,
        vhist=1,
        state_dim=8,
        latent_dim=32,
        noise=1.0,
        enc_d=128,
        enc_depth=2,
        reg="kl",
        cond_prior=False,
        **kw,
    ):
        super().__init__()
        self.net = VLAPolicy(
            n_tasks=n_tasks,
            action_dim=action_dim,
            chunk=chunk,
            p_drop=p_drop,
            hist=hist,
            vhist=vhist,
        )  # decoder (warm-startable backbone)
        self.chunk, self.action_dim, self.hist, self.vhist = chunk, action_dim, hist, vhist
        self.latent_dim, self.noise, self.reg = latent_dim, noise, reg
        self.cond_prior = cond_prior  # conditional prior p(z|x) (Sohn CVAE)
        self.recog = ActionEncoder(
            action_dim, chunk, state_dim * hist, d=enc_d, depth=enc_depth, latent_dim=latent_dim
        )
        if cond_prior:
            self.prior = PriorTransformer(
                self.net.d, self.net.blocks[0].sa.h, state_dim * hist, n_tasks, chunk, latent_dim
            )
        # z -> decoder conditioning (per token); bias-free so z=0 -> 0 (inference == base policy)
        self.z_proj = nn.Linear(latent_dim, self.net.d, bias=False)
        nn.init.zeros_(self.z_proj.weight)  # start as pure regression, ramp in

    def _ctx(self, imgs, state, task):
        """Encode vision ONCE; return (vis, vcoords, mu_p) with per-token mu_p (B,chunk,latent)."""
        vis, vcoords = self.net.encode_vision(imgs)
        mu_p = self.prior(vis, vcoords, state, task)
        return vis, vcoords, mu_p

    def train_forward(self, imgs, state, task, action):
        """CVAE training pass -> (pred, reg). Latents are PER chunk token (B,chunk,latent).
        cond_prior=False : z=mu_q+eps; reg = 0.5||mu_q||^2 (KL to N(0,I)) or MMD(mu_q,N(0,I)).
        cond_prior=True  : z=mu_q+eps; reg = 0.5||mu_q - mu_p(x)||^2 = KL(N(mu_q,I)||N(mu_p,I)),
                           mu_p from the cross-attention prior; z=mu_p(x) drives deterministic test."""
        mu_q = self.recog(action, state)  # (B,chunk,latent)
        z = mu_q + self.noise * torch.randn_like(mu_q) if self.training else mu_q
        if self.cond_prior:
            vis, vcoords, mu_p = self._ctx(imgs, state, task)
            pred = self.net(imgs, state, task, cond=self.z_proj(z), vis=vis, vcoords=vcoords)
            reg = 0.5 * (mu_q - mu_p).pow(2).sum(dim=(1, 2)).mean()
        else:
            pred = self.net(imgs, state, task, cond=self.z_proj(z))
            reg = (
                imq_mmd(mu_q.flatten(0, 1), torch.randn_like(mu_q).flatten(0, 1))
                if self.reg == "mmd"
                else 0.5 * mu_q.pow(2).sum(dim=(1, 2)).mean()
            )
        return pred, reg

    # ---- deterministic inference ----
    @torch.no_grad()
    def predict(self, imgs, state, task):
        if self.cond_prior:  # z = mu_p(x): context-aware latent
            vis, vcoords, mu_p = self._ctx(imgs, state, task)
            return self.net(imgs, state, task, cond=self.z_proj(mu_p), vis=vis, vcoords=vcoords)
        return self.net(imgs, state, task, cond=None)  # unconditional: z = 0

    @torch.no_grad()
    def prior_post(self, imgs, state, task, action):
        """(mu_p, mu_q) for diagnostics: how well the prior predicts the posterior latent."""
        mu_q = self.recog(action, state)
        if self.cond_prior:
            _, _, mu_p = self._ctx(imgs, state, task)
        else:
            mu_p = torch.zeros_like(mu_q)
        return mu_p, mu_q

    @torch.no_grad()
    def decode_z(self, imgs, state, task, z):
        """Decode a given latent z -> action chunk (z_proj conditioning). Used to probe the
        oracle reconstruction (z=mu_q) and prior sampling."""
        return self.net(imgs, state, task, cond=self.z_proj(z))

    @torch.no_grad()
    def sample(self, imgs, state, task, steps=None, deterministic=True):
        return self.predict(imgs, state, task)
