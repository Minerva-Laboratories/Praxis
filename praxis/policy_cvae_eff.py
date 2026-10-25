"""Standalone conditional CVAE for action chunks, assembled from the efficient building blocks (BlockEff
cross-attention, xATLU, ScaleNorm, ConvActionHead, EfficientEncoderV2 vision). This is NOT a head bolted onto
the regression policy -- it is a conditional VAE in its own right, trained from scratch.

Conditioning: the CONDITION is the current observation. Dense vision tokens enter via CROSS-ATTENTION (not by
concatenating a pooled vector); the task-id and proprio enter as prefix tokens in the self-attention stream.
Three streams share the vision encoding:
  * encoder  q(z | obs, a)  -- a latent-query token + the (embedded) target action chunk self-attend and
        cross-attend to vision -> posterior (mu_q, logvar_q). SEES the action -> the latent carries the MODE.
        Train-only; discarded at inference.
  * prior    p(z | obs)      -- a prior-query token cross-attends to vision -> (mu_p, logvar_p). A CONDITIONAL
        prior, so its mean mu_p(obs) is generally NON-ZERO (this is what "pick the mean, not 0" means).
  * decoder  p(a | z, obs)   -- z seeds the action queries; they cross-attend to vision -> action chunk.
Loss = recon (log-cosh on pos/rot + gripper BCE) + beta * MMD( q(z|obs,a) samples, p(z|obs) samples ).
MMD (InfoVAE/WAE-MMD), NOT KL: it matches the aggregate code across all moments and does not mean-collapse.
Inference: z = prior MEAN mu_p(obs) (deterministic) or z ~ p(z|obs) (sample a mode); decode -> chunk.
"""

import torch, torch.nn as nn, torch.nn.functional as F
from .rope import build_1d, build_2d
from .policy_eff import BlockEff, ConvActionHead, _norm, _act, _logcosh


def imq_mmd(z, zp, scales=(0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)):
    """Unbiased MMD^2 between aggregate samples z and prior samples zp, summed over a bank of inverse-
    multiquadratic kernels k(x,y)=C/(C+||x-y||^2), C=2*d*s (WAE-MMD, Tolstikhin 2018 / InfoVAE). Matches the
    AGGREGATE code across ALL moments -> not absorbable and PREVENTS the posterior mean-collapse that KL
    (which only matches the 1st moment) suffers. z, zp: (n, d)."""
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


class Stream(nn.Module):
    """Prefix+query tokens self-attend (1D RoPE) and cross-attend to shared vision tokens (2D RoPE), reusing
    BlockEff. Returns the full normalized sequence; the caller slices the token(s) it needs."""

    def __init__(self, d, heads, depth, S, norm="scale", act="xatlu", ls_init=0.1):
        super().__init__()
        self.d, self.heads, self.S = d, heads, S
        self.q_ref = nn.Parameter(torch.rand(S, 2))
        self.blocks = nn.ModuleList(
            [BlockEff(d, heads, norm=norm, act=act, ls_init=ls_init) for _ in range(depth)]
        )
        self.norm = _norm(d, norm)

    def forward(self, x, vis, vcoords):
        dh = self.d // self.heads
        rs = build_1d(torch.arange(self.S, device=x.device), dh)
        rq = build_2d(self.q_ref.to(x.device), dh)
        rv = build_2d(vcoords, dh)
        for blk in self.blocks:
            x = blk(x, vis, rs, rq, rv)
        return self.norm(x)


class CVAEEff(nn.Module):
    def __init__(
        self,
        n_tasks,
        action_dim=7,
        state_dim=8,
        chunk=8,
        d=256,
        heads=4,
        enc_depth=2,
        prior_depth=2,
        dec_depth=5,
        n_prefix=4,
        latent_dim=32,
        beta=1.0,
        w=2.15,
        head_depth=2,
        head_k=3,
        norm="scale",
        act="xatlu",
        ls_init=0.1,
        vision_v2=True,
        vis_attn=True,
        vis_multicat=True,
        vis_act="hswish",
        p_drop=0.1,
        hist=1,
        vhist=1,
        **kw,
    ):
        super().__init__()
        self.d, self.chunk, self.action_dim = d, chunk, action_dim
        self.latent_dim, self.n_prefix, self.beta = latent_dim, n_prefix, beta
        if vision_v2:
            from .vision_eff import EfficientEncoderV2

            self.vision = EfficientEncoderV2(
                d_model=d, w=w, attn=vis_attn, multicat=vis_multicat, act=vis_act
            )
        else:
            from .vision import EfficientEncoder

            self.vision = EfficientEncoder(d_model=d, w=w)
        self.cam_emb = nn.Parameter(0.02 * torch.randn(2, d))
        self.task_emb = nn.Embedding(n_tasks, n_prefix * d)  # condition: task-id (no language)
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, d), _act(act), nn.Linear(d, d)
        )  # condition: proprio
        # encoder q(z|obs,a): [prefix, state, latent-query, action-tokens]
        self.act_in = nn.Linear(action_dim, d)
        self.enc_latq = nn.Parameter(0.02 * torch.randn(1, d))
        self.enc = Stream(d, heads, enc_depth, n_prefix + 2 + chunk, norm, act, ls_init)
        self.enc_mu = nn.Linear(d, latent_dim)
        self.enc_ls = nn.Linear(d, latent_dim)
        # conditional prior p(z|obs): [prefix, state, prior-query]
        self.prior_q = nn.Parameter(0.02 * torch.randn(1, d))
        self.prior = Stream(d, heads, prior_depth, n_prefix + 2, norm, act, ls_init)
        self.prior_mu = nn.Linear(d, latent_dim)
        self.prior_ls = nn.Linear(d, latent_dim)
        # decoder p(a|z,obs): [prefix, state, z-seeded action queries]
        self.z_proj = nn.Linear(latent_dim, d)
        self.act_query = nn.Parameter(0.02 * torch.randn(chunk, d))
        self.dec = Stream(d, heads, dec_depth, n_prefix + 1 + chunk, norm, act, ls_init)
        self.head = ConvActionHead(
            d, action_dim, k=head_k, depth=head_depth, norm=norm, act=act, ls_init=ls_init
        )

    def encode_vision(self, imgs):
        if imgs.dim() == 5:
            imgs = imgs.unsqueeze(1)
        Fr = imgs.shape[1]
        toks, crds = [], []
        for f in range(Fr):
            for cam in range(2):
                t, c = self.vision(imgs[:, f, cam])
                toks.append(t + self.cam_emb[cam])
                crds.append(c)
        return torch.cat(toks, 1), torch.cat(crds, 1)

    def _cond(self, task, state):
        B = state.shape[0]
        prefix = self.task_emb(task).view(B, self.n_prefix, self.d)
        st = self.state_mlp(state).unsqueeze(1)
        return prefix, st

    def _prior(self, prefix, st, vis, vc):
        B = st.shape[0]
        x = torch.cat([prefix, st, self.prior_q.expand(B, -1, -1)], 1)
        h = self.prior(x, vis, vc)[:, -1]  # prior-query token
        return self.prior_mu(h), self.prior_ls(h)

    def _encode(self, prefix, st, action, vis, vc):
        B = st.shape[0]
        x = torch.cat([prefix, st, self.enc_latq.expand(B, -1, -1), self.act_in(action)], 1)
        h = self.enc(x, vis, vc)[:, self.n_prefix + 1]  # latent-query token
        return self.enc_mu(h), self.enc_ls(h)

    def _decode(self, prefix, st, z, vis, vc):
        B = st.shape[0]
        aq = self.act_query.expand(B, -1, -1) + self.z_proj(z).unsqueeze(
            1
        )  # z seeds every action query
        x = torch.cat([prefix, st, aq], 1)
        h = self.dec(x, vis, vc)[:, self.n_prefix + 1 :]
        return self.head(h)  # (B,chunk,action_dim)

    def loss(self, imgs, state, task, a1, rot_w=2.0, grip_w=1.0, beta=None, **kw):
        beta = self.beta if beta is None else beta
        vis, vc = self.encode_vision(imgs)
        prefix, st = self._cond(task, state)
        mu_q, ls_q = self._encode(prefix, st, a1, vis, vc)
        mu_p, ls_p = self._prior(prefix, st, vis, vc)
        zq = mu_q + torch.randn_like(mu_q) * (0.5 * ls_q).exp() if self.training else mu_q
        pred = self._decode(prefix, st, zq, vis, vc)  # reconstruct from the POSTERIOR sample
        w = torch.ones(6, device=pred.device)
        w[3:6] = rot_w
        w = w / w.mean()
        cont = (_logcosh(pred[..., :6] - a1[..., :6]) * w).mean()
        grip = F.binary_cross_entropy_with_logits(pred[..., 6], (a1[..., 6] > 0).float())
        zp = (
            mu_p + torch.randn_like(mu_p) * (0.5 * ls_p).exp()
        )  # samples from the conditional prior p(z|obs)
        mmd = imq_mmd(zq, zp)  # MMD: match AGGREGATE posterior <-> prior (no mean-collapse)
        return cont + grip_w * grip + beta * mmd

    @torch.no_grad()
    def sample(self, imgs, state, task, steps=None, deterministic=True, temp=1.0):
        vis, vc = self.encode_vision(imgs)
        prefix, st = self._cond(task, state)
        mu_p, ls_p = self._prior(prefix, st, vis, vc)
        z = mu_p if deterministic else mu_p + temp * torch.randn_like(mu_p) * (0.5 * ls_p).exp()
        pred = self._decode(prefix, st, z, vis, vc)
        grip = 2 * torch.sigmoid(pred[..., 6:7]) - 1
        return torch.cat([pred[..., :6], grip], -1)

    @torch.no_grad()
    def diag(self, imgs, state, task, a1, rot_w=2.0):
        """orc = recon at z=mu_q (posterior, sees target = capacity ceiling); pm = recon at z=mu_p (prior mean,
        the deployed prediction); pp = ||mu_q - mu_p|| (does the prior predict the posterior)."""
        vis, vc = self.encode_vision(imgs)
        prefix, st = self._cond(task, state)
        mu_q, _ = self._encode(prefix, st, a1, vis, vc)
        mu_p, _ = self._prior(prefix, st, vis, vc)
        w = torch.ones(6, device=a1.device)
        w[3:6] = rot_w
        w = w / w.mean()
        rl = lambda z: (
            (_logcosh(self._decode(prefix, st, z, vis, vc)[..., :6] - a1[..., :6]) * w)
            .mean()
            .item()
        )
        return dict(
            orc=rl(mu_q),
            pm=rl(mu_p),
            pp=(mu_q - mu_p).norm(dim=-1).mean().item(),
            muq_norm=mu_q.norm(dim=-1).mean().item(),
        )
