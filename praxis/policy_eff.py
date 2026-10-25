"""Efficient VLA backbone variant (param-matched to VLAPolicy) — better tricks per parameter.

Swaps vs praxis/policy.py, keeping the sequence design (prefix task | state | action queries) and the
cheap cross-attention to dense vision identical:
  * ScaleNorm prenorm      (Nguyen & Salazar 2019): g*x/||x|| -- 1 scalar vs LayerNorm's 2d.
  * LayerScale residuals   (Touvron 2021, CaiT):   x + diag(lambda)*sublayer(x), lambda small init.
  * xATLU activation       (Huang 2024, arXiv:2405.20768): x*((atan(x)+pi/2)/pi*(1+2a)-a), a per-block, init 0.
  * ConvActionHead: the action chunk is decoded by DEPTHWISE-SEPARABLE TEMPORAL convolutions over the
    chunk axis (depthwise Conv1d groups=d -> xATLU -> pointwise 1x1, residual+LayerScale), so adjacent
    action steps share temporal structure -- vs the baseline's per-token Linear readout.

Same vision encoder (praxis/vision.py) and RoPE. Depth/width tuned so total params ~= the baseline.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import build_1d, build_2d
from .vision import EfficientEncoder
from .policy import Attention  # reuse RoPE self/cross attention


class XATLU(nn.Module):
    """Expanded ArcTan Linear Unit: x*(g(x)*(1+2a)-a), g(x)=(atan(x)+pi/2)/pi in (0,1). a init 0."""

    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        g = (torch.atan(x) + math.pi / 2) / math.pi
        return x * (g * (1 + 2 * self.alpha) - self.alpha)


class ScaleNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.g = nn.Parameter(torch.tensor(float(d) ** 0.5))
        self.eps = eps

    def forward(self, x):
        return self.g * x / (x.norm(dim=-1, keepdim=True) + self.eps)


class LayerScale(nn.Module):
    def __init__(self, d, init=0.1):
        super().__init__()
        self.lam = nn.Parameter(init * torch.ones(d))

    def forward(self, x):
        return self.lam * x


def _norm(d, kind):
    return ScaleNorm(d) if kind == "scale" else nn.LayerNorm(d)


def _act(kind):
    return XATLU() if kind == "xatlu" else nn.GELU()


def _ls(d, init):
    return nn.Identity() if init is None else LayerScale(d, init)


class BlockEff(nn.Module):
    def __init__(self, d, heads, mlp=4, p_drop=0.0, ls_init=0.1, norm="scale", act="xatlu"):
        super().__init__()
        self.n1 = _norm(d, norm)
        self.sa = Attention(d, heads, cross=False, p_drop=p_drop)
        self.n2 = _norm(d, norm)
        self.ca = Attention(d, heads, cross=True, p_drop=p_drop)
        self.n3 = _norm(d, norm)
        self.ff = nn.Sequential(
            nn.Linear(d, mlp * d), _act(act), nn.Dropout(p_drop), nn.Linear(mlp * d, d)
        )
        self.drop = nn.Dropout(p_drop)
        self.ls1, self.ls2, self.ls3 = _ls(d, ls_init), _ls(d, ls_init), _ls(d, ls_init)

    def forward(self, x, vis, rope_self, rope_qref, rope_vis):
        x = x + self.drop(self.ls1(self.sa(self.n1(x), self.n1(x), None, rope_self)))
        x = x + self.drop(self.ls2(self.ca(self.n2(x), vis, rope_qref, rope_vis)))
        x = x + self.drop(self.ls3(self.ff(self.n3(x))))
        return x


class ConvActionHead(nn.Module):
    """Decode the H-step action chunk with depthwise-separable temporal convs + xATLU."""

    def __init__(
        self, d, action_dim, k=3, depth=2, p_drop=0.0, ls_init=0.1, norm="scale", act="xatlu"
    ):
        super().__init__()
        self.norm = nn.ModuleList([_norm(d, norm) for _ in range(depth)])
        self.dw = nn.ModuleList(
            [nn.Conv1d(d, d, k, padding=k // 2, groups=d) for _ in range(depth)]
        )
        self.act = nn.ModuleList([_act(act) for _ in range(depth)])
        self.pw = nn.ModuleList([nn.Conv1d(d, d, 1) for _ in range(depth)])
        self.ls = nn.ModuleList([_ls(d, ls_init) for _ in range(depth)])
        self.drop = nn.Dropout(p_drop)
        self.proj = nn.Linear(d, action_dim)

    def forward(self, x):  # (B,chunk,d)
        for i in range(len(self.dw)):
            h = self.norm[i](x).transpose(1, 2)  # (B,d,chunk)
            h = self.pw[i](self.act[i](self.dw[i](h))).transpose(
                1, 2
            )  # depthwise-sep temporal conv
            x = x + self.drop(self.ls[i](h))
        return self.proj(x)  # (B,chunk,action_dim)


class VLAPolicyEff(nn.Module):
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
        mlp=4,
        head_depth=2,
        head_k=3,
        ls_init=0.1,
        norm="scale",
        act="xatlu",
        vision_v2=False,
        vis_attn=True,
        vis_multicat=True,
        vis_act="hswish",
        temporal=False,
        t_heads=4,
        cam_res=None,
        rot6d=False,
        stats=None,
    ):
        super().__init__()
        self.d, self.chunk, self.action_dim, self.hist, self.vhist = (
            d,
            chunk,
            action_dim,
            hist,
            vhist,
        )
        self.n_prefix = n_prefix
        self.cam_res = cam_res  # per-camera [global_res, wrist_res] or None (symmetric)
        self.rot6d = bool(
            rot6d
        )  # feed EE orientation as 6D (uniform) instead of axis-angle (non-uniform @ theta~pi)
        if self.rot6d:  # need raw-state un-normalize for the aa dims -> R -> 6D
            assert hist == 1, "rot6d state input assumes hist==1"
            sm, ss = (
                (stats[2], stats[3])
                if stats is not None
                else (torch.zeros(state_dim), torch.ones(state_dim))
            )
            self.register_buffer(
                "s_mean6", torch.as_tensor(sm, dtype=torch.float32), persistent=False
            )
            self.register_buffer(
                "s_std6", torch.as_tensor(ss, dtype=torch.float32), persistent=False
            )
        self.temporal_vis = bool(temporal)
        if temporal:  # V3: per-cam temporal delta-attention over T frames
            from .vision_eff import EfficientEncoderV3

            self.vision = EfficientEncoderV3(
                d_model=d, w=w, attn=vis_attn, multicat=vis_multicat, act=vis_act, t_heads=t_heads
            )
        elif vision_v2:
            from .vision_eff import EfficientEncoderV2  # local import avoids circular dep

            self.vision = EfficientEncoderV2(
                d_model=d, w=w, attn=vis_attn, multicat=vis_multicat, act=vis_act
            )
        else:
            self.vision = EfficientEncoder(d_model=d, w=w)
        self.cam_emb = nn.Parameter(0.02 * torch.randn(2, d))
        if vhist > 1 and not temporal:  # temporal encodes time via RoPE, not frame_emb
            self.frame_emb = nn.Parameter(0.02 * torch.randn(vhist, d))
        self.task_emb = nn.Embedding(n_tasks, n_prefix * d)
        state_in = (state_dim + 3) if self.rot6d else (state_dim * hist)
        self.state_mlp = nn.Sequential(nn.Linear(state_in, d), _act(act), nn.Linear(d, d))
        self.act_query = nn.Parameter(0.02 * torch.randn(chunk, d))
        self.q_ref = nn.Parameter(torch.rand(n_prefix + 1 + chunk, 2))
        self.blocks = nn.ModuleList(
            [
                BlockEff(d, heads, mlp=mlp, p_drop=p_drop, ls_init=ls_init, norm=norm, act=act)
                for _ in range(depth)
            ]
        )
        self.norm = _norm(d, norm)
        self.head = ConvActionHead(
            d,
            action_dim,
            k=head_k,
            depth=head_depth,
            p_drop=p_drop,
            ls_init=ls_init,
            norm=norm,
            act=act,
        )
        self.S = n_prefix + 1 + chunk

    def encode_vision(self, imgs):
        if imgs.dim() == 5:
            imgs = imgs.unsqueeze(1)  # (B,1,2,3,H,W)
        if self.temporal_vis:  # V3: temporal-integrate each cam over the T frames
            toks, crds = [], []
            for cam in range(2):
                t, c = self.vision(imgs[:, :, cam])  # (B,T,3,H,W) -> (B,64,d),(B,64,2)
                toks.append(t + self.cam_emb[cam])
                crds.append(c)
            return torch.cat(toks, 1), torch.cat(crds, 1)
        Fr = imgs.shape[1]
        toks, crds = [], []
        for f in range(Fr):
            fe = self.frame_emb[f] if hasattr(self, "frame_emb") else 0.0
            for cam in range(2):
                x = imgs[:, f, cam]
                if (
                    self.cam_res is not None and x.shape[-1] != self.cam_res[cam]
                ):  # per-camera resolution
                    x = F.interpolate(
                        x, size=(self.cam_res[cam],) * 2, mode="bilinear", align_corners=False
                    )
                t, c = self.vision(x)
                toks.append(t + self.cam_emb[cam] + fe)
                crds.append(c)
        return torch.cat(toks, 1), torch.cat(crds, 1)

    def _to6d(self, state):
        """(B,state_dim) normalized state -> (B,state_dim+3): replace the axis-angle EE orientation (dims 3-5,
        non-uniform @ theta~pi) with the CONTINUOUS, uniform 6D rep (first two columns of R; Zhou et al. 2019)."""
        aa_raw = state[:, 3:6] * self.s_std6[3:6] + self.s_mean6[3:6]  # un-normalize axis-angle
        R = _aa_to_R(aa_raw)  # (B,3,3)
        rot6 = R[:, :, :2].reshape(state.shape[0], 6)  # first two columns -> 6D
        return torch.cat([state[:, :3], rot6, state[:, 6:]], dim=-1)  # [pos, 6D, grip...]

    def features(self, imgs, state, task_id, cond=None, vis=None, vcoords=None, prefix=None):
        B = imgs.shape[0]
        if vis is None:
            vis, vcoords = self.encode_vision(imgs)
        if prefix is None:  # default: static per-task prefix tokens
            prefix = self.task_emb(task_id).view(B, self.n_prefix, self.d)
        st = self.state_mlp(self._to6d(state) if self.rot6d else state).unsqueeze(1)
        aq = self.act_query.expand(B, -1, -1)
        if cond is not None:
            aq = aq + (cond.unsqueeze(1) if cond.dim() == 2 else cond)
        x = torch.cat([prefix, st, aq], 1)
        dh = self.d // self.blocks[0].sa.h
        rope_self = build_1d(torch.arange(self.S, device=imgs.device), dh)
        rope_qref = build_2d(self.q_ref.to(imgs.device), dh)
        rope_vis = build_2d(vcoords, dh)
        for blk in self.blocks:
            x = blk(x, vis, rope_self, rope_qref, rope_vis)
        return self.norm(x)[:, self.n_prefix + 1 :]  # (B,chunk,d)

    def forward(self, imgs, state, task_id, cond=None, vis=None, vcoords=None, prefix=None):
        return self.head(self.features(imgs, state, task_id, cond, vis, vcoords, prefix))


def _logcosh(x):
    """Stable log(cosh(x)) = |x| + softplus(-2|x|) - log2. ~x^2/2 near 0 (smooth like L2), ~|x| far."""
    ax = x.abs()
    return ax + F.softplus(-2 * ax) - math.log(2)


def _aa_to_R(aa):
    """axis-angle (...,3) -> rotation matrix (...,3,3) via Rodrigues."""
    th = aa.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    k = aa / th
    O = torch.zeros_like(k[..., 0])
    K = torch.stack(
        [
            torch.stack([O, -k[..., 2], k[..., 1]], -1),
            torch.stack([k[..., 2], O, -k[..., 0]], -1),
            torch.stack([-k[..., 1], k[..., 0], O], -1),
        ],
        -2,
    )  # (...,3,3)
    s = th.sin().unsqueeze(-1)
    c = th.cos().unsqueeze(-1)
    I = torch.eye(3, device=aa.device, dtype=aa.dtype).expand_as(K)
    return I + s * K + (1 - c) * (K @ K)


class ContPolicyEff(nn.Module):
    """Efficient backbone; continuous action dims (0-5 = pos+rot) via log-cosh, gripper (dim 6) as a
    BCE logit. Grippers are bimodal {-1,+1} (normalized ~ same) so classification > L1 there.

    local_frame (MCF-lite): predict the POSITION delta (dims 0-2) in the EE-LOCAL frame (rotated by the
    current gripper orientation from proprio) instead of the base frame -> canonical, position-invariant
    action mapping, less compounding error. Targets rotated to local in loss; predictions rotated back to
    base in sample. Needs `stats` (a_mean,a_std,s_mean,s_std) to un/re-normalize around the rotation."""

    def __init__(
        self,
        n_tasks,
        action_dim=7,
        chunk=8,
        p_drop=0.1,
        hist=1,
        vhist=1,
        local_frame=False,
        stats=None,
        **kw,
    ):
        super().__init__()
        self.net = VLAPolicyEff(
            n_tasks=n_tasks,
            action_dim=action_dim,
            chunk=chunk,
            p_drop=p_drop,
            hist=hist,
            vhist=vhist,
            stats=stats,
            **kw,
        )
        self.chunk, self.action_dim, self.hist, self.vhist = chunk, action_dim, hist, vhist
        self.local_frame = local_frame
        if local_frame and stats is not None:
            am, as_, sm, ss = stats  # persistent=False -> not in state_dict
            self.register_buffer(
                "a_mean", torch.as_tensor(am, dtype=torch.float32), persistent=False
            )
            self.register_buffer(
                "a_std", torch.as_tensor(as_, dtype=torch.float32), persistent=False
            )
            self.register_buffer(
                "s_mean", torch.as_tensor(sm, dtype=torch.float32), persistent=False
            )
            self.register_buffer(
                "s_std", torch.as_tensor(ss, dtype=torch.float32), persistent=False
            )

    def _R(self, state):  # normalized state -> EE rotation (B,3,3)
        aa = state[:, 3:6] * self.s_std[3:6] + self.s_mean[3:6]  # raw axis-angle of the gripper
        return _aa_to_R(aa)

    def _rot_pos(self, a, R):  # rotate normalized POS delta (0-2) by R
        raw = a[..., :3] * self.a_std[:3] + self.a_mean[:3]
        rot = torch.einsum("bij,bcj->bci", R.to(raw.dtype), raw)
        return torch.cat([(rot - self.a_mean[:3]) / self.a_std[:3], a[..., 3:]], -1)

    def loss(
        self,
        imgs,
        state,
        task,
        a1,
        rot_w=2.0,
        grip_w=1.0,
        grasp_w=1.0,
        grasp_win=1,
        state_w=None,
        state_traj=False,
    ):
        if self.local_frame:
            a1 = self._rot_pos(a1, self._R(state).transpose(-1, -2))  # base target -> local (R^T)
        pred = self.net(imgs, state, task)  # (B,chunk,7): [:6]=cont, [6]=grip logit
        return _eff_action_loss(
            pred, a1, rot_w, grip_w, grasp_w, grasp_win, state_w=state_w, state_traj=state_traj
        )

    @torch.no_grad()
    def sample(self, imgs, state, task, steps=None, deterministic=True):
        pred = self.net(imgs, state, task)
        if self.local_frame:
            pred = self._rot_pos(pred, self._R(state))  # local prediction -> base (R)
        grip = 2 * torch.sigmoid(pred[..., 6:7]) - 1  # logit -> normalized {-1,+1}-ish
        return torch.cat([pred[..., :6], grip], -1)


def _eff_action_loss(
    pred, a1, rot_w=2.0, grip_w=1.0, grasp_w=1.0, grasp_win=1, state_w=None, state_traj=False
):
    """log-cosh (pos/rot) + gripper-BCE, with optional grasp-phase reweighting. Shared by the
    regression and sub-task policies (identical to ContPolicyEff.loss's body).

    state_w (per-dim tensor, len 6): STATE-SPACE supervision. Weight each continuous dim by its state-space
    impact B*a_std (so ~ ||B(a_pi-a*)|| = the next-position error, 'supervise where the EE goes, not the action').
    state_traj: the TRAJECTORY (shape) form -- supervise the CUMULATIVE position path along the chunk
    (cumsum of the per-step deltas) rather than each step in isolation. Gradient flows THROUGH the cumsum, so an
    early action is weighted by its DOWNSTREAM effect on the whole recovery path -- not a per-dim reweighting and
    not per-step imitation. This is the genuine shape difference from action-SEAL (see [[seal-as-state-supervision]])."""
    if state_w is not None:
        w = state_w.to(pred.device).float()
        w = w / w.mean()
    else:
        w = torch.ones(6, device=pred.device)
        w[3:6] = rot_w
        w = w / w.mean()
    de = pred[..., :6] - a1[..., :6]
    if state_traj:  # cumulative position-path deviation along the chunk
        de = torch.cumsum(de, dim=-2)
    cont_s = (_logcosh(de) * w).mean(-1)
    grip_s = F.binary_cross_entropy_with_logits(
        pred[..., 6], (a1[..., 6] > 0).float(), reduction="none"
    )
    if grasp_w > 1.0:
        g = (a1[..., 6] > 0).float()
        dd = (g[:, 1:] != g[:, :-1]).float()
        tr = torch.zeros_like(g)
        tr[:, 1:] = dd
        tr[:, :-1] = torch.maximum(tr[:, :-1], dd)
        if grasp_win > 1:
            tr = F.max_pool1d(tr.unsqueeze(1), 2 * grasp_win + 1, 1, grasp_win).squeeze(1)
        sw = 1.0 + (grasp_w - 1.0) * tr
        sw = sw / sw.mean()
        return (cont_s * sw).mean() + grip_w * (grip_s * sw).mean()
    return cont_s.mean() + grip_w * grip_s.mean()


class ContSubtaskEff(nn.Module):
    """Subtask-conditioned policy with independent per-phase task tokens.

    Training uses hard phase labels and a learned selector. Inference mixes phase
    tokens using the selector belief. Single-phase tasks use their base token.
    Warm starts copy the base task token into each phase slot, with optional small
    perturbations to break symmetry."""

    def __init__(
        self,
        n_tasks,
        action_dim=7,
        chunk=8,
        p_drop=0.1,
        hist=1,
        vhist=1,
        n_sub=None,
        k_max=3,
        lambda_ce=1.0,
        **kw,
    ):
        super().__init__()
        self.net = VLAPolicyEff(
            n_tasks=n_tasks,
            action_dim=action_dim,
            chunk=chunk,
            p_drop=p_drop,
            hist=hist,
            vhist=vhist,
            **kw,
        )
        d, npx = self.net.d, self.net.n_prefix
        self.chunk, self.action_dim, self.k_max, self.lambda_ce = (
            chunk,
            action_dim,
            k_max,
            lambda_ce,
        )
        # INDEPENDENT per-(task, phase) full prefix token set -- NOT a residual/delta on a shared token.
        # Each phase develops its own token. Default std-0.02 init (used when training a fresh model);
        # when warm-starting a checkpoint, call init_phase_from_base() to copy the task token into each slot.
        self.phase_tok = nn.Parameter(torch.empty(n_tasks, k_max, npx * d))
        nn.init.normal_(self.phase_tok, std=0.02)
        self.selector = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, k_max))
        ns = (
            torch.ones(n_tasks, dtype=torch.long)
            if n_sub is None
            else torch.as_tensor(n_sub, dtype=torch.long)
        )
        self.register_buffer("n_sub", ns.clamp(1, k_max))

    @torch.no_grad()
    def init_phase_from_base(self, perturb=0.05):
        """Initialize phase tokens from a loaded base policy, then break phase symmetry."""
        te = self.net.task_emb.weight  # (n_tasks, npx*d)
        self.phase_tok.data.copy_(te.unsqueeze(1).expand(-1, self.k_max, -1))
        if perturb > 0:
            self.phase_tok.data += perturb * torch.randn_like(self.phase_tok)

    def _belief(self, logits, task):  # masked softmax over the task's K sub-goals
        K = self.n_sub[task]
        ar = torch.arange(self.k_max, device=logits.device)
        mask = ar[None, :] < K[:, None]
        return torch.softmax(logits.masked_fill(~mask, float("-inf")), -1)

    def _prefix(self, task, w):  # select/combine INDEPENDENT phase tokens
        B = task.shape[0]
        pt = torch.einsum("bk,bkp->bp", w, self.phase_tok[task]).view(
            B, self.net.n_prefix, self.net.d
        )
        base = self.net.task_emb(task).view(B, self.net.n_prefix, self.net.d)
        single = (self.n_sub[task] < 2).view(
            B, 1, 1
        )  # single-phase tasks: use base token untouched
        return torch.where(single, base, pt)

    def _select(self, imgs, state):  # -> (vis, vcoords, logits)
        vis, vcoords = self.net.encode_vision(imgs)
        feat = torch.cat([vis.mean(1), self.net.state_mlp(state)], -1)
        return vis, vcoords, self.selector(feat)

    def loss(
        self, imgs, state, task, a1, sub=None, rot_w=2.0, grip_w=1.0, grasp_w=1.0, grasp_win=1
    ):
        vis, vcoords, logits = self._select(imgs, state)
        belief = self._belief(logits, task)
        if sub is None:
            w = belief.detach()
            ce = torch.zeros((), device=imgs.device)
        else:
            onehot = F.one_hot(sub.clamp(min=0), self.k_max).float()
            lab = (sub >= 0).float().unsqueeze(-1)
            w = lab * onehot + (1 - lab) * belief.detach()  # teacher-force true phase where labeled
            m = sub >= 0
            ce = (
                F.cross_entropy(logits[m], sub[m])
                if m.any()
                else torch.zeros((), device=imgs.device)
            )
        prefix = self._prefix(task, w)
        pred = self.net.head(
            self.net.features(imgs, state, task, vis=vis, vcoords=vcoords, prefix=prefix)
        )
        return _eff_action_loss(pred, a1, rot_w, grip_w, grasp_w, grasp_win) + self.lambda_ce * ce

    @torch.no_grad()
    def sample(self, imgs, state, task, steps=None, deterministic=True, phase=None):
        vis, vcoords, logits = self._select(imgs, state)
        belief = self._belief(logits, task)
        if phase is not None:  # ORACLE / manual switching: use the given phase
            oh = F.one_hot(phase.clamp(min=0), self.k_max).float()
            valid = (phase >= 0).float().unsqueeze(-1)
            w = valid * oh + (1 - valid) * belief  # phase<0 -> fall back to selector belief
        else:
            w = belief  # normal: selector belief
        prefix = self._prefix(task, w)
        pred = self.net.head(
            self.net.features(imgs, state, task, vis=vis, vcoords=vcoords, prefix=prefix)
        )
        grip = 2 * torch.sigmoid(pred[..., 6:7]) - 1
        return torch.cat([pred[..., :6], grip], -1)

    @torch.no_grad()
    def sel_acc(self, imgs, state, task, sub):  # selector accuracy on labeled timesteps
        _, _, logits = self._select(imgs, state)
        m = sub >= 0
        if not m.any():
            return None
        pred = self._belief(logits, task).argmax(-1)
        return (pred[m] == sub[m]).float().mean().item()


class ContCVAEEff(nn.Module):
    """CVAE-MMD action head on the EFFICIENT backbone (VLAPolicyEff). Plain conditional VAE (ACT/Sohn-style)
    with the latent regularized by MMD (InfoVAE / WAE-MMD), NOT KL: MMD(q(mu), N(0,I)) matches the aggregate
    code to the prior across all moments and prevents the mean-collapse that KL suffers. The decoder is the
    efficient backbone, conditioned on the CURRENT observation via its existing cross-attention (z is added
    to the action queries, cond=). The recognition encoder q(z|a,state) sees the target chunk so the per-token
    latent mu_q carries the MODE -> the decoder can commit to a mode instead of mode-averaging (the failure of
    a plain L1 regressor on multimodal teleop data). z_proj is bias-free + zero-init so z=0 reduces EXACTLY to
    the base regression policy -> warm-starts from a ContPolicyEff checkpoint with no jump.
    Inference: deterministic=True -> z=0 == base policy (control); deterministic=False -> z~N(0,I) samples a
    mode (generative, the point on genuinely multimodal data)."""

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
        beta=1.0,
        **kw,
    ):
        super().__init__()
        from .policy_cont_vae import ActionEncoder, imq_mmd

        self._mmd = imq_mmd
        self.net = VLAPolicyEff(
            n_tasks=n_tasks,
            action_dim=action_dim,
            chunk=chunk,
            p_drop=p_drop,
            hist=hist,
            vhist=vhist,
            **kw,
        )
        self.chunk, self.action_dim = chunk, action_dim
        self.latent_dim, self.noise, self.beta = latent_dim, noise, beta
        self.recog = ActionEncoder(
            action_dim, chunk, state_dim * hist, d=enc_d, depth=enc_depth, latent_dim=latent_dim
        )
        self.z_proj = nn.Linear(latent_dim, self.net.d, bias=False)
        nn.init.zeros_(self.z_proj.weight)  # z=0 -> base policy (perfect warm-start)

    def loss(self, imgs, state, task, a1, rot_w=2.0, grip_w=1.0, beta=None, **kw):
        beta = self.beta if beta is None else beta
        mu_q = self.recog(a1, state)  # (B,chunk,latent) -- sees the target chunk
        z = mu_q + self.noise * torch.randn_like(mu_q) if self.training else mu_q
        pred = self.net(
            imgs, state, task, cond=self.z_proj(z)
        )  # decode conditioned on z + current obs
        w = torch.ones(6, device=pred.device)
        w[3:6] = rot_w
        w = w / w.mean()
        cont = (_logcosh(pred[..., :6] - a1[..., :6]) * w).mean()
        grip = F.binary_cross_entropy_with_logits(pred[..., 6], (a1[..., 6] > 0).float())
        mmd = self._mmd(
            mu_q.flatten(0, 1), torch.randn_like(mu_q).flatten(0, 1)
        )  # aggregate code -> N(0,I)
        return cont + grip_w * grip + beta * mmd

    @torch.no_grad()
    def sample(self, imgs, state, task, steps=None, deterministic=True):
        if deterministic:
            pred = self.net(imgs, state, task, cond=None)  # z=0 == base regression policy
        else:
            B = imgs.shape[0]
            z = torch.randn(
                B, self.chunk, self.latent_dim, device=imgs.device, dtype=self.z_proj.weight.dtype
            )
            pred = self.net(imgs, state, task, cond=self.z_proj(z))  # sample a mode
        grip = 2 * torch.sigmoid(pred[..., 6:7]) - 1
        return torch.cat([pred[..., :6], grip], -1)

    @torch.no_grad()
    def diag(self, imgs, state, task, a1, rot_w=2.0):
        """Latent capacity probe on a batch: oracle recon (z=mu_q, sees target) vs base (z=0); mu norm."""
        mu_q = self.recog(a1, state)
        w = torch.ones(6, device=a1.device)
        w[3:6] = rot_w
        w = w / w.mean()
        rl = lambda pred: (_logcosh(pred[..., :6] - a1[..., :6]) * w).mean().item()
        return dict(
            orc=rl(self.net(imgs, state, task, cond=self.z_proj(mu_q))),  # capacity ceiling
            base=rl(self.net(imgs, state, task, cond=None)),  # z=0
            mu_norm=mu_q.flatten(0, 1).norm(dim=-1).mean().item(),
        )  # ~sqrt(dim) if filling N(0,I)


class ContPolicyBase(nn.Module):
    """BASELINE VLAPolicy backbone + the SAME log-cosh/gripper-BCE loss & inference as ContPolicyEff.
    Isolates arch vs pipeline: if this works but ContPolicyEff doesn't, the efficient BACKBONE is at
    fault; if this also fails, the loss/optimizer/lineage is. Warm-starts a cont2_* (VLAPolicy) ckpt."""

    def __init__(self, n_tasks, action_dim=7, chunk=8, p_drop=0.1, hist=1, vhist=1, **kw):
        super().__init__()
        from .policy import VLAPolicy

        self.net = VLAPolicy(
            n_tasks=n_tasks,
            action_dim=action_dim,
            chunk=chunk,
            p_drop=p_drop,
            hist=hist,
            vhist=vhist,
        )
        self.chunk, self.action_dim, self.hist, self.vhist = chunk, action_dim, hist, vhist

    loss = ContPolicyEff.loss
    sample = ContPolicyEff.sample
