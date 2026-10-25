"""Inference wrapper for the continuous regression VLA: obs -> action chunk.

Temporal memory:
  * hist>1  : the state token encodes the last `hist` proprio states.
  * vhist>1 : VISUAL memory -- the vision stream sees the last `vhist` STRIDED frames (t, t-S, ...).
Both are buffered per episode keyed by (task, demo_idx), reset at t==0; the rollout must call act with
pass_meta=True. hist==vhist==1 keeps the original single-frame/single-state path.
"""

from __future__ import annotations
from collections import deque
from contextlib import nullcontext
import pickle
import numpy as np
import torch
from .policy_cont import ContPolicy
from .policy_cont_vae import ContCVAEPolicy

from .config import CHECKPOINT_DIR, load_stats, resolve_asset

CKPT = str(CHECKPOINT_DIR)


class ContInfer:
    """Normalize observations and run a checkpoint on CPU or CUDA.

    Images are RGB uint8 arrays; proprioception contains position, axis-angle
    rotation and both gripper joints. Task IDs index the checkpoint vocabulary.
    """

    def __init__(
        self, ckpt, steps=5, dev="cuda", stats="normalization.npy", deterministic=False, temp=1.0
    ):
        device = torch.device(dev)
        if device.type not in {"cpu", "cuda"}:
            raise ValueError("Inference supports CPU and CUDA devices")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError(
                "CUDA is unavailable; use dev='cpu' or install a CUDA-enabled PyTorch build"
            )
        if steps <= 0 or not np.isfinite(temp) or temp <= 0:
            raise ValueError("steps and temperature must be positive")
        self.temp = temp
        # Checkpoints contain tensor state dictionaries and plain metadata only.
        checkpoint_path = resolve_asset(ckpt, CKPT)
        try:
            ck = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except (OSError, RuntimeError, EOFError, pickle.UnpicklingError) as exc:
            raise ValueError(f"Cannot read tensor checkpoint {checkpoint_path}: {exc}") from exc
        if not isinstance(ck, dict) or not isinstance(ck.get("model"), dict):
            raise ValueError("Checkpoint must contain a model state dictionary")
        for key in ("n_tasks", "chunk", "res", "hist", "vhist", "vstride", "latent_dim"):
            if key in ck and (type(ck[key]) is not int or ck[key] <= 0):
                raise ValueError(f"Checkpoint {key} must be a positive integer")
        if "n_tasks" not in ck or "chunk" not in ck:
            raise ValueError("Checkpoint requires n_tasks and chunk metadata")
        policy = ck.get("policy")
        if policy is not None and (
            not isinstance(policy, str)
            or policy not in {"eff", "eff_base", "cvae", "cvae_eff", "subtask_eff"}
        ):
            raise ValueError(f"Unknown checkpoint policy {policy!r}")
        if not isinstance(ck.get("eff_cfg", {}), dict):
            raise ValueError("Checkpoint eff_cfg must be a dictionary")
        self.hist = ck.get("hist", 1)
        self.vhist = ck.get("vhist", 1)
        self.vstride = ck.get("vstride", 16)
        self.a_mean, self.a_std, self.s_mean, self.s_std = load_stats(resolve_asset(stats, CKPT))
        _stats = (self.a_mean, self.a_std, self.s_mean, self.s_std)
        if ck.get("policy") == "eff":  # efficient backbone (ScaleNorm/xATLU/conv head, gripper BCE)
            from .policy_eff import ContPolicyEff

            self.m = (
                ContPolicyEff(
                    n_tasks=ck["n_tasks"],
                    chunk=ck["chunk"],
                    p_drop=0.0,
                    hist=self.hist,
                    vhist=self.vhist,
                    local_frame=ck.get("local_frame", False),
                    stats=_stats,
                    temporal=ck.get("temporal", False),
                    t_heads=ck.get("t_heads", 4),
                    cam_res=ck.get("cam_res"),
                    rot6d=ck.get("rot6d", False),
                    **ck.get("eff_cfg", {}),
                )
                .to(dev)
                .eval()
            )
        elif (
            ck.get("policy") == "eff_base"
        ):  # baseline VLAPolicy backbone + eff loss (arch-vs-pipeline test)
            from .policy_eff import ContPolicyBase

            self.m = (
                ContPolicyBase(
                    n_tasks=ck["n_tasks"],
                    chunk=ck["chunk"],
                    p_drop=0.0,
                    hist=self.hist,
                    vhist=self.vhist,
                )
                .to(dev)
                .eval()
            )
        elif ck.get("policy") == "cvae":  # fixed-var CVAE head: eval decodes at z=0 (.sample)
            self.m = (
                ContCVAEPolicy(
                    n_tasks=ck["n_tasks"],
                    chunk=ck["chunk"],
                    p_drop=0.0,
                    hist=self.hist,
                    vhist=self.vhist,
                    latent_dim=ck.get("latent_dim", 32),
                    noise=ck.get("noise", 1.0),
                )
                .to(dev)
                .eval()
            )
        elif (
            ck.get("policy") == "cvae_eff"
        ):  # standalone conditional CVAE; det=True -> z=mu_p(obs) mean, det=False -> z~p(z|obs)
            from .policy_cvae_eff import CVAEEff

            self.m = (
                CVAEEff(
                    n_tasks=ck["n_tasks"],
                    chunk=ck["chunk"],
                    p_drop=0.0,
                    latent_dim=ck.get("latent_dim", 32),
                    **ck.get("eff_cfg", {}),
                )
                .to(dev)
                .eval()
            )
        elif (
            ck.get("policy") == "subtask_eff"
        ):  # sub-task-conditioned: selector belief -> convex prefix tokens
            from .policy_eff import ContSubtaskEff

            self.m = (
                ContSubtaskEff(
                    n_tasks=ck["n_tasks"],
                    chunk=ck["chunk"],
                    p_drop=0.0,
                    hist=self.hist,
                    vhist=self.vhist,
                    n_sub=ck.get("n_sub"),
                    k_max=ck.get("k_max", 3),
                    **ck.get("eff_cfg", {}),
                )
                .to(dev)
                .eval()
            )
        else:
            self.m = (
                ContPolicy(
                    n_tasks=ck["n_tasks"],
                    chunk=ck["chunk"],
                    p_drop=0.0,
                    hist=self.hist,
                    vhist=self.vhist,
                )
                .to(dev)
                .eval()
            )
        self.m.load_state_dict(ck["model"])
        self.steps, self.dev, self.deterministic = steps, device, deterministic
        self.res, self.chunk, self.n_tasks = ck.get("res", 128), ck["chunk"], ck["n_tasks"]
        self.hbuf, self.fbuf = {}, {}

    def reset(self, task=None, demo=None):
        """Clear all temporal memory, or memory for one (task, episode) pair."""
        if task is None and demo is None:
            self.hbuf.clear()
            self.fbuf.clear()
        elif task is None or demo is None:
            raise ValueError("reset requires both task and demo, or neither")
        else:
            key = (int(task), int(demo))
            self.hbuf.pop(key, None)
            self.fbuf.pop(key, None)

    def _autocast(self):
        device = torch.device(self.dev)
        if device.type == "cuda":
            with torch.cuda.device(device):
                supported = torch.cuda.is_bf16_supported()
            if supported:
                return torch.autocast("cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _hist_state(self, raw, task, demo, t):
        key = (int(task), int(demo))
        if t == 0 or key not in self.hbuf:
            self.hbuf[key] = deque([raw] * self.hist, maxlen=self.hist)
        else:
            self.hbuf[key].append(raw)
        return ((np.stack(self.hbuf[key]) - self.s_mean) / self.s_std).reshape(-1)

    def _frame_stack(self, cur, task, demo, t):
        """cur (2,H,W,3) uint8 -> (vhist,2,H,W,3) strided history, clamped at episode start."""
        key = (int(task), int(demo))
        ml = (self.vhist - 1) * self.vstride + 1
        if t == 0 or key not in self.fbuf:
            self.fbuf[key] = deque([cur] * ml, maxlen=ml)
        else:
            self.fbuf[key].append(cur)
        buf = list(self.fbuf[key])
        n = len(buf)
        idx = [max(0, n - 1 - k * self.vstride) for k in range(self.vhist - 1, -1, -1)]
        return np.stack([buf[j] for j in idx])

    @torch.no_grad()
    def act(self, imgs_u8, states, tasks, metas=None, phases=None):
        """Return clipped (batch, chunk, 7) actions in controller coordinates.

        Temporal models require one (episode ID, step) pair per observation.
        A zero step starts a new episode and initializes its history buffers.
        """
        imgs_u8 = np.asarray(imgs_u8)  # (B,2,H,W,3)
        if (
            imgs_u8.ndim != 5
            or imgs_u8.shape[0] == 0
            or imgs_u8.shape[1] != 2
            or imgs_u8.shape[-1] != 3
            or min(imgs_u8.shape[2:4]) < 1
            or imgs_u8.dtype != np.uint8
        ):
            raise ValueError("images must be uint8 with shape (batch, 2, height, width, 3)")
        batch = imgs_u8.shape[0]
        states = np.asarray(states)
        if (
            states.shape != (batch, 8)
            or states.dtype.kind not in "fiu"
            or not np.isfinite(states).all()
        ):
            raise ValueError("states must be finite numeric values with shape (batch, 8)")
        tasks = np.asarray(tasks)
        if (
            tasks.shape != (batch,)
            or tasks.dtype.kind not in "iu"
            or np.any(tasks < 0)
            or np.any(tasks >= self.n_tasks)
        ):
            raise ValueError(f"tasks must contain {batch} integer IDs in [0, {self.n_tasks})")
        if metas is None and (self.hist > 1 or self.vhist > 1):
            raise ValueError(
                "Temporal policies require (episode, step) metadata for each observation"
            )
        if metas is not None:
            meta_array = np.asarray(metas)
            if (
                meta_array.shape != (batch, 2)
                or meta_array.dtype.kind not in "iu"
                or np.any(meta_array < 0)
            ):
                raise ValueError(
                    "metas must have shape (batch, 2) with nonnegative integer episode/step values"
                )
        if phases is not None and (
            np.asarray(phases).shape != (batch,) or np.asarray(phases).dtype.kind not in "iu"
        ):
            raise ValueError("phases must contain one integer per observation")
        if phases is not None and type(self.m).__name__ != "ContSubtaskEff":
            raise ValueError("phases require a subtask_eff checkpoint")
        if self.vhist > 1 and metas is not None:
            st_ = np.stack(
                [
                    self._frame_stack(imgs_u8[i], tasks[i], metas[i][0], metas[i][1])
                    for i in range(len(imgs_u8))
                ]
            )  # (B,vhist,2,H,W,3)
            x = (
                torch.from_numpy(st_.astype(np.float32) / 255.0)
                .permute(0, 1, 2, 5, 3, 4)
                .to(self.dev)
            )
        else:
            x = (
                torch.from_numpy(imgs_u8.astype(np.float32) / 255.0)
                .permute(0, 1, 4, 2, 3)
                .to(self.dev)
            )
        if self.hist > 1 and metas is not None:
            st = np.stack(
                [
                    self._hist_state(states[i], tasks[i], metas[i][0], metas[i][1])
                    for i in range(len(states))
                ]
            ).astype(np.float32)
        else:
            st = ((states - self.s_mean) / self.s_std).astype(np.float32)
        st = torch.from_numpy(st).to(self.dev)
        tk = torch.tensor(tasks.astype(np.int64), device=self.dev, dtype=torch.long)
        kw = {}
        if phases is not None:  # oracle / manual phase switching
            kw["phase"] = torch.tensor(list(phases), device=self.dev, dtype=torch.long)
        if (
            getattr(self, "temp", 1.0) != 1.0 and type(self.m).__name__ == "CVAEEff"
        ):  # CVAE exploration temperature (self-active collection)
            kw["temp"] = self.temp
        with self._autocast():
            a = (
                self.m.sample(x, st, tk, steps=self.steps, deterministic=self.deterministic, **kw)
                .float()
                .cpu()
                .numpy()
            )
        if a.shape != (batch, self.chunk, 7) or not np.isfinite(a).all():
            raise ValueError(
                "Policy returned invalid actions; expected finite (batch, chunk, 7) output"
            )
        a = a * self.a_std + self.a_mean
        return np.clip(a, -1.0, 1.0)
