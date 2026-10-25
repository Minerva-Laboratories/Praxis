"""Behavior cloning and SEAL training for Praxis regression and generative policies."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
SEALDIR = "data/seal_eff"


def aug_photo(x, s):
    """Jitter brightness, contrast, and saturation with shared temporal noise.

    Accepts (B, cameras, C, H, W) or (B, T, cameras, C, H, W).
    """
    import torch

    B = x.shape[0]
    shp = (B,) + (1,) * (x.dim() - 1)
    rnd = lambda: 1 + s * (torch.rand(shp, device=x.device) * 2 - 1)
    m = x.mean(dim=(-2, -1), keepdim=True)
    x = ((x - m) * rnd() + m) * rnd()
    gray = x.mean(dim=-3, keepdim=True)
    x = gray + rnd() * (x - gray)
    return x.clamp(0, 1)


def positive_int(value):
    """Parse a strictly positive integer for counts and dimensions."""
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def nonnegative_int(value):
    """Parse an integer count that can be disabled with zero."""
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return result


def positive_float(value):
    """Parse a finite positive scale or learning rate."""
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return result


def build_parser():
    """Build the CLI without importing training or simulator dependencies."""
    ap = argparse.ArgumentParser(
        description="Train Praxis with behavior cloning and optional SEAL self-exposure rounds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path(os.environ.get("PRAXIS_CHECKPOINT_DIR", str(CHECKPOINT_DIR))),
        help="Directory for checkpoints and default normalization/task-map assets",
    )
    ap.add_argument("--device", default="cuda", help="CUDA device, for example cuda or cuda:1")
    ap.add_argument(
        "--task-map",
        type=Path,
        default=None,
        help="Task mapping JSON (default: checkpoint directory/task_map.json)",
    )
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--det", type=int, default=0)

    ap.add_argument("--pack", default="data/libero_merged")
    ap.add_argument(
        "--stats",
        type=Path,
        default=None,
        help="Normalization file (default: checkpoint directory/normalization.npy)",
    )
    ap.add_argument("--out", default="eff")
    ap.add_argument("--init", default="")
    ap.add_argument("--bc_epochs", type=nonnegative_int, default=0)
    ap.add_argument("--rounds", type=nonnegative_int, default=0)
    ap.add_argument("--seal_epochs", type=nonnegative_int, default=2)
    ap.add_argument("--seal_sched_global", type=int, default=0)
    ap.add_argument("--episodes", type=positive_int, default=8)
    ap.add_argument("--rsi", action="store_true")
    ap.add_argument("--rsi_lo", type=float, default=0.0)
    ap.add_argument("--rsi_hi", type=float, default=0.98)
    ap.add_argument("--init_perturb_norm", type=float, default=0.0)
    ap.add_argument("--ctrl_nullspace", type=float, default=0.0)
    ap.add_argument("--ctrl_perturb", type=float, default=0.0)
    ap.add_argument("--ctrl_perturb_j", type=float, default=0.0)
    ap.add_argument("--roll_chunks", type=positive_int, default=1)
    ap.add_argument("--sample_depth", type=int, default=0)
    ap.add_argument("--deep_roll_chunks", type=positive_int, default=45)
    ap.add_argument("--deep_suites", default="")
    ap.add_argument("--gb_cap", type=float, default=1.0)
    ap.add_argument("--gb_align", type=int, default=0)

    ap.add_argument("--gb_horizon", type=nonnegative_int, default=0)

    ap.add_argument("--dump_labels", default="")

    ap.add_argument("--gb_exact_rot", type=int, default=0)
    ap.add_argument("--seal_no_correct", type=int, default=0)
    ap.add_argument("--selfdemo", default="")
    ap.add_argument("--selfseal", default="")
    ap.add_argument("--lora", default="", choices=["", "singlora", "rslora", "lora"])
    ap.add_argument("--lora_rank", type=positive_int, default=16)
    ap.add_argument("--lora_alpha", type=float, default=16.0)
    ap.add_argument("--lora_ramp", type=nonnegative_int, default=200)
    ap.add_argument("--lora_weak_demos", type=int, default=0)
    ap.add_argument("--self_only", type=int, default=0)
    ap.add_argument("--selfdemo_reps", type=nonnegative_int, default=0)
    ap.add_argument("--init_perturb", type=float, default=0.0)
    ap.add_argument("--init_perturb_frac", type=float, default=0.5)
    ap.add_argument("--seal_inits", default="data/seal_inits.npz")
    ap.add_argument("--seal_states", default="data/seal_states.npz")
    ap.add_argument("--seal_ref", default="data/seal_ref.npz")
    ap.add_argument("--seal_dir", default=SEALDIR)
    ap.add_argument("--noise_sigma", type=float, default=0.0)
    ap.add_argument("--noise_steps", type=nonnegative_int, default=6)
    ap.add_argument("--rot6d", type=int, default=0)
    ap.add_argument("--local_frame", type=int, default=0)
    ap.add_argument("--opt", default="adamw", choices=["adamw", "nadamw", "musgd"])
    ap.add_argument("--bs", type=positive_int, default=96)
    ap.add_argument("--lr", type=positive_float, default=3e-4)
    ap.add_argument("--sgd_lr", type=positive_float, default=3e-3)
    ap.add_argument("--ls_init", default="none")
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--ema", type=float, default=0.9995)
    ap.add_argument("--rot_w", type=float, default=2.0)
    ap.add_argument("--grip_w", type=float, default=1.0)
    ap.add_argument("--state_loss", type=int, default=0)
    ap.add_argument("--state_traj", type=int, default=0)
    ap.add_argument("--grasp_w", type=float, default=1.0)
    ap.add_argument("--grasp_win", type=nonnegative_int, default=1)
    ap.add_argument("--subtask", type=int, default=0)
    ap.add_argument("--sublabels", default="data/subtask_labels.npz")
    ap.add_argument("--lambda_ce", type=float, default=1.0)
    ap.add_argument("--k_max", type=positive_int, default=3)
    ap.add_argument("--sub_perturb", type=float, default=0.05)
    ap.add_argument("--head_lr_mult", type=positive_float, default=1.0)
    ap.add_argument("--cvae", type=int, default=0)
    ap.add_argument("--latent_dim", type=positive_int, default=32)
    ap.add_argument("--noise_cvae", type=float, default=1.0)
    ap.add_argument("--beta_cvae", type=float, default=1.0)
    ap.add_argument("--enc_d", type=positive_int, default=128)
    ap.add_argument("--temporal", type=int, default=0)
    ap.add_argument("--vhist", type=positive_int, default=1)
    ap.add_argument("--vstride", type=positive_int, default=4)
    ap.add_argument("--t_heads", type=positive_int, default=4)
    ap.add_argument("--global_res", type=positive_int, default=128)
    ap.add_argument("--wrist_res", type=nonnegative_int, default=0)
    ap.add_argument("--jitter", type=float, default=1.0)
    ap.add_argument("--pad", type=nonnegative_int, default=8)
    ap.add_argument("--photo", type=float, default=0.2)
    ap.add_argument("--aug_apr", type=float, default=0.0)
    ap.add_argument("--aug_cutout", type=float, default=0.0)
    ap.add_argument("--cutout_frac", type=float, default=0.25)
    ap.add_argument("--aug_affine", type=float, default=0.0)
    ap.add_argument("--state_bias", type=float, default=0.0)
    ap.add_argument("--state_equiv", type=float, default=0.0)
    ap.add_argument("--workers", type=nonnegative_int, default=8)
    ap.add_argument("--n_tasks", type=nonnegative_int, default=0)
    ap.add_argument("--task_lo", type=int, default=-1)
    ap.add_argument("--task_hi", type=int, default=10**9)
    ap.add_argument("--only_tasks", default="")
    ap.add_argument("--save_epochs", type=int, default=0)

    ap.add_argument("--w", type=positive_float, default=1.85)
    ap.add_argument("--head_depth", type=positive_int, default=2)
    ap.add_argument("--vision_v2", type=int, default=1)
    ap.add_argument("--arch", default="eff", choices=["eff", "base"])
    descriptions = {
        "seed": "Random seed; negative values leave RNGs unseeded",
        "det": "Enable deterministic algorithms and seeded rollout workers (0/1)",
        "pack": "Directory containing packed LIBERO demonstrations",
        "out": "Checkpoint name prefix",
        "init": "Warm-start checkpoint name or explicit .pt path",
        "bc_epochs": "Behavior-cloning epochs; zero skips behavior cloning",
        "rounds": "Number of SEAL self-exposure rounds",
        "seal_epochs": "Training epochs per SEAL round; zero collects rollouts only",
        "seal_sched_global": "Use one learning-rate schedule across all SEAL rounds (0/1)",
        "episodes": "Rollouts per task per round",
        "rsi": "Start rollouts at demonstration anchors",
        "rsi_lo": "Lower RSI fraction bound (retained for recipe compatibility)",
        "rsi_hi": "Upper RSI fraction bound (retained for recipe compatibility)",
        "roll_chunks": "Open-loop action chunks to collect from each anchor",
        "sample_depth": "Sample rollout depth uniformly up to roll_chunks (0/1)",
        "deep_roll_chunks": "Rollout chunks for suites selected by deep_suites",
        "deep_suites": "Comma-separated suite names for full-trajectory rollouts",
        "gb_cap": "Direction-preserving get-back correction cap in (0, 1]",
        "gb_align": "Align the reference by path progress (0/1)",
        "gb_horizon": "Get-back return horizon; zero uses the legacy gain",
        "gb_exact_rot": "Use quaternion-composition rotation correction (0/1)",
        "seal_no_correct": "Use reference actions without corrective labels (0/1)",
        "dump_labels": "Optional output directory for raw relabeling inputs",
        "selfdemo": "Directory of successful self-demonstration chunks",
        "selfseal": "Directory of successful reference trajectories for SEAL",
        "self_only": "Exclude human demonstrations from the SEAL training mixture (0/1)",
        "selfdemo_reps": "Self-demonstration repetition count; zero balances automatically",
        "lora": "Optional low-rank adaptation method",
        "lora_rank": "Low-rank adapter rank",
        "lora_alpha": "Low-rank adapter scale",
        "lora_ramp": "Adapter ramp-in steps",
        "lora_weak_demos": "Restrict LoRA demonstration training to only_tasks (0/1)",
        "bs": "Training batch size",
        "lr": "Main optimizer learning rate",
        "sgd_lr": "MuSGD learning rate for one-dimensional parameters and embeddings",
        "ls_init": "LayerScale initialization, or none to disable LayerScale",
        "wd": "Optimizer weight decay",
        "ema": "Exponential moving-average decay",
        "state_loss": "Weight action errors by their state-space effect (0/1)",
        "state_traj": "Use cumulative trajectory state loss (0/1)",
        "subtask": "Enable the subtask-conditioned policy (0/1)",
        "cvae": "Enable the generative CVAE-MMD action head (0/1)",
        "temporal": "Enable temporal delta-attention vision (0/1)",
        "vhist": "Number of visual history frames",
        "vstride": "Stride between visual history frames",
        "global_res": "Global-camera resolution for asymmetric vision",
        "wrist_res": "Wrist-camera resolution; zero disables asymmetric vision",
        "pad": "Replicate padding for random-shift image augmentation",
        "photo": "Brightness, contrast, and saturation jitter strength",
        "workers": "Data-loader and rollout worker count",
        "n_tasks": "Model task count; zero infers it from the demonstration pack",
        "task_lo": "Lowest demonstration task ID to train on",
        "task_hi": "Highest demonstration task ID to train on",
        "only_tasks": "Comma-separated task IDs for SEAL/self-data focus",
        "save_epochs": "Save each behavior-cloning epoch (0/1)",
        "w": "Efficient vision backbone width multiplier",
        "head_depth": "Action-head depth",
        "vision_v2": "Enable the v2 vision encoder (0/1)",
        "arch": "Policy backbone architecture",
    }
    for action in ap._actions:
        if action.type is int and action.dest not in {"seed", "task_lo", "task_hi"}:
            action.choices = (0, 1)
        if action.help is None:
            action.help = descriptions.get(action.dest, action.dest.replace("_", " ").capitalize())
    return ap


def validate_args(a, parser):
    """Reject invalid schedules, shapes, and augmentation ranges before loading data."""
    positive = (
        "bs",
        "episodes",
        "roll_chunks",
        "deep_roll_chunks",
        "lora_rank",
        "lr",
        "sgd_lr",
        "k_max",
        "head_lr_mult",
        "latent_dim",
        "enc_d",
        "vhist",
        "vstride",
        "t_heads",
        "global_res",
        "w",
        "head_depth",
    )
    nonnegative = (
        "bc_epochs",
        "rounds",
        "seal_epochs",
        "workers",
        "n_tasks",
        "save_epochs",
        "lora_ramp",
        "selfdemo_reps",
        "gb_horizon",
        "noise_steps",
        "wrist_res",
        "pad",
        "grasp_win",
        "init_perturb_norm",
        "ctrl_nullspace",
        "ctrl_perturb",
        "ctrl_perturb_j",
        "init_perturb",
        "noise_sigma",
        "wd",
        "rot_w",
        "grip_w",
        "grasp_w",
        "lambda_ce",
        "sub_perturb",
        "noise_cvae",
        "beta_cvae",
        "jitter",
        "photo",
        "state_bias",
        "state_equiv",
    )
    for name in positive + nonnegative:
        value = getattr(a, name)
        if not math.isfinite(value) or (value <= 0 if name in positive else value < 0):
            parser.error(f"--{name} must be {'positive' if name in positive else 'nonnegative'}")
    for name in (
        "rsi_lo",
        "rsi_hi",
        "init_perturb_frac",
        "dropout",
        "ema",
        "aug_apr",
        "aug_cutout",
        "aug_affine",
        "cutout_frac",
    ):
        if not 0 <= getattr(a, name) <= 1:
            parser.error(f"--{name} must be between 0 and 1")
    if a.rounds and not a.workers:
        parser.error("--workers must be positive when --rounds is nonzero")
    if a.dropout == 1:
        parser.error("--dropout must be less than 1")
    if not 0 < a.gb_cap <= 1:
        parser.error("--gb_cap must be in (0, 1]")
    if a.rsi_lo > a.rsi_hi:
        parser.error("--rsi_lo must not exceed --rsi_hi")
    if a.task_lo > a.task_hi:
        parser.error("--task_lo must not exceed --task_hi")
    try:
        if a.only_tasks and any(int(x) < 0 for x in a.only_tasks.split(",")):
            raise ValueError
        if a.ls_init != "none" and not math.isfinite(float(a.ls_init)):
            raise ValueError
    except ValueError:
        parser.error(
            "--only_tasks requires nonnegative comma-separated integers; --ls_init requires a finite number or none"
        )
    if not a.out or Path(a.out).name != a.out or a.out in (".", ".."):
        parser.error("--out must be a checkpoint name without directory components")
    if a.device != "cuda" and not (a.device.startswith("cuda:") and a.device[5:].isdigit()):
        parser.error("Training currently requires a CUDA device (--device cuda or cuda:N)")


def main(argv=None):
    parser = build_parser()
    a = parser.parse_args(argv)
    validate_args(a, parser)

    import gc
    import time

    import numpy as np
    import torch
    from torch.utils.data import ConcatDataset, DataLoader, Subset

    from .augment import affine as aug_affine
    from .augment import apr as aug_apr
    from .augment import cutout as aug_cutout
    from .augment import state_bias as aug_state_bias
    from .augment import state_equivariant as aug_state_equiv
    from .config import load_stats, load_task_map, resolve_asset
    from .data import LiberoChunks
    from .musgd import MuSGD, musgd_param_groups
    from .policy_eff import ContPolicyEff, ContSubtaskEff
    from .training_utils import EMA, ActSet, SealFlow
    from .utils import SUITE_MAX, build_demo_lookup, rand_shift

    if not torch.cuda.is_available():
        parser.error(
            "CUDA is unavailable; training requires a CUDA-enabled PyTorch installation and GPU"
        )
    DEV = torch.device(a.device)
    torch.cuda.set_device(DEV)
    AMP = torch.bfloat16
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = not bool(a.det)
    torch.backends.cudnn.deterministic = bool(a.det)
    CK = a.checkpoint_dir.expanduser().resolve()
    CK.mkdir(parents=True, exist_ok=True)
    a.stats = resolve_asset(a.stats or "normalization.npy", CK)
    a.task_map = resolve_asset(a.task_map or "task_map.json", CK)
    for name in (
        "pack",
        "seal_dir",
        "seal_inits",
        "seal_states",
        "seal_ref",
        "sublabels",
        "dump_labels",
        "selfdemo",
        "selfseal",
    ):
        value = getattr(a, name)
        if value:
            setattr(a, name, str(Path(value).expanduser().resolve()))

    if a.seed >= 0:
        import random

        random.seed(a.seed)
        np.random.seed(a.seed)
        torch.manual_seed(a.seed)
        torch.cuda.manual_seed_all(a.seed)
        print(f"seed set to {a.seed}", flush=True)
    if a.det:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except (RuntimeError, AttributeError) as _e:
            print(f"[det] use_deterministic_algorithms unavailable: {_e}", flush=True)
        print(
            "[det] cudnn.benchmark=False, deterministic algorithms on, rollout workers seeded from --seed",
            flush=True,
        )
        print(f"seed set to {a.seed}", flush=True)
    os.makedirs(a.seal_dir, exist_ok=True)
    stats = load_stats(a.stats)
    a_mean, a_std, s_mean, s_std = stats

    EQUIV_GAIN = torch.tensor(
        (1.0 / np.array([0.05, 0.05, 0.05], np.float32))
        * np.asarray(s_std[:3], np.float32)
        / np.asarray(a_std[:3], np.float32),
        device=DEV,
    )

    A_LO = torch.tensor(
        (-1.0 - np.asarray(a_mean[:3], np.float32)) / np.asarray(a_std[:3], np.float32),
        device=DEV,
    )
    A_HI = torch.tensor(
        (1.0 - np.asarray(a_mean[:3], np.float32)) / np.asarray(a_std[:3], np.float32),
        device=DEV,
    )
    STATE_W = torch.tensor(
        np.array([0.05, 0.05, 0.05, 0.5, 0.5, 0.5], np.float32) * np.asarray(a_std[:6], np.float32),
        device=DEV,
    )
    cam_res = [a.global_res, a.wrist_res] if a.wrist_res > 0 else None
    data_res = a.wrist_res if a.wrist_res > 0 else 128
    sublabels = None
    n_sub_model = None
    if a.subtask:
        sl = np.load(a.sublabels)
        sublabels = sl["labels"]
        _built = sl["n_sub"]
    ds = LiberoChunks(
        chunk=8,
        res=data_res,
        stats=stats,
        pack=a.pack,
        vhist=a.vhist,
        vstride=a.vstride,
        sublabels=sublabels,
    )
    NT = a.n_tasks if a.n_tasks > 0 else ds.n_tasks
    if a.subtask:
        n_sub_model = np.ones(NT, dtype=np.int64)
        n_sub_model[: len(_built)] = _built
        print(
            f"subtask: {(sublabels >= 0).sum()} labeled rows | compound tasks {(n_sub_model >= 2).sum()} "
            f"| n_sub[:10]={n_sub_model[:10].tolist()}",
            flush=True,
        )
    print(f"{len(ds)} samples, {ds.n_tasks} tasks (model NT={NT})", flush=True)

    ls_init = None if a.ls_init == "none" else float(a.ls_init)
    if a.arch == "base":
        from .policy_eff import ContPolicyBase

        ecfg = {}
        POLICY = "eff_base"
        mk = lambda pd: ContPolicyBase(n_tasks=NT, chunk=8, p_drop=pd).to(DEV)
    else:
        ecfg = {
            "w": a.w,
            "head_depth": a.head_depth,
            "vision_v2": bool(a.vision_v2),
            "ls_init": ls_init,
        }
        if a.subtask:
            POLICY = "subtask_eff"
            mk = lambda pd: ContSubtaskEff(
                n_tasks=NT,
                chunk=8,
                p_drop=pd,
                n_sub=n_sub_model,
                k_max=a.k_max,
                lambda_ce=a.lambda_ce,
                **ecfg,
            ).to(DEV)
        elif a.cvae:
            from .policy_cvae_eff import CVAEEff

            POLICY = "cvae_eff"
            mk = lambda pd: CVAEEff(
                n_tasks=NT,
                chunk=8,
                p_drop=pd,
                latent_dim=a.latent_dim,
                beta=a.beta_cvae,
                **ecfg,
            ).to(DEV)
        else:
            POLICY = "eff"
            mk = lambda pd: ContPolicyEff(
                n_tasks=NT,
                chunk=8,
                p_drop=pd,
                local_frame=bool(a.local_frame),
                stats=stats,
                temporal=bool(a.temporal),
                t_heads=a.t_heads,
                cam_res=cam_res,
                rot6d=bool(a.rot6d),
                **ecfg,
            ).to(DEV)
    model = mk(a.dropout)
    if a.opt == "musgd":
        opt = MuSGD(musgd_param_groups(model, lr=a.lr, sgd_lr=a.sgd_lr, weight_decay=a.wd))
    elif a.opt == "nadamw":
        opt = torch.optim.NAdam(
            model.parameters(), lr=a.lr, weight_decay=a.wd, decoupled_weight_decay=True
        )
    elif a.subtask and a.head_lr_mult != 1.0:
        head = [p for n, p in model.named_parameters() if n.startswith(("phase_tok", "selector"))]
        base = [
            p for n, p in model.named_parameters() if not n.startswith(("phase_tok", "selector"))
        ]
        opt = torch.optim.AdamW(
            [
                {"params": base, "lr": a.lr},
                {"params": head, "lr": a.lr * a.head_lr_mult},
            ],
            weight_decay=a.wd,
            fused=True,
        )
        print(
            f"subtask param groups: base lr {a.lr:.1e} ({len(base)}), head lr {a.lr * a.head_lr_mult:.1e} ({len(head)})",
            flush=True,
        )
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd, fused=True)
    _base = [(g.get("lr"), g.get("sgd_lr")) for g in opt.param_groups]

    def set_lr(mult):
        for g, (bl, bs) in zip(opt.param_groups, _base):
            if bl is not None:
                g["lr"] = bl * mult
            if bs is not None:
                g["sgd_lr"] = bs * mult

    print(
        f"ContPolicyEff {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M  opt={a.opt} lr={a.lr} "
        f"wd={a.wd} ls_init={ls_init} loss=logcosh+bce w={a.w} head_depth={a.head_depth}",
        flush=True,
    )
    if a.init:
        init_path = Path(a.init).expanduser()
        if not init_path.suffix:
            init_path = init_path.with_suffix(".pt")
        ick = torch.load(resolve_asset(init_path, CK), map_location=DEV, weights_only=True)
        istate = ick["model"]
        drop_smlp = bool(a.rot6d) and not ick.get("rot6d", False)
        if drop_smlp:
            istate = {k: v for k, v in istate.items() if not k.startswith("net.state_mlp")}
        strict_ = not (bool(a.cvae) or bool(a.temporal) or bool(a.subtask) or drop_smlp)
        res_ = model.load_state_dict(istate, strict=strict_)
        tag_ = (
            f"CVAE (missing {len(res_.missing_keys)} recog/z_proj)"
            if a.cvae
            else f"subtask (missing {len(res_.missing_keys)} sub_delta/selector)"
            if a.subtask
            else f"temporal (missing {len(res_.missing_keys)} temporal keys)"
            if a.temporal
            else "ContPolicyEff"
        )
        print(f"warm-start {tag_} from {a.init}", flush=True)
        if a.subtask:
            model.init_phase_from_base(a.sub_perturb)
            print(f"phase tokens <- base task token (perturb {a.sub_perturb})", flush=True)
    _gsched = {"tot": None, "warm": None, "step": 0}
    ema = EMA(model, decay=a.ema)
    ema_model = mk(0.0).eval()
    sync_ema = lambda: ema_model.load_state_dict(ema.shadow)

    if a.lora:
        from .lora import apply_lora

        for _m in (model, ema_model):
            apply_lora(
                _m,
                r=a.lora_rank,
                alpha=a.lora_alpha,
                variant=a.lora,
                ramp_steps=a.lora_ramp,
            )
        ema = EMA(model, decay=a.ema)
        opt = torch.optim.NAdam(
            [p for p in model.parameters() if p.requires_grad],
            lr=a.lr,
            weight_decay=a.wd,
            decoupled_weight_decay=True,
        )
        _base = [(g.get("lr"), g.get("sgd_lr")) for g in opt.param_groups]

    rng = np.random.default_rng(0)
    n_ep = len(ds.ep_start)
    val = np.zeros(n_ep, bool)
    val[rng.choice(n_ep, max(1, n_ep // 20), replace=False)] = True
    m = val[ds.ep_of]
    tr, va = np.flatnonzero(~m), np.flatnonzero(m)
    only_set = {int(x) for x in a.only_tasks.split(",")} if a.only_tasks else None

    if only_set is not None and a.lora and a.lora_weak_demos:
        ok = np.isin(ds.task[ds.rows], list(only_set))
    else:
        ok = (ds.task[ds.rows] >= a.task_lo) & (ds.task[ds.rows] <= a.task_hi)
    tr = tr[ok[tr]]
    va = va[ok[va]]
    print(
        f"task filter {'weak-only ' + str(sorted(only_set)) if (only_set and a.lora) else [a.task_lo, a.task_hi]} "
        f"(SEAL/self focus={sorted(only_set) if only_set else 'all'}): train {len(tr)} / val {len(va)}",
        flush=True,
    )
    aset = ActSet(ds, None, stats)
    gstep = [0]

    self_ds = None
    self_sel = None
    if a.selfdemo:
        _sz = np.load(f"{a.selfdemo}/idx.npz")
        Nself = int(_sz["state"].shape[0])
        self_ds = SealFlow(a.selfdemo, Nself, 128, stats)
        self_sel = (
            np.flatnonzero(np.isin(_sz["task"], list(only_set)))
            if only_set is not None
            else np.arange(Nself)
        )
        print(f"self-demo: {len(self_sel)}/{Nself} chunks after task filter", flush=True)

    SS = None
    if a.selfseal:
        _q = np.load(f"{a.selfseal}/idx.npz")
        _qi = np.load(f"{a.selfseal}/init.npz", allow_pickle=True)
        SS = {
            "state": _q["state"],
            "act": _q["act"],
            "ep_start": _q["ep_start"],
            "ep_len": _q["ep_len"],
            "task": _q["task"],
            "init": {int(e): v for e, v in zip(_qi["ep_idx"], _qi["init"])},
        }
        keep = [
            e
            for e in range(len(SS["ep_start"]))
            if only_set is None or int(SS["task"][e]) in only_set
        ]
        SS["eps"] = keep
        print(
            f"self-seal: {len(keep)}/{len(SS['ep_start'])} reference episodes after task filter",
            flush=True,
        )

    def step(b, train=True, plain=False):
        im = b["imgs"].to(DEV, non_blocking=True)
        if train:
            im = rand_shift(im, pad=a.pad)
            if a.jitter > 0:
                im = aug_photo(im, a.photo)

            if a.aug_apr > 0:
                im = aug_apr(im, p=a.aug_apr)
            if a.aug_cutout > 0:
                im = aug_cutout(im, p=a.aug_cutout, max_frac=a.cutout_frac)
            if a.aug_affine > 0:
                im = aug_affine(im, p=a.aug_affine)
        st, tk, act = (
            b["state"].to(DEV, non_blocking=True),
            b["task"].to(DEV, non_blocking=True),
            b["action"].to(DEV, non_blocking=True),
        )
        if train and a.state_bias > 0:
            st = aug_state_bias(st, sigma=a.state_bias)
        if train and a.state_equiv > 0:
            st, act = aug_state_equiv(st, act, EQUIV_GAIN, A_LO, A_HI, sigma=a.state_equiv)
        kw = (
            {"grasp_w": a.grasp_w, "grasp_win": a.grasp_win}
            if (a.grasp_w > 1.0 and not plain)
            else {}
        )
        if a.subtask and not plain and "sub" in b:
            kw["sub"] = b["sub"].to(DEV, non_blocking=True)
        if a.state_loss or a.state_traj:
            kw["state_w"] = STATE_W
            if a.state_traj:
                kw["state_traj"] = True
        with torch.autocast("cuda", dtype=AMP):
            return model.loss(im, st, tk, act, rot_w=a.rot_w, grip_w=a.grip_w, **kw)

    def save(tag):
        sync_ema()
        if a.lora:
            from .lora import merged_plain_state_dict

            _sd = merged_plain_state_dict(ema_model)
        else:
            _sd = ema.shadow
        torch.save(
            {
                "model": _sd,
                "n_tasks": NT,
                "chunk": 8,
                "res": data_res,
                "policy": POLICY,
                "eff_cfg": ecfg,
                "local_frame": bool(a.local_frame),
                "gstep": gstep[0],
                "latent_dim": a.latent_dim,
                "noise": a.noise_cvae,
                "enc_d": a.enc_d,
                "temporal": bool(a.temporal),
                "t_heads": a.t_heads,
                "cam_res": cam_res,
                "vhist": a.vhist,
                "vstride": a.vstride,
                "subtask": bool(a.subtask),
                "k_max": a.k_max,
                "rot6d": bool(a.rot6d),
                "n_sub": (n_sub_model.tolist() if n_sub_model is not None else None),
            },
            f"{CK}/{tag}.pt",
        )

    if a.bc_epochs > 0:
        bc_parts = [Subset(aset, tr.tolist())]
        if self_ds is not None and len(self_sel) > 0:
            reps = (
                a.selfdemo_reps if a.selfdemo_reps > 0 else max(1, len(tr) // max(1, len(self_sel)))
            )
            bc_parts.append(Subset(self_ds, np.tile(self_sel, reps).tolist()))
            print(f"BC + {len(self_sel)} self-demo chunks x{reps}", flush=True)
        dl = DataLoader(
            ConcatDataset(bc_parts),
            batch_size=a.bs,
            shuffle=True,
            num_workers=a.workers,
            drop_last=True,
            pin_memory=True,
            persistent_workers=bool(a.workers),
            prefetch_factor=6 if a.workers else None,
        )
        vdl = DataLoader(
            Subset(
                aset,
                np.random.default_rng(1).choice(va, min(6000, len(va)), replace=False).tolist(),
            ),
            batch_size=a.bs,
            num_workers=4,
            pin_memory=True,
        )
        total = a.bc_epochs * len(dl)
        warm = int(0.1 * total)

        def lr_mult(s):
            if s < warm:
                return s / max(1, warm)
            return max(0.0, (total - s) / max(1, total - warm))

        best = 1e9
        sstep = 0
        for ep in range(a.bc_epochs):
            model.train()
            t0 = time.time()
            run = 0.0
            for i, b in enumerate(dl):
                mult = lr_mult(sstep)
                sstep += 1
                set_lr(mult)
                loss = step(b)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                ema.update(model)
                gstep[0] += 1
                run += loss.item()
                if i % 300 == 0:
                    print(
                        f"  bc ep{ep} {i}/{len(dl)} loss {loss.item():.4f} lr {opt.param_groups[0]['lr']:.2e} "
                        f"{a.bs * (i + 1) / (time.time() - t0):.0f} smp/s",
                        flush=True,
                    )
            sync_ema()
            vl = 0.0
            k = 0
            with torch.no_grad(), torch.autocast("cuda", dtype=AMP):
                for b in vdl:
                    vl += ema_model.loss(
                        b["imgs"].to(DEV),
                        b["state"].to(DEV),
                        b["task"].to(DEV),
                        b["action"].to(DEV),
                        rot_w=a.rot_w,
                        grip_w=a.grip_w,
                    ).item()
                    k += 1
            vl /= max(1, k)
            print(
                f"bc ep{ep} DONE train {run / len(dl):.4f} VAL {vl:.4f} ({time.time() - t0:.0f}s)",
                flush=True,
            )
            if a.cvae:
                bd = ema_model.diag(
                    b["imgs"].to(DEV),
                    b["state"].to(DEV),
                    b["task"].to(DEV),
                    b["action"].to(DEV),
                )
                print(
                    f"   [cvae diag] orc {bd['orc']:.4f} (posterior)  pm {bd['pm']:.4f} (prior-mean deployed)  "
                    f"gap {bd['pm'] - bd['orc']:+.4f}  |muq-mup| {bd['pp']:.2f}  muq_norm {bd['muq_norm']:.2f}",
                    flush=True,
                )
            save(f"{a.out}_bc_last")
            if a.save_epochs:
                save(f"{a.out}_bc_ep{ep}")
            if vl < best:
                best = vl
                save(f"{a.out}_bc")
                print("   ^ new best val", flush=True)

    if a.rounds > 0:
        from robosuite.utils.transform_utils import quat2axisangle

        from .cont_infer import ContInfer
        from .rollout import SelfExposure
        from .seal import getback_action

        if a.bc_epochs > 0:
            bv = torch.load(f"{CK}/{a.out}_bc.pt", map_location=DEV, weights_only=True)["model"]
            model.load_state_dict(bv)
            ema.shadow = {k: v.detach().clone() for k, v in bv.items()}
            print(
                f"SEAL init from best-val {a.out}_bc (not the overfit final BC epoch)",
                flush=True,
            )
        tmap = load_task_map(a.task_map)
        by_task = build_demo_lookup(ds, tmap)
        _deep_suites = {s for s in a.deep_suites.split(",") if s}
        if _deep_suites:
            print(
                f"DEEP SEAL (full-episode rollout, {a.deep_roll_chunks} chunks) for suites: {sorted(_deep_suites)}",
                flush=True,
            )
        _imap = {}
        try:
            _si = np.load(a.seal_inits, allow_pickle=True)
            _imap = {int(e): s for e, s in zip(_si["ep_idx"], _si["init"])}
            print(
                f"loaded {len(_imap)} recovered TRUE inits from {a.seal_inits} (fixes demo/init alignment)",
                flush=True,
            )
        except FileNotFoundError:
            print(f"no {a.seal_inits}; using default inits (misaligned)", flush=True)

        _smap = {}
        try:
            _ss = np.load(a.seal_states, allow_pickle=True)
            _smap = {int(e): s for e, s in zip(_ss["ep_idx"], _ss["states"])}
            print(
                f"loaded per-step demo states for {len(_smap)} episodes from {a.seal_states} "
                f"(RSI anchors are loaded, not replayed)",
                flush=True,
            )
        except FileNotFoundError:
            print(
                f"no {a.seal_states}; RSI anchors will be reached by replaying the demo (slow)",
                flush=True,
            )

        _rmap = {}
        if _smap:
            try:
                _rf = np.load(a.seal_ref, allow_pickle=True)
                _rmap = {int(e): r for e, r in zip(_rf["ep_idx"], _rf["ref"])}
                print(
                    f"loaded matching EE reference for {len(_rmap)} episodes from {a.seal_ref}",
                    flush=True,
                )
            except FileNotFoundError:
                print(
                    f"no {a.seal_ref} -- REFUSING to use cached anchors with the pack reference "
                    f"(that combination inflates every corrective label); falling back to replay",
                    flush=True,
                )
                _smap = {}

        def build_jobs(rnd):

            jobs = []
            jrng = np.random.default_rng(1000 + rnd + 100_000 * max(a.seed, 0))
            if SS is not None:
                R = a.roll_chunks * 8
                for e in SS["eps"]:
                    ti = int(SS["task"][e])
                    su, tid = tmap[ti]
                    s0, L = int(SS["ep_start"][e]), int(SS["ep_len"][e])
                    ref_act = SS["act"][s0 : s0 + L].astype(np.float32)
                    init_ov = SS["init"][e]
                    for ep in range(a.episodes):
                        anchor = int(jrng.uniform(0.0, 1.0) * max(1, L - R))
                        job_max = min(anchor + R, L)
                        jobs.append((su, tid, ep, ti, e, job_max, anchor, ref_act, init_ov))
                return jobs
            for ti, (su, tid) in sorted(tmap.items()):
                if only_set is not None and ti not in only_set:
                    continue
                deep = su in _deep_suites
                eps_full = by_task.get(ti, [])
                eps = [e for e in eps_full if e in _imap] if (deep and _imap) else eps_full
                use_rsi = a.rsi and len(eps) > 0
                R = (a.deep_roll_chunks if deep else a.roll_chunks) * 8
                for ep in range(a.episodes):
                    if not use_rsi:
                        jobs.append((su, tid, ep, ti, ep, SUITE_MAX[su], 0, None, None))
                    else:
                        e = eps[ep % len(eps)]
                        L = int(ds.ep_len[e])
                        s0 = int(ds.ep_start[e])
                        init_ov = _imap.get(e)
                        if deep:
                            anchor = 0
                            job_max = min(R, min(L, SUITE_MAX[su]))
                        else:
                            if a.sample_depth:
                                Rj = int(jrng.integers(1, a.roll_chunks + 1)) * 8
                                anchor = int(jrng.uniform(0.0, 1.0) * max(1, L - 8))
                                job_max = min(anchor + Rj, L)
                            else:
                                anchor = int(jrng.uniform(0.0, 1.0) * max(1, L - R))
                                job_max = min(anchor + R, L)

                        if (
                            a.init_perturb > 0
                            and init_ov is not None
                            and jrng.uniform() < a.init_perturb_frac
                        ):
                            init_ov = np.asarray(init_ov, np.float64).copy()
                            if a.init_perturb_norm > 0:
                                _p = jrng.normal(0.0, 1.0, 7)
                                _p /= np.linalg.norm(_p) + 1e-12
                                init_ov[1:8] += _p * a.init_perturb_norm
                            else:
                                init_ov[1:8] += jrng.normal(0.0, a.init_perturb, 7)
                            anchor = 0
                            job_max = min(R, L)

                        _sa = _smap.get(e)
                        anchor_st = (
                            _sa[min(anchor, len(_sa) - 1)]
                            if (_sa is not None and a.init_perturb <= 0)
                            else None
                        )
                        jobs.append(
                            (
                                su,
                                tid,
                                ep,
                                ti,
                                e,
                                job_max,
                                anchor,
                                ds.act[s0 : s0 + L].astype(np.float32),
                                init_ov,
                                anchor_st,
                            )
                        )
            return jobs

        va_sub = np.random.default_rng(1).choice(va, min(4000, len(va)), replace=False)
        vdl = DataLoader(
            Subset(aset, va_sub.tolist()),
            batch_size=a.bs,
            num_workers=4,
            pin_memory=True,
        )

        def seal_val():
            model.eval()
            tot = 0.0
            n = 0
            with torch.no_grad():
                for b in vdl:
                    nb = int(b["task"].shape[0])
                    tot += step(b, train=False, plain=True).item() * nb
                    n += nb
            model.train()
            return tot / max(1, n)

        for rnd in range(a.rounds):
            sync_ema()
            pol = ContInfer.__new__(ContInfer)
            pol.m, pol.steps, pol.dev = ema_model, 5, DEV
            pol.a_mean, pol.a_std, pol.s_mean, pol.s_std = a_mean, a_std, s_mean, s_std
            pol.res, pol.chunk, pol.n_tasks, pol.deterministic = 128, 8, NT, True
            pol.hist, pol.vhist, pol.vstride = 1, 1, 16
            pol.hbuf, pol.fbuf = {}, {}
            bi, bs_, bc, bt, bsub = [], [], [], [], []
            t0 = time.time()
            DUMP_W = 24

            dmp_pos, dmp_quat, dmp_ds, dmp_da = [], [], [], []

            def act_fn(ob, tb, pol=pol):
                ims = np.stack([np.stack([o["img"], o["wrist"]]) for o in ob])
                st = np.stack(
                    [np.concatenate([o["pos"], quat2axisangle(o["quat"]), o["grip"]]) for o in ob]
                )
                return pol.act(ims, st, tb)

            def on_chain(
                payload,
                task,
                didx,
                t0,
                *,
                DUMP_W=DUMP_W,
                dmp_pos=dmp_pos,
                dmp_quat=dmp_quat,
                dmp_ds=dmp_ds,
                dmp_da=dmp_da,
                bi=bi,
                bs_=bs_,
                bc=bc,
                bt=bt,
                bsub=bsub,
            ):
                os0, poses = payload
                e = int(didx)
                if SS is not None:
                    if e < 0 or e >= len(SS["ep_start"]) or int(SS["task"][e]) != int(task):
                        return
                    s0, L = int(SS["ep_start"][e]), int(SS["ep_len"][e])
                    d_state, d_act = SS["state"][s0 : s0 + L], SS["act"][s0 : s0 + L]
                else:
                    if (
                        e < 0
                        or e >= len(ds.ep_start)
                        or int(ds.task[int(ds.ep_start[e])]) != int(task)
                    ):
                        return
                    s0, L = int(ds.ep_start[e]), int(ds.ep_len[e])
                    d_state, d_act = ds.state[s0 : s0 + L], ds.act[s0 : s0 + L]

                    _r = _rmap.get(e)
                    if _r is not None:
                        d_state = (
                            _r[:L]
                            if len(_r) >= L
                            else np.concatenate([_r, np.repeat(_r[-1:], L - len(_r), axis=0)])
                        )

                _hz = a.gb_horizon if a.gb_horizon > 0 else (8 if a.gb_align else 0)
                gbs = [
                    getback_action(
                        poses[i][:3],
                        poses[i][3:7],
                        d_state,
                        d_act,
                        t0 + i,
                        cap=a.gb_cap,
                        exact_rot=a.gb_exact_rot,
                        no_correct=a.seal_no_correct,
                        horizon=(_hz if _hz > 0 else None),
                        align=bool(a.gb_align),
                    )
                    for i in range(len(poses))
                ]
                if a.dump_labels:
                    for _i in range(len(poses)):
                        _k = int(min(max(t0 + _i, 0), len(d_act) - 1))
                        _w = min(DUMP_W, len(d_act) - _k)
                        _ds = np.zeros((DUMP_W, d_state.shape[1]), np.float32)
                        _da = np.zeros((DUMP_W, d_act.shape[1]), np.float32)
                        _ds[:_w] = d_state[_k : _k + _w]
                        _da[:_w] = d_act[_k : _k + _w]
                        if _w < DUMP_W:
                            _ds[_w:] = d_state[-1]
                            _da[_w:] = d_act[-1]
                        dmp_pos.append(poses[_i][:3])
                        dmp_quat.append(poses[_i][3:7])
                        dmp_ds.append(_ds)
                        dmp_da.append(_da)
                bi.append(np.stack([os0["img"], os0["wrist"]]))
                bs_.append(np.concatenate([os0["pos"], quat2axisangle(os0["quat"]), os0["grip"]]))
                bc.append(np.stack(gbs).astype(np.float32))
                bt.append(task)
                if sublabels is not None:
                    bsub.append(int(sublabels[min(s0 + t0, s0 + L - 1)]))

            se = SelfExposure(
                build_jobs(rnd),
                ctrl_perturb=a.ctrl_perturb,
                ctrl_perturb_j=a.ctrl_perturb_j,
                ctrl_nullspace=a.ctrl_nullspace,
                n_workers=a.workers,
                res=128,
                exec_steps=8,
                max_steps=600,
                worker_seed=(None if not a.det or a.seed < 0 else a.seed + 7919 * rnd),
                noise_sigma=a.noise_sigma,
                noise_steps=a.noise_steps,
            )
            se.on_chain = on_chain
            se.run(act_fn, lambda *a_: None)
            se.close()
            N = len(bi)
            print(
                f"[eff r{rnd}] self-exposure: {N} corrective samples ({time.time() - t0:.0f}s)",
                flush=True,
            )
            if a.dump_labels and dmp_pos:
                os.makedirs(a.dump_labels, exist_ok=True)
                np.savez(
                    f"{a.dump_labels}/r{rnd}.npz",
                    pos=np.stack(dmp_pos).astype(np.float32),
                    quat=np.stack(dmp_quat).astype(np.float32),
                    dstate=np.stack(dmp_ds),
                    dact=np.stack(dmp_da),
                )
                print(
                    f"[eff r{rnd}] dumped {len(dmp_pos)} label inputs -> {a.dump_labels}/r{rnd}.npz",
                    flush=True,
                )
                dmp_pos, dmp_quat, dmp_ds, dmp_da = [], [], [], []
            if N == 0:
                break
            im = np.memmap(
                f"{a.seal_dir}/imgs.u8",
                dtype=np.uint8,
                mode="w+",
                shape=(N, 2, 128, 128, 3),
            )
            im[:] = np.stack(bi)
            im.flush()
            del im
            np.savez(
                f"{a.seal_dir}/idx.npz",
                state=np.stack(bs_).astype(np.float32),
                act=np.stack(bc).astype(np.float32),
                task=np.array(bt, np.int16),
                **({"sub": np.array(bsub, np.int64)} if sublabels is not None else {}),
            )
            seal = SealFlow(a.seal_dir, N, 128, stats)
            d_idx = np.random.choice(tr, min(60000, len(tr)), replace=False)
            parts = [Subset(seal, np.tile(np.arange(N), 4).tolist())]
            if not a.self_only:
                parts.insert(0, Subset(aset, d_idx.tolist()))
            if self_ds is not None and len(self_sel) > 0:
                reps = a.selfdemo_reps if a.selfdemo_reps > 0 else max(1, N // len(self_sel))
                parts.append(Subset(self_ds, np.tile(self_sel, reps).tolist()))
                print(
                    f"[eff r{rnd}] + {len(self_sel)} self-demo chunks x{reps}",
                    flush=True,
                )
            mix = ConcatDataset(parts)
            sdl = DataLoader(
                mix,
                batch_size=a.bs,
                shuffle=True,
                num_workers=a.workers,
                drop_last=True,
                pin_memory=True,
                prefetch_factor=6 if a.workers else None,
            )
            model.train()
            if a.seal_sched_global:
                if _gsched["tot"] is None:
                    _gsched["tot"] = a.rounds * a.seal_epochs * len(sdl)
                    _gsched["warm"] = int(0.1 * _gsched["tot"])
                    print(
                        f"[eff] global SEAL schedule: {_gsched['tot']} steps, warmup {_gsched['warm']}",
                        flush=True,
                    )
                rtot, rwarm = _gsched["tot"], _gsched["warm"]
                rs = _gsched["step"]
            else:
                rtot = a.seal_epochs * len(sdl)
                rwarm = int(0.1 * rtot)
                rs = 0
            run = 0.0
            for ep in range(a.seal_epochs):
                t1 = time.time()
                run = 0.0
                for b in sdl:
                    mult = (
                        (rs / max(1, rwarm))
                        if rs < rwarm
                        else max(0.0, (rtot - rs) / max(1, rtot - rwarm))
                    )
                    set_lr(mult)
                    rs += 1
                    if a.seal_sched_global:
                        _gsched["step"] = rs
                    loss = step(b)
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    ema.update(model)
                    gstep[0] += 1
                    run += loss.item()
                print(
                    f"[eff r{rnd}] ep{ep} loss {run / len(sdl):.4f} ({time.time() - t1:.0f}s)",
                    flush=True,
                )
            tl = run / len(sdl)
            vl = seal_val()
            print(
                f"[eff r{rnd}] DONE  train {tl:.4f}  VAL {vl:.4f}  gap {vl - tl:+.4f}",
                flush=True,
            )
            del sdl, mix, seal
            gc.collect()
            torch.cuda.empty_cache()
            save(f"{a.out}_seal_r{rnd + 1}")
            save(f"{a.out}_seal_last")
            print(f"[eff r{rnd}] saved {a.out}_seal_r{rnd + 1}", flush=True)


if __name__ == "__main__":
    main()
