"""Small shared utilities: image-shift augmentation, per-suite horizons, and the demo lookup."""

import torch

# Max control steps per LIBERO suite (libero_10 is long-horizon, so it gets more).
SUITE_MAX = {"libero_spatial": 400, "libero_object": 400, "libero_goal": 400, "libero_10": 600}


def build_demo_lookup(ds, tmap):
    """Map task_index -> list of pack episode ids (kept in LIBERO demo order == init-state order)."""
    by_task = {}
    for e in range(len(ds.ep_start)):
        ti = int(ds.task[ds.ep_start[e]])
        by_task.setdefault(ti, []).append(e)
    return by_task


def rand_shift(x, pad=8):
    """DrQ per-sample random image shift (Kostrikov et al. 2020): replicate-pad, then crop each image at its own
    random offset. Per-sample (not per-batch) gives ~B x more augmentation diversity, which matters because the
    dataset is only ~a few thousand trajectories with near-duplicate consecutive frames. For a 6-D input
    (B,T,cams,C,H,W) the same shift is shared across the T history frames of a (sample,cam) so it adds no fake
    inter-frame motion."""
    temporal = x.dim() == 6
    if temporal:
        B, T, C2, C, H, W = x.shape
    else:
        B, C2, C, H, W = x.shape
        T = 1
    xf = torch.nn.functional.pad(x.reshape(-1, C, H, W), (pad,) * 4, mode="replicate")
    oy = torch.randint(0, 2 * pad + 1, (B, C2), device=x.device)
    ox = torch.randint(0, 2 * pad + 1, (B, C2), device=x.device)
    if temporal:
        oy = oy[:, None, :].expand(B, T, C2).reshape(-1)
        ox = ox[:, None, :].expand(B, T, C2).reshape(-1)
    else:
        oy = oy.reshape(-1)
        ox = ox.reshape(-1)
    N = xf.shape[0]
    ar = torch.arange(H, device=x.device)
    ry = oy[:, None] + ar[None, :]
    rx = ox[:, None] + ar[None, :]
    xf = xf[torch.arange(N)[:, None, None], :, ry[:, :, None], rx[:, None, :]]
    return xf.permute(0, 3, 1, 2).contiguous().view(*x.shape)
