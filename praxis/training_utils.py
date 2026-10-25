"""Dataset adapters and moving-average weights shared by training workflows."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class ActSet(Dataset):
    """continuous action-chunk targets (from demos OR corrective rollout)."""

    def __init__(self, ds, rows_or_none, stats, pack_imgs=None):
        self.ds, self.rows = ds, rows_or_none
        self.a_mean, self.a_std, self.s_mean, self.s_std = stats

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        d = self.ds[i]
        out = {
            "imgs": d["imgs"],
            "state": d["state"],
            "task": d["task"],
            "action": d["action"],
        }
        if "sub" in d:
            out["sub"] = d["sub"]
        return out


class SealFlow(Dataset):
    def __init__(self, path, n, res, stats):
        self.n, self.res, self.path = n, res, path
        self.a_mean, self.a_std, self.s_mean, self.s_std = stats
        with np.load(Path(path) / "idx.npz") as d:
            self.state, self.act, self.task = d["state"], d["act"], d["task"]
            self.sub = d.get("sub", None)
        self._im = None

    @property
    def im(self):
        if self._im is None:
            self._im = np.memmap(
                f"{self.path}/imgs.u8",
                dtype=np.uint8,
                mode="r",
                shape=(self.n, 2, self.res, self.res, 3),
            )
        return self._im

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        imgs = torch.from_numpy(np.asarray(self.im[i], np.float32) / 255.0).permute(0, 3, 1, 2)
        st = (self.state[i] - self.s_mean) / self.s_std
        ac = (self.act[i] - self.a_mean) / self.a_std
        sub = (
            int(self.sub[i]) if self.sub is not None else -1
        )  # always emit sub (-1) so mixes collate cleanly
        return {
            "imgs": imgs,
            "state": torch.from_numpy(st.astype(np.float32)),
            "action": torch.from_numpy(ac.astype(np.float32)),
            "task": torch.tensor(int(self.task[i]), dtype=torch.long),
            "sub": torch.tensor(sub, dtype=torch.long),
        }


class EMA:
    """Average floating weights and copy discrete buffers without changing state keys."""

    def __init__(self, model, decay=0.9995):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                s.copy_(v)
