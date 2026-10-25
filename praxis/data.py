"""LIBERO action chunks backed by a memory-mapped JPEG blob.

Samples contain two RGB views, normalized robot state, a task ID and an action
chunk. The expected pack schema is documented in docs/data-format.md. Each worker
opens the blob lazily so image bytes do not need to be copied into worker memory."""

from __future__ import annotations
import io, os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

PACK = "data/libero_pack"


class LiberoChunks(Dataset):
    def __init__(
        self,
        chunk=8,
        res=128,
        pack=PACK,
        max_episodes=None,
        stats=None,
        hist=1,
        vhist=1,
        vstride=16,
        sublabels=None,
    ):
        self.chunk, self.res, self.pack, self.hist = chunk, res, pack, hist
        self.vhist, self.vstride = vhist, vstride
        self.sublabels = sublabels  # per-global-row sub-task label (int8, -1 = none)
        ix = np.load(os.path.join(pack, "index.npz"))
        self.img_off, self.img_len = ix["img_off"], ix["img_len"]
        self.wr_off, self.wr_len = ix["wr_off"], ix["wr_len"]
        self.state, self.act, self.task = ix["state"], ix["act"], ix["task"]
        ep_start, ep_len = ix["ep_start"], ix["ep_len"]
        if max_episodes:
            ep_start, ep_len = ep_start[:max_episodes], ep_len[:max_episodes]
        self.ep_start, self.ep_len = ep_start, ep_len
        # valid (episode, t) starts where a full chunk fits inside the episode
        eps = np.repeat(np.arange(len(ep_start)), np.maximum(ep_len - chunk, 0))
        ts = np.concatenate([np.arange(max(int(l) - chunk, 0)) for l in ep_len])
        self.ep_of, self.t_of = eps, ts
        self.rows = ep_start[eps] + ts  # global row per sample
        self._blob = None  # memmap opened lazily per worker
        if stats is None:
            s, a = self.state, self.act
            self.s_mean, self.s_std = s.mean(0), s.std(0).clip(1e-3)
            self.a_mean, self.a_std = a.mean(0), a.std(0).clip(1e-3)
        else:
            (self.a_mean, self.a_std, self.s_mean, self.s_std) = stats
        self.n_tasks = int(self.task.max()) + 1

    def stats(self):
        return (self.a_mean, self.a_std, self.s_mean, self.s_std)

    @property
    def blob(self):
        if self._blob is None:  # each worker maps it itself
            self._blob = np.memmap(os.path.join(self.pack, "blob.u8"), dtype=np.uint8, mode="r")
        return self._blob

    def _img(self, off, ln):
        im = Image.open(io.BytesIO(self.blob[off : off + ln].tobytes())).convert("RGB")
        if im.size != (self.res, self.res):
            im = im.resize((self.res, self.res), Image.BILINEAR)
        return torch.from_numpy(np.asarray(im, np.float32) / 255.0).permute(2, 0, 1)

    def __len__(self):
        return len(self.rows)

    def _state_hist(self, r, ep_start):
        """last `hist` states ending at r, clamped to the episode start (front-padded)."""
        lo = max(ep_start, r - self.hist + 1)
        seg = self.state[lo : r + 1]  # (<=hist, 8)
        if len(seg) < self.hist:  # front-pad with earliest in-episode
            seg = np.concatenate([np.repeat(seg[:1], self.hist - len(seg), 0), seg], 0)
        return ((seg - self.s_mean) / self.s_std).reshape(-1)  # (hist*8,)

    def _frame(self, row):
        """base+wrist images at a global row -> (2,3,res,res)."""
        return torch.stack(
            [
                self._img(int(self.img_off[row]), int(self.img_len[row])),
                self._img(int(self.wr_off[row]), int(self.wr_len[row])),
            ]
        )

    def __getitem__(self, i):
        r = int(self.rows[i])
        if self.vhist > 1:  # strided visual history, clamped to ep
            ep0 = int(self.ep_start[self.ep_of[i]])
            rowsK = [max(ep0, r - k * self.vstride) for k in range(self.vhist - 1, -1, -1)]
            imgs = torch.stack([self._frame(rw) for rw in rowsK])  # (vhist,2,3,res,res)
        else:
            imgs = self._frame(r)
        if self.hist > 1:
            state = self._state_hist(r, int(self.ep_start[self.ep_of[i]]))
        else:
            state = (self.state[r] - self.s_mean) / self.s_std
        act = (self.act[r : r + self.chunk] - self.a_mean) / self.a_std
        sub = int(self.sublabels[r]) if self.sublabels is not None else -1
        return dict(
            imgs=imgs,
            state=torch.from_numpy(state.astype(np.float32)),
            action=torch.from_numpy(act.astype(np.float32)),
            task=torch.tensor(int(self.task[r]), dtype=torch.long),
            sub=torch.tensor(sub, dtype=torch.long),
        )
