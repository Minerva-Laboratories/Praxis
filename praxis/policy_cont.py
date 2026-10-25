"""Continuous regression policy with the SAME backbone as FlowPolicy/MaskGITPolicy, exposing a
FlowPolicy-compatible interface (loss, sample) so it runs through the identical SEAL pipeline.
This is the FAIR baseline: direct L1 regression of the action chunk (the 'tokenizer-free ceiling'
and the strongest small-data conditioner), same pretrain->FT->SEAL, seeded, matched regs."""

import torch, torch.nn as nn, torch.nn.functional as F
from .policy import VLAPolicy


class ContPolicy(nn.Module):
    def __init__(self, n_tasks, action_dim=7, chunk=8, p_drop=0.1, hist=1, vhist=1, **kw):
        super().__init__()
        self.net = VLAPolicy(
            n_tasks=n_tasks,
            action_dim=action_dim,
            chunk=chunk,
            p_drop=p_drop,
            hist=hist,
            vhist=vhist,
        )
        self.chunk, self.action_dim, self.hist, self.vhist = chunk, action_dim, hist, vhist

    def loss(self, imgs, state, task, a1):
        return F.l1_loss(self.net(imgs, state, task), a1)

    @torch.no_grad()
    def sample(self, imgs, state, task, steps=None, deterministic=True):
        return self.net(imgs, state, task)  # deterministic regression, no ODE
