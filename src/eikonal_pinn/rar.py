"""Residual-based adaptive refinement (Lu, Meng, Mao, Karniadakis 2021, SIAM Rev., Sec. 2.8).

Every `rar_every` Adam steps: evaluate |f_T| on `n_candidates` uniform points and append the
`m` points with the largest |f_T| to a persistent set X_rar. The Adam minibatch is then the
fresh LHS batch concatenated with X_rar, and L-BFGS uses the grid concatenated with X_rar.
"""
from __future__ import annotations

import numpy as np
import torch


class RARSet:
    def __init__(self, seed: int, n_candidates: int = 10000, m: int = 50):
        self.rng = np.random.default_rng(seed)
        self.n_candidates = int(n_candidates)
        self.m = int(m)
        self.X = np.zeros((0, 2))
        self.history = []  # (step, added points, their |f_T|)

    def refine(self, model, step: int, dtype=torch.float32, device="cpu"):
        cand = self.rng.random((self.n_candidates, 2))
        xy = torch.tensor(cand, dtype=dtype, device=device, requires_grad=True)
        with torch.enable_grad():
            f_T, _, _, _ = model.residual(xy)
        r = f_T.detach().abs().cpu().numpy()[:, 0]
        idx = np.argsort(r)[-self.m:]
        self.X = np.vstack([self.X, cand[idx]])
        self.history.append((step, cand[idx], r[idx]))
        return cand[idx], r[idx]

    def tensor(self, dtype=torch.float32, device="cpu"):
        return torch.tensor(self.X, dtype=dtype, device=device)

    def __len__(self):
        return len(self.X)
