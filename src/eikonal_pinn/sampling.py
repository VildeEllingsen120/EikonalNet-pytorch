"""Seeded samplers for collocation points."""
from __future__ import annotations

import numpy as np
import torch
from scipy.stats import qmc


class LHSampler:
    """Fresh Latin hypercube batch on each call; the underlying generator advances each draw."""

    def __init__(self, seed: int, d: int = 2):
        self._sampler = qmc.LatinHypercube(d=d, seed=seed)

    def draw(self, n: int) -> np.ndarray:
        return self._sampler.random(n)

    def draw_torch(self, n: int, dtype=torch.float32, device="cpu") -> torch.Tensor:
        return torch.as_tensor(self.draw(n), dtype=dtype, device=device)

