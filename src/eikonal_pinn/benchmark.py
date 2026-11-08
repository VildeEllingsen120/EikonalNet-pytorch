"""Benchmark problem of Sahli Costabal et al. (2020), Eqs. (11) and (12).

    T(x, y) = min(r0, 0.7 r1)
    V(x, y) = 1       if r0 <  0.7 r1
            = 1/0.7   otherwise

with r0 = sqrt(x^2 + y^2), r1 = sqrt((x-1)^2 + (y-1)^2) on [0, 1]^2.
All functions accept an (N, 2) array and return (N, 1).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import qmc

# Max of exact T on [0,1]^2: where the ridge meets the edge y = 0 (or x = 0), i.e. the root of
# x^2 = 0.49 ((x-1)^2 + 1)  ->  0.51 x^2 + 0.98 x - 0.98 = 0. Numerically 0.72583.
T_MAX = float((-0.98 + np.sqrt(0.98**2 + 4 * 0.51 * 0.98)) / (2 * 0.51))


def _radii(xy: np.ndarray):
    xy = np.asarray(xy, dtype=float)
    r0 = np.sqrt(xy[:, 0] ** 2 + xy[:, 1] ** 2)
    r1 = np.sqrt((xy[:, 0] - 1.0) ** 2 + (xy[:, 1] - 1.0) ** 2)
    return r0, r1


def exact_T(xy: np.ndarray) -> np.ndarray:
    r0, r1 = _radii(xy)
    return np.minimum(r0, 0.7 * r1)[:, None]


def exact_V(xy: np.ndarray) -> np.ndarray:
    r0, r1 = _radii(xy)
    return np.where(r0 < 0.7 * r1, 1.0, 1.0 / 0.7)[:, None]


def ridge_distance(xy: np.ndarray, n_ridge: int = 4000) -> np.ndarray:
    """Euclidean distance from each point to the ridge r0 = 0.7 r1, (N, 1).

    The ridge is the circle |x - c| = R with c = (-0.49, -0.49)/0.51 ... derived by squaring
    r0^2 = 0.49 r1^2; we sample it densely and take the nearest sample.
    """
    # r0^2 = 0.49 r1^2  ->  0.51 (x^2 + y^2) + 0.98 (x + y) - 0.98 = 0
    # -> (x + c)^2 + (y + c)^2 = R^2 with c = 0.49/0.51, R^2 = 2 c^2 + 0.98/0.51
    c = 0.49 / 0.51
    R = np.sqrt(2 * c**2 + 0.98 / 0.51)
    theta = np.linspace(0, 2 * np.pi, n_ridge, endpoint=False)
    ridge = np.stack([-c + R * np.cos(theta), -c + R * np.sin(theta)], axis=1)
    ridge = ridge[(ridge[:, 0] >= -0.05) & (ridge[:, 0] <= 1.05) & (ridge[:, 1] >= -0.05) & (ridge[:, 1] <= 1.05)]
    xy = np.asarray(xy, dtype=float)
    d2 = ((xy[:, None, :] - ridge[None, :, :]) ** 2).sum(-1)
    return np.sqrt(d2.min(axis=1))[:, None]


def make_grid(n: int = 50):
    """50x50 meshgrid on [0,1]^2 as in the notebook: returns (xy (n*n, 2), X_m, Y_m)."""
    x = np.linspace(0.0, 1.0, n)
    X_m, Y_m = np.meshgrid(x, x)
    xy = np.stack([X_m.flatten(), Y_m.flatten()], axis=1)
    return xy, X_m, Y_m


def sample_data(n: int = 50, seed: int = 1234) -> np.ndarray:
    """Latin hypercube sample of n points in [0,1]^2 (scipy, not pyDOE)."""
    sampler = qmc.LatinHypercube(d=2, seed=seed)
    return sampler.random(n)


def add_noise(T: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Add N(0, (sigma * T_max)^2) noise; T_max is the max of the exact T over the domain."""
    if sigma == 0.0:
        return T.copy()
    return T + rng.normal(0.0, sigma * T_MAX, size=T.shape)
