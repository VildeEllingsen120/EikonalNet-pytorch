import numpy as np
import torch

from eikonal_pinn.benchmark import exact_T, ridge_distance, sample_data
from eikonal_pinn.losses import loss_terms
from eikonal_pinn.model import EikonalPINN


def exact_T_torch(xy):
    r0 = torch.sqrt(xy[:, :1] ** 2 + xy[:, 1:] ** 2)
    r1 = torch.sqrt((xy[:, :1] - 1) ** 2 + (xy[:, 1:] - 1) ** 2)
    return torch.minimum(r0, 0.7 * r1)


def exact_V_torch(xy):
    r0 = torch.sqrt(xy[:, :1] ** 2 + xy[:, 1:] ** 2)
    r1 = torch.sqrt((xy[:, :1] - 1) ** 2 + (xy[:, 1:] - 1) ** 2)
    # + 0*r0 keeps V attached to xy so autograd.grad has a path (gradient is exactly zero)
    return torch.where(r0 < 0.7 * r1, torch.ones_like(r0), torch.full_like(r0, 1 / 0.7)) + 0.0 * r0


def _off_ridge(n, seed):
    rng = np.random.default_rng(seed)
    pts = rng.random((4 * n, 2))
    pts = pts[ridge_distance(pts)[:, 0] > 0.05][:n]
    assert len(pts) == n
    return pts


def test_data_and_residual_terms_vanish_with_exact_T():
    torch.manual_seed(0)
    m = EikonalPINN().double()
    m.T_override = exact_T_torch
    m.forward_V = exact_V_torch  # exact V too, so f_T = ||grad T|| - 1/V is exactly zero off-ridge
    xy_d = torch.tensor(_off_ridge(50, 1), dtype=torch.float64)
    T_d = torch.tensor(exact_T(xy_d.numpy()), dtype=torch.float64)
    xy_c = torch.tensor(_off_ridge(500, 2), dtype=torch.float64)
    total, terms = loss_terms(m, xy_d, T_d, xy_c, alpha=0.0, alpha_L2=0.0)
    assert terms["data"].item() < 1e-4
    assert terms["res_data"].item() < 1e-4
    assert terms["res_colloc"].item() < 1e-4
    assert terms["tv"].item() == 0.0 and terms["l2"].item() == 0.0
    assert total.item() < 3e-4


def test_terms_are_separate_and_weighted():
    torch.manual_seed(0)
    m = EikonalPINN()
    xy_d = torch.tensor(sample_data(50), dtype=torch.float32)
    T_d = torch.tensor(exact_T(xy_d.numpy()), dtype=torch.float32)
    xy_c = torch.rand(100, 2)
    total, terms = loss_terms(m, xy_d, T_d, xy_c, alpha=1e-7, alpha_L2=1e-9)
    assert set(terms) == {"data", "res_data", "res_colloc", "tv", "l2"}
    assert torch.isclose(total, sum(terms.values()))
    total2, _ = loss_terms(m, xy_d, T_d, xy_c, alpha=1e-7, alpha_L2=1e-9, lambdas={"data": 3.0})
    assert torch.isclose(total2 - total, 2.0 * terms["data"])
    assert total.requires_grad
