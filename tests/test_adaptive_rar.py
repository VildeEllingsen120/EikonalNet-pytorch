import numpy as np
import torch

from eikonal_pinn.adaptive import AdaptiveWeights
from eikonal_pinn.benchmark import exact_T, sample_data
from eikonal_pinn.losses import WEIGHTED_TERMS, loss_terms
from eikonal_pinn.model import EikonalPINN
from eikonal_pinn.rar import RARSet


def test_adaptive_formula_matches_manual():
    torch.manual_seed(0)
    m = EikonalPINN()
    params = list(m.parameters())
    xy_d = torch.tensor(sample_data(50), dtype=torch.float32)
    T_d = torch.tensor(exact_T(xy_d.numpy()), dtype=torch.float32)
    xy_c = torch.rand(100, 2)
    _, terms = loss_terms(m, xy_d, T_d, xy_c, 1e-7, 1e-9)
    aw = AdaptiveWeights(WEIGHTED_TERMS, params, beta=0.9)
    lam = aw.update(terms)
    # manual: max |grad L_r| / mean |grad L_i| over all params (zeros for unused)
    def flat(t):
        gs = torch.autograd.grad(t, params, retain_graph=True, allow_unused=True)
        return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1).abs() for g, p in zip(gs, params)])
    num = flat(terms["res_colloc"]).max().item()
    for k in WEIGHTED_TERMS:
        hat = num / flat(terms[k]).mean().item()
        assert np.isclose(lam[k], 0.1 * 1.0 + 0.9 * hat, rtol=1e-4)
    assert lam["data"] > 0


def test_rar_appends_and_targets_large_residual():
    torch.manual_seed(0)
    m = EikonalPINN()
    rar = RARSet(seed=0, n_candidates=500, m=20)
    pts, res = rar.refine(m, step=1)
    assert pts.shape == (20, 2) and len(rar) == 20
    rar.refine(m, step=2)
    assert len(rar) == 40 and rar.tensor().shape == (40, 2)
    # selected residuals are the largest among the candidates
    xy = torch.tensor(rar.rng.random((500, 2)), dtype=torch.float32)  # fresh points, just a sanity bound
    f_T, _, _, _ = m.residual(xy)
    assert res.min() >= np.quantile(f_T.detach().abs().numpy(), 0.5)
