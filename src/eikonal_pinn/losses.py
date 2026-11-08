"""Five-term loss of models_tf.py lines 74-79.

    L = lambda_data      * mean((T(xy_d) - T_d)^2)
      + lambda_res_data  * mean(f_T(xy_d)^2)
      + lambda_res_colloc* mean(f_T(xy_c)^2)                     (reference term, lambda = 1)
      + lambda_tv        * alpha * (mean(f_V(xy_d)^2) + mean(f_V(xy_c)^2))
      + alpha_L2 * sum_W 0.5 ||W||_F^2   over net_T weight matrices only

All lambdas are 1 in the baseline. Each term is returned separately (unweighted) for logging.
"""
from __future__ import annotations

import torch

TERM_NAMES = ("data", "res_data", "res_colloc", "tv", "l2")
WEIGHTED_TERMS = ("data", "res_data", "tv")  # terms that get adaptive lambdas


def loss_terms(model, xy_data, T_data, xy_colloc, alpha: float, alpha_L2: float, lambdas=None):
    """Returns (total, terms) where terms is a dict of unweighted scalar tensors.

    lambdas: dict name -> float weight for the WEIGHTED_TERMS (default all 1).
    """
    lam = {k: 1.0 for k in WEIGHTED_TERMS}
    if lambdas:
        lam.update(lambdas)

    n_d = xy_data.shape[0]
    f_T, f_V, T_pred, _ = model.residual(torch.cat([xy_data, xy_colloc], dim=0))
    f_T_d, f_T_c = f_T[:n_d], f_T[n_d:]
    f_V_d, f_V_c = f_V[:n_d], f_V[n_d:]
    T_d_pred = T_pred[:n_d]

    terms = {
        "data": torch.mean((T_d_pred - T_data) ** 2),
        "res_data": torch.mean(f_T_d**2),
        "res_colloc": torch.mean(f_T_c**2),
        "tv": alpha * (torch.mean(f_V_d**2) + torch.mean(f_V_c**2)),
        "l2": alpha_L2 * sum(0.5 * torch.sum(W**2) for W in model.net_T.weight_matrices()),
    }
    total = (
        lam["data"] * terms["data"]
        + lam["res_data"] * terms["res_data"]
        + terms["res_colloc"]
        + lam["tv"] * terms["tv"]
        + terms["l2"]
    )
    return total, terms
