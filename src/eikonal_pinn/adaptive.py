"""Learning-rate annealing of Wang, Teng, Perdikaris (2021), Algorithm 1 (arXiv:2001.04536v1, p.10).

    lambda_hat_i = max_theta |grad_theta L_r|  /  mean_theta |grad_theta L_i|        (Eq. 40)
    lambda_i    <- (1 - alpha) lambda_i + alpha lambda_hat_i,  alpha = 0.9           (Eq. 41)

L_r is the PDE residual at the collocation points. The denominator uses the UNWEIGHTED L_i
(the paper's Eq. 40), not lambda_i L_i as the project spec draft said. Gradients are taken
with respect to all trainable parameters, flattened and concatenated; parameters a term does
not depend on contribute zeros.
"""
from __future__ import annotations

import torch


class AdaptiveWeights:
    def __init__(self, names, params, beta: float = 0.9):
        self.names = list(names)
        self.params = list(params)
        self.beta = float(beta)
        self.lam = {k: 1.0 for k in self.names}
        self.n_updates = 0
        self.n_skipped = {k: 0 for k in self.names}

    def _flat_abs_grad(self, scalar):
        grads = torch.autograd.grad(scalar, self.params, retain_graph=True, allow_unused=True)
        parts = [(g if g is not None else torch.zeros_like(p)).reshape(-1).abs() for g, p in zip(grads, self.params)]
        return torch.cat(parts)

    @torch.no_grad()
    def _apply(self, k, hat):
        if not (hat == hat) or hat == float("inf") or hat <= 0:  # nan / inf / zero
            self.n_skipped[k] += 1
            return
        self.lam[k] = (1.0 - self.beta) * self.lam[k] + self.beta * hat

    def update(self, terms: dict, ref: str = "res_colloc"):
        """terms: dict of unweighted scalar loss terms that still hold their graph."""
        g_ref = self._flat_abs_grad(terms[ref])
        num = g_ref.max().item()
        for k in self.names:
            g = self._flat_abs_grad(terms[k])
            den = g.mean().item()
            self._apply(k, num / den if den > 0 else float("inf"))
        self.n_updates += 1
        return dict(self.lam)
