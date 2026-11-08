"""MLP and the two-network Eikonal PINN, ported from Eikonal2DnetCV2 in models_tf.py."""
from __future__ import annotations

import math
from typing import Callable, Optional

import torch
import torch.nn as nn


def xavier_normal_(w: torch.Tensor) -> None:
    """TF code: std = 1/sqrt((fan_in+fan_out)/2) = sqrt(2/(fan_in+fan_out)), zero-mean normal."""
    fan_out, fan_in = w.shape
    std = math.sqrt(2.0 / (fan_in + fan_out))
    with torch.no_grad():
        w.normal_(0.0, std)


class MLP(nn.Module):
    """layers = [in, h1, ..., out]; activation between layers, linear output.

    act: 'tanh' (Xavier normal init) or 'relu' (Kaiming normal init, fan_in). Biases zero.
    """

    def __init__(self, layers, act: str = "tanh"):
        super().__init__()
        self.layers = list(layers)
        self.act_name = act
        self.linears = nn.ModuleList(
            [nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]
        )
        if act == "tanh":
            self.act = torch.tanh
        elif act == "relu":
            self.act = torch.relu
        else:
            raise ValueError(f"unknown activation {act!r}")
        self.reset_parameters()

    def reset_parameters(self):
        for i, lin in enumerate(self.linears):
            is_hidden = i < len(self.linears) - 1
            if self.act_name == "relu" and is_hidden:
                nn.init.kaiming_normal_(lin.weight, mode="fan_in", nonlinearity="relu")
            else:
                xavier_normal_(lin.weight)
            nn.init.zeros_(lin.bias)

    def weight_matrices(self):
        return [lin.weight for lin in self.linears]

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        for lin in self.linears[:-1]:
            h = self.act(lin(h))
        return self.linears[-1](h)


class EikonalPINN(nn.Module):
    """T = net_T(H), V = v_max * sigmoid(net_V(H)), H = 2 (xy - lb)/(ub - lb) - 1.

    residual_form: 'tf'    -> f_T = ||grad T|| - 1/V   (models_tf.py line 144)
                   'paper' -> f_T = V ||grad T|| - 1   (paper Eq. 2)
    f_V = ||grad V|| in both cases. eps is added inside the sqrt only if > 0.
    """

    def __init__(
        self,
        layers_T=(2, 20, 20, 20, 20, 20, 1),
        layers_V=(2, 5, 5, 5, 5, 1),
        v_max: float = 1.5,
        lb=(0.0, 0.0),
        ub=(1.0, 1.0),
        act_T: str = "tanh",
        act_V: str = "tanh",
        residual_form: str = "tf",
        eps: float = 0.0,
    ):
        super().__init__()
        self.net_T = MLP(layers_T, act_T)
        self.net_V = MLP(layers_V, act_V)
        self.v_max = float(v_max)
        self.register_buffer("lb", torch.tensor(lb, dtype=torch.get_default_dtype()))
        self.register_buffer("ub", torch.tensor(ub, dtype=torch.get_default_dtype()))
        if residual_form not in ("tf", "paper"):
            raise ValueError(residual_form)
        self.residual_form = residual_form
        self.eps = float(eps)
        # optional override used by tests: callable xy -> T (N,1) replacing net_T
        self.T_override: Optional[Callable[[torch.Tensor], torch.Tensor]] = None

    def _scale(self, xy: torch.Tensor) -> torch.Tensor:
        return 2.0 * (xy - self.lb) / (self.ub - self.lb) - 1.0

    def forward_T(self, xy: torch.Tensor) -> torch.Tensor:
        if self.T_override is not None:
            return self.T_override(xy)
        return self.net_T(self._scale(xy))

    def forward_V(self, xy: torch.Tensor) -> torch.Tensor:
        return self.v_max * torch.sigmoid(self.net_V(self._scale(xy)))

    def forward(self, xy: torch.Tensor):
        return self.forward_T(xy), self.forward_V(xy)

    @staticmethod
    def _grad(out: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
        """(N, 2) gradient of a (N, 1) field wrt the (N, 2) input, graph kept for the loss."""
        return torch.autograd.grad(out, xy, grad_outputs=torch.ones_like(out), create_graph=True)[0]

    def _norm(self, g: torch.Tensor) -> torch.Tensor:
        sq = (g**2).sum(dim=1, keepdim=True)
        if self.eps > 0:
            sq = sq + self.eps
        return torch.sqrt(sq)

    def residual(self, xy: torch.Tensor):
        """Returns f_T, f_V, T, V, all (N, 1). xy must have requires_grad=True (set here if not)."""
        if not xy.requires_grad:
            xy = xy.detach().requires_grad_(True)
        T, V = self.forward(xy)
        grad_T = self._grad(T, xy)
        grad_V = self._grad(V, xy)
        nT = self._norm(grad_T)
        if self.residual_form == "tf":
            f_T = nT - 1.0 / V
        else:
            f_T = V * nT - 1.0
        f_V = self._norm(grad_V)
        return f_T, f_V, T, V

    @torch.no_grad()
    def predict(self, xy: torch.Tensor):
        T, V = self.forward(xy)
        return T, V
