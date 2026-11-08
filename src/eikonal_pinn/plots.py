"""Figures: 2x2 field panels and loss curves."""
from __future__ import annotations

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

T_ERR_LIM = 0.05
V_ERR_LIM = 0.5


def plot_fields(path, X_m, Y_m, T_pred, T_true, V_pred, V_true, xy_data, X_rar=None, title=None):
    shape = X_m.shape
    T_pred = np.asarray(T_pred).reshape(shape)
    T_true = np.asarray(T_true).reshape(shape)
    V_pred = np.asarray(V_pred).reshape(shape)
    V_true = np.asarray(V_true).reshape(shape)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8.5))
    panels = [
        (axes[0, 0], T_pred, "T predicted", dict(levels=20, cmap="viridis")),
        (axes[0, 1], T_pred - T_true, "T_pred - T_exact", dict(levels=np.linspace(-T_ERR_LIM, T_ERR_LIM, 21), cmap="RdBu_r", extend="both")),
        (axes[1, 0], V_pred, "V predicted", dict(levels=np.linspace(0.0, 1.5, 31), cmap="viridis")),
        (axes[1, 1], V_pred - V_true, "V_pred - V_exact", dict(levels=np.linspace(-V_ERR_LIM, V_ERR_LIM, 21), cmap="RdBu_r", extend="both")),
    ]
    for ax, field, name, kw in panels:
        cs = ax.contourf(X_m, Y_m, field, **kw)
        fig.colorbar(cs, ax=ax)
        ax.scatter(xy_data[:, 0], xy_data[:, 1], facecolors="none", edgecolors="k", s=25, linewidths=0.8)
        if X_rar is not None and len(X_rar) > 0:
            ax.scatter(X_rar[:, 0], X_rar[:, 1], marker="x", c="red", s=20, linewidths=0.8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_title(name)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def read_loss_csv(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows


def plot_loss(path_png, loss_csv, title=None):
    rows = read_loss_csv(loss_csv)
    if not rows:
        return
    terms = ["total", "data", "res_data", "res_colloc", "tv", "l2"]
    lambdas = [k for k in rows[0].keys() if k.startswith("lambda_")]
    phases = [r["phase"] for r in rows]
    n_adam = sum(1 for p in phases if p == "adam")
    adam_steps = np.array([int(r["step"]) for r in rows if r["phase"] == "adam"])
    last_adam = int(adam_steps.max()) if n_adam else 0
    # x-axis: Adam step, then L-BFGS closure count appended after the last Adam step
    x = np.array([int(r["step"]) if r["phase"] == "adam" else last_adam + int(r["step"]) for r in rows])

    has_lambda = bool(lambdas) and any(float(r[lambdas[0]]) != 1.0 for r in rows)
    fig, axes = plt.subplots(2 if has_lambda else 1, 1, figsize=(9, 7 if has_lambda else 4.5), squeeze=False)
    ax = axes[0, 0]
    for t in terms:
        y = np.array([float(r[t]) for r in rows])
        y = np.where(y > 0, y, np.nan)
        ax.plot(x, y, label=t, lw=1)
    if n_adam and n_adam < len(rows):
        ax.axvline(last_adam, color="k", ls="--", lw=1)
        ax.text(last_adam, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1, "  L-BFGS", va="top")
    ax.set_yscale("log")
    ax.set_xlabel("Adam step  |  L-BFGS closure call")
    ax.set_ylabel("loss term")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(alpha=0.3)
    if title:
        ax.set_title(title)
    if has_lambda:
        ax2 = axes[1, 0]
        for k in lambdas:
            ax2.plot(x, [float(r[k]) for r in rows], label=k, lw=1)
        ax2.set_yscale("log")
        ax2.set_ylabel("adaptive weight")
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path_png, dpi=130)
    plt.close(fig)
