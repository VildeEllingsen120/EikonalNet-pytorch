#!/usr/bin/env python3
"""Stage 9: report assets into report_assets/ (tables, figures as PDF+PNG at 300 dpi, facts.md)."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)
from eikonal_pinn.benchmark import exact_T, exact_V  # noqa: E402
from eikonal_pinn.model import EikonalPINN  # noqa: E402
from summarize import VARIANTS, collect, load  # noqa: E402

OUT = os.path.join(ROOT, "report_assets")
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "pdf.fonttype": 42})
SINGLE, DOUBLE = 3.5, 7.0
FACTS = []


def fact(what, value, source):
    FACTS.append((what, value, source))


def savefig(fig, name):
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300)
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    plt.close(fig)


def present_variants():
    return [v for v in VARIANTS if any(load(f"{v[1]}_d1234_s{s}") for s in range(5))]


# ---------------------------------------------------------------- 1. tables
def tables(variants):
    md = ["| variant | nRMSE_T [%] | nMAE_V [%] | max err T | max err V | failed seeds |", "|---|---|---|---|---|---|"]
    tex = ["\\begin{tabular}{lcccc}", "\\toprule",
           "variant & nRMSE$_T$ [\\%] & nMAE$_V$ [\\%] & max err $T$ & max err $V$ \\\\", "\\midrule"]
    stats = {}
    for label, prefix, flags in variants:
        ok, failed = collect(prefix)
        if not ok:
            continue
        g = lambda k: np.array([m[k] for _, m in ok])
        cell = lambda k, f: f"{f % g(k).mean()} ({f % g(k).min()} to {f % g(k).max()})"
        fs = ", ".join(f"seed {s}" for s, _ in failed) or "none"
        md.append(f"| {label} | {cell('nrmse_T', '%.2f')} | {cell('nmae_V', '%.2f')} | {cell('maxerr_T', '%.3f')} | {cell('maxerr_V', '%.2f')} | {fs} |")
        tl = label.replace("_", "\\_").replace("%", "\\%")
        tex.append(f"{tl} & {cell('nrmse_T', '%.2f')} & {cell('nmae_V', '%.2f')} & {cell('maxerr_T', '%.3f')} & {cell('maxerr_V', '%.2f')} \\\\")
        stats[label] = {k: (g(k).mean(), g(k).min(), g(k).max()) for k in ("nrmse_T", "nmae_V", "maxerr_T", "maxerr_V")}
        for k in ("nrmse_T", "nmae_V", "maxerr_T", "maxerr_V"):
            fact(f"{label}: {k} mean/min/max over seeds {[s for s, _ in ok]}", f"{g(k).mean():.4f} / {g(k).min():.4f} / {g(k).max():.4f}",
                 ", ".join(f"results/{prefix}_d1234_s{s}/metrics.json" for s, _ in ok))
        for s, why in failed:
            fact(f"{label}: failed seed", f"seed {s}: {why}", f"results/{prefix}_d1234_s{s}/")
    tex += ["\\bottomrule", "\\end{tabular}"]
    # hard layout (stage 6): data seed 2, init seeds 1234, 1, 2
    base = {1234: "baseline_tanh_fixed_d2_s1234", 1: "baseline_tanh_fixed_d2_s1", 2: "baseline_tanh_fixed_seed2"}
    hl = [("baseline", None), ("adaptive (paper)", "adaptive_datafit"), ("relu", "relu"), ("adaptive+rar+relu", "adaptive_rar_relu")]
    md += ["", "Hard layout (data seed 2), cells nRMSE_T / nMAE_V [%]:", "", "| variant | init 1234 | init 1 | init 2 |", "|---|---|---|---|"]
    tex += ["", "% hard layout, data seed 2", "\\begin{tabular}{lccc}", "\\toprule", "variant & init 1234 & init 1 & init 2 \\\\", "\\midrule"]
    for label, prefix in hl:
        cells = []
        for k in (1234, 1, 2):
            run = base[k] if prefix is None else f"{prefix}_d2_s{k}"
            m = load(run)
            if m is None:
                cells.append("-"); continue
            cells.append(f"{m['nrmse_T']:.2f} / {m['nmae_V']:.2f}")
            fact(f"hard layout, {label}, data seed 2, init {k}: nrmse_T / nmae_V", f"{m['nrmse_T']:.4f} / {m['nmae_V']:.4f}", f"results/{run}/metrics.json")
        md.append(f"| {label} | " + " | ".join(cells) + " |")
        tex.append(f"{label.replace('_', chr(92) + '_')} & " + " & ".join(cells) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(OUT, "summary_table.md"), "w").write(
        "Mean (min to max) over data seed 1234, init seeds 0 to 4. Paper Table 1 (0 % noise, PINN): nRMSE_T 0.49, nMAE_V 3.56.\n\n" + "\n".join(md) + "\n")
    open(os.path.join(OUT, "summary_table.tex"), "w").write(
        "% mean (min to max) over data seed 1234, init seeds 0-4; requires \\usepackage{booktabs}\n" + "\n".join(tex) + "\n")
    return stats


# ---------------------------------------------------------------- helpers
def fields(run):
    p = os.path.join(ROOT, "results", run, "fields.npz")
    return np.load(p) if os.path.exists(p) else None


def model_for(run):
    p = os.path.join(ROOT, "results", run, "model.pt")
    if not os.path.exists(p) and os.path.exists(os.path.join(ROOT, "results", run + "_rerun", "model.pt")):
        a, b = load(run), load(run + "_rerun")
        fact(f"cross-section weights for {run} taken from {run}_rerun (original run predates weight saving); original vs rerun nrmse_T / nmae_V",
             f"{a['nrmse_T']:.4f} / {a['nmae_V']:.4f} vs {b['nrmse_T']:.4f} / {b['nmae_V']:.4f}"
             + (" (identical: both single-thread)" if abs(a['nrmse_T'] - b['nrmse_T']) < 1e-6 else " (original used default torch threads, rerun one thread)"),
             f"results/{run}_rerun/")
        run = run + "_rerun"
        p = os.path.join(ROOT, "results", run, "model.pt")
    cfg = json.load(open(os.path.join(ROOT, "results", run, "config.json")))
    if not os.path.exists(p):
        return None
    m = EikonalPINN(v_max=cfg["v_max"], act_T=cfg["act"], residual_form=cfg["residual"], eps=cfg["eps"])
    m.load_state_dict(torch.load(p, map_location="cpu"))
    m.eval()
    return m


def short(label):
    return (label.replace("adaptive (spec: data, res_data, tv)", "adaptive (spec)")
                 .replace("adaptive_datafit (paper: data, res_data)", "adaptive (paper)")
                 .replace("residual paper (ablation)", "residual paper"))


def pick_rows(variants, best_combo_label=None):
    """baseline, adaptive (paper version), rar, relu, best combination (if present)."""
    want = ["baseline", "adaptive_datafit (paper: data, res_data)", "rar", "relu"]
    rows = [v for w in want for v in variants if v[0] == w]
    if best_combo_label:
        rows += [v for v in variants if v[0] == best_combo_label]
    return rows


# ---------------------------------------------------------------- 2. field figure
def field_figure(rows, seed=0):
    rows = [r for r in rows if fields(f"{r[1]}_d1234_s{seed}") is not None]
    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(DOUBLE, 2.1 * n + 0.9), squeeze=False, layout="constrained")
    for i, (label, prefix, _) in enumerate(rows):
        run = f"{prefix}_d1234_s{seed}"
        f = fields(run)
        X, Y = f["X_m"], f["Y_m"]
        Vp, Ve, Tp, Te = f["V_pred"].reshape(X.shape), f["V_exact"].reshape(X.shape), f["T_pred"].reshape(X.shape), f["T_exact"].reshape(X.shape)
        panels = [(Vp, "V predicted", dict(levels=np.linspace(0, 1.5, 31), cmap="viridis")),
                  (Vp - Ve, "V error", dict(levels=np.linspace(-0.5, 0.5, 21), cmap="RdBu_r", extend="both")),
                  (Tp - Te, "T error", dict(levels=np.linspace(-0.05, 0.05, 21), cmap="RdBu_r", extend="both"))]
        for j, (fld, name, kw) in enumerate(panels):
            ax = axes[i, j]
            cs = ax.contourf(X, Y, fld, **kw)
            ax.scatter(f["xy_data"][:, 0], f["xy_data"][:, 1], facecolors="none", edgecolors="k", s=6, linewidths=0.4)
            if "X_rar" in f and len(f["X_rar"]):
                ax.scatter(f["X_rar"][:, 0], f["X_rar"][:, 1], marker="x", c="red", s=5, linewidths=0.4)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
            ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
            if i < n - 1:
                ax.set_xticklabels([])
            if j > 0:
                ax.set_yticklabels([])
            if i == 0:
                ax.set_title(name)
            if j == 0:
                ax.set_ylabel(short(label))
            if i == n - 1:
                cb = fig.colorbar(cs, ax=axes[:, j].tolist(), orientation="horizontal", shrink=0.9, pad=0.02, aspect=25,
                                  ticks=[0, 0.5, 1.0, 1.5] if j == 0 else ([-0.5, -0.25, 0, 0.25, 0.5] if j == 1 else [-0.05, -0.025, 0, 0.025, 0.05]))
                cb.ax.tick_params(labelsize=8)
        m = load(run)
        fact(f"field figure row '{label}' (seed {seed}): nrmse_T / nmae_V", f"{m['nrmse_T']:.3f} / {m['nmae_V']:.3f}", f"results/{run}/")
    fig.suptitle(f"data seed 1234, init seed {seed}; circles = data points, red crosses = RAR points", fontsize=8)
    savefig(fig, "fig_fields_variants")


# ---------------------------------------------------------------- 3. cross-section
def cross_section(rows, seed=0, n=801, p0=(0.9, 0.1), p1=(0.1, 0.9), name="fig_cross_section", note=""):
    s = np.linspace(0, 1, n)
    xy = np.stack([p0[0] + (p1[0] - p0[0]) * s, p0[1] + (p1[1] - p0[1]) * s], 1)
    L = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
    arc = s * L
    Te, Ve = exact_T(xy)[:, 0], exact_V(xy)[:, 0]
    g = np.hypot(xy[:, 0], xy[:, 1]) - 0.7 * np.hypot(xy[:, 0] - 1, xy[:, 1] - 1)
    crosses = bool(np.any(g <= 0) and np.any(g >= 0))
    k = int(np.argmin(np.abs(g)))
    fact(f"{name}: line", f"from {p0} to {p1}, {n} points, arc length {L:.4f}; crosses the ridge: {crosses}"
         + (f" at arc length {arc[k]:.4f}, point ({xy[k, 0]:.4f}, {xy[k, 1]:.4f})" if crosses else f"; closest approach g = r0 - 0.7 r1 = {g[k]:.4f} at ({xy[k, 0]:.3f}, {xy[k, 1]:.3f})"), "analytic")
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(SINGLE, 4.6), sharex=True)
    a1.plot(arc, Te, "k--", lw=1.2, label="exact")
    a2.plot(arc, Ve, "k--", lw=1.2, label="exact")
    xt = torch.tensor(xy, dtype=torch.float32)
    for label, prefix, _ in rows:
        run = f"{prefix}_d1234_s{seed}"
        m = model_for(run)
        if m is None:
            print(f"{name}: no model.pt for {run}, skipped")
            continue
        with torch.no_grad():
            Tp, Vp = m(xt)
        Tp, Vp = Tp.numpy()[:, 0], Vp.numpy()[:, 0]
        lab = short(label)
        a1.plot(arc, Tp, lw=1, label=lab)
        a2.plot(arc, Vp, lw=1, label=lab)
        lo, hi = 1 + 0.1 * (1 / 0.7 - 1), 1 + 0.9 * (1 / 0.7 - 1)
        inside = np.where((Vp > lo) & (Vp < hi))[0]
        width = (arc[inside].max() - arc[inside].min()) if (len(inside) and crosses) else float("nan")
        fact(f"{name} '{lab}' (seed {seed}): max |T_pred - T| on line; V transition width (10-90 % of the jump, arc length); V min/max on line",
             f"{np.abs(Tp - Te).max():.4f}; {width:.4f}; {Vp.min():.3f}/{Vp.max():.3f}", f"results/{run}/model.pt")
    for a in (a1, a2):
        if crosses:
            a.axvline(arc[k], color="0.6", lw=0.8, ls=":")
        a.grid(alpha=0.3)
    a1.set_ylabel("T"); a2.set_ylabel("V"); a2.set_xlabel(f"arc length from {p0} to {p1}")
    a2.set_ylim(0, 1.6)
    a2.legend(ncol=2, frameon=False, loc="lower right", fontsize=7)
    if note:
        a1.set_title(note, fontsize=8, loc="left")
    fig.tight_layout()
    savefig(fig, name)


# ---------------------------------------------------------------- 4. seed spread
def seed_spread(variants):
    labels, T, V = [], [], []
    for label, prefix, _ in variants:
        ok, _ = collect(prefix)
        if not ok:
            continue
        labels.append(short(label).replace("adaptive", "ad").replace("baseline", "base").replace("residual paper", "res. paper")
                      .replace(", noise 1 % (indicative)", " σ1%").replace(" (spec)", " spec").replace(" (paper)", ""))
        T.append([m["nrmse_T"] for _, m in ok]); V.append([m["nmae_V"] for _, m in ok])
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(SINGLE, 5.8), sharex=True)
    for ax, data, name, ref in ((a1, T, "nRMSE_T [%]", 0.49), (a2, V, "nMAE_V [%]", 3.56)):
        ax.boxplot(data, widths=0.5, showfliers=False, medianprops=dict(color="k"))
        for i, d in enumerate(data, 1):
            ax.scatter(np.full(len(d), i) + np.random.default_rng(0).uniform(-0.12, 0.12, len(d)), d, s=12, zorder=3, color="tab:blue")
        ax.axhline(ref, color="tab:red", lw=0.8, ls="--", label=f"paper {ref}")
        ax.set_ylabel(name); ax.set_yscale("log"); ax.grid(alpha=0.3, which="both")
    a1.legend(frameon=False, loc="upper left")
    a2.set_xticks(range(1, len(labels) + 1)); a2.set_xticklabels(labels, rotation=90, fontsize=8)
    fig.tight_layout()
    savefig(fig, "fig_seed_spread")
    fact("seed-spread figure", "boxes = quartiles over the finished seeds, dots = individual seeds, dashed = paper Table 1; labels: base = baseline, ad = adaptive (paper version unless 'spec'), σ1% = 1 % noise (indicative)", "results/<variant>_d1234_s<k>/metrics.json")


# ---------------------------------------------------------------- 5. loss histories
def loss_histories(seed=0, adaptive_prefix="adaptive_datafit"):
    runs = [("baseline", f"baseline_tanh_fixed_d1234_s{seed}"), ("adaptive (paper)", f"{adaptive_prefix}_d1234_s{seed}")]
    fig, axes = plt.subplots(3, 1, figsize=(SINGLE, 6.6), sharex=True)
    terms = ["data", "res_data", "res_colloc", "tv", "l2"]
    for (label, run), ax in zip(runs, axes[:2]):
        rows = list(csv.DictReader(open(os.path.join(ROOT, "results", run, "loss.csv"))))
        adam = [r for r in rows if r["phase"] == "adam"]
        last = int(adam[-1]["step"])
        x = np.array([int(r["step"]) if r["phase"] == "adam" else last + int(r["step"]) for r in rows])
        for t in terms:
            y = np.array([float(r[t]) for r in rows]); y[y <= 0] = np.nan
            ax.plot(x, y, lw=0.7, label=t)
        ax.axvline(last, color="k", ls="--", lw=0.7)
        ax.set_yscale("log"); ax.set_ylabel("loss term"); ax.set_title(label, loc="left")
        ax.grid(alpha=0.3)
        if label == "baseline":
            ax.legend(ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.32), handlelength=1.2, columnspacing=1.0)
        else:
            lam_rows = rows
            for k in ("lambda_data", "lambda_res_data"):
                axes[2].plot(x, [float(r[k]) for r in lam_rows], lw=0.8, label=k)
            fact(f"loss-history figure, {label} seed {seed}: final lambda_data / lambda_res_data",
                 f"{float(rows[-1]['lambda_data']):.4g} / {float(rows[-1]['lambda_res_data']):.4g}", f"results/{run}/loss.csv")
        m = load(run)
        fact(f"loss-history figure, {label} seed {seed}: final grid loss total / data / res_colloc",
             f"{m['final_loss_total_grid']:.3e} / {m['final_loss_data']:.3e} / {m['final_loss_res_colloc']:.3e}", f"results/{run}/metrics.json")
    axes[2].set_yscale("log"); axes[2].set_ylabel("adaptive weight"); axes[2].legend(frameon=False)
    axes[2].set_xlabel("Adam step | L-BFGS call (after dashed line)"); axes[2].grid(alpha=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    savefig(fig, "fig_loss_histories")


# ---------------------------------------------------------------- 6. facts
def write_facts():
    # keep the section appended by scripts/paper_style_figures.py, if present
    p = os.path.join(OUT, "facts.md")
    keep = ""
    if os.path.exists(p):
        s = open(p).read()
        if "<!-- PAPER_STYLE_FIGURES_START" in s:
            keep = "\n" + s[s.index("<!-- PAPER_STYLE_FIGURES_START"):]
    with open(p, "w") as f:
        f.write("# Facts behind the report assets\n\nEvery number in the tables and figures, with the run directory it comes from.\n"
                "Metrics: nRMSE_T = 100 x RMSE of T, nMAE_V = 100 x MAE of V over the 50x50 grid. Protocol: data seed 1234, init seeds 0 to 4.\n\n"
                "| quantity | value | source |\n|---|---|---|\n")
        for what, value, src in FACTS:
            f.write(f"| {what} | {value} | {src} |\n")
        f.write(keep)


def paper_facts():
    fact("paper Table 1, PINN, 0 % noise: nRMSE_T / nMAE_V", "0.49 / 3.56", "Sahli Costabal et al. 2020, Table 1, p. 6")
    fact("paper Table 1, PINN, 1 % noise: nRMSE_T (range) / nMAE_V (range)", "1.91 (1.09 to 3.09) / 10.24 (6.09 to 17.86); 100k Adam, no L-BFGS, 30 repeats", "Sahli Costabal et al. 2020, Table 1 and p. 6")
    fact("field figure width", "7.0 in (two-column); other figures 3.5 in (single column); fonts 9 pt (ticks 8 pt)", "scripts/make_report_assets.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--best_combo", default=None, help="label of the best combination (as in summarize.VARIANTS)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    variants = present_variants()
    paper_facts()
    tables(variants)
    rows = pick_rows(variants, a.best_combo)
    field_figure(rows)
    cross_section(rows, p0=(0.9, 0.1), p1=(0.1, 0.9), name="fig_cross_section",
                  note="requested line: does not cross the ridge")
    cross_section(rows, p0=(0.1, 0.1), p1=(0.7, 0.7), name="fig_cross_section_diagonal",
                  note="diagonal: crosses ridge at (0.412, 0.412)")
    seed_spread(variants)
    loss_histories()
    write_facts()
    print("wrote", sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
