#!/usr/bin/env python3
"""Paper-style figures (layout of Figs. 2 and 3 in Sahli Costabal et al. 2020) from existing results only.

No training, no reruns. Fields come from results/<run>/fields.npz; the optional ||grad T|| curve in the
section figures is an autograd evaluation of saved weights (results/<run>/model.pt), never a retrain.

Outputs (PDF + PNG, 300 dpi) in --out (default report_assets/):
  fields_main       2 x 3: exact | baseline | adaptive weights          (5.5 in, 9 pt)
  fields_appendix   2 x 6: exact | baseline | adaptive | RAR | ReLU | adaptive+RAR+ReLU (7 in, 8 pt)
  section_main      1 x 3: exact | baseline + exact | adaptive + exact, along x = y (5.5 in, 9 pt)
  section_appendix  1 x 5: one panel per variant, along x = y          (7 in, 8 pt)
  figure_stubs.tex  LaTeX stubs with \\label{fig:...}, captions TODO
and a marked section (one per seed) in report_assets/facts.md listing every run directory used.
Runs are results/<prefix>_d1234_s<seed>; the seed-1234 baseline is results/baseline_tanh_fixed (see ALIASES).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)
from eikonal_pinn.benchmark import exact_T, exact_V  # noqa: E402
from eikonal_pinn.model import EikonalPINN  # noqa: E402
from summarize import collect, load  # noqa: E402

FACTS_DIR = os.path.join(ROOT, "report_assets")
OUT = FACTS_DIR  # figure directory, overridden by --out
# runs that do not follow the <prefix>_d1234_s<seed> naming
ALIASES = {("baseline_tanh_fixed", 1234): "baseline_tanh_fixed", ("baseline_tanh_fixed", 1): "baseline_tanh_fixed_seed1",
           ("baseline_tanh_fixed", 2): "baseline_tanh_fixed_seed2"}

# (title used in the figures, run prefix); data seed 1234 throughout
VARIANTS = [
    ("baseline", "baseline_tanh_fixed"),
    ("adaptive weights", "adaptive_datafit"),
    ("RAR", "rar"),
    ("ReLU", "relu"),
    ("adaptive+RAR+ReLU", "adaptive_rar_relu"),
]
T_LEVELS = np.linspace(0.0, 0.75, 16)
SQRT2 = float(np.sqrt(2.0))


# ---------------------------------------------------------------- loading
def run_name(prefix, seed):
    name = f"{prefix}_d1234_s{seed}"
    if not os.path.isdir(os.path.join(ROOT, "results", name)) and (prefix, seed) in ALIASES:
        return ALIASES[(prefix, seed)]
    return name


def have(prefix, seed):
    return os.path.exists(os.path.join(ROOT, "results", run_name(prefix, seed), "fields.npz"))


def load_fields(run):
    p = os.path.join(ROOT, "results", run, "fields.npz")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    d = np.load(p)
    n = int(round(np.sqrt(d["xy_grid"].shape[0])))
    f = {k: d[k] for k in d.files}
    f["n"] = n
    f["T2"] = d["T_pred"].reshape(n, n)
    f["V2"] = d["V_pred"].reshape(n, n)
    f["Te2"] = d["T_exact"].reshape(n, n)
    f["Ve2"] = d["V_exact"].reshape(n, n)
    assert np.allclose(d["xy_grid"][:, 0].reshape(n, n), d["X_m"]) and np.allclose(d["xy_grid"][:, 1].reshape(n, n), d["Y_m"])
    return f


def weights_for(run, allow_rerun):
    """Path to saved weights for `run`, or (None, reason). A `<run>_rerun` is accepted only if its metrics
    are identical to the original run (same single-thread arithmetic, hence the same weights)."""
    p = os.path.join(ROOT, "results", run, "model.pt")
    if os.path.exists(p):
        return p, f"results/{run}/model.pt"
    pr = os.path.join(ROOT, "results", run + "_rerun", "model.pt")
    if os.path.exists(pr):
        a, b = load(run), load(run + "_rerun")
        same = a and b and abs(a["nrmse_T"] - b["nrmse_T"]) < 1e-9 and abs(a["nmae_V"] - b["nmae_V"]) < 1e-9
        if same:
            return pr, f"results/{run}_rerun/model.pt (metrics identical to results/{run})"
        if allow_rerun:
            return pr, (f"results/{run}_rerun/model.pt (NOT identical to results/{run}: "
                        f"{a['nrmse_T']:.4f} / {a['nmae_V']:.4f} vs {b['nrmse_T']:.4f} / {b['nmae_V']:.4f}; used because --allow_rerun_weights)")
        return None, (f"no model.pt in results/{run}; results/{run}_rerun/model.pt exists but its metrics differ "
                      f"({a['nrmse_T']:.4f} / {a['nmae_V']:.4f} vs {b['nrmse_T']:.4f} / {b['nmae_V']:.4f}), so it is not the same network; skipped")
    return None, f"no model.pt in results/{run}; skipped"


def grad_norm_on_line(run, wpath, s):
    """||grad T|| along x = y = s / sqrt(2), from saved weights, by autograd (evaluation only)."""
    cfg = json.load(open(os.path.join(ROOT, "results", run, "config.json")))
    dtype = torch.float64 if cfg.get("dtype") == "float64" else torch.float32
    m = EikonalPINN(v_max=cfg["v_max"], act_T=cfg["act"], residual_form=cfg["residual"], eps=cfg["eps"]).to(dtype)
    m.load_state_dict(torch.load(wpath, map_location="cpu"))
    m.eval()
    x = s / SQRT2
    xy = torch.tensor(np.stack([x, x], 1), dtype=dtype, requires_grad=True)
    T = m.forward_T(xy)
    g = torch.autograd.grad(T, xy, grad_outputs=torch.ones_like(T))[0]
    return g.norm(dim=1).detach().numpy()


# ---------------------------------------------------------------- figures A, B
def fields_figure(cols, seed, v_lim, width, font, name, facts, t_extend="neither"):
    """cols: list of (title, prefix); exact column is prepended."""
    v_levels = np.linspace(v_lim[0], v_lim[1], 16)
    data = [(t, run_name(p, seed), load_fields(run_name(p, seed))) for t, p in cols]
    ref = data[0][2]
    X, Y = ref["X_m"], ref["Y_m"]
    for _, run, f in data:
        assert np.allclose(f["xy_data"], ref["xy_data"]), f"{run}: data points differ"
    xy_data = ref["xy_data"]
    ncol = len(data) + 1
    panel = (width - 0.55) / (ncol + 0.25)
    rc = {"font.size": font, "axes.titlesize": font, "axes.labelsize": font, "xtick.labelsize": font,
          "ytick.labelsize": font, "pdf.fonttype": 42}
    with plt.rc_context(rc):
        fig, axes = plt.subplots(2, ncol, figsize=(width, 2 * panel + 0.45), layout="constrained")
        titles = ["exact"] + [t for t, _, _ in data]
        for j in range(ncol):
            if j == 0:
                T2, V2 = ref["Te2"], ref["Ve2"]
            else:
                T2, V2 = data[j - 1][2]["T2"], data[j - 1][2]["V2"]
            csT = axes[0, j].contourf(X, Y, T2, levels=T_LEVELS, cmap="jet_r", extend=t_extend)
            csV = axes[1, j].contourf(X, Y, V2, levels=v_levels, cmap="jet", extend="both")
            for cs in (csT, csV):  # fill the sub-pixel gaps between very thin bands on the steep V front
                cs.set_edgecolor("face"); cs.set_linewidth(0.3)
            axes[0, j].set_title(titles[j])
            for ax in axes[:, j]:
                if j > 0:
                    ax.scatter(xy_data[:, 0], xy_data[:, 1], s=18, facecolors="none", edgecolors="k", linewidths=0.6)
                ax.set_aspect("equal")
                ax.set_xlim(0, 1); ax.set_ylim(0, 1)
                ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
                ax.tick_params(labelbottom=False, labelleft=False, length=2)
            if j > 0:
                tmin, tmax, vmin, vmax = T2.min(), T2.max(), V2.min(), V2.max()
                facts.append((f"{name} column '{titles[j]}' (seed {seed}): T_pred min / max; V_pred min / max on the grid",
                              f"{tmin:.4f} / {tmax:.4f}; {vmin:.4f} / {vmax:.4f}", f"results/{data[j - 1][1]}/fields.npz"))
                if tmin < T_LEVELS[0] or tmax > T_LEVELS[-1]:
                    facts.append((f"{name} column '{titles[j]}': T_pred outside the contour levels [0, 0.75] (unfilled unless --t_extend)",
                                  f"min {tmin:.4f}, max {tmax:.4f}", f"results/{data[j - 1][1]}/fields.npz"))
        ax = axes[1, 0]
        ax.tick_params(labelbottom=True, labelleft=True)
        ax.set_xlabel("X [mm]"); ax.set_ylabel("Y [mm]")
        cbT = fig.colorbar(csT, ax=list(axes[0, :]), pad=0.02, shrink=0.9)
        cbT.set_label("activation\ntime [ms]"); cbT.set_ticks([0, 0.75])
        cbV = fig.colorbar(csV, ax=list(axes[1, :]), pad=0.02, shrink=0.9)
        cbV.set_label("conduction\nvelocity [mm/ms]"); cbV.set_ticks(list(v_lim))
        fig.savefig(os.path.join(OUT, name + ".png"), dpi=300)
        fig.savefig(os.path.join(OUT, name + ".pdf"))
        plt.close(fig)
    facts.append((f"{name}: data points overlaid", f"{xy_data.shape[0]} points, identical in every run (data seed 1234)",
                  f"results/{data[0][1]}/fields.npz (xy_data)"))
    return [run for _, run, _ in data]


# ---------------------------------------------------------------- figures C, D
def line_data(f):
    """Diagonal nodes of the meshgrid (exactly on x = y): s = sqrt(2) x, T_pred[i, i], V_pred[i, i]."""
    n = f["n"]
    x = np.diag(f["X_m"])
    assert np.allclose(x, np.diag(f["Y_m"]))
    return SQRT2 * x, np.diag(f["T2"]), np.diag(f["V2"])


def exact_line(npts=400):
    s = np.linspace(0, SQRT2, npts)
    xy = np.stack([s / SQRT2, s / SQRT2], 1)
    return s, exact_T(xy).ravel(), exact_V(xy).ravel()


def draw_section(ax, curves, s_ex, T_ex, V_ex, t_lim, grad_curves=(), styles=None, legend=False, show_exact=True, legend_loc="upper right"):
    """curves: list of (label, s, T, V); plotted on twin axes. styles: per-curve linestyle."""
    styles = styles or ["-"] * len(curves)
    axr = ax.twinx()
    handles = []
    if show_exact:
        ax.plot(s_ex, T_ex, "--", color="tab:blue", lw=1.0)
        axr.plot(s_ex, 1.0 / V_ex, "--", color="tab:red", lw=1.0)
        handles.append(Line2D([], [], color="k", ls="--", lw=1.0, label="exact"))
    for (label, s, T, V), ls in zip(curves, styles):
        ax.plot(s, T, ls, color="tab:blue", lw=1.2)
        axr.plot(s, 1.0 / V, ls, color="tab:red", lw=1.2)
        handles.append(Line2D([], [], color="k", ls=ls, lw=1.2, label=label))
    for s, g in grad_curves:
        axr.plot(s, g, "-.", color="red", lw=0.7)
    ax.set_xlim(0, SQRT2); ax.set_xticks([0, SQRT2]); ax.set_xticklabels(["0", r"$\sqrt{2}$"])
    ax.set_ylim(0, t_lim); axr.set_ylim(0, 1.2)
    ax.tick_params(axis="y", colors="tab:blue", length=2); axr.tick_params(axis="y", colors="tab:red", length=2)
    ax.tick_params(axis="x", length=2)
    if legend:
        ax.legend(handles=handles, loc=legend_loc, frameon=False, fontsize="small", handlelength=1.8, borderaxespad=0.3)
    return axr


def section_main_figure(seed, width, font, name, facts):
    """1 x 3: exact only | baseline + exact | adaptive weights + exact. No autograd curves."""
    s_ex, T_ex, V_ex = exact_line()
    t_lim = max(0.6, float(np.ceil(T_ex.max() * 10) / 10))
    runs, curves = [], []
    for title, prefix in VARIANTS[:2]:
        run = run_name(prefix, seed)
        s, T, V = line_data(load_fields(run))
        curves.append((title, s, T, V)); runs.append(run)
        facts.append((f"{name} '{title}' (seed {seed}): max |T_pred - T_exact| and max |1/V_pred - 1/V_exact| on the 50 diagonal nodes",
                      f"{np.abs(T - exact_T(np.stack([s / SQRT2] * 2, 1)).ravel()).max():.4f} / "
                      f"{np.abs(1 / V - 1 / exact_V(np.stack([s / SQRT2] * 2, 1)).ravel()).max():.4f}", f"results/{run}/fields.npz"))
    rc = {"font.size": font, "axes.titlesize": font, "axes.labelsize": font, "xtick.labelsize": font,
          "ytick.labelsize": font, "legend.fontsize": font - 1, "pdf.fonttype": 42}
    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, 3, figsize=(width, 0.42 * width + 0.3), layout="constrained")
        rights = [draw_section(axes[0], [], s_ex, T_ex, V_ex, t_lim),
                  draw_section(axes[1], curves[:1], s_ex, T_ex, V_ex, t_lim, legend=True, legend_loc="lower center"),
                  draw_section(axes[2], curves[1:], s_ex, T_ex, V_ex, t_lim, legend=True, legend_loc="lower center")]
        for ax, t in zip(axes, ["exact", "baseline", "adaptive weights"]):
            ax.set_title(t); ax.set_xlabel("s")
        for j, ax in enumerate(axes):
            ax.tick_params(axis="y", labelleft=(j == 0))
            rights[j].tick_params(axis="y", labelright=(j == 2))
        axes[0].set_ylabel("activation time [ms]", color="tab:blue")
        rights[-1].set_ylabel(r"$\|\nabla T\| = 1/V$ [s/m]", color="tab:red")
        fig.savefig(os.path.join(OUT, name + ".png"), dpi=300)
        fig.savefig(os.path.join(OUT, name + ".pdf"))
        plt.close(fig)
    facts.append((f"{name}: panels", "exact only | baseline (solid) + exact (dashed) | adaptive weights (solid) + exact (dashed); "
                  "x = y, s = sqrt(2) x; predictions at the 50 diagonal nodes; no ||grad T|| autograd curves in this figure", "analytic"))
    facts.append((f"{name}: left axis limit", f"0 to {t_lim} (max exact T on the line {T_ex.max():.4f}, rounded up to 0.6)", "analytic"))
    return runs


def section_figure(panels, seed, width, font, name, facts, allow_rerun, overlay=False):
    """panels: list of (title, prefix). If overlay, a third panel shows all panels' curves together."""
    s_ex, T_ex, V_ex = exact_line()
    t_lim = max(0.6, float(np.ceil(T_ex.max() * 10) / 10))
    runs, curves, grads, lacking = [], [], [], []
    for title, prefix in panels:
        run = run_name(prefix, seed)
        f = load_fields(run)
        s, T, V = line_data(f)
        curves.append((title, s, T, V)); runs.append(run)
        wpath, why = weights_for(run, allow_rerun)
        if wpath is None:
            grads.append(None); lacking.append((run, why))
            facts.append((f"{name} panel '{title}': ||grad T|| autograd curve", "skipped: " + why, f"results/{run}/"))
        else:
            g = grad_norm_on_line(run, wpath, s_ex)
            grads.append((s_ex, g))
            facts.append((f"{name} panel '{title}': ||grad T|| from saved weights along x = y, min / max on 400 points",
                          f"{g.min():.4f} / {g.max():.4f}", why))
        facts.append((f"{name} panel '{title}' (seed {seed}): max |T_pred - T_exact| and max |1/V_pred - 1/V_exact| on the 50 diagonal nodes",
                      f"{np.abs(T - exact_T(np.stack([s / SQRT2] * 2, 1)).ravel()).max():.4f} / "
                      f"{np.abs(1 / V - 1 / exact_V(np.stack([s / SQRT2] * 2, 1)).ravel()).max():.4f}", f"results/{run}/fields.npz"))
    ncol = len(panels) + (1 if overlay else 0)
    rc = {"font.size": font, "axes.titlesize": font, "axes.labelsize": font, "xtick.labelsize": font,
          "ytick.labelsize": font, "legend.fontsize": font - 1, "pdf.fonttype": 42}
    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, ncol, figsize=(width, 0.42 * width / ncol * 3 + 0.3 if ncol == 3 else 1.9), layout="constrained")
        axes = np.atleast_1d(axes)
        rights = []
        for j, (title, _) in enumerate(panels):
            g = [grads[j]] if grads[j] is not None else []
            rights.append(draw_section(axes[j], [curves[j]], s_ex, T_ex, V_ex, t_lim, grad_curves=g))
            axes[j].set_title(title)
        if overlay:
            # adaptive solid, baseline dotted (panels are ordered baseline, adaptive)
            styles = [":" if t == "baseline" else "-" for t, _ in panels]
            g = [gr for gr in grads if gr is not None]
            rights.append(draw_section(axes[-1], curves, s_ex, T_ex, V_ex, t_lim, grad_curves=g, styles=styles, legend=True))
            axes[-1].set_title("both")
        for j, ax in enumerate(axes):
            ax.tick_params(axis="y", labelleft=(j == 0))
            rights[j].tick_params(axis="y", labelright=(j == ncol - 1))
        axes[0].set_ylabel("activation time [ms]", color="tab:blue")
        rights[-1].set_ylabel(r"$\|\nabla T\| = 1/V$ [s/m]", color="tab:red")
        for ax in axes:
            ax.set_xlabel("s")
        fig.savefig(os.path.join(OUT, name + ".png"), dpi=300)
        fig.savefig(os.path.join(OUT, name + ".pdf"))
        plt.close(fig)
    facts.append((f"{name}: line", "x = y, s = sqrt(2) x in [0, sqrt(2)]; predictions at the 50 diagonal meshgrid nodes (no interpolation); "
                  "exact curves and ||grad T|| on 400 points; dashed = exact, solid = predicted, red dash-dot = ||grad T|| by autograd", "analytic"))
    facts.append((f"{name}: left axis limit", f"0 to {t_lim} (max exact T on the line {T_ex.max():.4f}, rounded up to 0.6)", "analytic"))
    return runs, lacking


# ---------------------------------------------------------------- report helpers
def seed_check(seed, facts):
    rows = []
    for title, prefix in VARIANTS:
        if not have(prefix, seed):
            continue
        ok, _ = collect(prefix)
        m0 = load(run_name(prefix, seed))
        vals = np.array([m["nmae_V"] for _, m in ok])
        mean = vals.mean()
        ratio = m0["nmae_V"] / mean
        flag = ratio > 2 or ratio < 0.5
        rows.append((title, m0["nmae_V"], mean, ratio, flag))
        facts.append((f"seed {seed} representativeness, {title}: nMAE_V seed {seed} / 5-seed mean (seeds {[s for s, _ in ok]})",
                      f"{m0['nmae_V']:.4f} / {mean:.4f} = {ratio:.2f}" + (" (more than 2x off)" if flag else ""),
                      ", ".join(f"results/{run_name(prefix, s)}/metrics.json" for s, _ in ok)))
    return rows


def v_minima(seed, facts):
    out = []
    for title, prefix in VARIANTS:
        if not have(prefix, seed):
            continue
        run = run_name(prefix, seed)
        f = load_fields(run)
        k = int(np.argmin(f["V_pred"]))
        out.append((title, run, float(f["V_pred"].min()), tuple(f["xy_grid"][k]), float(f["V_pred"].max())))
        facts.append((f"{title} (seed {seed}): min V_pred on the grid (location); max V_pred", f"{f['V_pred'].min():.4f} ({f['xy_grid'][k][0]:.3f}, {f['xy_grid'][k][1]:.3f}); {f['V_pred'].max():.4f}",
                      f"results/{run}/fields.npz"))
    return out


def write_facts(facts, args):
    """Replace (or append) the section for this seed in report_assets/facts.md; other seeds' sections are kept."""
    start, end = f"<!-- PAPER_STYLE_FIGURES_START seed={args.seed} -->", f"<!-- PAPER_STYLE_FIGURES_END seed={args.seed} -->"
    p = os.path.join(FACTS_DIR, "facts.md")
    s = open(p).read() if os.path.exists(p) else "# Facts behind the report assets\n"
    if start in s and end in s:
        s = s[:s.index(start)].rstrip("\n") + "\n" + s[s.index(end) + len(end):].lstrip("\n")
    block = [start, "", f"## Paper-style figures (scripts/paper_style_figures.py, seed {args.seed})", "",
             f"Flags: --seed {args.seed} --v_lim {args.v_lim[0]} {args.v_lim[1]} --t_extend {args.t_extend} --out {os.path.relpath(OUT, ROOT)}" + (" --allow_rerun_weights" if args.allow_rerun_weights else "") +
             ". Fields from results/<run>/fields.npz; ||grad T|| curves from saved weights by autograd (evaluation only, no training).", "",
             "| quantity | value | source |", "|---|---|---|"]
    block += [f"| {w} | {v} | {src} |" for w, v, src in facts]
    block += ["", end, ""]
    open(p, "w").write(s.rstrip("\n") + "\n\n" + "\n".join(block))


def write_stubs():
    tex = r"""% LaTeX stubs for the paper-style figures (report_assets/). Captions TODO.
\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{report_assets/fields_main.pdf}
  \caption{TODO}
  \label{fig:fields_main}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{report_assets/section_main.pdf}
  \caption{TODO}
  \label{fig:section_main}
\end{figure}

\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{report_assets/fields_appendix.pdf}
  \caption{TODO}
  \label{fig:fields_appendix}
\end{figure*}

\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{report_assets/section_appendix.pdf}
  \caption{TODO}
  \label{fig:section_appendix}
\end{figure*}
"""
    open(os.path.join(FACTS_DIR, "figure_stubs.tex"), "w").write(tex)
    return tex


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0, help="init seed of the runs shown (data seed is 1234)")
    ap.add_argument("--v_lim", type=float, nargs=2, default=(0.9, 1.5), help="V colour limits and colourbar ticks")
    ap.add_argument("--out", default="report_assets", help="directory for the figures (relative to the repo root); facts.md always goes to report_assets/")
    ap.add_argument("--t_extend", default="neither", choices=["neither", "min", "max", "both"],
                    help="contourf 'extend' for the T row (default neither, as in the paper; T_pred above 0.75 is then left white)")
    ap.add_argument("--allow_rerun_weights", action="store_true",
                    help="use results/<run>_rerun/model.pt for the ||grad T|| curve even if the rerun's metrics differ from <run>")
    args = ap.parse_args()
    global OUT
    OUT = os.path.join(ROOT, args.out)
    os.makedirs(OUT, exist_ok=True); os.makedirs(FACTS_DIR, exist_ok=True)
    facts = []
    seed = args.seed
    main_cols = VARIANTS[:2]
    missing = [run_name(p, seed) for _, p in VARIANTS if not have(p, seed)]
    if any(run_name(p, seed) in missing for _, p in main_cols):
        sys.exit(f"missing results for the main figures at seed {seed}: {missing}")
    runsA = fields_figure(main_cols, seed, args.v_lim, 5.5, 9, "fields_main", facts, args.t_extend)
    runsC, lackC = section_main_figure(seed, 5.5, 9, "section_main", facts), []
    if missing:
        print(f"appendix figures skipped at seed {seed}: no results for {missing}")
        runsB, runsD, lackD = [], [], []
    else:
        runsB = fields_figure(VARIANTS, seed, args.v_lim, 7.0, 8, "fields_appendix", facts, args.t_extend)
        runsD, lackD = section_figure(VARIANTS, seed, 7.0, 8, "section_appendix", facts, args.allow_rerun_weights)
    vm = v_minima(seed, facts)
    sc = seed_check(seed, facts)
    write_facts(facts, args)
    write_stubs()

    used = sorted(set(runsA + runsB + runsC + runsD))
    print("run directories used (fields.npz):")
    for r in used:
        print(f"  results/{r}")
    print("\nweights used for ||grad T|| curves:")
    seen = set()
    for w, v, src in facts:
        if "||grad T||" in w and src not in seen and "skipped" not in v:
            seen.add(src); print(f"  {src}")
    lacking = {r: why for r, why in lackC + lackD}
    print("\nruns without usable weights (||grad T|| curve skipped):")
    for r, why in lacking.items():
        print(f"  results/{r}: {why}")
    if not lacking:
        print("  none")
    print(f"\nmin / max of V_pred on the grid (seed {seed}):")
    for title, run, vmin, loc, vmax in vm:
        print(f"  {title:20s} {vmin:.4f} at ({loc[0]:.3f}, {loc[1]:.3f})   max {vmax:.4f}   results/{run}")
    print(f"\nseed {seed} vs 5-seed mean, nMAE_V (ratio; flagged if > 2x or < 0.5x):")
    for title, v0, mean, ratio, flag in sc:
        print(f"  {title:20s} {v0:.3f} vs {mean:.3f}  ratio {ratio:.2f}" + ("  <-- more than 2x off" if flag else ""))


if __name__ == "__main__":
    main()
