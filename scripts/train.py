#!/usr/bin/env python3
"""Train the 2D Eikonal PINN: Adam with fresh LHS minibatches, then one L-BFGS call on the grid."""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from eikonal_pinn.adaptive import AdaptiveWeights  # noqa: E402
from eikonal_pinn.benchmark import add_noise, exact_T, exact_V, make_grid, ridge_distance, sample_data  # noqa: E402
from eikonal_pinn.rar import RARSet  # noqa: E402
from eikonal_pinn.io import RunWriter  # noqa: E402
from eikonal_pinn.losses import TERM_NAMES, WEIGHTED_TERMS, loss_terms  # noqa: E402
from eikonal_pinn.metrics import compute_metrics  # noqa: E402
from eikonal_pinn.model import EikonalPINN  # noqa: E402
from eikonal_pinn.plots import plot_fields, plot_loss  # noqa: E402
from eikonal_pinn.sampling import LHSampler  # noqa: E402

LOSS_COLUMNS = ["phase", "step", "total", *TERM_NAMES, *[f"lambda_{k}" for k in WEIGHTED_TERMS], "wall_s"]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--results_dir", default=os.path.join(os.path.dirname(__file__), "..", "results"))
    ap.add_argument("--seed", type=int, default=1234, help="torch init, noise, collocation sampler")
    ap.add_argument("--data_seed", type=int, default=None, help="LHS seed for the 50 data points (default: --seed)")
    ap.add_argument("--sigma", type=float, default=0.0, help="noise std as fraction of max T")
    ap.add_argument("--n_data", type=int, default=50)
    ap.add_argument("--batch", type=int, default=100, help="fresh LHS collocation points per Adam step")
    ap.add_argument("--adam_steps", type=int, default=50000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lbfgs_iters", type=int, default=10000)
    ap.add_argument("--lbfgs_max_eval", type=int, default=50000)
    ap.add_argument("--lbfgs_history", type=int, default=50)
    ap.add_argument("--alpha", type=float, default=1e-7)
    ap.add_argument("--alpha_L2", type=float, default=1e-9)
    ap.add_argument("--v_max", type=float, default=1.5)
    ap.add_argument("--residual", choices=["tf", "paper"], default="tf")
    ap.add_argument("--eps", type=float, default=0.0, help="eps inside the gradient-norm sqrt (0 = off)")
    ap.add_argument("--act", choices=["tanh", "relu"], default="tanh", help="activation of net_T")
    ap.add_argument("--weighting", choices=["fixed", "adaptive"], default="fixed")
    ap.add_argument("--anneal_every", type=int, default=10)
    ap.add_argument("--adaptive_terms", default="data,res_data",
                    help="comma list of terms that get an adaptive lambda (default: the data-fit terms of Wang et al. Alg. 1; data,res_data,tv also puts a lambda on the TV term, the failed variant in RESULTS.md)")
    ap.add_argument("--beta", type=float, default=0.9)
    ap.add_argument("--rar", type=int, choices=[0, 1], default=0)
    ap.add_argument("--rar_every", type=int, default=5000)
    ap.add_argument("--rar_m", type=int, default=50)
    ap.add_argument("--rar_candidates", type=int, default=10000)
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = torch default)")
    return ap.parse_args(argv)


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                       cwd=os.path.dirname(os.path.abspath(__file__))).strip()
    except Exception:  # noqa: BLE001
        return None


def main(argv=None):
    a = parse_args(argv)
    data_seed = a.seed if a.data_seed is None else a.data_seed
    dtype = torch.float32 if a.dtype == "float32" else torch.float64
    torch.set_default_dtype(dtype)
    if a.threads > 0:
        torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = torch.device(a.device)

    # ---- data ---------------------------------------------------------------
    xy_d_np = sample_data(a.n_data, seed=data_seed)
    T_d_exact = exact_T(xy_d_np)
    T_d_np = add_noise(T_d_exact, a.sigma, np.random.default_rng(a.seed))
    xy_g_np, X_m, Y_m = make_grid(50)
    T_g, V_g = exact_T(xy_g_np), exact_V(xy_g_np)

    xy_d = torch.tensor(xy_d_np, dtype=dtype, device=dev)
    T_d = torch.tensor(T_d_np, dtype=dtype, device=dev)
    xy_g = torch.tensor(xy_g_np, dtype=dtype, device=dev)
    colloc_sampler = LHSampler(seed=a.seed + 1)

    model = EikonalPINN(v_max=a.v_max, act_T=a.act, residual_form=a.residual, eps=a.eps).to(dev)
    params = list(model.parameters())
    lambdas = {k: 1.0 for k in WEIGHTED_TERMS}
    adaptive_terms = [t for t in a.adaptive_terms.split(",") if t]
    assert all(t in WEIGHTED_TERMS for t in adaptive_terms), adaptive_terms
    adapt = AdaptiveWeights(adaptive_terms, params, beta=a.beta) if a.weighting == "adaptive" else None
    rar = RARSet(seed=a.seed + 2, n_candidates=a.rar_candidates, m=a.rar_m) if a.rar else None
    adam_log = []  # (step, total) for the stability statistics

    writer = RunWriter(a.results_dir, a.run_name, LOSS_COLUMNS)
    cfg = vars(a).copy()
    cfg["results_dir"] = os.path.relpath(a.results_dir)
    cfg.update(dict(data_seed_effective=data_seed, colloc_seed=a.seed + 1, rar_seed=a.seed + 2, git=git_hash(),
                    torch=torch.__version__, numpy=np.__version__, python=sys.version.split()[0],
                    layers_T=model.net_T.layers, layers_V=model.net_V.layers, n_params=sum(p.numel() for p in params)))
    writer.write_config(cfg)

    def log_row(phase, step, total, terms, t0):
        row = {"phase": phase, "step": step, "total": float(total), "wall_s": round(time.time() - t0, 3)}
        row.update({k: float(v.detach()) for k, v in terms.items()})
        row.update({f"lambda_{k}": lambdas[k] for k in WEIGHTED_TERMS})
        writer.log(row)

    status = {"nan_at_step": None}
    t_start = time.time()

    # ---- phase 1: Adam -------------------------------------------------------
    opt = torch.optim.Adam(params, lr=a.lr)
    t0 = time.time()
    for step in range(a.adam_steps):
        if rar is not None and step > 0 and step % a.rar_every == 0:
            pts, res = rar.refine(model, step, dtype=dtype, device=dev)
            print(f"rar step {step}: added {len(pts)} points (|f_T| {res.min():.2e} to {res.max():.2e}), total {len(rar)}", flush=True)
        xy_c = colloc_sampler.draw_torch(a.batch, dtype=dtype, device=dev)
        if rar is not None and len(rar) > 0:
            xy_c = torch.cat([xy_c, rar.tensor(dtype=dtype, device=dev)], dim=0)
        opt.zero_grad(set_to_none=True)
        total, terms = loss_terms(model, xy_d, T_d, xy_c, a.alpha, a.alpha_L2, lambdas)
        if not torch.isfinite(total):
            status["nan_at_step"] = step
            print(f"non-finite loss at Adam step {step}; stopping", file=sys.stderr)
            break
        if adapt is not None and step % a.anneal_every == 0:
            lambdas.update(adapt.update(terms, ref="res_colloc"))
            total = (lambdas["data"] * terms["data"] + lambdas["res_data"] * terms["res_data"]
                     + terms["res_colloc"] + lambdas["tv"] * terms["tv"] + terms["l2"])
        total.backward()
        opt.step()
        adam_log.append((step, float(total.detach())))
        if step % a.log_every == 0 or step == a.adam_steps - 1:
            log_row("adam", step, total.item(), terms, t0)
            if step % (10 * a.log_every) == 0:
                print(f"adam {step:6d}  total {total.item():.3e}  data {terms['data'].item():.2e}  "
                      f"res_c {terms['res_colloc'].item():.2e}  {time.time() - t0:.0f}s", flush=True)
    t_adam = time.time() - t0

    # ---- phase 2: L-BFGS on the fixed grid -----------------------------------
    n_closure = [0]
    t0 = time.time()
    lbfgs_info = {}
    if a.lbfgs_iters > 0 and status["nan_at_step"] is None:
        lbfgs = torch.optim.LBFGS(params, lr=1.0, max_iter=a.lbfgs_iters, max_eval=a.lbfgs_max_eval,
                                  history_size=a.lbfgs_history, tolerance_grad=1e-9, tolerance_change=1e-12,
                                  line_search_fn="strong_wolfe")

        xy_lbfgs = torch.cat([xy_g, rar.tensor(dtype=dtype, device=dev)], dim=0) if (rar is not None and len(rar) > 0) else xy_g

        def closure():
            lbfgs.zero_grad(set_to_none=True)
            total, terms = loss_terms(model, xy_d, T_d, xy_lbfgs, a.alpha, a.alpha_L2, lambdas)
            total.backward()
            n_closure[0] += 1
            log_row("lbfgs", n_closure[0], total.item(), terms, t0)
            if n_closure[0] % 1000 == 0:
                print(f"lbfgs eval {n_closure[0]:6d}  total {total.item():.3e}  {time.time() - t0:.0f}s", flush=True)
            return total

        lbfgs.step(closure)
        st = lbfgs.state[params[0]]
        lbfgs_info = {"lbfgs_n_iter": int(st.get("n_iter", 0)), "lbfgs_func_evals": int(st.get("func_evals", 0))}
    t_lbfgs = time.time() - t0
    t_total = time.time() - t_start

    # ---- evaluation ---------------------------------------------------------
    model.eval()
    T_pred, V_pred = model.predict(xy_g)
    T_pred, V_pred = T_pred.cpu().numpy(), V_pred.cpu().numpy()
    metrics = compute_metrics(T_pred, V_pred, T_g, V_g)
    with torch.enable_grad():
        final_total, final_terms = loss_terms(model, xy_d, T_d, xy_g, a.alpha, a.alpha_L2, lambdas)
    metrics.update({
        "wall_time_s": round(t_total, 2), "wall_adam_s": round(t_adam, 2), "wall_lbfgs_s": round(t_lbfgs, 2),
        "final_loss_total_grid": float(final_total),
        **{f"final_loss_{k}": float(v) for k, v in final_terms.items()},
        "adam_steps_done": min(a.adam_steps, status["nan_at_step"] or a.adam_steps),
        "lbfgs_closure_calls": n_closure[0], **lbfgs_info, **status,
        "nan_in_prediction": bool(not (np.isfinite(T_pred).all() and np.isfinite(V_pred).all())),
    })
    # stability: loss spikes during Adam (total > 10x the median of the previous 100 steps)
    tot = np.array([t for _, t in adam_log])
    n_spikes = 0
    for i in range(100, len(tot)):
        if tot[i] > 10.0 * np.median(tot[i - 100:i]):
            n_spikes += 1
    metrics["adam_loss_spikes"] = int(n_spikes)
    metrics["adam_loss_max_after_1000"] = float(tot[1000:].max()) if len(tot) > 1000 else None
    # dead hidden units of net_T on the grid (ReLU: zero for every grid point; tanh: |h| < 1e-6)
    with torch.no_grad():
        h = model._scale(xy_g)
        dead, n_units = 0, 0
        for lin in model.net_T.linears[:-1]:
            h = model.net_T.act(lin(h))
            dead += int((h.abs().max(dim=0).values < 1e-6).sum())
            n_units += h.shape[1]
    metrics["dead_units_T"] = dead
    metrics["dead_units_T_frac"] = dead / n_units
    # adaptive weights
    if adapt is not None:
        metrics.update({f"lambda_final_{k}": v for k, v in lambdas.items()})
        metrics["adaptive_updates"] = adapt.n_updates
        metrics["adaptive_skipped"] = adapt.n_skipped
        metrics["adaptive_terms"] = adaptive_terms
    # RAR statistics: do the added points sit on the ridge?
    X_rar = rar.X if rar is not None else np.zeros((0, 2))
    if len(X_rar):
        d = ridge_distance(X_rar)[:, 0]
        d_grid = ridge_distance(xy_g_np)[:, 0]
        metrics.update({"n_rar": int(len(X_rar)),
                        "rar_frac_within_0.05": float((d < 0.05).mean()), "rar_frac_within_0.10": float((d < 0.10).mean()),
                        "grid_frac_within_0.05": float((d_grid < 0.05).mean()), "grid_frac_within_0.10": float((d_grid < 0.10).mean()),
                        "rar_median_ridge_dist": float(np.median(d)),
                        "rar_frac_near_sources": float(((np.linalg.norm(X_rar, axis=1) < 0.1) | (np.linalg.norm(X_rar - 1, axis=1) < 0.1)).mean())})
    writer.write_metrics(metrics)
    writer.write_fields(xy_grid=xy_g_np, X_m=X_m, Y_m=Y_m, T_pred=T_pred, V_pred=V_pred, T_exact=T_g, V_exact=V_g,
                        xy_data=xy_d_np, T_data=T_d_np, T_data_exact=T_d_exact, X_rar=X_rar,
                        lambdas_final=np.array([lambdas[k] for k in WEIGHTED_TERMS]))
    torch.save(model.state_dict(), writer.path("model.pt"))
    writer.close()
    plot_fields(writer.path("fig_fields.png"), X_m, Y_m, T_pred, T_g, V_pred, V_g, xy_d_np,
                X_rar=X_rar if len(X_rar) else None, title=a.run_name)
    plot_loss(writer.path("fig_loss.png"), writer.loss_path, title=a.run_name)

    print(json.dumps(metrics, indent=2))
    print(f"| {a.run_name} | {a.sigma:g} | {a.adam_steps} | {a.lbfgs_iters} | {metrics['nrmse_T']:.3f} | "
          f"{metrics['nmae_V']:.3f} | {metrics['maxerr_T']:.4f} | {metrics['maxerr_V']:.3f} | {t_total:.0f} s | |")
    return metrics


if __name__ == "__main__":
    main()
