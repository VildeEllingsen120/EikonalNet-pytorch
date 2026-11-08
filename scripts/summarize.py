#!/usr/bin/env python3
"""Build the shared variant table (mean / min / max over seeds) and insert it into RESULTS.md."""
import argparse
import json
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
START, END = "<!-- VARIANT_TABLE_START -->", "<!-- VARIANT_TABLE_END -->"


def load(run):
    p = os.path.join(ROOT, "results", run, "metrics.json")
    return json.load(open(p)) if os.path.exists(p) else None


def collect(prefix, data_seed=1234, seeds=(0, 1, 2, 3, 4)):
    ok, failed = [], []
    for s in seeds:
        run = f"{prefix}_d{data_seed}_s{s}"
        m = load(run)
        if m is None:
            failed.append((s, "no metrics.json (run missing or crashed)"))
        elif m.get("nan_at_step") is not None:
            failed.append((s, f"non-finite loss at Adam step {m['nan_at_step']}"))
        elif m.get("nan_in_prediction"):
            failed.append((s, "NaN in prediction"))
        else:
            ok.append((s, m))
    return ok, failed


def fmt(vals, f="%.2f"):
    if not vals:
        return "-"
    v = np.array(vals)
    return f"{f % v.mean()} ({f % v.min()} to {f % v.max()})"


def table(variants, data_seed=1234, seeds=(0, 1, 2, 3, 4), extra_cols=()):
    lines = ["| variant | flags | nRMSE_T [%] mean (min to max) | nMAE_V [%] mean (min to max) | max err T | max err V | wall [s] | failed seeds |",
             "|---|---|---|---|---|---|---|---|"]
    rows = {}
    for label, prefix, flags in variants:
        ok, failed = collect(prefix, data_seed, seeds)
        g = lambda k: [m[k] for _, m in ok]
        fs = "; ".join(f"seed {s}: {why}" for s, why in failed) or "none"
        lines.append(f"| {label} | `{flags}` | {fmt(g('nrmse_T'))} | {fmt(g('nmae_V'))} | {fmt(g('maxerr_T'), '%.3f')} | "
                     f"{fmt(g('maxerr_V'), '%.2f')} | {fmt(g('wall_time_s'), '%.0f')} | {fs} |")
        rows[label] = (ok, failed)
    return "\n".join(lines), rows


VARIANTS = [
    ("baseline", "baseline_tanh_fixed", "(none)"),
    ("adaptive (spec: data, res_data, tv)", "adaptive", "--weighting adaptive --adaptive_terms data,res_data,tv"),
    ("adaptive_datafit (paper: data, res_data)", "adaptive_datafit", "--weighting adaptive --adaptive_terms data,res_data"),
    ("rar", "rar", "--rar 1"),
    ("relu", "relu", "--act relu"),
    ("residual paper (ablation)", "residual_paper", "--residual paper"),
    ("adaptive+rar", "adaptive_rar", "--weighting adaptive --rar 1"),
    ("adaptive+relu", "adaptive_relu", "--weighting adaptive --act relu"),
    ("rar+relu", "rar_relu", "--rar 1 --act relu"),
    ("adaptive+rar+relu", "adaptive_rar_relu", "--weighting adaptive --rar 1 --act relu"),
    ("baseline, noise 1 % (indicative)", "baseline_sigma0.01", "--sigma 0.01"),
    ("adaptive+rar+relu, noise 1 % (indicative)", "adaptive_rar_relu_sigma0.01", "--sigma 0.01 --weighting adaptive --rar 1 --act relu"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="insert into RESULTS.md between the markers")
    a = ap.parse_args()
    present = [v for v in VARIANTS if any(load(f"{v[1]}_d1234_s{s}") for s in range(5))]
    txt, _ = table(present)
    print(txt)
    if a.write:
        p = os.path.join(ROOT, "RESULTS.md")
        s = open(p).read()
        assert START in s and END in s, "markers missing in RESULTS.md"
        pre, rest = s.split(START)
        _, post = rest.split(END)
        open(p, "w").write(pre + START + "\n" + txt + "\n" + END + post)


if __name__ == "__main__":
    main()
