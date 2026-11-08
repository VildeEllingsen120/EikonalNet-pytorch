#!/usr/bin/env python3
"""Write the 50 LHS data points (with exact T) and the 50x50 evaluation grid to data/."""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from eikonal_pinn.benchmark import add_noise, exact_T, exact_V, make_grid, sample_data  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_data", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--sigma", type=float, default=0.0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    xy_d = sample_data(a.n_data, seed=a.seed)
    T_d_exact = exact_T(xy_d)
    T_d = add_noise(T_d_exact, a.sigma, np.random.default_rng(a.seed))
    xy_g, X_m, Y_m = make_grid(50)
    T_g, V_g = exact_T(xy_g), exact_V(xy_g)

    tag = "" if a.sigma == 0 else f"_sigma{a.sigma:g}"
    np.savez(os.path.join(a.out, f"data_points{tag}.npz"), xy=xy_d, T=T_d, T_exact=T_d_exact, seed=a.seed, sigma=a.sigma)
    np.savetxt(os.path.join(a.out, f"data_points{tag}.csv"), np.hstack([xy_d, T_d, T_d_exact]),
               delimiter=",", header="x,y,T_measured,T_exact", comments="")
    np.savez(os.path.join(a.out, "grid.npz"), xy=xy_g, T=T_g, V=V_g, X_m=X_m, Y_m=Y_m)
    np.savetxt(os.path.join(a.out, "grid.csv"), np.hstack([xy_g, T_g, V_g]), delimiter=",",
               header="x,y,T_exact,V_exact", comments="")
    print(f"wrote {a.n_data} data points (sigma={a.sigma}) and {len(xy_g)} grid points to {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()
