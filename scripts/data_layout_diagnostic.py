#!/usr/bin/env python3
"""Stage 7 diagnostic (no training): how the 50 LHS data points of data seeds 1234, 1, 2 cover the domain."""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from eikonal_pinn.benchmark import ridge_distance, sample_data  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "data_layout_diagnostic")


def ridge_curve(n=400):
    c = 0.49 / 0.51
    R = np.sqrt(2 * c**2 + 0.98 / 0.51)
    th = np.linspace(0, 2 * np.pi, 4000)
    pts = np.stack([-c + R * np.cos(th), -c + R * np.sin(th)], 1)
    keep = (pts[:, 0] >= 0) & (pts[:, 0] <= 1) & (pts[:, 1] >= 0) & (pts[:, 1] <= 1)
    pts = pts[keep]
    return pts[np.argsort(pts[:, 0])]


def main():
    os.makedirs(OUT, exist_ok=True)
    seeds = [1234, 1, 2]
    stats = {}
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
    rc = ridge_curve()
    for ax, ds in zip(axes, seeds):
        p = sample_data(50, seed=ds)
        r0 = np.hypot(p[:, 0], p[:, 1])
        r1 = np.hypot(p[:, 0] - 1, p[:, 1] - 1)
        slow = r0 < 0.7 * r1  # V = 1 region (origin side)
        d = ridge_distance(p)[:, 0]
        stats[str(ds)] = {
            "n_region_V1 (r0 < 0.7 r1)": int(slow.sum()),
            "n_region_V1.43 (r0 >= 0.7 r1)": int((~slow).sum()),
            "dist_source_(0,0)_to_nearest_point": float(r0.min()),
            "dist_source_(1,1)_to_nearest_point": float(r1.min()),
            "n_within_0.05_of_ridge": int((d < 0.05).sum()),
            "n_within_0.10_of_ridge": int((d < 0.10).sum()),
            "min_dist_to_ridge": float(d.min()),
        }
        ax.plot(rc[:, 0], rc[:, 1], "k-", lw=1.2, label="ridge r0 = 0.7 r1")
        ax.scatter(p[slow, 0], p[slow, 1], facecolors="none", edgecolors="tab:blue", s=30, label="V = 1 side")
        ax.scatter(p[~slow, 0], p[~slow, 1], facecolors="none", edgecolors="tab:red", s=30, label="V = 1/0.7 side")
        ax.plot([0, 1], [0, 1], "k*", ms=9)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_title(f"data seed {ds}")
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "data_layouts.png"), dpi=200)
    fig.savefig(os.path.join(OUT, "data_layouts.pdf"))
    json.dump(stats, open(os.path.join(OUT, "stats.json"), "w"), indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
