# STATUS (2026-11-07)

## What ran

80 training runs in total, all on CPU (Apple M2, one torch thread per run, four runs at a time),
plus the no-training data-layout diagnostic. Summed training wall time 8160 s (2.27 h); elapsed
session time was longer because of the sequential stages and the write-ups.

| stage | runs | result (mean, min to max over data seed 1234, init seeds 0 to 4; nRMSE_T / nMAE_V in %) |
|---|---|---|
| baseline (earlier) | 12 (protocol 5 + 3x3 seed grid, 2 overlap) | 0.70 (0.42 to 0.84) / 5.78 (2.56 to 9.74) |
| 1 adaptive, spec version (lambda on data, res_data, tv) | 5 | 2.32 (1.71 to 3.12) / 15.44 (13.70 to 16.92) |
| 1 adaptive, paper version (lambda on data-fit terms only) | 5 | 0.53 (0.20 to 0.89) / 2.75 (1.95 to 3.87) |
| 2 rar | 5 | 2.81 (0.75 to 6.30) / 10.55 (6.98 to 14.81) |
| 3 relu | 5 | 1.40 (0.86 to 2.33) / 6.74 (4.48 to 8.97) |
| 4 residual paper (ablation) | 5 | 2.40 (1.23 to 3.41) / 10.33 (7.31 to 11.55) |
| 5 adaptive+rar | 5 | 0.62 (0.26 to 0.93) / 3.42 (1.31 to 5.66) |
| 5 adaptive+relu | 5 | 0.66 (0.54 to 0.89) / 3.16 (1.78 to 4.23) |
| 5 rar+relu | 5 | 1.92 (0.93 to 3.06) / 9.22 (6.90 to 12.36) |
| 5 adaptive+rar+relu | 5 | 0.70 (0.61 to 0.83) / 2.60 (2.25 to 3.13) |
| 6 hard layout (data seed 2; adaptive, relu, adaptive+rar+relu; init 1234, 1, 2) | 9 | adaptive removes the baseline failure in both failed cells; see RESULTS.md stage 6 |
| 7 data-layout diagnostic | 0 | table and figure in RESULTS.md stage 7 |
| 8 noise 1 %: baseline / adaptive+rar+relu | 10 | 1.33 (0.58 to 2.91) / 9.42 (4.40 to 17.17) and 1.11 (0.76 to 1.58) / 5.37 (2.75 to 7.09); indicative only |
| 9 report assets | 3 reruns (seed 0, weights) | `report_assets/` |

Paper Table 1, PINN, 0 % noise: 0.49 / 3.56.

## What failed

- No run produced NaNs or diverged; no seed is marked failed by the protocol's definition.
- The spec version of adaptive weighting (lambda also on the TV term) does not train: lambda_tv
  grows to 1e19 and V is pinned at 1.5. The paper's Algorithm 1 restricts the lambdas to
  data-fit terms, so the paper version was run as well and used in all combinations. Flagged in
  RESULTS.md stage 1.
- With ReLU in net_T, L-BFGS stops after 2 to 30 iterations (strong-Wolfe line search fails),
  so every relu variant is effectively Adam-only. 25 to 47 of 100 hidden units are dead.
- RAR alone anchors wrong kinks in 3 of 5 seeds; its added points sit on the network's own
  kink, which is the true ridge only when the ridge has already been found.
- The requested cross-section line from (0.9, 0.1) to (0.1, 0.9) does not cross the ridge (it lies
  entirely on the V = 1/0.7 side). It is provided as asked (`fig_cross_section`) together with a
  diagonal line from (0.1, 0.1) to (0.7, 0.7) that crosses the ridge perpendicularly at
  (0.412, 0.412) (`fig_cross_section_diagonal`).
- The baseline seed-0 weights come from a single-thread rerun (0.427 / 4.660) because the
  original run (0.423 / 4.883, default threads) predates weight saving; the other two reruns
  reproduce the originals exactly. Recorded in `report_assets/facts.md`.
- The cross-section figures named above and the three rerun directories are not shipped in this
  repository; `scripts/make_report_assets.py` regenerates the figures from the kept runs.

## What the results say

- Adaptive loss weighting restricted to the data-fit terms is the only single modification that
  improves on the baseline: 0.53 / 2.75 against 0.70 / 5.78, and 0.33 to 0.63 / 2.12 to 3.78 on
  the data layout where the baseline failed in two of three cells (4.96 / 18.98 and 4.95 / 16.86).
- RAR (2.81 / 10.55) and ReLU (1.40 / 6.74) alone are worse than the baseline; combined with
  adaptive weighting they are not worse than it (adaptive+rar+relu 0.70 / 2.60, the lowest mean
  and the narrowest seed range on V of all variants, but with larger maximum T errors, 0.075
  against 0.058).
- Seed-to-seed spread is of the same size as the differences between variants: the baseline's
  five seeds span 0.42 to 0.84 on T and 2.56 to 9.74 on V, and the paper's single 0 % value
  (0.49 / 3.56) lies at the good end of that range.
