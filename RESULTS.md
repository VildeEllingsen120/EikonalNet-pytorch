# Results

Target: paper Table 1, PINN column, 0 % noise: nRMSE_T = 0.49 %, nMAE_V = 3.56 %.
Metrics are RMSE of T and MAE of V over the 50x50 grid, in percent (see README, "Metrics").
Hardware: Apple M2 (8 cores), CPU only, float32. Software: Python 3.14.3, torch 2.11.0.
All runs: 50 LHS data points, 100 fresh LHS collocation points per Adam step, Adam lr 1e-3,
then one L-BFGS call on the 2500-point grid (max_iter 10000, max_eval 50000, history 50,
strong Wolfe). alpha = 1e-7, alpha_L2 = 1e-9, V_max = 1.5, residual form `tf`
(`|grad T| - 1/V`, as in the original code), no eps inside the sqrt (no NaNs occurred).

| run | sigma | adam steps | lbfgs iters | nRMSE_T [%] | nMAE_V [%] | max err T | max err V | wall time | notes |
|---|---|---|---|---|---|---|---|---|---|
| baseline_tanh_fixed | 0 | 50000 | 10000 | 0.424 | 3.521 | 0.0548 | 1.198 | 112 s | seed 1234; L-BFGS hit the 10000-iteration cap (10654 evaluations) |
| baseline_tanh_fixed_seed1 | 0 | 50000 | 10000 | 1.521 | 6.378 | 0.0984 | 0.874 | 109 s | seed 1 (init, data points, collocation all reseeded); L-BFGS hit the cap (10630 evaluations) |
| baseline_tanh_fixed_seed2 | 0 | 50000 | 10000 | 4.954 | 16.863 | 0.2347 | 1.379 | 86 s | seed 2; L-BFGS stopped on its own after 4537 iterations (4896 evaluations), final grid loss 1.0e-3: poor local minimum |
| **mean (range) of the 3 seeds** | 0 | 50000 | 10000 | 2.30 (0.42 to 4.95) | 8.92 (3.52 to 16.86) | 0.129 (0.055 to 0.235) | 1.15 (0.87 to 1.38) | 102 (86 to 112) s | paper: 0.49 / 3.56 |

## baseline_tanh_fixed (seed 1234)

Command: `python scripts/train.py --run_name baseline_tanh_fixed --sigma 0`

Reproduces the paper's row within 15 % on both metrics. The T error is essentially zero
over the domain except for a narrow band along the collision ridge r0 = 0.7 r1, where the
network smooths the kink of the exact T; the largest error (about 0.05) sits where the ridge
meets the bottom edge near (0.75, 0), where there are no data points. The V error is confined
to a band of width about 0.05 around the ridge: the predicted V jumps from 1 to 1.43 slightly
on the wrong side of the true ridge over part of its length, and at the two ends of the ridge
on the boundary, where no data constrain it, V collapses towards the sigmoid floor of 0: the
largest V error (1.198) is at grid point (0, 0.755), where the ridge meets the left edge, with
V_pred = 0.23 against 1.43 exact; a second trough reaches V_pred = 0.38 at (0.84, 0) on the
bottom edge. 16 of the 2500 grid points have V_pred < 0.9, all in these two troughs. Away
from the ridge, V is within about 0.03 of the exact value on both sides. (The V panel of
`fig_fields.png` originally used a 0.9 to 1.5 colour scale that hid the troughs; it now spans
0 to 1.5 and the figures of all runs were regenerated from `fields.npz`.)

Loss history: Adam brings the total loss from 2.3 to about 1e-3 with the usual minibatch
noise; the TV term grows from 1e-10 to 1e-5 as V develops its step; L-BFGS then lowers the
total loss from about 1e-4 to 1.0e-5 in 10000 iterations and was still decreasing slowly
when it hit the iteration cap (the original notebook reached 7.2e-6 under the same cap).

Deviations from the project brief: none in the model or loss. `--lbfgs_max_eval` is 50000 (matches
the original scipy `maxfun`; the brief's draft said 10000, changed after discussion). The paper
PDF is not part of this repository.

## Seed study (seeds 1 and 2, same configuration)

Commands: `python scripts/train.py --run_name baseline_tanh_fixed_seed1 --sigma 0 --seed 1`
and the same with `seed2` / `--seed 2`. The seed changes the network initialisation, the
collocation stream and the 50 data-point locations together (`--data_seed` defaults to `--seed`).

Seed 1234 reproduces the paper's row; seeds 1 and 2 do not. The spread (nRMSE_T 0.42 to 4.95 %,
nMAE_V 3.5 to 16.9 %) is of the same order as the ranges the paper reports over 30 repeats for
the noisy rows (e.g. 1 % noise: T 1.09 to 3.09, V 6.09 to 17.86), so the single 0 % value in
Table 1 should be read as one realisation, not a typical one. Only seed 1234 is within a factor
of 2 of the paper on both metrics; seed 1 is within a factor of 2 on V and about 3x on T;
seed 2 is about 10x on T and 5x on V.

seed 1: T is accurate except for a band along the ridge, with the largest errors (about -0.1)
in a diamond-shaped region around the ridge centre (0.45, 0.4) and (+0.1) where the ridge meets
the bottom edge near (0.6, 0); both regions have no data points on the ridge itself. V has the
correct plateaus on both sides and the ridge in roughly the right place, with the usual
over/undershoot band; the lower-right end of the ridge is bent towards (0.7, 0), and along
the bottom edge between x = 0.5 and 0.75 V falls to 0.38 (max V error 0.874 at (0.735, 0));
190 grid points have V_pred < 0.9.

seed 2: a qualitatively wrong solution. The predicted V ridge is a thin trough (V down to 0.05
at (0.755, 0.367); the sigmoid floor is 0) displaced from the true ridge, V is about 1.4 instead of 1.0 over most of the
lower-left region (only a patch near the origin reaches 1.0), and the T error forms broad bands of
+-0.05 and more on both sides of the ridge over the whole lower-right half. L-BFGS stopped after
4537 iterations with the loss still at 1.0e-3 (two orders of magnitude above the other seeds),
i.e. Adam had converged to a basin that L-BFGS could not leave.

Whether the variability comes from the data-point locations or from the initialisation is
separated in the 3x3 grid below.

## Data seed vs initialisation seed (3x3 grid)

Runs `baseline_tanh_fixed_d<data_seed>_s<seed>` (the three diagonal cells are the runs above).
`--data_seed` sets the Latin hypercube of the 50 data points; `--seed` sets the network
initialisation, the collocation stream and the noise generator. Cell = nRMSE_T / nMAE_V in percent.

| data seed \ init seed | 1234 | 1 | 2 |
|---|---|---|---|
| **1234** | 0.42 / 3.52 | 0.62 / 4.71 | 0.79 / 2.56 |
| **1** | 0.58 / 4.69 | 1.52 / 6.38 | 0.57 / 4.83 |
| **2** | 4.96 / 18.98 | 0.84 / 4.54 | 4.95 / 16.86 |

Row means (data seed): nRMSE_T 0.61, 0.89, 3.58, nMAE_V 3.60, 5.30, 13.46.
Column means (init seed): nRMSE_T 1.99, 0.99, 2.10, nMAE_V 9.06, 5.21, 8.08.

The spread follows the data seed, not the initialisation. With data seeds 1234 and 1 every
initialisation lands between 0.42 and 1.52 % on T and between 2.6 and 6.4 % on V, i.e. within
the paper's factor-of-2 window or close to it. Data seed 2 produces the two failures
(4.96 / 18.98 and 4.95 / 16.86) with final losses of about 1e-3, two orders of magnitude above
the other cells, and only one of its three initialisations (seed 1) escapes to 0.84 / 4.54; so
that data layout has a competing basin that two of three initialisations fall into. Sparse
coverage of the ridge is not the explanation: data seed 2 has more points near the ridge than
the other two layouts (13 within 0.10 and 6 within 0.05, against 9 / 5 for seed 1234 and 8 / 5
for seed 1), and more points in the bottom-right region x > 0.6, y < 0.3 where the ridge meets
the boundary (8 against 5 and 5). Why this layout is hard for the optimiser is not identified.
The column means show no systematic effect of the initialisation once the failed data layout is set aside.

## Protocol for all variants from now on: data seed 1234, init seeds 0 to 4

Commands: `python scripts/train.py --run_name baseline_tanh_fixed_d1234_s<k> --sigma 0 --data_seed 1234 --seed <k>`
for k = 0..4 (seeds 1 and 2 are the grid runs above). Every variant is reported as mean / min / max over these five seeds.

| init seed | nRMSE_T [%] | nMAE_V [%] | max err T | max err V | wall time | L-BFGS iters | final grid loss |
|---|---|---|---|---|---|---|---|
| 0 | 0.423 | 4.883 | 0.0349 | 0.470 | 103 s | 7137 | 8.1e-06 |
| 1 | 0.616 | 4.709 | 0.0702 | 0.947 | 96 s | 6778 | 1.6e-05 |
| 2 | 0.793 | 2.560 | 0.0724 | 0.421 | 108 s | 8852 | 6.7e-05 |
| 3 | 0.828 | 6.983 | 0.0729 | 0.741 | 115 s | 10000 | 3.1e-05 |
| 4 | 0.840 | 9.741 | 0.0394 | 0.687 | 93 s | 6203 | 1.6e-05 |
| **mean / min / max** | 0.700 / 0.423 / 0.840 | 5.775 / 2.560 / 9.741 | 0.0580 / 0.0349 / 0.0729 | 0.653 / 0.421 / 0.947 | 103 / 93 / 115 s | | |

Baseline protocol summary: **nRMSE_T 0.70 % (0.42 to 0.84), nMAE_V 5.78 % (2.56 to 9.74)**;
paper 0.49 / 3.56. Three of the five seeds are within a factor of 2 of the paper on both metrics;
seeds 3 and 4 miss on V (7.0 and 9.7 %) while matching on T. L-BFGS stopped on its tolerance
criterion before the 10000-iteration cap in four of five runs. The seed-1234 run (0.42 / 3.52)
is the best of the five on T and second-best on V, so the single-run reproduction above is a
favourable draw, not the typical outcome.

# Variants

Protocol: data seed 1234, init seeds 0 to 4, 50k Adam + L-BFGS as for the baseline, one run per
seed; cells are mean (min to max) over the seeds that finished. A seed counts as failed if the
loss became non-finite or the prediction contains NaNs; failed seeds are excluded from the
statistics and listed in the last column. Table generated by `scripts/summarize.py --write`.

<!-- VARIANT_TABLE_START -->
| variant | flags | nRMSE_T [%] mean (min to max) | nMAE_V [%] mean (min to max) | max err T | max err V | wall [s] | failed seeds |
|---|---|---|---|---|---|---|---|
| baseline | `(none)` | 0.70 (0.42 to 0.84) | 5.78 (2.56 to 9.74) | 0.058 (0.035 to 0.073) | 0.65 (0.42 to 0.95) | 103 (93 to 115) | none |
| adaptive (spec: data, res_data, tv) | `--weighting adaptive --adaptive_terms data,res_data,tv` | 2.32 (1.71 to 3.12) | 15.44 (13.70 to 16.92) | 0.217 (0.119 to 0.311) | 0.42 (0.36 to 0.46) | 105 (83 to 118) | none |
| adaptive_datafit (paper: data, res_data) | `--weighting adaptive --adaptive_terms data,res_data` | 0.53 (0.20 to 0.89) | 2.75 (1.95 to 3.87) | 0.035 (0.015 to 0.049) | 0.42 (0.37 to 0.48) | 109 (85 to 123) | none |
| rar | `--rar 1` | 2.81 (0.75 to 6.30) | 10.55 (6.98 to 14.81) | 0.162 (0.056 to 0.297) | 0.78 (0.50 to 1.18) | 165 (129 to 185) | none |
| relu | `--act relu` | 1.40 (0.86 to 2.33) | 6.74 (4.48 to 8.97) | 0.127 (0.059 to 0.298) | 0.30 (0.26 to 0.33) | 77 (61 to 84) | none |
| residual paper (ablation) | `--residual paper` | 2.40 (1.23 to 3.41) | 10.33 (7.31 to 11.55) | 0.188 (0.128 to 0.239) | 0.97 (0.55 to 1.32) | 126 (93 to 147) | none |
| adaptive+rar | `--weighting adaptive --rar 1` | 0.62 (0.26 to 0.93) | 3.42 (1.31 to 5.66) | 0.049 (0.021 to 0.070) | 0.38 (0.31 to 0.45) | 121 (94 to 131) | none |
| adaptive+relu | `--weighting adaptive --act relu` | 0.66 (0.54 to 0.89) | 3.16 (1.78 to 4.23) | 0.062 (0.048 to 0.078) | 0.45 (0.41 to 0.50) | 80 (64 to 87) | none |
| rar+relu | `--rar 1 --act relu` | 1.92 (0.93 to 3.06) | 9.22 (6.90 to 12.36) | 0.162 (0.037 to 0.320) | 0.33 (0.30 to 0.37) | 87 (71 to 94) | none |
| adaptive+rar+relu | `--weighting adaptive --rar 1 --act relu` | 0.70 (0.61 to 0.83) | 2.60 (2.25 to 3.13) | 0.075 (0.058 to 0.093) | 0.46 (0.43 to 0.47) | 94 (73 to 101) | none |
| baseline, noise 1 % (indicative) | `--sigma 0.01` | 1.33 (0.58 to 2.91) | 9.42 (4.40 to 17.17) | 0.072 (0.030 to 0.138) | 0.67 (0.48 to 0.98) | 118 (87 to 144) | none |
| adaptive+rar+relu, noise 1 % (indicative) | `--sigma 0.01 --weighting adaptive --rar 1 --act relu` | 1.11 (0.76 to 1.58) | 5.37 (2.75 to 7.09) | 0.119 (0.063 to 0.211) | 0.45 (0.42 to 0.48) | 93 (77 to 101) | none |
<!-- VARIANT_TABLE_END -->

## Stage 1: adaptive loss weighting (Wang, Teng, Perdikaris 2021, Algorithm 1)

**Verification against the paper.** Checked against arXiv:2001.04536v1 (January 2020), p. 10,
Algorithm 1, Eqs. (40) and (41): lambda_hat_i = max_theta |grad_theta L_r| / mean_theta |grad_theta L_i|,
lambda_i <- (1 - alpha) lambda_i + alpha lambda_hat_i, recommended alpha = 0.9, update every
iteration or every 10, L_r the PDE residual. Two differences from the spec draft, both resolved
in favour of the paper as the project brief requires:
1. the denominator is the gradient of the *unweighted* L_i (Eq. 40), not of lambda_i L_i;
2. the paper defines the lambda_i for "data-fit terms (e.g., measurements, initial or boundary
   conditions)". The TV penalty is a regulariser, not a data-fit term. The spec asked for a
   lambda on the TV term as well, so both versions were run:
   `adaptive` (spec: lambdas on data, res_data, tv) and `adaptive_datafit`
   (paper: lambdas on data and res_data only; flag `--adaptive_terms data,res_data`).
   The SIAM print version of the paper was not checked; only the arXiv preprint was available.
Implementation in `src/eikonal_pinn/adaptive.py`; gradients over all parameters of both
networks, flattened; parameters a term does not touch count as zeros. Updates every 10 Adam
steps, beta = 0.9, lambdas frozen during L-BFGS, logged in `loss.csv`
(`lambda_data`, `lambda_res_data`, `lambda_tv`). Terms that are not re-weighted keep lambda = 1.

**adaptive (spec version): fails on every seed.** nRMSE_T 2.08, 1.71, 2.01, 2.67, 3.12 %, nMAE_V
16.92, 14.25, 13.70, 15.89, 16.44 % (seeds 0 to 4). Because the TV term carries alpha = 1e-7 inside it,
its gradient is 1e-7 times smaller than the residual gradient and the annealing compensates:
lambda_tv reaches 1e9 at step 0 and ends at 6.5e+19, 6.5e+22, 8.9e+18, 1.9e+19, 1.4e+21. The weighted TV
term then dominates the loss, V is pinned flat at V_max = 1.5 over the whole domain (V error
+0.5 on the whole r0 < 0.7 r1 side) and the Eikonal residual is never satisfied (final residual
term on the grid 6.2e-02, 6.0e-02, 7.9e-02, 1.7e-01, 1.5e-01, against 2e-6 for the baseline).
lambda_data ends at 1.3e+05, 3.4e+03, 4.2e+05, 2.9e+04, 1.6e+05 and lambda_res_data at
1.8e+04, 1.3e+03, 3.2e+04, 3.0e+03, 1.2e+04. Adam loss spikes (total > 10x the median of the previous
100 steps): 1334, 1202, 1250, 1052, 1194 per run, against 0 to 2 for the baseline. L-BFGS
stopped after 4203, 1168, 3601, 24, 2831 iterations with a total loss of 0.07 to 0.6. No NaNs,
so no seed counts as failed by the protocol's definition, but the variant does not work.

**adaptive_datafit (paper version): better than the baseline on both metrics.**
nRMSE_T 0.89, 0.20, 0.66, 0.48, 0.44 %, nMAE_V 3.87, 2.16, 3.29, 1.95, 2.47 %; mean 0.53 / 2.75 against
0.70 / 5.78 for the baseline, and the worst seed (0.89 / 3.87) is better than the baseline's
median. lambda_data ends at 2.4e+05, 1.2e+04, 1.4e+05, 2.8e+05, 1.6e+05 and lambda_res_data at
5.5e+04, 8.5e+02, 4.3e+02, 4.9e+04, 4.4e+03: the data terms are up-weighted by 1e3 to 1e5, the data
misfit is driven to 1e-8 (baseline 6e-6) and the residual term on the grid ends at
2.3e-02, 2.2e-03, 1.4e-02, 6.6e-03, 8.7e-03 (baseline 2e-6), so the better field errors come with a
worse PDE fit. The V field is a smooth ramp across the ridge rather than a sharp step with
over/undershoot: the V error band is wider (about 0.1) but lower (within +-0.3 instead of
+-0.5 and the troughs to 0.2 of the baseline), which is what lowers the MAE. The T error is
concentrated at the bottom-edge end of the ridge as in the baseline (max 0.049, 0.015, 0.045, 0.036, 0.032).
Loss spikes 1057, 796, 1005, 1081, 1221 per run: the weighted total jumps whenever the
lambdas are updated, so this variant is markedly less smooth to train than the baseline.

Decision for the combinations (stage 5): "adaptive" means the paper version
(`--weighting adaptive --adaptive_terms data,res_data`), since the spec version does not train.

**Additional run at init seed 1234** (the seed of the baseline row above), `adaptive_datafit_d1234_s1234`,
command `python scripts/train.py --run_name adaptive_datafit_d1234_s1234 --sigma 0 --data_seed 1234 --seed 1234 --weighting adaptive --adaptive_terms data,res_data`:
nRMSE_T 0.625, nMAE_V 1.403, max err T 0.043, max err V 0.394, wall 84 s (76 s Adam, 8 s L-BFGS, 1771 iterations), no NaNs.
Against the baseline at the same seed (0.424 / 3.521): worse on T, better on V; the minimum V_pred on the grid is 0.96
against 0.23 for the baseline. This run and the baseline row are the two runs behind `figs/fields_main` and `figs/section_main`.

## Stage 2: residual-based adaptive refinement (Lu et al. 2021, Sec. 2.8)

Implementation in `src/eikonal_pinn/rar.py`: every 5000 Adam steps (steps 5000 to 45000, nine
refinements), |f_T| is evaluated on 10000 fresh uniform points (generator seeded with seed + 2)
and the 50 largest are appended to a persistent set X_rar; the Adam minibatch is the fresh LHS
batch of 100 plus all of X_rar (450 points at the end), and L-BFGS uses the 2500-point grid plus
X_rar. X_rar is saved in `fields.npz` and drawn as red crosses on `fig_fields.png`.

**Result: worse than the baseline on average, and more variable.** nRMSE_T 6.30, 0.75, 2.09, 4.06, 0.85 %,
nMAE_V 12.72, 7.19, 11.03, 14.81, 6.98 % (seeds 0 to 4); mean 2.81 / 10.55 against 0.70 / 5.78 for the baseline.
The two best seeds (1 and 4: 0.75 / 7.19 and 0.85 / 6.98) match the baseline on T and are worse
on V; seeds 0, 2 and 3 end in wrong solutions (T error bands of 0.05 and more, V plateaus on the
wrong side of the ridge over large areas), with L-BFGS grid losses of 2e-4 to 2e-3 against 1e-5
for the baseline. Wall time 165 s per run against 103 s (the minibatch grows to 550 points).

**Where the added points cluster.** The expectation was clustering along the collision ridge.
Measured fraction of X_rar within 0.05 of the true ridge: 0.20, 0.66, 0.30, 0.45, 0.36
(a uniform set would give 0.10); within 0.10:
0.37, 0.76, 0.57, 0.66, 0.76 (uniform 0.21); median distance to the
ridge 0.146, 0.030, 0.082, 0.057, 0.064. So the points are strongly concentrated near the
ridge in seed 1 (66 % within 0.05) and moderately in the others, but the figures show the rest of
the mechanism: the points line up along whatever kink the *network* currently has, which is the
true ridge where the network has found it (seed 1, upper-left half of the ridge) and a spurious
straight line elsewhere (seed 0: a band from about (0.25, 1.0) to (0.75, 0.0); seed 3: a cloud
around (0.8, 0.2) inside a region where T is off by more than 0.05). Because the set is
persistent, the residual is then enforced hardest exactly along the wrong kink, which anchors it:
in seeds 0, 2 and 3 the final T error is largest where the RAR points are densest. A few points
also sit at the source corner (1, 1) in seed 1 (4 % within 0.1 of a source), where the exact T has
a cone tip and |f_T| of any smooth network is large. Loss spikes: 104, 102, 121, 76, 113
per run (baseline 0 to 2), one per refinement plus the steps after it.

## Stage 3: ReLU activation in net_T

`--act relu`: ReLU between the layers of net_T (Kaiming-normal init, fan-in, for the hidden
layers; the output layer keeps Xavier), net_V stays tanh. The residual needs only first
derivatives of T, so ReLU is admissible, and grad T is then piecewise constant, as for the exact T.

**Result: worse than the baseline on both metrics.** nRMSE_T 1.23, 1.33, 1.25, 0.86, 2.33 %, nMAE_V
7.96, 7.60, 4.71, 4.48, 8.97 % (seeds 0 to 4); mean 1.40 / 6.74 against 0.70 / 5.78. The T error is no longer
a thin band at the ridge but a patchwork of +-0.03 facets over the whole domain (the piecewise
linear T has kinks between data points that the exact T does not have), with the largest errors at
the source corners and at the ridge; the V field is a smooth ramp across the ridge without the
baseline's over/undershoot, so max err V is lower (0.26, 0.32, 0.29, 0.28, 0.33) while the MAE is higher.

**Stability.** Measured on the loss logged every 100 Adam steps, with the same definition for
both variants (total > 10x the median of the previous 10 logged values): ReLU
1, 0, 1, 6, 8 spikes per run against 0, 0, 1, 0, 0 for the baseline;
largest total loss after step 1000: 1.2e-2 to 7.8e-2 (baseline 2.1e-2 to 3.7e-2). So Adam is only
mildly less smooth. Adam in fact ends lower than with tanh (median total over the last 5000 steps
1.1e-4 to 2.7e-4 against 3.8e-4 to 1.0e-3). Dead units: 47, 46, 39, 39, 47 of the 100
hidden units of net_T are zero for every one of the 2500 grid points (tanh: none by the same
test). The decisive instability is in the second phase: L-BFGS terminates after
13, 5, 15, 11, 20 iterations (53, 21, 57, 65, 55 evaluations) because the
strong-Wolfe line search fails on the piecewise-smooth loss of a ReLU network (the loss is not
twice differentiable and its gradient jumps whenever a kink crosses a collocation point), so the
final grid loss stays at 1.3e-04, 3.0e-04, 1.2e-04, 2.1e-04, 2.0e-04 against 1e-5 for tanh. The ReLU
runs are therefore effectively Adam-only. Wall time 77 s against 103 s for the same reason.

## Stage 4: residual form of the paper (ablation, not a contribution)

`--residual paper`: f_T = V |grad T| - 1 (paper Eq. 2) instead of the original code's
f_T = |grad T| - 1/V; everything else as in the baseline. The two residuals differ by the
factor V (between 1 and 1.5 for the exact solution), so for a converged solution they are
equivalent up to that scaling; they differ in what they penalise far from convergence.

**Result: clearly worse than the original code's form.** nRMSE_T 2.38, 1.23, 2.46, 2.54, 3.41 %, nMAE_V
11.12, 7.31, 10.18, 11.47, 11.55 % (seeds 0 to 4); mean 2.40 / 10.33 against 0.70 / 5.78, max err V
1.23, 1.32, 0.55, 1.10, 0.67. Every seed develops a narrow trough in V (down to 0.18 at the
worst) along a line that is not the true ridge, with T error bands of +-0.05 on both sides of it.
The mechanism is visible in the residual: with V |grad T| - 1 the residual is bounded below by -1
as V -> 0, so wherever the network's |grad T| is too large, shrinking V towards 0 cancels the
residual cheaply; with |grad T| - 1/V the same move costs 1/V -> infinity. The 1/V form of the
original code therefore acts as a barrier against small V, which the paper's form lacks. Loss
spikes 6, 19, 23, 53, 11 per run (baseline 0 to 2); L-BFGS reached
10000, 10000, 8874, 3372, 4653 iterations with final grid losses 1.8e-04, 6.0e-05, 9.2e-05, 6.5e-05, 5.6e-04.
The `tf` form stays the default for all other variants.

## Stage 5: combinations

"adaptive" is the paper version throughout (`--weighting adaptive --adaptive_terms data,res_data`).
Mean (min to max) over seeds 0 to 4, from the shared table:

| combination | nRMSE_T [%] | nMAE_V [%] |
|---|---|---|
| adaptive+rar | 0.62 (0.26 to 0.93) | 3.42 (1.31 to 5.66) |
| adaptive+relu | 0.66 (0.54 to 0.89) | 3.16 (1.78 to 4.23) |
| rar+relu | 1.92 (0.93 to 3.06) | 9.22 (6.90 to 12.36) |
| adaptive+rar+relu | 0.70 (0.61 to 0.83) | 2.60 (2.25 to 3.13) |

Ranking of all variants by mean nMAE_V: adaptive+rar+relu 2.60, adaptive (paper) 2.75,
adaptive+relu 3.16, adaptive+rar 3.42, baseline 5.78, relu 6.74, rar+relu 9.22, rar 10.55,
residual paper 10.33 (ablation), adaptive (spec) 15.44. By mean nRMSE_T: adaptive (paper) 0.53,
adaptive+rar 0.62, adaptive+relu 0.66, baseline 0.70, adaptive+rar+relu 0.70, relu 1.40,
rar+relu 1.92, rar 2.81.

Every combination that contains adaptive weighting beats the baseline on V, and none of them is
worse than the baseline on T by more than 0.01 (mean). The combination without adaptive
weighting (rar+relu) inherits the failures of both parts (three of five seeds with T error bands
of 0.05 or more; L-BFGS stops after 9 to 20 iterations). adaptive+rar+relu has the smallest seed
spread of all variants on V (2.25 to 3.13) and the best mean, but not the best single seed
(adaptive+rar seed 1: 0.26 / 1.31 is the best run of the whole study). With adaptive weighting
the RAR points concentrate on the true ridge in most seeds (0.33, 0.82, 0.70, 0.41, 0.64
of X_rar within 0.05 of it for adaptive+rar; 0.49, 0.30, 0.62, 0.33, 0.24 for the triple),
i.e. the up-weighted data terms fix the ridge position before the persistent points can anchor a
wrong one. The relu combinations still lose L-BFGS (5 to 30 iterations) and keep 25 to 45 dead
units; their results are Adam-only, so the gain over plain relu comes from the weighting.
Weighted-loss spikes during Adam: 1562, 1481, 1372, 1542, 1688 for adaptive+rar,
591, 787, 545, 702, 881 for adaptive+relu, 763, 773, 745, 606, 976
for the triple (the lambda updates every 10 steps change the weighted total; the unweighted
terms do not spike).

Selection for stage 6 (hard layout): best two single variants by mean nMAE_V are adaptive (paper,
2.75) and relu (6.74; rar is 10.55); best combination is adaptive+rar+relu (2.60).

## Stage 6: hard layout (data seed 2, init seeds 1234, 1, 2)

The two best single variants and the best combination by mean nMAE_V (stage 5) were run on
data seed 2, where the baseline failed for init seeds 1234 and 2. Run names `<variant>_d2_s<k>`.
Cells: nRMSE_T / nMAE_V in percent.

| variant | init 1234 | init 1 | init 2 | failure removed? |
|---|---|---|---|---|
| baseline | 4.96 / 18.98 | 0.84 / 4.54 | 4.95 / 16.86 | (reference: fails in cells 1234 and 2) |
| adaptive (paper) | 0.47 / 2.12 | 0.63 / 3.78 | 0.33 / 2.39 | yes, both failed cells; all three cells within a factor of 2 of the paper |
| relu | 1.64 / 5.19 | 1.08 / 5.99 | 1.66 / 9.75 | partly: errors in the failed cells drop by 3x on T and 2 to 4x on V but stay 2 to 3x above the paper |
| adaptive+rar+relu | 0.51 / 2.40 | 1.15 / 7.55 | 0.59 / 1.77 | yes in both failed cells; but the cell where the baseline worked (init 1) gets worse (1.15 / 7.55 vs 0.84 / 4.54; L-BFGS did 2 iterations) |

Adaptive weighting (paper version) removes the failure: on the hard layout its three cells
(0.33 to 0.63 / 2.12 to 3.78) are as good as its five seeds on data seed 1234 (0.20 to 0.89 /
1.95 to 3.87), so with this weighting data seed 2 is no longer a hard layout. The V field in
the former failure cells has the plateaus on the correct sides and a single transition band of
width about 0.05 along the true ridge (figure `results/adaptive_datafit_d2_s1234/fig_fields.png`).
ReLU alone reduces but does not remove the failure. The triple combination removes it in the two
failed cells, with RAR points on the true ridge, but in the init-1 cell L-BFGS stopped after 2
iterations with the weighted loss at 0.4 and the result is worse than the baseline's; its
Adam-only character (stage 3) makes it the least predictable of the three.

## Stage 7: data seed 2 diagnostic (no training)

`scripts/data_layout_diagnostic.py`; figure `results/data_layout_diagnostic/data_layouts.png`
(three layouts side by side, ridge drawn, points coloured by side), numbers in `stats.json`.

| data seed | points with r0 < 0.7 r1 (V = 1 side) | points with r0 >= 0.7 r1 (V = 1/0.7 side) | distance (0,0) to nearest point | distance (1,1) to nearest point | points within 0.05 of ridge | within 0.10 | closest point to ridge |
|---|---|---|---|---|---|---|---|
| 1234 | 16 | 34 | 0.162 | 0.186 | 5 | 9 | 0.024 |
| 1 | 15 | 35 | 0.138 | 0.144 | 5 | 8 | 0.006 |
| 2 | 13 | 37 | 0.053 | 0.091 | 6 | 13 | 0.016 |

For reference, the V = 1 region covers 31.3 % of the unit square (783 of 2500 grid points), so a
uniform layout would put about 16 of 50 points there. Data seed 2 has the fewest points on the
V = 1 side (13 against 16 and 15), the data points closest to both sources (0.053 from (0, 0) and
0.091 from (1, 1), against 0.14 to 0.19 for the other two layouts), and the most points near the
ridge (6 within 0.05 and 13 within 0.10, against 5 / 9 and 5 / 8). The side counts and the ridge
coverage therefore do not single out data seed 2 as under-sampled; the one quantity in which it
is an outlier is the proximity of data points to the two source corners, where the exact T has
its cone tips. Whether that proximity is what makes the layout hard is not established by these
numbers.

## Stage 8: noise 1 % (indicative only)

`--sigma 0.01`: Gaussian noise with standard deviation 0.01 x max T (T_max = 0.7258) added to the
50 measured activation times; the noise realisation is drawn from a generator seeded with the init
seed, so the five seeds differ in noise, initialisation and collocation stream, with the same point
locations (data seed 1234). Everything else as in the protocol: 50k Adam + L-BFGS, 5 seeds. Run
names `baseline_sigma0.01_d1234_s<k>` and `adaptive_rar_relu_sigma0.01_d1234_s<k>`.

**Caveat.** The paper's 1 % row (PINN: nRMSE_T 1.91, range 1.09 to 3.09; nMAE_V 10.24, range 6.09
to 17.86) was produced with 100,000 Adam iterations, no L-BFGS, and 30 repeats (p. 6). The runs
here use the 0 % protocol (50k Adam + L-BFGS, 5 repeats), so the comparison is indicative only.

| variant, sigma = 0.01 | nRMSE_T [%] per seed 0..4 | nMAE_V [%] per seed 0..4 | mean (min to max) |
|---|---|---|---|
| baseline | 2.91, 0.70, 0.72, 0.58, 1.72 | 17.17, 4.54, 7.22, 4.40, 13.75 | 1.33 (0.58 to 2.91) / 9.42 (4.40 to 17.17) |
| adaptive+rar+relu | 1.58, 0.76, 1.16, 0.91, 1.12 | 7.09, 2.75, 5.42, 4.75, 6.85 | 1.11 (0.76 to 1.58) / 5.37 (2.75 to 7.09) |

Both lie inside the paper's reported ranges for this noise level. The noisy baseline has two
seeds (0 and 4) with the same kind of wrong-ridge solution seen on the hard layout (T error bands
over 0.1, V plateaus on the wrong side), which the combination does not produce; the combination
is better on V for every seed but has larger maximum T errors (0.211, 0.063, 0.115, 0.068, 0.137 against
0.138, 0.037, 0.040, 0.030, 0.117), as in the noise-free case. The final data misfit of the combination
(9.0e-06, 7.0e-06, 2.2e-06, 9.6e-06, 6.8e-06) is below the noise variance (sigma^2 T_max^2 = 5.3e-5), i.e. the
up-weighted data term fits the noise; the baseline's (4.0e-04, 2.4e-05, 4.0e-05, 5.0e-05, 1.1e-04) is at or
below it as well for three seeds.

## Status

All stages 1 to 9 done (2026-11-07). See STATUS.md for the summary and `report_assets/` for tables, figures and `facts.md`.
