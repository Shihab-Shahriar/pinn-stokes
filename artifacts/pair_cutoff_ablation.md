# Pair-cutoff ablation: moments v2 (all, r_c = 8), pair_cutoff 6 -> 8

Generated 2026-08-30 21:01 from `paper_accuracy_v2.csv` (1286 pc8 rows) by `benchmarks/pair_cutoff_ablation.py`.

Both operators share everything except the corrected-pair range: `M_mom_v2_kinf_rc8` corrects pairs with d <= 6 (2b NN base to 6, RPY beyond), `M_mom_v2_kinf_rc8_pc8` corrects d <= 8 and runs with `switch_dist=8` (the 2b NN, trained to d = 8, is the base wherever pairs are corrected; NN-vs-RPY base difference in the 6-8 shell is 2e-4 median, i.e. negligible). Both trained with the identical recipe on dataset v2; labels R = 0.5(M_ts + M_st^T) - M2b. Metric: PRMSE % (`rel_rmse`), mean over 10 seeds; MFS Xfine truths (cached, identical for every operator).

## Figure 3 protocol, N = 200 (PRMSE %, mean over 10 seeds)

| phi | NeMO 2-body | NeMO n-body (paper) | pc6 | pc8 | delta | improvement |
|---|---:|---:|---:|---:|---:|---:|
| 0.025 | 2.58 | 2.12 | 1.83 | 1.58 | +0.25 | 14 % |
| 0.05 | 5.10 | 3.93 | 3.26 | 2.74 | +0.53 | 16 % |
| 0.075 | 7.64 | 5.83 | 4.57 | 3.89 | +0.68 | 15 % |
| 0.1 | 10.64 | 7.77 | 5.79 | 4.78 | +1.01 | 17 % |
| 0.125 | 13.26 | 10.07 | 6.93 | 5.61 | +1.32 | 19 % |
| 0.15 | 15.98 | 11.98 | 8.10 | 6.57 | +1.53 | 19 % |
| 0.175 | 18.94 | 14.38 | 9.21 | 7.60 | +1.61 | 17 % |
| 0.2 | 19.54 | 15.01 | 9.08 | 7.41 | +1.67 | 18 % |

## Figure 3 protocol, N = 300 (PRMSE %, mean over 10 seeds)

| phi | NeMO 2-body | NeMO n-body (paper) | pc6 | pc8 | delta | improvement |
|---|---:|---:|---:|---:|---:|---:|
| 0.025 | 2.72 | 2.28 | 2.03 | 1.78 | +0.25 | 12 % |
| 0.05 | 5.49 | 4.30 | 3.56 | 3.04 | +0.52 | 15 % |
| 0.075 | 8.00 | 6.19 | 4.91 | 4.16 | +0.76 | 15 % |
| 0.1 | 10.32 | 7.93 | 6.06 | 5.14 | +0.92 | 15 % |
| 0.125 | 12.87 | 9.74 | 7.02 | 5.92 | +1.10 | 16 % |
| 0.15 | 15.69 | 12.16 | 8.63 | 7.21 | +1.42 | 16 % |
| 0.175 | 18.74 | 14.62 | 9.80 | 8.10 | +1.69 | 17 % |
| 0.2 | 22.70 | 18.08 | 11.94 | 9.89 | +2.05 | 17 % |

## Figure 3, N = 200: max per-particle error (max_rel_rmse %, mean over seeds)

| phi | pc6 | pc8 | improvement |
|---|---:|---:|---:|
| 0.025 | 6.2 | 5.5 | 12 % |
| 0.05 | 11.1 | 8.9 | 20 % |
| 0.075 | 13.3 | 12.3 | 7 % |
| 0.1 | 19.8 | 16.1 | 19 % |
| 0.125 | 20.6 | 15.9 | 23 % |
| 0.15 | 23.1 | 20.9 | 9 % |
| 0.175 | 28.6 | 24.2 | 15 % |
| 0.2 | 27.6 | 22.9 | 17 % |

## Figure 4 protocol (PRMSE %, mean over 10 seeds)

Mean over the 14 sizes N = 20..200 per phi:

| phi | pc6 | pc8 | improvement |
|---|---:|---:|---:|
| 0.025 | 1.62 | 1.35 | 17 % |
| 0.05 | 2.77 | 2.27 | 18 % |
| 0.075 | 3.63 | 2.97 | 18 % |
| 0.1 | 4.51 | 3.68 | 18 % |
| 0.125 | 5.20 | 4.23 | 19 % |
| 0.15 | 5.87 | 4.80 | 18 % |
| 0.175 | 6.41 | 5.28 | 18 % |
| 0.2 | 7.33 | 6.07 | 17 % |

Per size at phi = 0.1:

| N | pc6 | pc8 | improvement |
|---|---:|---:|---:|
| 20 | 3.06 | 2.67 | 13 % |
| 30 | 3.13 | 2.44 | 22 % |
| 40 | 3.66 | 2.97 | 19 % |
| 50 | 3.94 | 3.27 | 17 % |
| 60 | 4.10 | 3.47 | 15 % |
| 70 | 4.22 | 3.38 | 20 % |
| 80 | 4.26 | 3.43 | 20 % |
| 90 | 4.02 | 3.13 | 22 % |
| 100 | 5.55 | 4.60 | 17 % |
| 120 | 5.09 | 4.07 | 20 % |
| 140 | 5.06 | 4.14 | 18 % |
| 160 | 5.77 | 4.73 | 18 % |
| 180 | 6.04 | 4.87 | 19 % |
| 200 | 5.23 | 4.29 | 18 % |

Per size at phi = 0.2:

| N | pc6 | pc8 | improvement |
|---|---:|---:|---:|
| 20 | 4.74 | 4.42 | 7 % |
| 30 | 5.80 | 4.80 | 17 % |
| 40 | 6.46 | 5.32 | 18 % |
| 50 | 6.25 | 5.07 | 19 % |
| 60 | 6.01 | 4.97 | 17 % |
| 70 | 6.45 | 5.39 | 16 % |
| 80 | 7.44 | 5.95 | 20 % |
| 90 | 6.99 | 5.72 | 18 % |
| 100 | 8.85 | 7.33 | 17 % |
| 120 | 8.26 | 6.73 | 19 % |
| 140 | 8.28 | 6.81 | 18 % |
| 160 | 7.85 | 6.44 | 18 % |
| 180 | 10.05 | 8.22 | 18 % |
| 200 | 9.14 | 7.74 | 15 % |

## Clustered near-contact protocol (N = 10, PRMSE %)

| delta | NeMO n-body (paper) | moments v2 (pairs<=6) | moments v2 (pairs<=8) |
|---|---:|---:|---:|
| 0.1 | 3.30 | 2.65 | 2.56 |
| 0.2 | 2.47 | 1.82 | 1.46 |
| 0.5 | 1.51 | 1.19 | 0.76 |
| 1 | 0.97 | 0.58 | 0.50 |
| 2 | 0.43 | 0.32 | 0.24 |
| 3 | 0.40 | 0.24 | 0.21 |

## Exact-residual floor (uniform v2 boxes, P <= 64): what a *perfect* pairwise correction leaves

e_near from `experiments/nbody_v2_ceiling.py` on the two caches -- the model-independent floor of the
design. Raising the cutoff moves pairs from the never-corrected far field into the corrected set:

| phi (uniform) | e_2b | floor, pairs<=6 | floor, pairs<=8 |
|---|---:|---:|---:|
| 0.025 | 2.22 | 1.23 | 0.87 |
| 0.05 | 3.97 | 1.86 | 1.19 |
| 0.075 | 5.96 | 2.35 | 1.48 |
| 0.1 | 8.27 | 2.86 | 1.70 |
| 0.15 | 11.22 | 3.31 | 2.12 |
| 0.2 | 15.58 | 4.08 | 2.69 |
| 0.25 | 20.04 | 4.90 | 3.45 |

Caveat: v2 boxes are small (half-box ~5.5-9), so the d > 8 far field is a smaller share there than
at N = 200/300; the Fig-4 N-sweep above is the operative measurement of the remaining far-field term.

## Dataset-v2 validation (PRMSE % lin / ang)

| model | prmse_lin | prmse_ang | residual capture % |
|---|---:|---:|---:|
| 2-body only (pc6 rows) | 15.92 | 37.51 | -- |
| pc6 (`nbody_moments_v2_kinf_rc8`) | 3.98 | 15.42 | 28.0 |
| 2-body only (pc8 rows) | 15.03 | 37.15 | -- |
| pc8 (`nbody_moments_v2_kinf_rc8_pc8`) | 3.89 | 15.66 | 28.9 |

These validation sets differ (the pc8 rows include the easier 6-8 shell), so pc6-vs-pc8 is NOT
apples-to-apples here -- the paper protocols above, on identical truths, are the comparison.

## Models

- `nbody_moments_v2_kinf_rc8.pt`: max_neighbors=None, neighbor_cutoff=8.0, pair_cutoff=6.0 (run with switch_dist=max(6, pair_cutoff)); trained 2026-08-30 13:57:28.
- `nbody_moments_v2_kinf_rc8_pc8.pt`: max_neighbors=None, neighbor_cutoff=8.0, pair_cutoff=8.0 (run with switch_dist=max(6, pair_cutoff)); trained 2026-08-30 18:42:58.
