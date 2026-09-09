# FTS reflection ladder (N = 200, fig4g configurations, fine BatchedMFS grand mobility)

Mean over seeds of each metric (%), per forcing and volume fraction. Rungs are cumulative; see the
docstring of `benchmarks/fts_reflection_ladder.py` for their definitions.

## prmse_lin

**gravity** | φ=0.1 (n=3) | φ=0.2 (n=3)
|---|---|---|
| rpy | 3.95 | 7.72 |
| 2b | 3.49 | 6.97 |
| near_nn | 1.72 | 3.14 |
| stack_nn | 1.73 | 3.19 |
| near_ex | 1.28 | 2.28 |
| diag_ex | 1.27 | 2.25 |
| far_ex | 0.00 | 0.00 |
| rpy+ref1 | 0.30 | 0.55 |
| rpy+ref2 | 0.38 | 1.20 |
| rpy+full | 0.26 | 0.75 |
| sd_minf | 0.26 | 0.75 |
| stack+ref1 | 2.68 | 5.42 |
| stack+ref2 | 2.40 | 4.36 |
| stack+full | 2.52 | 4.77 |
| stack+ref1_uncov | 0.52 | 0.97 |
| pnear+ref1_far | 0.26 | 0.54 |
| pnear+full_far | 0.15 | 0.40 |

**random** | φ=0.1 (n=3) | φ=0.2 (n=3)
|---|---|---|
| rpy | 10.38 | 22.74 |
| 2b | 8.12 | 19.41 |
| near_nn | 3.36 | 7.82 |
| stack_nn | 3.35 | 7.71 |
| near_ex | 2.86 | 6.29 |
| diag_ex | 2.81 | 6.07 |
| far_ex | 0.00 | 0.00 |
| rpy+ref1 | 2.33 | 5.35 |
| rpy+ref2 | 2.47 | 6.54 |
| rpy+full | 2.15 | 5.15 |
| sd_minf | 2.15 | 5.15 |
| stack+ref1 | 8.37 | 18.99 |
| stack+ref2 | 7.32 | 14.67 |
| stack+full | 7.72 | 16.29 |
| stack+ref1_uncov | 1.89 | 4.43 |
| pnear+ref1_far | 0.62 | 1.51 |
| pnear+full_far | 0.34 | 1.01 |

## prmse_fluct

**gravity** | φ=0.1 (n=3) | φ=0.2 (n=3)
|---|---|---|
| rpy | 20.90 | 48.97 |
| 2b | 19.90 | 47.12 |
| near_nn | 13.23 | 29.61 |
| stack_nn | 13.27 | 29.90 |
| near_ex | 10.14 | 17.41 |
| diag_ex | 10.03 | 17.33 |
| far_ex | 0.00 | 0.00 |
| rpy+ref1 | 2.41 | 5.37 |
| rpy+ref2 | 2.42 | 8.54 |
| rpy+full | 1.78 | 5.61 |
| sd_minf | 1.78 | 5.61 |
| stack+ref1 | 10.10 | 23.67 |
| stack+ref2 | 8.61 | 16.55 |
| stack+full | 9.18 | 19.07 |
| stack+ref1_uncov | 3.18 | 9.44 |
| pnear+ref1_far | 2.07 | 4.66 |
| pnear+full_far | 1.10 | 2.47 |

**random** | φ=0.1 (n=3) | φ=0.2 (n=3)
|---|---|---|
| rpy | 20.43 | 46.19 |
| 2b | 15.86 | 39.55 |
| near_nn | 6.66 | 15.72 |
| stack_nn | 6.63 | 15.49 |
| near_ex | 5.58 | 12.12 |
| diag_ex | 5.45 | 11.67 |
| far_ex | 0.00 | 0.01 |
| rpy+ref1 | 4.73 | 11.00 |
| rpy+ref2 | 5.03 | 13.46 |
| rpy+full | 4.37 | 10.52 |
| sd_minf | 4.37 | 10.52 |
| stack+ref1 | 16.48 | 39.02 |
| stack+ref2 | 14.42 | 29.65 |
| stack+full | 15.21 | 33.12 |
| stack+ref1_uncov | 3.78 | 8.74 |
| pnear+ref1_far | 1.23 | 3.02 |
| pnear+full_far | 0.69 | 1.97 |

## err_mean_pct

**gravity** | φ=0.1 (n=3) | φ=0.2 (n=3)
|---|---|---|
| rpy | 3.04 | 5.97 |
| 2b | 2.53 | 5.13 |
| near_nn | 0.61 | 1.01 |
| stack_nn | 0.62 | 1.09 |
| near_ex | 0.34 | 1.47 |
| diag_ex | 0.37 | 1.43 |
| far_ex | 0.00 | 0.00 |
| rpy+ref1 | 0.03 | 0.08 |
| rpy+ref2 | 0.24 | 0.84 |
| rpy+full | 0.15 | 0.49 |
| sd_minf | 0.15 | 0.49 |
| stack+ref1 | 2.40 | 4.90 |
| stack+ref2 | 2.18 | 4.05 |
| stack+full | 2.27 | 4.39 |
| stack+ref1_uncov | 0.35 | 0.19 |
| pnear+ref1_far | 0.05 | 0.27 |
| pnear+full_far | 0.08 | 0.31 |

**random** | φ=0.1 (n=3) | φ=0.2 (n=3)
|---|---|---|
| rpy | 2.94 | 5.09 |
| 2b | 2.51 | 4.47 |
| near_nn | 1.19 | 1.84 |
| stack_nn | 1.20 | 1.87 |
| near_ex | 1.15 | 1.89 |
| diag_ex | 1.17 | 1.87 |
| far_ex | 0.00 | 0.00 |
| rpy+ref1 | 0.19 | 0.62 |
| rpy+ref2 | 0.27 | 0.80 |
| rpy+full | 0.19 | 0.65 |
| sd_minf | 0.19 | 0.65 |
| stack+ref1 | 2.15 | 3.68 |
| stack+ref2 | 1.87 | 3.23 |
| stack+full | 1.95 | 3.41 |
| stack+ref1_uncov | 0.32 | 1.43 |
| pnear+ref1_far | 0.22 | 0.35 |
| pnear+full_far | 0.11 | 0.26 |

## prmse_ang

**gravity** | φ=0.1 (n=3) | φ=0.2 (n=3)
|---|---|---|
| rpy | 21.79 | 46.70 |
| 2b | 21.19 | 45.70 |
| near_nn | 14.21 | 25.37 |
| stack_nn | 14.29 | 25.63 |
| near_ex | 9.83 | 11.52 |
| diag_ex | 9.83 | 11.64 |
| far_ex | 0.01 | 0.01 |
| rpy+ref1 | 3.86 | 8.82 |
| rpy+ref2 | 2.49 | 7.58 |
| rpy+full | 1.91 | 4.85 |
| sd_minf | 1.91 | 4.85 |
| stack+ref1 | 11.44 | 27.28 |
| stack+ref2 | 8.56 | 16.90 |
| stack+full | 9.23 | 19.92 |
| stack+ref1_uncov | 5.16 | 13.25 |
| pnear+ref1_far | 2.34 | 4.33 |
| pnear+full_far | 1.03 | 2.23 |

**random** | φ=0.1 (n=3) | φ=0.2 (n=3)
|---|---|---|
| rpy | 9.60 | 22.07 |
| 2b | 7.36 | 17.00 |
| near_nn | 3.27 | 6.39 |
| stack_nn | 3.16 | 5.86 |
| near_ex | 1.68 | 3.57 |
| diag_ex | 1.47 | 2.21 |
| far_ex | 0.00 | 0.01 |
| rpy+ref1 | 3.69 | 8.25 |
| rpy+ref2 | 3.73 | 9.06 |
| rpy+full | 3.56 | 8.06 |
| sd_minf | 3.56 | 8.06 |
| stack+ref1 | 7.34 | 18.83 |
| stack+ref2 | 6.34 | 14.05 |
| stack+full | 6.65 | 15.60 |
| stack+ref1_uncov | 2.83 | 5.39 |
| pnear+ref1_far | 0.52 | 1.14 |
| pnear+full_far | 0.24 | 0.58 |

## truth gap ‖MF − v_file‖/‖v_file‖ (fine M vs the cached truth files)

                mean      max
forcing phi                  
gravity 0.1 2.70e-05 3.09e-05
        0.2 3.68e-06 4.31e-06
random  0.1 3.55e-04 4.12e-04
        0.2 6.56e-04 8.33e-04
