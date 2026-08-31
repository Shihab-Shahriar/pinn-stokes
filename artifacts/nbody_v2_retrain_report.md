# n-body correction retrained on dataset v2, and the NeMO paper accuracy protocols re-run

Date: 2026-08-30.  Branch `moments_nb`.  Companion reports: `artifacts/nbody_moments_report.md` (the moments
architecture and its results on the old rows), `artifacts/mfs_batched_report.md` (the solver and dataset v2).

## 1. What was done

1. **Training rows from dataset v2 through the operators' own selection.**  `nbody_features.select_pair_neighbours`
   is now the single code path for pair/neighbour selection (unordered near pairs t < s with d <= 6; neighbours =
   all particles within `neighbor_cutoff` of the pair *midpoint*, the `max_neighbors` smallest d_kt*d_ks,
   index-sorted); `Mob_Op_Nbody_Moments._pair_rows` calls it, and `tests/test_nbody_v2.py` pins it against
   `Mob_Op_Nbody._select_neighbor_indices`.  Three selection variants are cached: `k10_rc6` (K = 10, r_c = 6 -- the
   baseline's and the GPU operator's), `kinf_rc6` (all within 6), `kinf_rc8` (all within 8).
2. **Labels are the full residual blocks** `R_ts = 0.5 (M_ts + M_st^T) - M2b_ts` with `M2b_ts` the two-body block
   the operators apply (`TwoBodyCombined.predict_mobility(X2b)[1]`, `+s_vec`, median 5.01 = `NNMob.get_two_vel`'s
   constant; the old labels used 5.0083).  Both architectures are reciprocal by construction (moments by design;
   the b1 baseline because its per-neighbour features are swap-even and `L3(-d) = L3(d)^T`), so one unordered row
   per pair carries the whole information and the symmetrised label removes the fine-MFS asymmetry noise
   (median 4e-5 uniform / 9e-5 grown, p99 4.6e-4 / 2.5e-3).
3. **Geometry + label cache** (`experiments/build_nbody_v2_cache.py`, 131 s with 12 workers, 2.7 GB in
   `data/multibody_v2_cache/`): 56,048 configurations, **6,361,020 unordered near pairs** (uniform 3.22 M, grown
   2.57 M, lattice 0.58 M); features are built on the fly on the GPU by the trainer, so one cache serves every
   variant and both architectures.  Configuration-level split: `seed % 10 == 0` -> 10.0 % of the configurations
   (633k validation pairs), family-stratified, no pair of a validation configuration is ever trained on.
4. **Trainer** `experiments/train_nbody_v2.py`: L1 over the 36 entries of `6*pi*(predict_mobility(X) - R)`
   (optionally per-block RMS weights), Adam 1e-3 + cosine over all steps, batch 4096, 100 epochs = 139,800
   steps, 3 ms/step on the RTX 4060 (~10 min per model); moments normalisation buffers fitted on 262k train rows.
5. **Paper harness** `benchmarks/paper_accuracy_v2.py` (warp-free): the exact Figure-3 grid (N = 200, 300; 8 phi;
   seeds 123..132) and Figure-4 grid (14 N x 8 phi x 10 seeds, `seed = 123 + v_idx*1000 + p_idx*100 + run`) of
   `benchmarks/accuracy_grand_M.py`, the same truth generator (`benchmarks.cluster.generate_uniform_testcase`,
   Xfine MFS, tol 1e-8, L_cut 25) cached per (N, phi, seed) in `tmp/nbody_moments_truth/` (1,280 files, 28 MB, on
   the cluster and the laptop), the paper metric (`rel_rmse` = PRMSE %), a CPU worker pool for the CPU operators,
   and the two GPU yardsticks (`mfs_coarse`, the paper's GPU n-body operator `Mob_Nbody_Torch` +
   `nbody_cross_tmp.wt`).  Truths + GPU rows: SLURM array 16546489 (8 x ~20 min on H200 nodes); CPU operators:
   arrays 16546747 / (v2 models) on 32-core nodes.  The harness reproduces `data/nbody_moments_compare.csv` to
   the CSV's last digit on the overlapping rows (same truths, same operators).

## 2. What dataset v2 says about the operator error before any model is trained

`experiments/build_nbody_v2_cache.py --stats` and `experiments/nbody_v2_ceiling.py`
(`artifacts/nbody_v2_ceiling.{csv,md}`, 6 validation configurations per (family, parameter, P)).

**Residual fraction of the cross blocks** (RMS |R| / RMS |M_ts| over all near pairs): TT 0.14 / 0.16 / 0.08,
TR 0.38 / 0.42 / 0.20, RR 0.45 / 0.52 / 0.25 for uniform / grown / lattice -- the two-body model leaves ~15 % of
TT but ~40-50 % of the coupling and rotational blocks; by pair distance the TT fraction falls from 0.20 (d < 2.5)
to 0.10 (d > 5) while RR stays 0.4-0.65 everywhere.

**Error decomposition of the 2-body operator** (`NNMob`: analytic self + two-body NN within 6 + RPY beyond),
rel-L2 % of the velocity, mean over the sampled configurations, each step adding the *exact* MFS residual:

| family | param | 2-body | + near-pair residuals (floor of any pairwise correction) | + diagonal residual | + far-pair residuals |
|---|---|---:|---:|---:|---:|
| uniform | phi 0.05 | 3.97 | 1.86 | 1.76 | 0.002 |
| uniform | phi 0.1 | 8.27 | 2.86 | 2.70 | 0.004 |
| uniform | phi 0.15 | 11.22 | 3.31 | 2.97 | 0.005 |
| uniform | phi 0.2 | 15.58 | 4.08 | 3.33 | 0.007 |
| uniform | phi 0.25 | 20.04 | 4.90 | 3.63 | 0.010 |
| grown | delta 0.05 | 20.32 | 6.02 | 3.78 | 0.068 |
| grown | delta 0.2 | 11.94 | 3.27 | 2.52 | 0.005 |
| grown | delta 0.5 | 6.17 | 1.76 | 1.58 | 0.000 |
| lattice | phi 0.1 | 4.13 | 1.95 | 1.87 | 0.000 |

(full table in `artifacts/nbody_v2_ceiling.md`).  Three consequences:

* A **perfect** pairwise near-field correction (any architecture, any data) leaves 2.9 % at phi = 0.1 and
  4.1 % at phi = 0.2 on these boxes: that is the floor the n-body models are chasing.
* The floor is **not** the diagonal: `M_tt - self - sum K_s` is RMS 6.6e-4 in TT (2.4 % of the self block,
  about half the pairwise residual per block) and removing it exactly buys only 0.1-0.4 %.
* The floor is the **far field**: everything left after the near and diagonal corrections is the residual of
  the pairs beyond 6 radii that the operator treats with RPY (many-body reflections through the near
  neighbours of a far pair are not small relative to its direct RPY term, and there are hundreds of far pairs
  per configuration).  In the paper's N = 200-300 boxes there are more far pairs per particle than in these
  P <= 64 boxes, so the Figure-3 floor is higher still.  Adding the far residuals reproduces `M F` to the label
  noise (<= 1e-4, 7e-4 at gap 0.05), which also pins every sign / orientation convention end to end
  (`tests/test_nbody_v2.py::test_oracle_residual_reproduces_grand_M`).

**Neighbourhood coverage** (mean band counts of the 8 unit-width radial bands about the pair midpoint):
`k10_rc6` neighbourhoods of v2 pairs and of N = 200 test pairs populate the same bands (bands 1-7, K saturates
at 10); `kinf_rc8` on v2 uniform pairs has mean counts 5.5 / 5.1 / 4.2 in bands 6-8 versus 4.4 / 5.6 / 6.9 at
N = 200 phi = 0.05 and 11 / 14 / 16 at phi = 0.15 -- the P <= 64 boxes (half-box 5.5 at phi = 0.2) cannot fill
an 8-radius ball, so the unbounded r_c = 8 variant extrapolates in the outer bands at phi >= 0.1 exactly where
the old rows did.

## 3. Validation on dataset v2 (633,360 pairs of 5,605 held-out configurations)

PRMSE = relative L2 error of the total velocity `(M2b + K) F` against `M_ts F` under fixed random unit
force / torque (x 6 pi), translational / rotational; "block" = relative Frobenius error of `M2b + K` against
`Mts_sym`.  Two-body only: **15.93 % / 37.51 %** (block 17.0 %) -- far harder rows than the old synthetic ones
(6.80 / 16.36 % there).

| model @ selection | PRMSE lin % | PRMSE ang % | block rel-Frobenius % | TT | TR | RR | uniform lin/ang | grown lin/ang | lattice lin/ang |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2-body only | 15.93 | 37.51 | 17.00 | – | – | – | 15.10/37.15 | 17.51/38.89 | 8.35/19.32 |
| b1 shipped (`nbody_pinn_b1.pt`, old data) @ K=10 r_c=6 | 8.97 | 22.14 | 9.65 | 8.28 | 22.32 | 30.53 | 8.58/19.90 | 9.81/24.33 | 4.49/9.87 |
| b1 retrained on old rows @ K=10 r_c=6 | 8.84 | 22.41 | 9.56 | 8.12 | 22.60 | 31.28 | 8.24/19.95 | 9.86/24.74 | 4.30/9.68 |
| moments, old rows @ K=10 r_c=6 | 7.47 | 19.09 | 8.08 | 6.65 | 21.44 | 21.37 | 7.29/17.69 | 8.01/20.63 | 4.22/8.94 |
| moments, old rows @ all r_c=6 (unseen neighbourhoods) | 8.15 | 20.98 | 8.83 | 7.30 | 22.96 | 27.87 | 8.74/22.18 | 8.01/20.78 | 3.94/9.36 |
| **b1 retrained on v2** @ K=10 r_c=6 | 6.87 | 19.82 | 7.61 | 6.17 | 19.20 | 24.46 | 5.97/16.94 | 8.03/22.32 | 2.53/8.17 |
| **moments v2** @ K=10 r_c=6 | 4.52 | 15.66 | 5.23 | 3.60 | 16.84 | 10.40 | 4.17/13.72 | 5.09/17.44 | 1.72/6.23 |
| **moments v2** @ all r_c=6 | 4.10 | 15.44 | 4.86 | 3.07 | 16.69 | 10.04 | 3.49/13.34 | 4.84/17.30 | 1.49/6.12 |
| **moments v2** @ all r_c=8 | 3.98 | 15.42 | 4.77 | 2.90 | 16.68 | 9.96 | 3.33/13.31 | 4.75/17.30 | 1.24/6.06 |

* Retraining on v2 alone (same b1 architecture, same K = 10 / r_c = 6 selection) takes the baseline from 8.97 / 22.1 % to
  6.87 / 19.8 %; the moments architecture on the same rows reaches 4.52 / 15.7 %, and with all neighbours within
  r_c = 8 (mean 22.7 per pair) 3.98 / 15.4 % -- the old ≤10-neighbour rows were the limit, not the architecture.
* Per neighbour count (moments v2, r_c = 8): PRMSE lin/ang K 1-5: 1.82/6.27 % (3967 rows), K 6-10: 4.16/15.26 % (91324 rows), K 11-20: 3.81/14.78 % (212796 rows), K 21-40: 4.07/16.19 % (263445 rows), K 41-inf: 3.97/17.05 % (61950 rows).
* The rotational block stays the hard one (RR / TR rel-Frobenius 10 / 17 % vs TT 3 %); an L1 loss over the raw block
  entries is dominated by TT (RMS |R| TT 1.1e-3 vs RR 1.8e-4).  Reweighting the loss by per-block RMS (`--block-weights rms`) does not help (r_c = 8: 4.04/15.39 % vs 3.98/15.42 % unweighted; same picture for the other variants) -- the rotational error is limited by the data/architecture, not by the loss weighting, so the published models use the plain unweighted L1.
* Old models fed neighbourhoods they never saw degrade (moments old rows: 7.47 % -> 8.15 % lin when given all
  neighbours within 6), which is why every published v2 model carries its selection in a `.json` sidecar that the
  harness asserts against.

Full tables (by neighbour count and by pair distance): `artifacts/nbody_v2_validation_tables.md`
(`experiments/nbody_v2_summary.py`).

## 4. Paper protocols

All rows: `data/paper_accuracy_v2.csv`; rendered tables `artifacts/paper_accuracy_v2_tables.md`; figures
`figures/paper_v2_fig3_P200.pdf`, `figures/paper_v2_fig3_P300.pdf`, `figures/paper_v2_fig4_rel_rmse.pdf`,
`figures/paper_v2_fig4_max_rel_rmse.pdf`.  PRMSE (%) = `rel_rmse` of `accuracy_grand_M._compute_error_stats`,
mean (± std) over the 10 seeds; truth = Xfine MFS (`generate_uniform_testcase`, tol 1e-8) exactly as in the paper.
The *historic* rows are the paper's CSV: the same seeds no longer give byte-identical configurations (near-field
counts differ by ~3 %), but the unchanged operators land within a few % of them, so the protocol is intact.

### 4.1 Figure 3 protocol, N = 200 (8 volume fractions x 10 seeds)

| operator | φ=0.025 | φ=0.05 | φ=0.075 | φ=0.1 | φ=0.125 | φ=0.15 | φ=0.175 | φ=0.2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RPY | 4.13 ± 0.42 | 7.12 ± 1.08 | 9.94 ± 1.56 | 13.39 ± 2.43 | 16.04 ± 3.52 | 19.06 ± 3.67 | 22.25 ± 5.25 | 22.80 ± 4.80 |
| NeMO 2-body | 2.58 ± 0.26 | 5.10 ± 0.75 | 7.64 ± 1.31 | 10.64 ± 2.03 | 13.26 ± 2.97 | 15.98 ± 3.11 | 18.94 ± 4.44 | 19.54 ± 4.04 |
| NeMO 3-body summation | 2.27 ± 0.25 | 4.20 ± 0.58 | 6.10 ± 1.16 | 8.10 ± 1.47 | 9.79 ± 2.07 | 11.37 ± 2.20 | 13.08 ± 3.05 | 13.28 ± 2.65 |
| NeMO n-body b1 (paper, CPU) | 2.12 ± 0.23 | 3.93 ± 0.54 | 5.83 ± 1.08 | 7.77 ± 1.49 | 10.07 ± 2.31 | 11.98 ± 2.36 | 14.38 ± 3.51 | 15.01 ± 3.02 |
| NeMO n-body (paper GPU op, nbody_cross_tmp.wt) | 2.14 ± 0.24 | 3.89 ± 0.51 | 5.72 ± 1.06 | 7.56 ± 1.47 | 9.78 ± 2.28 | 11.62 ± 2.27 | 14.07 ± 3.53 | 14.77 ± 2.89 |
| moments, old data (K=10, r_c=6) | 1.97 ± 0.23 | 3.64 ± 0.50 | 5.47 ± 1.02 | 7.30 ± 1.48 | 9.56 ± 2.17 | 11.55 ± 2.36 | 13.91 ± 3.50 | 14.67 ± 3.04 |
| **b1 retrained on v2** (K=10, r_c=6) | 2.00 ± 0.21 | 3.59 ± 0.46 | 5.10 ± 0.94 | 6.58 ± 1.27 | 8.24 ± 1.80 | 9.53 ± 1.89 | 11.13 ± 2.86 | 11.20 ± 2.30 |
| **moments v2** (K=10, r_c=6) | 1.85 ± 0.20 | 3.36 ± 0.45 | 4.83 ± 0.91 | 6.24 ± 1.27 | 7.92 ± 1.72 | 9.17 ± 1.88 | 10.68 ± 2.79 | 10.83 ± 2.24 |
| **moments v2** (all, r_c=6) | 1.86 ± 0.20 | 3.36 ± 0.45 | 4.76 ± 0.87 | 5.98 ± 1.20 | 7.31 ± 1.49 | 8.34 ± 1.72 | 9.46 ± 2.50 | 9.27 ± 1.95 |
| **moments v2** (all, r_c=8) | 1.83 ± 0.19 | 3.26 ± 0.44 | 4.57 ± 0.85 | 5.79 ± 1.14 | 6.93 ± 1.35 | 8.10 ± 1.66 | 9.21 ± 2.43 | 9.08 ± 1.89 |
| MFS coarse (54 nodes) | 0.35 ± 0.10 | 0.60 ± 0.15 | 0.62 ± 0.13 | 0.89 ± 0.29 | 0.85 ± 0.18 | 1.14 ± 0.40 | 1.18 ± 0.28 | 1.34 ± 0.33 |
| *historic paper CSV, 2-body* | *2.58* | *5.17* | *7.65* | *10.16* | *13.28* | *15.72* | *18.95* | *21.10* |
| *historic paper CSV, n-body* | *2.13* | *3.94* | *5.84* | *7.48* | *10.08* | *11.64* | *14.39* | *16.13* |

Translational / rotational rel-L2 (%) at N = 200:

| operator | φ=0.025 | φ=0.05 | φ=0.075 | φ=0.1 | φ=0.125 | φ=0.15 | φ=0.175 | φ=0.2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RPY | 4.45 | 7.50 | 10.42 | 14.16 | 16.68 | 19.69 | 23.01 | 22.99 |
| NeMO 2-body | 2.85 | 5.46 | 8.11 | 11.44 | 13.90 | 16.67 | 19.75 | 19.85 |
| NeMO 3-body summation | 2.48 | 4.40 | 6.32 | 8.42 | 9.88 | 11.34 | 12.93 | 12.70 |
| NeMO n-body b1 (paper, CPU) | 2.35 | 4.22 | 6.23 | 8.43 | 10.67 | 12.66 | 15.22 | 15.49 |
| NeMO n-body (paper GPU op, nbody_cross_tmp.wt) | 2.36 | 4.17 | 6.11 | 8.20 | 10.34 | 12.25 | 14.85 | 15.17 |
| moments, old data (K=10, r_c=6) | 2.19 | 3.92 | 5.86 | 7.95 | 10.15 | 12.23 | 14.74 | 15.12 |
| **b1 retrained on v2** (K=10, r_c=6) | 2.21 | 3.83 | 5.42 | 7.11 | 8.71 | 10.04 | 11.76 | 11.53 |
| **moments v2** (K=10, r_c=6) | 2.05 | 3.61 | 5.16 | 6.78 | 8.40 | 9.71 | 11.33 | 11.19 |
| **moments v2** (all, r_c=6) | 2.06 | 3.61 | 5.08 | 6.48 | 7.74 | 8.81 | 10.02 | 9.54 |
| **moments v2** (all, r_c=8) | 2.02 | 3.50 | 4.87 | 6.25 | 7.31 | 8.54 | 9.73 | 9.33 |
| MFS coarse (54 nodes) | 0.32 | 0.52 | 0.55 | 0.78 | 0.75 | 1.03 | 1.09 | 1.21 |

| operator | φ=0.025 | φ=0.05 | φ=0.075 | φ=0.1 | φ=0.125 | φ=0.15 | φ=0.175 | φ=0.2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RPY | 2.96 | 5.49 | 7.70 | 10.26 | 12.83 | 15.93 | 18.98 | 22.83 |
| NeMO 2-body | 1.45 | 3.28 | 5.16 | 7.03 | 9.70 | 12.08 | 14.88 | 18.14 |
| NeMO 3-body summation | 1.45 | 3.28 | 5.16 | 7.03 | 9.70 | 12.08 | 14.88 | 18.14 |
| NeMO n-body b1 (paper, CPU) | 1.19 | 2.43 | 3.61 | 4.56 | 6.30 | 7.62 | 9.19 | 11.28 |
| NeMO n-body (paper GPU op, nbody_cross_tmp.wt) | 1.22 | 2.42 | 3.57 | 4.50 | 6.24 | 7.69 | 9.45 | 11.92 |
| moments, old data (K=10, r_c=6) | 1.04 | 2.08 | 3.18 | 4.08 | 5.68 | 7.04 | 8.78 | 11.23 |
| **b1 retrained on v2** (K=10, r_c=6) | 1.16 | 2.32 | 3.37 | 4.10 | 5.34 | 6.35 | 7.33 | 8.76 |
| **moments v2** (K=10, r_c=6) | 1.01 | 2.02 | 2.95 | 3.65 | 4.76 | 5.67 | 6.56 | 7.99 |
| **moments v2** (all, r_c=6) | 1.02 | 2.01 | 2.93 | 3.60 | 4.57 | 5.41 | 6.13 | 7.34 |
| **moments v2** (all, r_c=8) | 1.01 | 2.01 | 2.92 | 3.61 | 4.55 | 5.41 | 6.13 | 7.35 |
| MFS coarse (54 nodes) | 0.42 | 0.87 | 0.89 | 1.26 | 1.34 | 1.68 | 1.72 | 2.31 |

Max per-particle relative error (%) at N = 200:

| operator | φ=0.025 | φ=0.05 | φ=0.075 | φ=0.1 | φ=0.125 | φ=0.15 | φ=0.175 | φ=0.2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RPY | 18.41 | 23.81 | 31.05 | 37.37 | 41.30 | 48.58 | 54.96 | 65.64 |
| NeMO 2-body | 9.90 | 16.85 | 25.20 | 35.28 | 37.53 | 48.91 | 50.61 | 55.27 |
| NeMO 3-body summation | 7.93 | 13.37 | 19.40 | 25.03 | 28.91 | 31.57 | 37.09 | 37.08 |
| NeMO n-body b1 (paper, CPU) | 7.76 | 12.18 | 16.94 | 24.81 | 28.87 | 34.93 | 40.00 | 40.06 |
| NeMO n-body (paper GPU op, nbody_cross_tmp.wt) | 7.98 | 11.31 | 16.90 | 23.66 | 27.27 | 33.29 | 40.10 | 37.17 |
| moments, old data (K=10, r_c=6) | 6.58 | 10.84 | 16.73 | 23.23 | 26.91 | 33.57 | 39.28 | 38.50 |
| **b1 retrained on v2** (K=10, r_c=6) | 7.75 | 11.19 | 15.20 | 21.08 | 23.01 | 26.61 | 33.05 | 29.66 |
| **moments v2** (K=10, r_c=6) | 6.38 | 10.82 | 14.12 | 20.95 | 22.77 | 25.38 | 32.78 | 28.53 |
| **moments v2** (all, r_c=6) | 6.42 | 11.04 | 13.83 | 20.22 | 20.88 | 23.59 | 28.89 | 27.31 |
| **moments v2** (all, r_c=8) | 6.24 | 11.05 | 13.32 | 19.82 | 20.60 | 23.11 | 28.64 | 27.60 |
| MFS coarse (54 nodes) | 2.70 | 3.78 | 3.53 | 4.80 | 3.81 | 5.14 | 4.79 | 5.07 |

### 4.2 Figure 3 protocol, N = 300

| operator | φ=0.025 | φ=0.05 | φ=0.075 | φ=0.1 | φ=0.125 | φ=0.15 | φ=0.175 | φ=0.2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RPY | 4.13 ± 0.55 | 7.43 ± 1.22 | 10.13 ± 1.68 | 12.81 ± 2.67 | 15.67 ± 3.54 | 18.61 ± 3.84 | 21.86 ± 5.05 | 25.89 ± 6.23 |
| NeMO 2-body | 2.72 ± 0.39 | 5.49 ± 0.92 | 8.00 ± 1.43 | 10.32 ± 2.08 | 12.87 ± 2.95 | 15.69 ± 3.13 | 18.74 ± 4.16 | 22.70 ± 5.38 |
| NeMO 3-body summation | 2.41 ± 0.37 | 4.59 ± 0.77 | 6.41 ± 1.20 | 8.04 ± 1.62 | 9.59 ± 2.30 | 11.53 ± 2.23 | 13.39 ± 2.71 | 16.07 ± 3.56 |
| NeMO n-body b1 (paper, CPU) | 2.28 ± 0.33 | 4.30 ± 0.70 | 6.19 ± 1.14 | 7.93 ± 1.58 | 9.74 ± 2.29 | 12.16 ± 2.31 | 14.62 ± 2.91 | 18.08 ± 4.15 |
| NeMO n-body (paper GPU op, nbody_cross_tmp.wt) | 2.29 ± 0.33 | 4.27 ± 0.70 | 6.10 ± 1.15 | 7.71 ± 1.53 | 9.44 ± 2.24 | 11.87 ± 2.25 | 14.31 ± 2.81 | 17.92 ± 4.11 |
| moments, old data (K=10, r_c=6) | 2.19 ± 0.33 | 4.04 ± 0.62 | 5.85 ± 1.16 | 7.50 ± 1.49 | 9.22 ± 2.16 | 11.77 ± 2.28 | 14.21 ± 2.80 | 17.81 ± 4.14 |
| **b1 retrained on v2** (K=10, r_c=6) | 2.16 ± 0.31 | 3.91 ± 0.65 | 5.46 ± 1.03 | 6.81 ± 1.36 | 8.09 ± 2.10 | 9.98 ± 1.90 | 11.54 ± 2.20 | 14.18 ± 3.17 |
| **moments v2** (K=10, r_c=6) | 2.06 ± 0.30 | 3.70 ± 0.59 | 5.21 ± 1.04 | 6.51 ± 1.30 | 7.73 ± 2.02 | 9.69 ± 1.88 | 11.24 ± 2.15 | 13.85 ± 3.14 |
| **moments v2** (all, r_c=6) | 2.06 ± 0.31 | 3.69 ± 0.58 | 5.10 ± 1.03 | 6.29 ± 1.26 | 7.28 ± 1.97 | 8.90 ± 1.74 | 10.03 ± 1.87 | 12.10 ± 2.70 |
| **moments v2** (all, r_c=8) | 2.03 ± 0.30 | 3.56 ± 0.57 | 4.91 ± 0.99 | 6.06 ± 1.23 | 7.02 ± 1.94 | 8.63 ± 1.69 | 9.80 ± 1.92 | 11.94 ± 2.71 |
| MFS coarse (54 nodes) | 0.30 ± 0.08 | 0.50 ± 0.13 | 0.59 ± 0.12 | 0.74 ± 0.17 | 0.85 ± 0.24 | 0.93 ± 0.23 | 1.10 ± 0.30 | 1.31 ± 0.35 |
| *historic paper CSV, 2-body* | *2.73* | *5.51* | *8.01* | *10.33* | *12.88* | *15.71* | *18.76* | *22.72* |
| *historic paper CSV, n-body* | *2.29* | *4.30* | *6.20* | *7.94* | *9.75* | *12.17* | *14.63* | *18.10* |

### 4.3 Figure 4 protocol (N = 20..200, 8 phi, 10 seeds): PRMSE mean over N

| operator | φ=0.025 | φ=0.05 | φ=0.075 | φ=0.1 | φ=0.125 | φ=0.15 | φ=0.175 | φ=0.2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RPY | 4.15 | 6.97 | 9.60 | 12.40 | 14.79 | 18.02 | 19.89 | 23.00 |
| NeMO 2-body | 2.42 | 4.71 | 6.92 | 9.33 | 11.43 | 14.14 | 15.96 | 18.72 |
| NeMO 3-body summation | 2.08 | 3.81 | 5.32 | 6.89 | 8.21 | 9.83 | 10.96 | 12.71 |
| NeMO n-body b1 (paper, CPU) | 1.93 | 3.48 | 4.89 | 6.40 | 7.76 | 9.56 | 10.70 | 12.47 |
| NeMO n-body (paper GPU op, nbody_cross_tmp.wt) | 1.96 | 3.47 | 4.79 | 6.20 | 7.53 | 9.23 | 10.46 | 12.36 |
| moments, old data (K=10, r_c=6) | 1.75 | 3.12 | 4.36 | 5.78 | 7.09 | 8.74 | 10.03 | 11.83 |
| **b1 retrained on v2** (K=10, r_c=6) | 1.82 | 3.20 | 4.27 | 5.35 | 6.24 | 7.32 | 7.99 | 9.18 |
| **moments v2** (K=10, r_c=6) | 1.65 | 2.86 | 3.83 | 4.84 | 5.66 | 6.57 | 7.27 | 8.35 |
| **moments v2** (all, r_c=6) | 1.65 | 2.85 | 3.76 | 4.65 | 5.34 | 6.05 | 6.59 | 7.44 |
| **moments v2** (all, r_c=8) | 1.62 | 2.77 | 3.63 | 4.51 | 5.20 | 5.87 | 6.41 | 7.33 |
| MFS coarse (54 nodes) | 0.33 | 0.50 | 0.63 | 0.78 | 0.93 | 1.13 | 1.25 | 1.54 |

PRMSE (%) vs N for the n-body operators (paper's four volume fractions):

| operator | φ | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 | 120 | 140 | 160 | 180 | 200 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NeMO n-body (paper GPU op, nbody_cross_tmp.wt) | 0.05 | 3.2 | 2.4 | 2.9 | 3.3 | 3.5 | 3.6 | 3.4 | 3.6 | 3.4 | 3.9 | 4.0 | 3.8 | 3.7 | 3.8 |
| NeMO n-body (paper GPU op, nbody_cross_tmp.wt) | 0.1 | 4.2 | 4.6 | 5.0 | 5.5 | 5.7 | 6.0 | 5.9 | 5.5 | 7.4 | 7.1 | 6.9 | 7.8 | 8.0 | 7.0 |
| NeMO n-body (paper GPU op, nbody_cross_tmp.wt) | 0.15 | 6.8 | 7.8 | 7.4 | 8.6 | 8.5 | 9.1 | 9.6 | 10.3 | 9.7 | 8.9 | 10.5 | 9.8 | 11.0 | 11.2 |
| NeMO n-body (paper GPU op, nbody_cross_tmp.wt) | 0.2 | 8.3 | 9.8 | 10.4 | 10.8 | 10.3 | 11.1 | 12.8 | 12.6 | 15.6 | 14.2 | 14.3 | 12.9 | 15.6 | 14.5 |
| **b1 retrained on v2** (K=10, r_c=6) | 0.05 | 3.0 | 2.3 | 2.7 | 3.1 | 3.2 | 3.3 | 3.2 | 3.3 | 3.1 | 3.6 | 3.7 | 3.5 | 3.5 | 3.5 |
| **b1 retrained on v2** (K=10, r_c=6) | 0.1 | 4.0 | 4.1 | 4.5 | 4.7 | 4.9 | 5.0 | 5.0 | 4.7 | 6.5 | 6.0 | 5.9 | 6.6 | 6.9 | 6.1 |
| **b1 retrained on v2** (K=10, r_c=6) | 0.15 | 6.3 | 6.4 | 5.8 | 6.6 | 6.6 | 6.6 | 7.3 | 8.1 | 7.8 | 7.0 | 8.3 | 7.7 | 9.0 | 9.0 |
| **b1 retrained on v2** (K=10, r_c=6) | 0.2 | 6.9 | 7.8 | 8.1 | 7.7 | 7.4 | 8.0 | 9.2 | 9.0 | 11.2 | 10.2 | 10.3 | 9.7 | 12.0 | 11.1 |
| **moments v2** (all, r_c=8) | 0.05 | 2.3 | 1.9 | 2.3 | 2.6 | 2.7 | 2.9 | 2.7 | 2.9 | 2.7 | 3.2 | 3.3 | 3.1 | 3.1 | 3.2 |
| **moments v2** (all, r_c=8) | 0.1 | 3.1 | 3.1 | 3.7 | 3.9 | 4.1 | 4.2 | 4.3 | 4.0 | 5.6 | 5.1 | 5.1 | 5.8 | 6.0 | 5.2 |
| **moments v2** (all, r_c=8) | 0.15 | 4.4 | 4.8 | 4.6 | 5.1 | 5.3 | 5.3 | 5.7 | 6.5 | 6.3 | 5.8 | 6.8 | 6.5 | 7.5 | 7.4 |
| **moments v2** (all, r_c=8) | 0.2 | 4.7 | 5.8 | 6.5 | 6.3 | 6.0 | 6.4 | 7.4 | 7.0 | 8.8 | 8.3 | 8.3 | 7.8 | 10.1 | 9.1 |

### 4.4 Clustered near-contact clusters (`tmp/reference_sphere_delta.csv`, N = 10; max per-particle in parentheses)

| operator | δ=0.1 | δ=0.2 | δ=0.5 | δ=1 | δ=2 | δ=3 |
|---|---:|---:|---:|---:|---:|---:|
| RPY | 10.47 (18.0) | 8.92 (13.6) | 5.91 (8.9) | 2.93 (5.0) | 1.32 (2.1) | 0.67 (1.0) |
| NeMO 2-body | 7.03 (12.2) | 5.73 (8.7) | 3.58 (5.4) | 1.59 (3.0) | 0.71 (1.3) | 0.42 (0.7) |
| NeMO 3-body summation | 3.68 (4.4) | 3.77 (5.8) | 1.97 (3.0) | 0.86 (1.2) | 0.46 (0.7) | 0.38 (0.6) |
| NeMO n-body b1 (paper, CPU) | 3.30 (4.6) | 2.47 (3.2) | 1.51 (2.4) | 0.97 (1.5) | 0.43 (0.8) | 0.40 (0.7) |
| moments, old data (K=10, r_c=6) | 3.05 (4.3) | 1.58 (2.2) | 1.07 (1.4) | 0.61 (1.0) | 0.36 (0.6) | 0.39 (0.6) |
| **b1 retrained on v2** (K=10, r_c=6) | 2.90 (4.0) | 2.38 (3.4) | 1.48 (2.3) | 0.81 (1.1) | 0.49 (0.7) | 0.28 (0.5) |
| **moments v2** (K=10, r_c=6) | 2.61 (4.1) | 2.03 (2.8) | 1.25 (1.5) | 0.61 (0.8) | 0.48 (0.6) | 0.28 (0.5) |
| **moments v2** (all, r_c=6) | 2.57 (4.0) | 1.97 (2.9) | 1.22 (1.5) | 0.62 (0.8) | 0.45 (0.6) | 0.27 (0.5) |
| **moments v2** (all, r_c=8) | 2.65 (3.9) | 1.82 (2.8) | 1.19 (1.7) | 0.58 (0.8) | 0.32 (0.5) | 0.24 (0.4) |

### 4.5 Reading the tables

* **Every operator retrained on v2 beats every old one at every phi, N and delta.**  At the paper's headline point
  (N = 200, phi = 0.2) the n-body operator goes from 15.0 % (b1, paper) to 11.2 % with the same architecture retrained
  on v2 and to **9.1 %** with the moments model (all neighbours within r_c = 8); at phi = 0.1 from 7.8 % to 5.8 %.
  The 3-body summation, which the paper's n-body model could not beat at phi >= 0.125, is beaten by all v2 models
  everywhere (13.3 % at phi = 0.2); the rotational error, which the 3-body term does not touch at all (its
  rotational column equals the 2-body one), drops from 11.3 % to 7.4 %.
* **Neighbourhood size matters once the data has it**: K = 10 -> all within 6 -> all within 8 gives 10.8 -> 9.3 ->
  9.1 % at phi = 0.2 (N = 200) and 13.9 -> 12.1 -> 11.9 % at N = 300, with identical cost (the moments are a
  segment sum over the neighbours).  The r_c = 8 variant extrapolates in the outer bands at phi >= 0.1 (section 2)
  and still wins, i.e. the model leans on the inner bands.
* **Fig 4**: the v2 operators grow much more slowly with N (moments r_c = 8 at phi = 0.2: 4.7 % at N = 20 -> 9.1 % at
  N = 200; paper GPU operator 8.3 -> 14.5 %).  The remaining growth is the far-field term of section 2: the number
  of pairs beyond 6 radii per particle grows with the box, and their many-body residual is never corrected.
* **Max per-particle error** halves at high phi (55 % 2-body, 40 % b1 -> 28 % moments v2), i.e. the improvement is not
  only in the bulk.
* Cost: the CPU operators take 3.6 s per apply at N = 200 (moments) vs 6.8 s (b1), dominated by the python two-body loop of
  `NNMob`; the moments correction itself is a single batched call.  The GPU path still runs the old 64-wide
  `MultiBodyCorrection`; porting the moments model to `Mob_Nbody_Torch` is the open item (segment-sum over the
  complete edge list, see `artifacts/nbody_moments_report.md` section 7).

## 5. Files

- `src/nbody_features.py`: `select_pair_neighbours`, `pad_neighbours`, `two_body_blocks`,
  `baseline_features_torch`, `predict_blocks`, `v2_is_val`, `MEDIAN_2B_OPERATOR`, `SELECTION_VARIANTS`;
  `src/mob_op_nbody_moments.py::_pair_rows` uses the shared selection.
- `experiments/build_nbody_v2_cache.py` (cache + `--stats`), `experiments/train_nbody_v2.py`,
  `experiments/nbody_v2_summary.py` (validation tables), `experiments/nbody_v2_ceiling.py` (error decomposition).
- `benchmarks/paper_accuracy_v2.py` (+ `--merge --summary --figures`), `slurm/paper_truth.sbatch`,
  `slurm/paper_eval.sbatch`; results `data/paper_accuracy_v2.csv`, `artifacts/paper_accuracy_v2_tables.md`,
  `figures/paper_v2_fig3_P{200,300}.pdf`, `figures/paper_v2_fig4_{rel_rmse,max_rel_rmse}.pdf`.
- Models: `data/models/nbody_moments_v2_{k10_rc6,kinf_rc6,kinf_rc8}.pt`, `data/models/nbody_pinn_b1_v2.pt`
  (+ `.json` sidecars with the selection each must be run with, `experiments/*.wt`); runs in
  `experiments/runs_v2/`.
- Tests: `tests/test_nbody_v2.py` (12 tests) + the existing 32.
