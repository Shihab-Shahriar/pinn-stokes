# Moments-based n-body correction (`MultiBodyMoments`): implementation and comparison with the b1 baseline

*2026-08-29 — RTX 4060 laptop, native python (torch 2.6+cu124, Triton MFS truth), CPU operators.*

## 1. What was built

`moments_for_nbody.md` (§3–§5) replaces the n-body correction's neighbourhood descriptor. Implemented, against the
**existing** training data (`data/multibody*`, 49,400 rows, ≤10 neighbours per row) and with the **baseline-matched**
inference neighbourhood (within 6.0 of the pair midpoint, top-10 by `d_kt·d_ks`) as the primary configuration:

| piece | file |
|---|---|
| band weights, moments `(s_a, v_a, Q_a)`, 72 invariants, 34 TT/RR + 25 TR bases, 6×6 assembly (pure torch, TorchScript-scriptable) | `src/nbody_moments.py` |
| `MultiBodyMoments` (76→128→64→128→64→93, Tanh; input standardisation + per-basis scaling buffers; zero-initialised head) and `MultiBodyCorrectionB1` (clean copy of the notebook baseline for retraining) + `.wt→.pt` export entries | `src/model_archs.py` |
| shared loader / features / residual labels / split (used by training, tests and the operator) | `src/nbody_features.py` |
| CPU operator `Mob_Op_Nbody_Moments(Mob_Op_Nbody)`: same pair gate and neighbour selection, moments once per unordered pair, `max_neighbors=None` and `neighbor_cutoff` variants | `src/mob_op_nbody_moments.py` |
| training / evaluation CLI (notebook recipe, deterministic split, metrics.json, TorchScript export) | `experiments/train_nbody_moments.py` |
| grand-mobility benchmark vs Xfine MFS truth (warp-free, truth cached in `tmp/nbody_moments_truth/`), incl. reciprocity checks | `benchmarks/compare_nbody_moments.py` |
| 16 structural tests (partition of unity, numpy-oracle parity, E-sign pin, permutation/padding invariance, reciprocity, O(3) incl. reflections, zero-moment reduction to Eq. 17, TorchScript round trip, feature parity with the operator, neighbour-selection parity, operator-level grand-M symmetry, label convention vs `NNMob`) | `tests/test_nbody_moments.py`, `tests/nbody_moments_ref.py` |
| table rendering for this report | `experiments/nbody_moments_summary.py` |

Published models: `data/models/nbody_moments.pt` (+ `experiments/nbody_moments.wt`) and the fair reference
`data/models/nbody_pinn_b1_retrained.pt` (+ `experiments/nbody_b1_retrained.wt`).

### Deviations from the design doc (all deliberate, all documented in code)

1. **No zeroing of the band weights beyond r_c = 8.** The tent partition of unity saturates in band 8 for r ≥ 7.5 and
   stays 1 beyond 8; the cutoff is enforced by neighbour *selection*. Reason: the existing rows were sampled in a ball
   about the *target* (radius 6 / 8), so 13 % / 28 % of their neighbours lie farther than 6 from the midpoint (up to
   11.8) and do affect the MFS label — dropping them from the features would make the labels noisy.
2. **Neighbour count stays capped at K = 10 (baseline selection) at inference.** The rows carry ≤ 10 neighbours, so
   Σ_a s_a ≤ 10 in training; the unbounded r_c = 8 neighbourhood of the doc (~100 neighbours at φ = 0.2) is far outside
   the training distribution. Both unbounded variants are reported as extra rows (`M_mom_kinf_rc6`, `M_mom_kinf_rc8`).
3. **Conventions.** `z = −s_vec/|s_vec|` (= the doc's ẑ = the baseline's `d_vec` after negation) and
   `E(u)_ab = ε_abc u_c` (= `model_archs.L3`; the notebook's L3 is −E). Same convention in training and inference.

### Training details that are not in the doc

* Residual labels `Y − v2b` with the two-body term in the **operator's** convention (§3 below), L1 loss, Adam 1e-3,
  cosine schedule, 500 epochs, batch 256, 80/20 split — the notebook's recipe, for both models.
* MLP inputs are standardised (`inv_mean`/`inv_std` buffers) and each of the 93 coefficients is divided by the RMS
  Frobenius norm of its basis over the training set (`basis_scale`), so raw coefficients are O(1). Both are fitted on
  the training split and travel with the state dict.
* The output layer is zero-initialised so the correction starts at exactly the two-body solution. Without it the random
  93-term head starts ~5× too large and the model is still far behind after 25 epochs (§4, ablation).
* One deterministic split (sorted file order, `numpy.random.default_rng(41)`), saved with every run; all validation
  numbers below are on the same 9,880 rows.

## 2. Data facts that matter

* Rows: `[s_vec, d, d−2, F(3), T(3), nk×(3)]`, target at origin, force/torque on the source only (|F| = |T| = 6π),
  `Y` = target 6-velocity from MFS (`acc="fine"`). nk ~ U{1..10}; never 0. `dist_s ∈ [2.05, 8.0]`, mean 4.6900.
* Neighbours were sampled about the target, not the midpoint; the inference selection (midpoint ball, top-10 by
  `d_kt·d_ks`) yields denser neighbourhoods near the pair: mean `s_a` per band at inference (N=30, φ=0.1) is
  `[0.13, 0.78, 2.19, 3.79, 2.57, 0.51, 0.01, 0]` vs `[0.05, 0.30, 0.81, 1.22, 1.18, 0.90, 0.56, 0.48]` in training.
  This shift is shared with the baseline (same selection) and is what the unbounded variants amplify.

## 3. Finding: the saved notebook's residual labels have the RT sign flipped

`branch1_multibody_pinn.ipynb::predict_two_body_from_triplet` feeds `−s_vec` into the two-body model; the operators
(`NNMob.get_two_vel`, `center2 = pos[s] − pos[t]`) feed `+s_vec`. `L1`/`L2` are even in the axis but `L3` is odd, so the
notebook's two-body prediction has its RT/TR blocks sign-flipped and its residual labels carry a spurious −2·RT term.

Measured on all 49,400 rows (identical features, verified bit-exact against `Mob_Op_Nbody._build_pair_feature_vector`):

| two-body convention | mean \|residual\| lin / ang | 2-body-only PRMSE lin / ang | shipped `nbody_pinn_b1.pt` 2b+nbody PRMSE lin / ang |
|---|---:|---:|---:|
| notebook, `−s_vec` | 0.0404 / 0.0399 | 38.3 % / 198.7 % | 39.5 % / 207 % |
| operator, `+s_vec` | 0.0064 / 0.0028 | **6.85 % / 16.5 %** | **3.82 % / 10.94 %** |

So the shipped b1 was trained with the *correct* convention (it reproduces the notebook's 3.8 % / 11.1 % only against the
correct residual), the notebook as saved is inconsistent with it (`experiments/nbody_cross_tmp.wt` is a flawed-label
model), and the "2-body only 38 % / 199 %" bars in `nbody_correction_effect.pdf` are the artefact: the true two-body-only
errors on this data are 6.8 % / 16.4 %, i.e. the n-body term buys ~1.8× / 1.5×, not 10× / 18×.

It is load-bearing for this work: a moments model trained on the notebook's labels reached 2.2 % / 8.3 % on validation
and then scored **28 % rel-RMSE inside the operator at φ = 0.05 (2-body alone: 6.8 %)** — it was adding a wrong RT term
to a correct two-body term. `nbody_features.two_body_velocity` uses `+s_vec`, and
`tests/test_nbody_moments.py::test_two_body_labels_match_operator_convention` pins it against `NNMob.apply`. All
numbers in §4–§5 use the corrected labels.

## 4. Validation results (identical 9,880-row split; 500 epochs; residual labels in the operator convention)

PRMSE = ‖pred − Y‖₂ / ‖Y‖₂ over the split (pooled over the three translational / rotational components, and
per component); RMSE over all six components. "2-body only" is the two-body NN alone; every other row is
two-body + the n-body correction.

| model | run | RMSE | PRMSE lin % | PRMSE ang % | Ux % | Uy % | Uz % | Ox % | Oy % | Oz % | note |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2-body only | – | 0.00847 | 6.80 | 16.36 | 6.83 | 6.84 | 6.74 | 16.29 | 16.52 | 16.28 | no n-body term |
| shipped nbody_pinn_b1.pt | b1_shipped | 0.00494 | 3.82 | 11.10 | 3.84 | 3.86 | 3.75 | 11.12 | 11.20 | 11.00 | shipped model: trained on an unknown 80% split of the same rows -> validation rows are partly in its training set (contaminated) |
| baseline | baseline_s411 | 0.00525 | 4.09 | 11.42 | 4.15 | 4.12 | 4.01 | 11.40 | 11.52 | 11.36 | seed 411 |
| baseline | baseline_s412 | 0.00520 | 4.05 | 11.43 | 4.07 | 4.10 | 3.99 | 11.44 | 11.47 | 11.38 | seed 412 |
| baseline (zero-init) | baseline_zeroinit_s411 | 0.00530 | 4.14 | 11.45 | 4.16 | 4.20 | 4.07 | 11.43 | 11.56 | 11.36 | seed 411 |
| moments (inv_norm) | moments_invnorm_s411 | 0.00311 | 2.23 | 8.43 | 2.17 | 2.31 | 2.19 | 8.45 | 8.46 | 8.39 | seed 411 |
| moments | moments_nozeroinit_s411_cpu | 0.00321 | 2.33 | 8.52 | 2.27 | 2.40 | 2.31 | 8.54 | 8.57 | 8.46 | seed 411 |
| moments | moments_s411 | 0.00311 | 2.24 | 8.39 | 2.19 | 2.31 | 2.20 | 8.44 | 8.42 | 8.31 | seed 411 |
| moments | moments_s412 | 0.00312 | 2.25 | 8.42 | 2.19 | 2.33 | 2.21 | 8.46 | 8.46 | 8.35 | seed 412 |

* The moments model halves the translational error of the retrained baseline (2.24 % vs 4.05–4.14 %) and cuts the
  rotational error by ~1.35× (8.4 % vs 11.4 %), with RMSE 0.00311 vs 0.00520–0.00530. Seed-to-seed spread is ±0.01 %
  for both models, so the gap is far outside noise.
* The shipped `nbody_pinn_b1.pt` (3.82 % / 11.10 %) is slightly better than the retrained baselines because ~80 % of
  the validation rows were in its own training set (its split cannot be reconstructed — the notebook iterates a
  `set` of file tags and draws the permutation on CUDA); the retrained rows are the fair reference.
* Ablations: the count-normalised invariants (`inv_norm`) change nothing in-distribution (2.23 % / 8.43 %); the
  design as written in the doc (no zero-initialised head) also converges (2.33 % / 8.52 %) but starts ~5× too
  large and is far behind for the first ~100 epochs; zero-initialising the *baseline's* head does not help it
  (4.14 % / 11.45 %), so the gain is not an initialisation artefact.

### Error vs. number of neighbours K (pooled PRMSE over all six components)

| K | n | 2-body only % | b1_shipped | baseline_s411 | moments_s411 |
|---:|---:|---:|---:|---:|---:|
| 1 | 986 | 3.07 | 2.38 | 2.39 | 1.28 |
| 2 | 906 | 4.34 | 3.18 | 3.20 | 1.88 |
| 3 | 1032 | 5.96 | 3.87 | 3.97 | 2.38 |
| 4 | 1054 | 6.32 | 3.97 | 4.13 | 2.50 |
| 5 | 1030 | 6.98 | 4.32 | 4.54 | 2.59 |
| 6 | 894 | 8.05 | 4.53 | 4.85 | 2.85 |
| 7 | 967 | 8.09 | 4.65 | 5.00 | 3.04 |
| 8 | 1013 | 8.74 | 4.84 | 5.20 | 3.20 |
| 9 | 1053 | 9.34 | 5.20 | 5.58 | 3.31 |
| 10 | 945 | 10.01 | 5.22 | 5.75 | 3.33 |

The baseline's error grows ~2.2× from K = 1 to K = 10 (2.38 → 5.22 %); the moments model's grows 2.6× from a
much lower start (1.28 → 3.33 %) and is below the baseline's K = 1 error up to K = 5.

## 5. Grand-mobility operator tests against MFS truth

All operators are the full CPU stack (analytic self term + two-body NN within 6 radii + RPY beyond + n-body
correction), applied to random unit force *and* torque on every particle; truth is Xfine MFS
(`benchmarks.cluster.generate_uniform_testcase`, cached in `tmp/nbody_moments_truth/`). Metrics are those of
`accuracy_grand_M._compute_error_stats` (rel-RMSE over all six components; max per-particle relative error) plus
the translational / rotational relative-L2 split. Operators:

| key | operator |
|---|---|
| `M_2b` | `NNMob`, no n-body term |
| `M_nbody_b1` | `Mob_Op_Nbody` with the shipped `nbody_pinn_b1.pt` (production reference) |
| `M_nbody_b1_retrained` | same operator, baseline retrained on the shared split (`nbody_pinn_b1_retrained.pt`) |
| `M_mom_k10_rc6` | `Mob_Op_Nbody_Moments`, **baseline-matched neighbourhood** (primary) |
| `M_mom_kinf_rc6` | all neighbours within 6 of the midpoint (no K cap) |
| `M_mom_kinf_rc8` | all neighbours within 8 of the midpoint (the doc's r_c) |

### 5a. Uniform suspensions, N = 200, φ ∈ {0.05, 0.1, 0.15, 0.2}, seeds 123–125

**N=200: rel-RMSE % (mean ± std over seeds)**

| operator | φ=0.05 | φ=0.1 | φ=0.15 | φ=0.2 |
|---|---:|---:|---:|---:|
| M_2b | 5.10 ± 0.59 (n=3) | 13.15 ± 0.57 (n=3) | 18.02 ± 0.12 (n=3) | 19.05 ± 3.57 (n=3) |
| M_mom_k10_rc6 | 3.80 ± 0.52 (n=3) | 9.07 ± 0.58 (n=3) | 13.15 ± 0.42 (n=3) | 15.04 ± 3.06 (n=3) |
| M_mom_kinf_rc6 | 3.79 ± 0.54 (n=3) | 8.68 ± 0.55 (n=3) | 12.55 ± 0.14 (n=3) | 14.59 ± 2.92 (n=3) |
| M_mom_kinf_rc8 | 3.82 ± 0.47 (n=3) | 12.35 ± 0.34 (n=3) | 22.15 ± 1.67 (n=3) | 25.93 ± 4.27 (n=3) |
| M_nbody_b1 | 4.04 ± 0.49 (n=3) | 9.62 ± 0.53 (n=3) | 13.55 ± 0.35 (n=3) | 15.31 ± 3.05 (n=3) |
| M_nbody_b1_retrained | 4.02 ± 0.52 (n=3) | 9.42 ± 0.47 (n=3) | 13.19 ± 0.38 (n=3) | 14.94 ± 2.91 (n=3) |

**N=200: translational rel-L2 %**

| operator | φ=0.05 | φ=0.1 | φ=0.15 | φ=0.2 |
|---|---:|---:|---:|---:|
| M_2b | 5.40 | 14.66 | 19.06 | 19.11 |
| M_mom_k10_rc6 | 4.05 | 10.23 | 14.11 | 15.28 |
| M_mom_kinf_rc6 | 4.04 | 9.79 | 13.39 | 14.72 |
| M_mom_kinf_rc8 | 4.04 | 13.59 | 22.83 | 24.64 |
| M_nbody_b1 | 4.29 | 10.83 | 14.49 | 15.57 |
| M_nbody_b1_retrained | 4.26 | 10.59 | 14.09 | 15.16 |

**N=200: rotational rel-L2 %**

| operator | φ=0.05 | φ=0.1 | φ=0.15 | φ=0.2 |
|---|---:|---:|---:|---:|
| M_2b | 3.36 | 7.39 | 12.40 | 19.38 |
| M_mom_k10_rc6 | 2.28 | 4.36 | 7.41 | 12.63 |
| M_mom_kinf_rc6 | 2.25 | 4.23 | 7.83 | 13.77 |
| M_mom_kinf_rc8 | 2.56 | 7.88 | 19.21 | 39.68 |
| M_nbody_b1 | 2.61 | 4.76 | 8.05 | 12.69 |
| M_nbody_b1_retrained | 2.61 | 4.72 | 8.04 | 12.91 |

**N=200: max per-particle rel. error %**

| operator | φ=0.05 | φ=0.1 | φ=0.15 | φ=0.2 |
|---|---:|---:|---:|---:|
| M_2b | 19.45 | 50.73 | 56.77 | 57.74 |
| M_mom_k10_rc6 | 11.67 | 32.00 | 37.56 | 41.96 |
| M_mom_kinf_rc6 | 11.38 | 33.83 | 37.96 | 47.93 |
| M_mom_kinf_rc8 | 14.64 | 42.18 | 73.25 | 86.10 |
| M_nbody_b1 | 13.76 | 34.80 | 36.72 | 40.49 |
| M_nbody_b1_retrained | 12.94 | 32.93 | 36.70 | 38.73 |

Reading the table (rel-RMSE, mean over three seeds; the seed spread at φ = 0.2 is dominated by one near-contact
configuration and is the same for every operator):

* **Baseline-matched moments (`M_mom_k10_rc6`) vs the shipped b1:** better at every φ — 3.80 vs 4.04, 9.07 vs 9.62,
  13.15 vs 13.55, 15.04 vs 15.31 % — with the largest relative gains in the rotational block at low φ
  (2.28 vs 2.61 % at φ = 0.05, 4.36 vs 4.76 % at φ = 0.1) and in the worst-particle error at φ = 0.05
  (11.7 vs 13.8 %).
* **Vs the retrained baseline** (same split/recipe): ahead at φ ≤ 0.1 (3.80 vs 4.02, 9.07 vs 9.42 %), tied at
  φ ≥ 0.15 (13.15 vs 13.19, 15.04 vs 14.94 %). The operator-level gap is much smaller than the in-distribution
  gap (§4) because both models are evaluated on neighbourhoods denser near the pair than anything in the
  training rows (§2); the moments model has more capacity to be hurt by that shift.
* **Unbounded r_c = 6 (`M_mom_kinf_rc6`)** is the best operator at every φ (3.79, 8.68, 12.55, 14.59 %) even
  though Σ s_a exceeds the training range — the extra neighbours carry real information and the bases are
  additive in them. **r_c = 8** is worse than the two-body operator alone at φ ≥ 0.15 (22 %, 26 %): the 6–8 shell
  about the midpoint holds ~60 neighbours at φ = 0.2 (~30 at φ = 0.1), whereas bands 7–8 carry ~0.5 neighbours on
  average (at most one or two) in the training rows — the moments there are 10–50× outside the fitted range and
  multiply extensive bases, so this is pure extrapolation.
* All learned operators improve over `M_2b` by 1.3–1.5× at every φ; the historic rows are lower for every
  operator including `M_2b` (§6, item 8) and are shown for orientation only.


### 5b. Clustered near-contact configurations (`tmp/reference_sphere_δ.csv`, N = 10, surface gap δ)

| δ | `M_2b` | `M_nbody_b1` | `M_nbody_b1_retrained` | `M_mom_k10_rc6` |
|---:|---:|---:|---:|---:|
| 0.1 | 7.03 % (max 12.2) | 3.30 % (max 4.6) | 3.21 % (max 4.0) | **3.05 %** (max 4.3) |
| 0.2 | 5.73 % (max 8.7) | 2.47 % (max 3.2) | 2.70 % (max 3.5) | **1.58 %** (max 2.2) |
| 0.5 | 3.58 % (max 5.4) | 1.51 % (max 2.4) | 1.57 % (max 2.5) | **1.07 %** (max 1.4) |
| 1.0 | 1.59 % (max 3.0) | 0.97 % (max 1.5) | 0.89 % (max 1.3) | **0.61 %** (max 1.0) |
| 2.0 | 0.71 % (max 1.3) | 0.43 % (max 0.8) | 0.43 % (max 0.8) | **0.36 %** (max 0.6) |
| 3.0 | 0.42 % (max 0.7) | 0.40 % (max 0.7) | 0.40 % (max 0.6) | **0.39 %** (max 0.6) |

(rel-RMSE %, max per-particle relative error in parentheses.)

### 5c. Reciprocity

`M_ji = M_ij^T` holds **exactly** (0.0 in float32) for every ordered pair and for the assembled n-body grand M
(`compare_nbody_moments.py --symmetry`; `tests/test_nbody_moments.py::test_operator_grand_M_symmetric`): the
(t, s) and (s, t) rows share the same moments, the invariants are even in the pair axis bit-for-bit, and every
basis tensor is an exact transpose under `z → −z`. No symmetrisation pass is needed — the mechanism of Eq. (17)
carries over unchanged.

### 5d. Cost

At N = 200 every operator's `apply` takes 17–20 s and is dominated by `NNMob`'s python two-body loop; the moments
correction itself is vectorised (one `band_moments` call per configuration, one model call for all ordered pairs)
and is cheaper than the baseline's O(N³) per-pair neighbour scan (`_select_neighbor_indices` is vectorised in the
new operator with identical semantics — pinned by a test). Per pair the model does 93 coefficients × 59 3×3 bases
vs 5 × 3; at inference this is a ~1.2× wider last layer plus ~600 FMAs of basis assembly, negligible against the
neighbour gather. The GPU port is not done in this pass (§7).

## 6. Pitfalls found on the way

1. **Residual-label sign (§3)** — the single load-bearing issue; it is now pinned by a test and documented in
   `CLAUDE.md`.
2. **Provenance of the baseline weights.** `data/models/nbody_pinn_b1.pt` matches the correct residual;
   `experiments/nbody_cross_tmp.wt` (128/64/128/64) and `data/models/nbody_cross_tmp.wt` (64/64/64/64, the one the
   GPU operator loads) are different trainings; the former is a flawed-label model. `Mob_Op_Nbody._load_nbody_model`'s
   `.wt` branch cannot load the notebook's `.wt` (wrong widths) — the new operator loads its own `.wt`.
3. **Two-body median.** Labels use the 2-body notebook's 5.008307682776568, the operator uses 5.01
   (`mob_op_2b_combined.py`); ~1e-4 relative difference in the two-body term, shared with b1, left as is.
4. **Notebook split is irreproducible** (set iteration order + CUDA `randperm`), so the shipped b1 is contaminated
   for any in-distribution comparison; the retrained baseline is the reference.
5. **TorchScript** rejects closed-over module constants; the scriptable core reads them through tiny functions.
6. **Initialisation.** A random 93-wide head on unit-RMS bases starts the correction ~5× too large; a zero
   head fixes it (the doc's design also converges without it, just later).
7. **Environment.** `accuracy_grand_M.py` imports warp at module import; the new benchmark copies its metric
   function instead. `PYTHONPATH=/home/shihab/repo` shadows this repo's `src` unless the repo root is inserted at
   `sys.path[0]`. `~/warp_env.sh` does not exist on this box; everything here ran natively (torch 2.6 + Triton).
8. **Historic CSV numbers** (`data/grand_M_acc_uniform_fixed_N.csv`) are lower than what the same seeds give today
   for *every* operator including `M_2b` (e.g. φ = 0.1: 10.2 % then vs 13.2 % now); the configurations or truth
   settings evidently changed since, so only within-run comparisons are meaningful.

## 7. Conclusions and next steps

* On the existing data the moments encoding + 93-coefficient bases is a clear improvement in distribution
  (translational error halved, rotational −27 %) and a consistent but much smaller improvement at the operator
  level with the baseline-matched neighbourhood (5–6 % relative vs the shipped b1 at every φ; ahead of the
  retrained baseline at φ ≤ 0.1, tied above) — the inference neighbourhoods (top-10 within 6 of the midpoint)
  are denser near the pair than the training rows (§2), and both models pay for that shift. The data, not the
  architecture, is now the limiting factor.
* The unbounded r_c = 6 variant is the best operator at every φ despite Σ s_a exceeding the training range,
  while r_c = 8 is worse than the two-body operator alone at φ ≥ 0.15: bands 7–8 hold ~0.5 neighbours per row in
  training but tens per pair at inference, so the model is asked to extrapolate 10–50× beyond what it has seen.
* **Regenerate the n-body training data** with the design doc's §8 recipe — neighbourhoods sampled about the
  *midpoint*, full shell up to r_c = 8, counts representative of φ up to 0.2 (tens of neighbours) — then retrain
  with `max_neighbors=None`. The operator, trainer and tests need no change for that.
* **GPU port**: `Mob_Nbody_Torch` variant with a segment-sum of the 104 raw moment components over the complete
  edge list (never per pair-chunk, see `CLAUDE.md`), compiled chunk loop outside the graph, then the treecode /
  1M-particle A/B.
* Cheap extensions once data allows: cross-band invariants (`v_a·v_b`, `tr Q_a Q_b`), count-scaled `s_a I`,
  `s_a zzᵀ`, `s_a E(z)` bases for explicit additivity in the neighbour count, and l = 3 moments.
