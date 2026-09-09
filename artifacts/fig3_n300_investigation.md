# Why Fig 3's N=300 error is consistently worse than N=200

**Date:** 2026-09-06.  **Scripts:** `tmp/fig3_n_scaling/phase_{a,b,c}.py` (tmp/, not committed).
**Data:** `data/paper_accuracy_v2.csv` (complete 13-op fig3/fig4 grid), cached Xfine truths in
`tmp/nbody_moments_truth/`, coarse-accuracy grand mobility matrices from `BatchedMFS` for the
term decomposition.  Errors are `rel_rmse` % (the harness metric) unless noted.

## TL;DR

The N-growth is **not the n-body model's fault and not a truth artifact**.  It is the uncorrected
**far field**: pairs beyond the switch distance (d > 8) are summed with plain pairwise RPY, which
ignores many-body screening.  That far-field term (a) already contributes 75–91 % of the velocity
magnitude, (b) grows with box size, and (c) carries a roughly constant *relative* error — so its
absolute error grows with N while everything the learned stack corrects stays N-flat.  Every
range-truncated operator pays the same penalty; the better the near field, the larger the *share*
of the total error the far field owns, which is why our best stack shows the steepest relative
growth.  N=300 is on the same smooth e(N) power-law trend as fig4's N=20→200 — nothing special
happens at 300.

## 1. The penalty is universal across operators (existing CSV)

fig3 mean rel_rmse at φ=0.2 (10 seeds):

| op | N=200 | N=300 | log-log slope of fig4 e(N), N≥60 |
|---|---|---|---|
| M_rpy | 22.80 | 25.89 | 0.08 |
| M_2b | 19.54 | 22.70 | 0.12 |
| M_3b | 13.28 | 16.07 | 0.15 |
| M_nbody_b1 | 15.01 | 18.08 | 0.26 |
| M_mom_v2_kinf_rc8_pc8 | 7.41 | 9.89 | 0.34 |
| M_mom_v2_kinf_rc8_pc8_diag | 7.29 | 9.86 | 0.39 |
| mfs_coarse | 1.34 | **1.31** | −0.10 |

- `mfs_coarse` — the only operator with no range truncation — gets slightly *better* at N=300.
  Rules out truth quality and the metric itself.
- The fig4 power-law fit extrapolated to N=300 predicts 9.0 % for the published stack (actual
  9.86): the fig3 gap is the continuation of a smooth trend in N, not a break.
- Slope hierarchy: the additive, N-growing far-field floor is shared by all truncated operators;
  as the near-field error shrinks (better ops), the floor dominates and the relative slope rises.

## 2. The growth is entirely translational

fig3 φ=0.2, published stack: `prmse_lin` 7.45 → 10.36 (N=200→300) while `prmse_ang` is flat
(6.28 → 6.17).  Same pattern for every operator.  This is the 1/r Stokeslet signature: the
UF far-field coupling decays as 1/r and its sums grow with box size; the angular couplings decay
as r⁻²/r⁻³ and converge.  Not a tail effect either: all 10 seeds shift up together (per-seed sd
1.53 → 2.20).

## 3. The far field dominates the velocity, and its share grows with N

|v_far,RPY(d>8)| / |v_truth| on new-vintage fig3 configs (seeds 126–128):

| φ | N=200 | N=300 |
|---|---|---|
| 0.1 | 0.848 | 0.914 |
| 0.2 | 0.745 | 0.813 |

At φ=0.2 the suspension's effective viscosity is ~2× the solvent's, but pairwise RPY propagates
every far interaction through plain solvent — a systematic O(φ) relative error on the dominant
and growing term of the velocity.

## 4. Per-particle profiles: collective, not compositional (phase_b)

Published stack, φ=0.2, seeds 123–128, per-particle error (÷ ensemble RMS, ×100) binned by
distance to the box wall:

| wall-distance bin | N=200 | N=300 |
|---|---|---|
| [1, 2) | 5.97 | 7.70 |
| [2, 3) | 6.88 | 9.11 |
| [3, 4) | 6.90 | 10.30 |
| [4, 6) | 8.45 | 12.71 |
| [6, 8) | 10.48 | 15.30 |

Error rises with depth at both N (deeper particles see more surrounding uncorrected far field),
and — decisively — N=300 is worse **at matched depth**, not merely because it has more interior
particles.  So the growth is a collective far-field effect (with a small composition assist),
not "interior particles are OOD for the model".  Spearman(err, wall) = +0.28 (N=200) / +0.43
(N=300); Spearman(err, nn8) = +0.35 / +0.49.

## 5. Exact term decomposition (phase_c)

Grand mobility M from `BatchedMFS(acc="coarse", triton32, tol_v=1e-4)` per config
(`truth_gap` = ‖MF − v_Xfine‖/‖v_Xfine‖ ≤ 1.2 % — coarse M is plenty for 3–10 % terms),
fig3 forces, truth `v = M F`.  Cumulative ladder at switch_dist = pair_cutoff = 8,
new-vintage seeds 126–128, mean rel-L2 %:

| φ | N | e_2b | e_near_nn (pair model) | e_stack_nn (pair+diag) | e_near_ex (exact pair floor) | **e_diag_ex (exact pair+diag floor = pure d>8 RPY residual)** | e_far12_ex (+exact 8–12 shell) | e_far_ex (sanity) |
|---|---|---|---|---|---|---|---|---|
| 0.1 | 200 | 8.44 | 3.91 | 3.87 | 3.26 | **3.14** | 1.74 | 0.06 |
| 0.1 | 300 | 9.74 | 4.95 | 4.88 | 4.51 | **4.43** | 2.60 | 0.07 |
| 0.2 | 200 | 16.08 | 6.46 | 6.29 | 4.51 | **4.10** | 2.35 | 0.11 |
| 0.2 | 300 | 21.89 | 10.14 | 10.01 | 6.87 | **6.57** | 3.50 | 0.12 |

Reading (φ=0.2):

- The **exact floor** — after replacing every near-pair (d ≤ 8) residual *and* the diagonal
  residual with their exact values, i.e. what remains is purely the d > 8 RPY far field —
  grows 4.10 → 6.57 (**×1.60**).  The published stack grows 6.29 → 10.01 (**×1.59**), and the
  harness on the same three seeds gives 6.41 → 10.24 (×1.60).  The stack sits at a constant
  ×1.53 over the floor at both N: **the entire N-growth rate is the far-field floor's growth
  rate**; a *perfect* pairwise near-field + diagonal correction would still pay it in full.
- The shell split shows both far ranges grow: exactly correcting the 8–12 shell as well leaves
  2.35 → 3.50, so the d > 12 tail alone also grows ~50 %; in variance terms the 8–12 shell owns
  ~two-thirds of the far floor at both N.
- The sanity term (`e_far_ex`: all exact residuals applied) closes to ≈ 0.1 — the ladder exactly
  reproduces MF.
- At φ=0.1 the same picture, milder: floor 3.14 → 4.43, and the floor owns 66–82 % of the
  stack's error variance outright.

## 5b. How the N-dependence scales with volume fraction (phase_d + extended ladder)

Three independent measurements, all agreeing: **the N-dependence strengthens roughly linearly
with φ, about doubling from dilute to φ=0.2.**

(i) fig4 power-law slope b (e ~ N^b, N≥60) for the published stack vs φ:
0.26 / 0.19 / 0.19 / 0.32 / 0.31 / 0.32 / 0.43 / 0.39 at φ = 0.025…0.2.
Pearson(b, φ) = **+0.86**, fit **b ≈ 0.17 + 1.2 φ**.  The φ-trend weakens down the op hierarchy
(b1 +0.67, 2b +0.47, RPY +0.65 with smaller slopes) and vanishes for the control
(`mfs_coarse`: b ≈ −0.1 at every φ, Pearson +0.19).

(ii) Direct exact-floor growth from the ladder, e(300)/e(200), seeds 126–128:

| φ | stack growth | **exact far-field floor growth** | floor share of stack variance (200 / 300) |
|---|---|---|---|
| 0.05 | 1.27 | **1.31** | 67 % / 71 % |
| 0.10 | 1.26 | **1.41** | 66 % / 82 % |
| 0.15 | 1.46 | **1.54** | 50 % / 55 % |
| 0.20 | 1.59 | **1.60** | 42 % / 43 % |

The floor's absolute size at N=300 is also ≈ linear in φ at the dilute end (2.42 / 4.43 / 5.16 /
6.57 % at φ = 0.05…0.2), consistent with an O(φ) screening error on the far field.

(iii) Ceiling-CSV floor exponent vs P (uniform family, fitted over P≥32 — P=16 boxes at high φ
have almost no far pairs and must be excluded): c ≈ 0.34 at φ ≤ 0.05 rising to ≈ 0.5–0.75 at
φ ≥ 0.075.

Two φ-regimes for *what dominates the stack's error*: at φ ≤ 0.1 the far-field floor owns
66–82 % of the error variance outright (the near-field models are nearly exact there); at
φ ≥ 0.15 the pair model's own residual grows to ~half the variance — but the *growth with N*
still comes almost entirely from the floor (stack and floor growth ratios match to ≤ 0.08).

## 6. Small-scale corroboration (existing ceiling CSV)

`artifacts/nbody_v2_ceiling.csv` (uniform family, exact near+diag corrections applied): the
remaining error — pure far-field RPY residual — grows with P at every φ, e.g. φ=0.2:
0.73 % (P=16) → 3.18 (P=32) → 4.15 (P=48) → 5.25 (P=64).  Same mechanism, small scale.

## 7. Side finding: 12 legacy truth files contaminate fig3 N=200

`tmp/nbody_moments_truth/uniform_N200_phi{0.05,0.1,0.15,0.2}_seed{123,124,125}.npz` (12 of the
80 fig3 N=200 truths; the only files without a `cluster_md5` key) predate the current generator:
they were made with the 2025-12 `uniform_sphere_cluster` (commit `18a5028`) — **uncentered box
[1, L−1]³ and min surface gap 0.05** instead of today's centered box with gap 0.1.  Same nominal
φ and homogeneous RSA, so they are valid physics, but the closer near-contact pairs make them
systematically harder: at φ=0.1 the published stack averages 5.97 % on them vs 3.97 % on the
new-vintage seeds.  They *inflate* fig3 N=200 (i.e. mask part of the true N-trend) at φ ≤ 0.15.
Vintage-matched (seeds 126–132 at both N) the gap is: φ=0.1: 4.22→5.07, φ=0.15: 5.94→6.81,
φ=0.2: 7.14→9.77.  **Recommendation:** delete the 12 files and regenerate with the current
generator (one GPU truth pass, ~10 min/config) next time the fig3 numbers are refreshed.

## 8. Implications

- The N=300-vs-200 gap is structural for any operator that corrects a finite range and sums RPY
  beyond: it will keep growing slowly (e ~ N^0.3–0.4 at φ=0.2 in this range) until the far-field
  share saturates.  It is not evidence of a defect in the moments/diag models — their increments
  over the exact floors are N-flat.
- Ways to actually attack it, in decreasing bang-per-effort: (a) treat the collective far field
  better than pairwise-RPY-through-solvent (effective-medium / screened far field), (b) push the
  correction range beyond 8 (the pc6→pc8 step already bought 15–19 % — diminishing but real),
  (c) periodic or Ewald-summed far field for bulk-suspension use cases.
- For the paper: report the N-dependence as a property of the truncation scheme (all baselines
  share it), and consider a sentence noting the far-field share of the velocity at fig3 sizes.
