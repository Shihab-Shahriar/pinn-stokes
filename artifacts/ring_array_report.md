# Ring-like array of spheres (Jordan & Lockerby 2025): NeMO vs RPY

**Date:** 2026-09-07. **Scripts:** `figures/fig_ring_array.py` (static coefficients), `figures/fig_ring_sedimentation.py`
(dynamic). **Reference:** J.J.P. Jordan & D.A. Lockerby, *The method of fundamental solutions for multi-particle Stokes
flows: application to a ring-like array of spheres*, JCP 520 (2025) 113487, Tables B.9/B.10 (transcribed with
`pdftotext` into `data/ring_array_jordan2025.csv`, 480 values, spot-checked against the page images).

## 1. Test

A planar regular P-gon of unit spheres with surface separation S between neighbours (side L = S + 2, ring radius
R_c = L / 2 sin(π/P)), every sphere carrying the same unit force: **parallel** to the ring plane (Fig. 11a of the paper)
or **perpendicular** to it (Fig. 11b). Symmetry reduces the per-sphere response to five coefficients (Eqs. 59–60):
the global mobilities M_∥, M_⊥, the in-plane anisotropy M_∘ (the ring's deformation pattern, Fig. 20: leading and
trailing spheres run ahead of the centre of mass for P < 7, behind it for P > 9) and two rotation coefficients
N_tz (perpendicular case) and N_zt (parallel case). The table values are exact to 5–6 digits for P = 3…10,
S/R = 0.1…1000 (B.10) and M_∥, M_⊥ for P up to 1000 (B.9).

Operators: `rpy` (`NNMob(rpy_only=True)`, full GRPY grand mobility, Stokes self term), `2b` (`NNMob`, pair NN to
switch 6), `nemo` (`Mob_Op_Nbody_Moments`, moments pc8c + learned diagonal, pair cutoff = switch 8), `b1_paper`
(`Mob_Op_Nbody` with `nbody_pinn_b1.pt`, the published operator). Coefficients are least-squares fits of the
per-sphere velocities onto the symmetry forms; the fit residual is ≤ 1e-8 for every operator (all are O(3)-equivariant,
so this is a consistency check, not a source of error). Touching rings (S = 0) are outside the learned near field's
training range (surface gap ≥ 0.1) and are not evaluated; S/R = 0.1 is the edge of that range.

## 2. Static coefficients (Table B.10 grid, P = 3…10)

Mean / max relative error over the 40 near-field cells S/R ≤ 2. M_∘ crosses zero with P, so its error is quoted
relative to the coefficient's scale at that S (max over P of |M_∘,exact|):

| operator | M_∥ | M_⊥ | M_∘ | N_tz | N_zt |
|---|---|---|---|---|---|
| RPY | 3.36 / 6.43 | 0.62 / 2.47 | 56.7 / 256 | 5.5 / 40.5 | 38.9 / 94.5 |
| 2-body only (switch 6) | 0.89 / 1.79 | 0.13 / 0.36 | 55.6 / 219 | 2.4 / 13.8 | 31.0 / 78.9 |
| NeMO b1 (published) | 0.30 / 1.14 | 0.39 / 1.14 | 53.3 / 208 | 2.9 / 7.1 | 27.6 / 76.6 |
| **NeMO (moments pc8c + diag)** | **0.28** / 1.57 | **0.18** / 0.85 | **17.7** / 109 | 5.6 / 19.9 | **19.0** / 49.7 |

Beyond S = 2 every operator is within 0.1 % on M_∥, M_⊥ and within 2 % on the rest (16 cells); NeMO carries a
2e-4 floor on M_⊥ from the self NN (1.00241 vs 1.00225 at S = 1000).

Mean over P at fixed S (RPY → NeMO): M_∥ error 5.5 → 0.6 % (S = 0.1), 4.8 → 0.35 (0.2), 3.5 → 0.14 (0.5),
2.1 → 0.12 (1), 0.9 → 0.22 (2). Absolute M_∘ error (× 6πμR): 0.040 → 0.017, 0.033 → 0.006, 0.024 → 0.003,
0.014 → 0.003, 0.005 → 0.005. N_zt error 87 → 41 %, 58 → 29, 29 → 15, 15 → 8, 6 → 4. N_tz: 14.6 → 9.1 % at
S = 0.1 but NeMO is *worse* than RPY for S ≥ 0.5 (6.1 vs 3.1 % at S = 0.5, 3.8 vs 1.0 at S = 1): the perpendicular
rotation is the leading-order rotlet coupling, which RPY has exactly, and NeMO's angular TR/RT residual shows.

Qualitative features:
- **Sign change of M_∘ with P** at S = 0.5: exact between P = 8 and 9, NeMO between 8 and 9, RPY between 7 and 8.
  At S = 0.1 the exact M_∘ stays negative up to P = 10; RPY turns positive at P = 9, NeMO stays negative.
- **Zero crossing S\* for P = 7** (exact 1.58 R): RPY 1.20, 2b 2.07, b1 2.21, NeMO 2.33. NeMO under-predicts M_∘
  at S = 1–2 for P ≥ 7 by about as much as RPY over-predicts it (P = 8, S = 2: exact 0.0217, NeMO 0.0137,
  RPY 0.0271, 2b 0.0187) — the same dilute-regime noise floor of the learned corrections seen in the Fig 9 study.
- **RPY's two rotation coefficients are identical by construction** (N_tz = −N_zt to all digits, a consequence of
  pairwise-additive coupling), while the exact ones differ by up to 90 % at S = 0.1 (P = 10: −0.2445 vs +0.1290);
  NeMO separates them (−0.293 vs +0.193).

Global mobility for large rings (Table B.9, P = 20 / 50 / 100 / 200), M_∥ error %: S = 0.5: RPY 1.99 / 1.17 / 0.86 /
0.69, NeMO 0.56 / 0.19 / 0.01 / 0.06; S = 1: RPY 1.17 / 0.64 / 0.45 / 0.37, NeMO 0.11 / 0.45 / 0.52 / 0.50 (NeMO's
moments term leaves a ~0.5 % floor at S = 1–2 for large P; the 2-body NN alone is at 0.1–0.3 %). M_⊥: RPY ~0.5 %,
NeMO 0.1–0.2 % at S = 0.5; all < 0.1 % for S ≥ 2.

Figures: `figures/fig_ring_array_A` (Fig.-20-style pattern, exact / NeMO / RPY, P = 6 and 10 at S = 0.5),
`fig_ring_array_B` (coefficients vs S/R for P = 4, 6, 8, 10), `fig_ring_array_C` (M_∥, M_⊥ vs P to 200),
`fig_ring_array_err` (error vs S/R). Data: `figures/fig_ring_array.csv`, `figures/fig_ring_array_zero_crossing.csv`.

## 3. Dynamics: ring sedimenting parallel to its plane (S = 0.5 R)

RK4 (Δt = 1, a = μ = F = 1) on the positions with accumulated rotation, truth = `BatchedMFS(acc="fine",
backend="triton32")` at every stage (0.1 s per solve; its t = 0 fit reproduces Table B.10 to five digits:
P = 6: M_∥ 2.51299, M_∘ −0.03422, N_zt 0.17124). For P < 7 the trailing sphere is squeezed by its two neighbours
(Fig. 20a), so every run ends when the smallest surface gap reaches 0.1 R (training range and MFS resolution).

Fall distance (radii) at which the gap reaches 0.1 R, and the mean sphere displacement relative to the centre of
mass versus truth at the RPY run's last common frame:

| P | truth | NeMO | RPY | 2-body only | shape error NeMO / RPY / 2b (R) |
|---|---|---|---|---|---|
| 4 | 12.8 | 12.1 | 6.4 | 6.8 | 0.010 / 0.126 / 0.095 (t = 60) |
| 5 | 17.7 | 17.0 | 9.5 | 8.5 | 0.016 / 0.113 / 0.127 (t = 70) |
| 6 | 24.9 | 21.6 | 16.7 | 12.5 | 0.013 / 0.083 / 0.154 (t = 95) |

RPY pinches the ring in half the fall distance of the truth (P = 4, 5); NeMO tracks the truth to within 4–13 %.
The 2-body NN alone is no better than RPY here: pairwise additivity, not the pair kernel, is the limitation, and the
many-body correction is what removes it. Accumulated rotation (sphere 2) runs ~10 % fast for NeMO and ~20 % fast
for RPY. Storyboards: `figures/fig_ring_sed_P{4,5,6}_S0.5.{png,pdf}`; frames in `figures/fig_ring_sed_P*_S0.5_*.npz`.

## 4. Verdict

The ring sedimentation problem is a clean multi-sphere showcase: on every quantity RPY is off by 3–6 % (global
settling) to 30–250 % (deformation, in-plane rotation) at S ≤ 0.5 R, and NeMO removes most of it — 10× on M_∥,
5–8× on the deformation coefficient at P ≤ 6, 2× on in-plane rotation — with the deformation visible as a 2× later
pinch in the dynamic run. Two honest limits: NeMO's perpendicular rotation is slightly worse than RPY's for
S ≥ 0.5 R, and its deformation coefficient is under-predicted at S = 1–2 R for P ≥ 7 (wrong zero crossing).
