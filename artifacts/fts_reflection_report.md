# The FTS stresslet reflection: closing NeMO's gap to Stokesian Dynamics' far field

*2026-09-08/09. Code: `src/fts_rpy.py` (FTS blocks + reflection), `benchmarks/fts_reflection_ladder.py` (diagnostic
ladder; tables `artifacts/fts_reflection_ladder.md`, raw `data/fts_reflection_ladder.csv`, fine grand-mobility truths
`tmp/fts_ladder/M_cache/`), `experiments/build_nbody_v2_cache.py --add-fts`, trainers `--label-base`, operator kwarg
`fts_reflection` (`src/mob_op_nbody_moments.py`, `src/gpu_nbody_moments.py`), harness ops `M_mom_v2_kinf_rc8_pc8c_fts_diag`
(CPU) / `M_mom_gpu_pc8c_fts_diag` (GPU, registered, not yet run), `experiments/finetune_config_velocity.py`. Tests
`tests/test_fts_rpy.py` (12), `tests/test_nbody_moments.py::test_operator_fts_reflection_term`,
`tests/test_nbody_v2.py::test_add_fts_augments_cache_and_labels`. Models `data/models/nbody_moments_v2_kinf_rc8_pc8c_fts.{pt,json}`,
`nbody_diag_v2_pc8c_fts.{pt,json}` (+ `.wt` in `experiments/`). Figures/tables regenerated with the new op:
`figures/fig_sd_compare_*`, `figures/fig_hignn_compare_*`, `artifacts/sd_comparison_tables.md`, `artifacts/hignn_comparison_tables.md`.*

## 1. Bottom line

Adding the leading many-body term of the force–torque–stresslet (FTS) hierarchy — one stresslet reflection, an
analytic O(N²) term with no parameters — and retraining the pair and diagonal models on labels with that term
removed makes NeMO **5–6× more accurate on the gravity comparison and 2.3–3× more accurate on the paper's Fig 3**,
with the same near-field models and the same neighbour lists:

| N = 200, translational PRMSE % | φ = 0.025 | 0.05 | 0.1 | 0.15 | 0.2 |
|---|---|---|---|---|---|
| NeMO (shipped: moments pair + learned diagonal) | 0.66 | 1.08 | 1.73 | 2.43 | 3.19 |
| **NeMO + FTS reflection** | **0.10** | **0.17** | **0.31** | **0.45** | **0.57** |
| SD far field only (`SD_Minf`, FTS solved exactly, O(N³)) | 0.08 | 0.13 | 0.27 | 0.50 | 0.77 |
| SD as shipped (far field + lubrication) | 2.16 | 4.88 | 11.87 | 18.91 | 25.18 |
| HIGNN full (2-body + 3-body + self) | 1.01 | 1.93 | 3.74 | 5.47 | — |
| RPY | 1.13 | 2.07 | 4.02 | 5.97 | 7.74 |

NeMO + FTS is now the most accurate operator in the table at φ ≥ 0.15 and within 1.1–1.4× of SD's exact far-field
solve at the dilute end, at O(N²) dense / O(N) with a treecode instead of O(N³). The gain comes from the far field:
the Fig 3 error no longer grows from N = 200 to 300 (φ = 0.2: 3.22 → 3.46 % instead of 7.51 → 10.29 %), and on the
Fig 4 gravity sweep the error now *decreases* with N (φ = 0.1: 0.48 % at N = 20 → 0.31 % at N = 200, against 0.66 → 1.73 %
for the shipped stack).

## 2. Why: the missing physics

The exact-residual ladder of `artifacts/fig3_n300_investigation.md` had shown that a *perfect* pairwise near field
(exact residual for every pair with d ≤ 8 plus exact diagonal) still leaves 3.1 / 4.1 % on the Fig 3 protocol at
φ = 0.1 / 0.2 — the pairwise RPY far field beyond 8 radii cannot represent many-body screening. The same ladder under
gravity (§4) puts that floor at 70–80 % of the shipped stack's error. SD's far field gets exactly this term right.

RPY moves particle t in the flow that the force on particle s makes. It ignores that every other particle k is
rigid: the flow from s arrives at k with a nonzero rate of strain, a rigid sphere cannot deform with it, so it
exerts a symmetric force dipole on the fluid (a stresslet), and that stresslet's flow reaches t. The path
s → k → t is the leading three-body effect and the origin of O(φ) hydrodynamic screening. Neither NeMO's RPY far
field nor HIGNN carries it.

## 3. The term

At the FTS level the velocities, angular velocities and rates of strain of N unit spheres are linear in the
forces, torques and stresslets:

```
[U; Ω] = A [F; T] + G S            A = RPY (grpy_tensors.mu); G_tk = [U; Ω] of t per unit stresslet on k  (~ 1/r²)
   E    = Gᵀ [F; T] + Mm S          Mm_tk = rate of strain at t per unit stresslet on k  (~ 1/r³)
Mm = D·I + Mm_off,  D = 3 / (20 π μ a³)       (single-sphere self term; condensed orthonormal 5-vector basis)
```

Rigid spheres have E = 0, hence `S = −Mm⁻¹ Gᵀ [F;T]` and the many-body correction to the FT mobility is
`−G Mm⁻¹ Gᵀ`: that is exactly what `SD_Minf` evaluates, with a dense inverse. Expanding the inverse in the
off-diagonal coupling, `Mm⁻¹ = (1/D)(I − Mm_off/D + …)`, and keeping the first term is Faxén's law for the stresslet
(a force-free sphere in an ambient strain rate E∞ carries `S = (20/3) π μ a³ E∞`). As an operator it is two global
passes:

```
pass 1:  E_k  = Σ_s G_skᵀ F_s        rate of strain at every particle from all Stokeslets and rotlets (Faxén terms included)
         S_k  = −E_k / D             the stresslet each sphere needs to stay rigid
pass 2:  ΔU_t = Σ_k G_tk S_k         the velocity those stresslets induce on every particle
```

i.e. the grand-mobility block `M_ref1(t, s) = −(1/D) Σ_k G_tk G_skᵀ` (all k; `G_kk = 0`), symmetric by construction
so reciprocity holds without fitting. A triplet contributes `a³ / (r_tk² r_ks²)` against the direct Stokeslet
`1 / r_ts`, i.e. `(a/r)³` per third body — summed over the neighbours, the O(φ) term the Fig 3 investigation had
isolated. Second-order term: `+(1/D²) G Mm_off Gᵀ` (stresslet-on-stresslet feedback); the series alternates.

`src/fts_rpy.py` transcribes SD's RPY-with-Faxén blocks (`generate_Minfinity.py`: `M13` g̃ with `(a₁²/6 + a₂²/10)∇²K`,
`M23` h̃, `M33` m with `(a₁²+a₂²)/10 ∇∇²K`; the `cond_E` condensation) in vectorised torch (float64, any device):
`fts_blocks(r, mu)` → G (n, 6, 5) and Mm (n, 5, 5) for `r = pos[velocity particle] − pos[stresslet particle]`;
`reflection_velocity(pos, F, mu, order, diag_exclude_within)` (matrix-free, target rows chunked; 0.14 s at N = 200 on
one CPU core, 7e-16 CPU/GPU agreement); `reflection_blocks(pos, mu, order, diag_exclude_within)` (dense, batched
configurations, for labels); `assemble_minfinity` (test helper). Validation (`tests/test_fts_rpy.py`): SD's 11N × 11N
matrix reproduced block by block to 5e-16 at μ = 1 and 2.5; `RPY + order "full"` equals `SDMob(minfinity_only=True)`
to 1e-14; chunked velocity = dense blocks to 1e-16; rotation/translation equivariance, 1/μ scaling, two-sphere 1/r⁴ decay.

**Diagonal two-body path (load-bearing).** For a pair block (t, s) every path t → k → s is a genuine three-body
term, but the diagonal path t → k → t is a *two-body* effect — the self-mobility reduction due to one neighbour —
which the two-body model's self correction K_s(t, k) already contains exactly for d_tk ≤ pair_cutoff. `diag_exclude_within
= pair_cutoff` removes `−G_tk G_tkᵀ / D` from the diagonal for those near pairs in the kernel, the cache labels and the
operator; for k beyond the cutoff the base is RPY and the term is genuinely missing, so it stays. Subtracting it
from the diagonal labels as well made the diagonal residual 4.4× larger (TT 5.4×); with the exclusion the diagonal
labels are unchanged (RMS ratio 0.997) while the pair labels shrink 2.5× in every block and every distance bin:

| pair residual RMS ratio, reflected base / 2b base (1 M sampled pc8c rows) | TT | TR | RT | RR | all |
|---|---|---|---|---|---|
| all rows | 0.40 | 0.38 | 0.35 | 0.45 | 0.39 |
| d ∈ [2, 3) / [3, 4) / [4, 6) / [6, 8] | 0.40 / 0.35 / 0.41 / 0.43 | | | | 0.39 / 0.36 / 0.43 / 0.44 |
| uniform / grown / lattice / chain | | | | | 0.31 / 0.45 / 0.16 / 0.41 |

Base error on the validation rows (relative Frobenius, %): TT 13.8 → 5.5, TR 38.8 → 14.8, RT 35.5 → 12.4,
RR 48.5 → 21.7; velocity PRMSE of the base alone with fixed random wrenches 14.9 / 36.8 → 5.9 / 13.5 (lin / ang).

## 4. Diagnostic ladder (N = 200 `fig4g` configurations, fine grand mobility, 3 seeds per φ)

Truth `v = M F` from a fine BatchedMFS grand mobility per configuration (`triton32`, tol_v 1e-5, ~4 min each on the
RTX 4060; gap to the cached gravity truth files 3e-6 … 3e-5, to the random-wrench files 4e-4 … 8e-4). Cumulative
rungs at switch_dist = pair_cutoff = 8, seed means in %, `lin` = translational PRMSE, `fluct` = PRMSE of the
fluctuations about the mean velocity. Full tables (also err_mean, ang) in `artifacts/fts_reflection_ladder.md`.

| rung | gravity φ=0.1 lin / fluct | gravity φ=0.2 lin / fluct | random φ=0.1 lin / fluct | random φ=0.2 lin / fluct |
|---|---|---|---|---|
| RPY | 3.95 / 20.9 | 7.72 / 49.0 | 10.38 / 20.4 | 22.74 / 46.2 |
| self + 2b NN (RPY beyond 8) | 3.49 / 19.9 | 6.97 / 47.1 | 8.12 / 15.9 | 19.41 / 39.6 |
| + pair moments model | 1.72 / 13.2 | 3.14 / 29.6 | 3.36 / 6.7 | 7.82 / 15.7 |
| **+ learned diagonal (shipped stack)** | **1.73 / 13.3** | **3.19 / 29.9** | **3.35 / 6.6** | **7.71 / 15.5** |
| exact pair residual (d ≤ 8) | 1.28 / 10.1 | 2.28 / 17.4 | 2.86 / 5.6 | 6.29 / 12.1 |
| + exact diagonal = **far-field floor** | **1.27 / 10.0** | **2.25 / 17.3** | **2.81 / 5.5** | **6.07 / 11.7** |
| + exact far residual (sanity) | 0.00 / 0.0 | 0.00 / 0.0 | 0.00 / 0.0 | 0.00 / 0.0 |
| RPY + single reflection | 0.30 / 2.4 | 0.55 / 5.4 | 2.33 / 4.7 | 5.35 / 11.0 |
| RPY + two reflections (Jacobi) | 0.38 / 2.4 | 1.20 / 8.5 | 2.47 / 5.0 | 6.54 / 13.5 |
| RPY + full series = SD_Minf | 0.26 / 1.8 | 0.75 / 5.6 | 2.15 / 4.4 | 5.15 / 10.5 |
| stack + ref1 unmasked (double counts) | 2.68 / 10.1 | 5.42 / 23.7 | 8.37 / 16.5 | 18.99 / 39.0 |
| **stack + ref1 on uncovered triplets (no retrain)** | **0.52 / 3.2** | **0.97 / 9.4** | **1.89 / 3.8** | **4.43 / 8.7** |
| **perfect near field + ref1 on far pair blocks (retrain floor)** | **0.26 / 2.1** | **0.54 / 4.7** | **0.62 / 1.2** | **1.51 / 3.0** |
| perfect near field + full series on far blocks | 0.15 / 1.1 | 0.40 / 2.5 | 0.34 / 0.7 | 1.01 / 2.0 |

- The far-field floor is 70–80 % of the shipped stack's gravity error and ~80 % of its random-wrench error; the
  learned near field is already close to its own floor.
- **RPY + one reflection alone** reaches 0.30 / 0.55 % on gravity — better than SD's converged series at φ = 0.2
  and equal at φ = 0.1. One reflection is both the cheapest and the best truncation: the two-reflection sweep
  overshoots (1.20 at φ = 0.2). The converged series is the exact FTS closure, which is itself truncated at the
  stresslet level; at close range its higher reflections compete with the higher multipoles it lacks, while the first
  reflection is the robust O(φ) piece.
- Adding the reflection everywhere on top of the shipped stack double-counts the triplets the learned models were
  trained on and hurts; restricted to uncovered triplets (k outside the pair's r_c = 8 midpoint shell, the diagonal's
  k beyond 8, every far pair) it already reaches 0.52 / 0.97 % (gravity) and 1.89 / 4.43 % (random) with no retraining.
- The floor after retraining on the reflected base is 0.26 / 0.54 % on gravity and 0.62 / 1.51 % on the random
  wrench — a 4–5× reduction of the Fig 3 floor.

## 5. Retrained models

Pipeline: `build_nbody_v2_cache.py --out data/multibody_v2_cache_pc8c --add-fts refl1` (93 s; writes
`Mref_refl1_ts (n,36)` and `Mref_refl1_tt (C,64,36)`, the latter without the two-body path within pair_cutoff = 8;
`meta["fts"]`) → `train_nbody_v2.py --label-base refl1` and `train_diag_v2.py --label-base refl1` with the published
pc8c recipe (100 epochs, batch 4096, lr 1e-3, L1 × 6π, seed 411, same 8,959,252 training rows) → sidecar
`fts_base: "refl1"`, asserted by the harness (`_fts_base_for`: pair and diag must agree) and required by the operator
(`fts_reflection="refl1"`, `pair_cutoff == switch_dist == 8`, same lock as the diagonal).

| validation (pc8c cache, config-level split) | published pc8c | on the reflected base |
|---|---|---|
| pair model: PRMSE lin / ang (%) | 3.87 / 15.5 | **2.89 / 9.38** |
| pair model: block rel. error total (TT / TR / RR) | 4.60 (2.89 / 16.6 / 10.4) | **3.25 (2.45 / 10.1 / 9.8)** |
| pair model: residual capture (100 = nothing learned) | 28.9 % of a large residual | 52.0 % of a 2.5× smaller residual |
| base alone: PRMSE lin / ang | 14.9 / 36.8 | 5.9 / 13.5 |
| diagonal model: capture (TT / TR / RR); velocity capture lin / ang | 53.9 (51.7 / 68.7 / 48.7); 54.4 / 53.3 | 53.9 (51.9 / 68.8 / 48.8); 54.5 / 53.2 |

The pair model's absolute block error is unchanged (0.29 × old residual ≈ 0.52 × new residual): the reflection
removes the smooth part it could learn anyway; the gain at N = 200 comes from the box-external part that only the
global term reaches. The diagonal labels barely change (far-k part only), so the diagonal model is a re-run.

## 6. Harness results (CPU op, all 10 seeds per cell)

### 6.1 Gravity, N = 200 (`fig4g`; SD / HIGNN comparison protocol)

| φ | metric | NeMO | **NeMO + FTS** | SD_Minf | SD | HIGNN_full | RPY |
|---|---|---|---|---|---|---|---|
| 0.025 | lin / fluct / mean / ang / max | 0.66 / 3.9 / 0.36 / 6.6 / 1.5 | **0.10 / 0.72 / 0.011 / 1.2 / 0.55** | 0.08 / 0.48 / 0.034 / 1.1 / 0.37 | 2.16 / 3.6 / 2.1 / 2.9 / 2.9 | 1.01 / 5.0 / 0.73 / — / 2.1 | 1.13 / 5.4 / 0.84 / 8.5 / 2.6 |
| 0.05 | | 1.08 / 7.1 / 0.52 / 9.4 / 2.0 | **0.17 / 1.23 / 0.024 / 2.1 / 0.59** | 0.13 / 0.86 / 0.065 / 1.3 / 0.50 | 4.88 / 7.1 / 4.8 / 6.0 / 5.8 | 1.93 / 9.8 / 1.4 / — / 3.5 | 2.07 / 10.1 / 1.6 / 12.7 / 3.9 |
| 0.075 | | 1.50 / 10.4 / 0.63 / 12.4 / 2.8 | **0.24 / 1.83 / 0.032 / 3.1 / 0.77** | 0.21 / 1.38 / 0.115 / 1.7 / 0.68 | 8.81 / 12.4 / 8.7 / 10.7 / 10.0 | — | 3.21 / 16.1 / 2.4 / 17.7 / 5.9 |
| 0.1 | | 1.73 / 13.1 / 0.64 / 14.4 / 3.1 | **0.31 / 2.46 / 0.052 / 3.8 / 0.97** | 0.27 / 1.83 / 0.156 / 1.9 / 0.83 | 11.87 / 16.2 / 11.8 / 14.3 / 13.2 | 3.74 / 20.7 / 2.8 / — / 6.7 | 4.02 / 21.2 / 3.1 / 22.2 / 7.1 |
| 0.125 | | 2.12 / 16.3 / 0.78 / 16.6 / 3.7 | **0.37 / 2.96 / 0.074 / 4.7 / 1.05** | 0.37 / 2.45 / 0.221 / 2.5 / 0.87 | 15.25 / 20.8 / 15.2 / 18.4 / 16.6 | — | 5.08 / 27.6 / 3.9 / 26.9 / 8.8 |
| 0.15 | | 2.43 / 20.2 / 0.84 / 19.5 / 4.2 | **0.45 / 3.91 / 0.078 / 5.8 / 1.14** | 0.50 / 3.47 / 0.316 / 3.3 / 1.05 | 18.91 / 25.2 / 18.8 / 22.4 / 20.3 | 5.47 / 33.0 / 4.0 / — / 9.7 | 5.97 / 34.2 / 4.6 / 33.1 / 10.2 |
| 0.175 | | 2.74 / 23.6 / 0.91 / 21.3 / 5.2 | **0.51 / 4.56 / 0.109 / 6.8 / 1.37** | 0.62 / 4.36 / 0.398 / 4.1 / 1.31 | 22.13 / 29.5 / 22.0 / 26.3 / 23.7 | — | 6.85 / 40.5 / 5.3 / 38.1 / 11.9 |
| 0.2 | | 3.19 / 29.3 / 1.11 / 25.1 / 6.6 | **0.57 / 5.50 / 0.108 / 8.2 / 1.52** | 0.77 / 5.66 / 0.509 / 5.0 / 1.48 | 25.18 / 33.2 / 25.1 / 29.7 / 26.7 | — | 7.74 / 48.4 / 6.0 / 45.7 / 13.5 |

(lin = translational PRMSE, fluct = fluctuation PRMSE, mean = error of the mean settling velocity, ang = rotational
PRMSE, max = max per-particle translational error; all %. HIGNN predicts no rotation.)

- Translational error 5.5–6.4× below the shipped stack at every φ; below SD_Minf for φ ≥ 0.15, within 1.1–1.4× below
  that; 10× below HIGNN full.
- The error of the collective settling velocity is 3–5× below SD_Minf everywhere (0.01–0.11 % vs 0.03–0.51 %) — the
  first reflection is the more accurate closure for the collective mode (§4).
- Fluctuations: 4–5× below the shipped stack; SD_Minf keeps a 1.1–1.5× edge at φ ≤ 0.15, equal at φ = 0.2.
- Rotation: 3–5× below the shipped stack; SD_Minf stays 1.1–1.7× better (its converged series carries the rotational
  reflections that one truncation does not).
- Worst particle: 2.7–4.3× below the shipped stack, equal to SD_Minf at φ ≥ 0.15.

### 6.2 Random wrench (paper Fig 3 protocol; PRMSE over all six components, %)

| N | op | φ = 0.025 | 0.05 | 0.075 | 0.1 | 0.125 | 0.15 | 0.175 | 0.2 |
|---|---|---|---|---|---|---|---|---|---|
| 200 | NeMO | 1.57 | 2.73 | 3.87 | 4.74 | 5.59 | 6.56 | 7.62 | 7.51 |
| 200 | **NeMO + FTS** | **0.60** | **0.91** | **1.25** | **1.85** | **2.03** | **2.58** | **3.05** | **3.22** |
| 300 | NeMO | 1.76 | 3.02 | 4.15 | 5.14 | 5.92 | 7.18 | 8.23 | 10.29 |
| 300 | **NeMO + FTS** | **0.71** | **0.99** | **1.27** | **1.68** | **2.00** | **2.45** | **2.99** | **3.46** |

Translational / rotational at N = 200, φ = 0.2: 7.68 / 6.30 → 3.14 / 4.11; max per-particle error 23.6 → 12.0 %.
The growth from N = 200 to 300 is gone (φ = 0.2: ×1.37 → ×1.07; φ = 0.1: ×1.08 → ×0.91). The shipped rows are the
`M_mom_v2_kinf_rc8_pc8_diag_final` certification part (bit-identical to pc8c) and include the 12 legacy gap-0.05
N = 200 truths, which both operators see alike.

### 6.3 Fig 4 grids, N ≤ 200

Random wrench (PRMSE %, vs N), shipped → FTS:

| φ | N = 20 | 50 | 100 | 200 |
|---|---|---|---|---|
| 0.05 | 1.85 → 0.96 | 2.07 → 0.92 | 2.14 → 0.73 | 2.67 → 0.80 |
| 0.1 | 2.55 → 1.54 | 3.19 → 1.66 | 4.55 → 2.04 | 4.28 → 1.55 |
| 0.15 | 3.70 → 2.51 | 3.97 → 2.56 | 4.99 → 2.40 | 6.09 → 2.50 |
| 0.2 | 3.88 → 2.78 | 4.73 → 3.09 | 6.97 → 3.96 | 7.75 → 3.48 |

Gravity (translational PRMSE / fluctuation PRMSE, vs N), shipped → FTS (HIGNN full for reference):

| φ | N = 20 | 50 | 100 | 200 | HIGNN full N = 200 |
|---|---|---|---|---|---|
| 0.025 | 0.43 / 2.9 → 0.17 / 1.3 | 0.50 / 3.7 → 0.12 / 0.9 | 0.57 / 3.9 → 0.11 / 0.8 | 0.66 / 3.9 → 0.10 / 0.7 | 1.01 / 5.0 |
| 0.05 | 0.68 / 4.5 → 0.33 / 2.6 | 0.75 / 5.5 → 0.27 / 2.1 | 0.78 / 5.7 → 0.18 / 1.3 | 1.08 / 7.1 → 0.17 / 1.2 | 1.93 / 9.8 |
| 0.1 | 0.66 / 5.5 → 0.48 / 4.1 | 1.02 / 7.3 → 0.43 / 3.6 | 1.23 / 10.3 → 0.37 / 3.0 | 1.73 / 13.1 → 0.31 / 2.5 | 3.74 / 20.7 |
| 0.15 | 0.87 / 8.5 → 0.69 / 6.7 | 1.22 / 8.8 → 0.63 / 5.7 | 1.48 / 13.0 → 0.52 / 4.5 | 2.43 / 20.2 → 0.45 / 3.9 | 5.47 / 33.0 |

With the reflection the gravity error falls with N where it used to grow: the large-N convergence of every pairwise
operator to the shared back-flow bias (`hignn_comparison_report.md` §4.3, ~3.3 % at φ = 0.1) no longer applies to
this stack, since that bias is the reflection term.

### 6.4 Clustered spheres (`cluster` protocol, PRMSE % / rotational PRMSE %, by surface gap δ)

| δ | 0.1 | 0.2 | 0.5 | 1.0 | 2.0 | 3.0 |
|---|---|---|---|---|---|---|
| NeMO (pc8 + diag) | 2.78 / 25.8 | 1.50 / 20.3 | 0.75 / 11.4 | 0.51 / 6.8 | 0.30 / 2.4 | 0.26 / 1.6 |
| **NeMO + FTS** | **2.45 / 18.0** | **0.95 / 9.2** | **0.36 / 4.2** | **0.16 / 2.1** | **0.08 / 0.49** | **0.03 / 0.18** |

At δ ≥ 2 (beyond the band support of the learned models) the analytic term does the work: 4–9× lower error.

### 6.5 Config-level velocity fine-tune (second lever)

`experiments/finetune_config_velocity.py` fine-tunes the FTS-base pair + diagonal models on the residual-space
velocity error of whole configurations (random-wrench + gravity patterns, L1 × 6π, plus the block L1 as anchor;
20 epochs, lr 1e-4, 64 configurations per step). On the training boxes it trades block accuracy for the coherent
gravity mode (gravity residual-velocity capture 85 → 63 %, random 50 → 55 %, pair PRMSE 2.89 → 3.16 %). On the harness
(`--models` overrides, tag `_ft`, part files `data/paper_accuracy_v2/parts/{fig4g_26583,fig3_27387}.csv`) it is a wash on
the gravity table (φ = 0.2: 0.565 vs 0.573 lin, 5.43 vs 5.50 fluct, rotation 7.0 vs 8.2) and a regression on Fig 3
(φ = 0.2: 3.44 vs 3.22 at N = 200, 3.76 vs 3.46 at N = 300). Not shipped; the velocity term dominates the loss ~50×, so
a re-run would need a much larger block anchor, and the headroom it targets (the near-field part of the gravity
fluctuations) is small next to what the reflection removed.

## 7. Cost and what is not done

- CPU operator: the dense chunked reflection adds ~0.14 s at N = 200 standalone (one core); inside the 8-worker
  harness the per-apply wall time went 3.4 → 5.7 s. It is O(N²) and stays usable to N ≈ 5000 (chunked, no dense G).
- GPU operator: `Mob_Nbody_Moments_Torch(fts_reflection="refl1")` carries the same dense torch term (registered as
  `M_mom_gpu_pc8c_fts_diag`); it has not been run here (no warp in the native env). The production far field still
  needs two widebvh treecode passes (a Stokeslet-gradient kernel for pass 1 and a stresslet kernel for pass 2 over the
  same tree and interaction lists); because the term is O(φ) of the velocity, both can run at loose accuracy (low
  `pdeg`, larger `mac`, fp32). Accepted budget ≤ ~15–20 % per step at N = 1M; the port is the next step.
- The Fig 4 grid was run to N = 200 (the random-wrench truths stop there except for the separately generated large-N
  cells); the chain suite / Fig 6 regression guard has not been re-run with the new models.
- Three tests in `tests/test_nbody_v2.py` (`test_v2_split_is_configuration_level`, `test_oracle_residual_reproduces_grand_M`,
  `test_harness_cases_and_metric`) fail on the committed code already (stale expected counts / smoke-cache composition);
  everything else in `tests/test_fts_rpy.py`, `test_nbody_moments.py`, `test_diag_moments.py`, `test_nbody_v2.py`, `test_sd_ops.py` passes.
- Harness gotcha met on the way: `paper_accuracy_v2.py` rewrites the whole main CSV after every configuration, so any
  read of `data/paper_accuracy_v2.csv` while a sweep is running sees a partially written file, and a kill mid-write
  truncates it. The file is git-tracked; it was restored once and the sweep re-run.
