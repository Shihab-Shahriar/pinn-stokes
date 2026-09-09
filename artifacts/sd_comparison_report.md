# Stokesian Dynamics vs NeMO on the Fig 3 protocol (N = 200, gravity forcing, MFS truth)

*2026-09-08. Code: `src/sd_ops.py` (Stokesian Dynamics as a harness operator), `benchmarks/paper_accuracy_v2.py`
(ops `SD`, `SD_Minf`), `benchmarks/sd_gravity_truth.py` (the four gravity truth cells the cluster never solved),
`benchmarks/sd_diagnostics.py` (evidence, `artifacts/sd_diagnostics.md`), `figures/fig_sd_compare.py` (figures +
`artifacts/sd_comparison_tables.md`), `tests/test_sd_ops.py`. Stokesian Dynamics checkout: Townsend,
github.com/Pecnut/stokesian-dynamics @ 6b9117d (JOSS 9(94) 6011, 2024), referenced through `SD_ROOT`, nothing copied
or edited.*

## 1. Bottom line

A new figure in the layout of the paper's Figure 3 — error vs volume fraction φ at N = 200, ten configurations per
point — compares the latest NeMO (moments + learned diagonal) with Stokesian Dynamics (SD), both scored against MFS
truth, under **uniform gravity** (F = (0, 0, −9.81), T = 0, the `fig4g` protocol). Figure 3 itself is untouched.

- **Standard SD (far field + pairwise lubrication) is *less* accurate than NeMO on this protocol at every volume
  fraction above the most dilute**, and the gap widens with φ: translational PRMSE 2.2 % at φ = 0.025, 11.9 % at φ = 0.1, 25.2 % at φ = 0.2, against NeMO's 0.66 / 1.73 / 3.19 % (3.3–8.1 × NeMO's error). Its error is a
  *collective* error — the settling speed of the whole cloud comes out ~10 % low at φ = 0.1 — and it is not a bug in
  the wrapper or in SD's two-body tables (§4). It is the known finite-cluster inconsistency of SD's pairwise-additive
  lubrication correction R₂B,exact − R₂B,∞ (Ichiki, *J. Fluid Mech.* 452, 2002): the correction is derived for an
  isolated pair and acts on absolute velocities, so it over-resists the strongly enhanced collective motion of an
  unbounded cloud (20 × the Stokes speed at φ = 0.1) and the error grows with every additional corrected pair.
- **SD's far-field part alone** — the many-body FTS (force–torque–stresslet) multipole inversion, an O(N³) dense
  solve with no learned component — **is the most accurate operator here on the collective mode**: translational
  PRMSE 0.08 / 0.27 / 0.77 % at φ = 0.025 / 0.1 / 0.2, versus NeMO 0.66 / 1.73 / 3.19 % and RPY 1.13 / 4.02 / 7.74 %. On the
  fluctuations about the mean settling velocity (the near-field, disorder part) it stays 0.5–5.7 % while NeMO is
  3.9–29.3 % (SD as shipped 3.6–33.2 %, RPY 5.4–48.4 %); on the induced rotations it is 1.1–5.0 % vs NeMO 6.6–25.1 % (SD as shipped 2.9–29.8 %).
- On the paper's own **random-wrench** protocol (random unit force and torque per sphere, same configurations) the
  same ordering holds for the translational block, with one reversal: SD's lubrication term makes its *rotational*
  velocities the most accurate of all operators (prmse_ang 1.3 % at φ = 0.1 vs NeMO 3.2 %, far field alone 3.7 %),
  while its translational error (9.4 %) is twice NeMO's (4.4 %) and 3.6 × the far field's (2.6 %). PRMSE over all six
  components at φ = 0.1: far field 2.8 %, NeMO 4.3 %, SD 8.7 %, RPY 12.2 %.

So the premise "SD should be more accurate than NeMO" holds only for SD's O(N³) far-field solve, not for SD as
shipped; and the far-field solve costs seconds per N = 200 evaluation (11N = 2200 unknowns, dense inverse) against
milliseconds for NeMO, and scales as N³.

## 2. Protocol

Configurations, seeds and truths are the harness's `fig4g` cells at N = 200 (the Fig 4 configurations,
seed = 123 + v_idx·1000 + 1300 + run, uniform RSA boxes with surface gap ≥ 0.1, radius 1, μ = 1), eight volume
fractions × ten seeds = 80 configurations, forcing F = (0, 0, −9.81), T = 0, unbounded fluid. Metrics are those of
`compute_error_stats` (`benchmarks/compare_nbody_moments.py`): `prmse_lin` = 100·‖U_pred − U_true‖_F/‖U_true‖_F
(headline; with T = 0 the six-component `rel_rmse` is the same number to 0.01), `prmse_fluct` (the same on
U − mean(U), i.e. the collective settling speed removed from both sides), `prmse_ang`, `max_rel_lin`, `err_mean_pct`.
The paper's random-wrench protocol (`fig4`, same configurations, |F| = |T| = 1 in random directions) is reported
alongside with the paper's `rel_rmse`.

Operators (harness names):

| op | what it is | cost at N = 200 |
|---|---|---|
| `M_mom_v2_kinf_rc8_pc8c_diag` | NeMO: 2-body NN to d ≤ 8, moments n-body pair correction, learned per-particle diagonal, RPY beyond 8 (CPU op) | ~5 s CPU op (ms on GPU) |
| `SD` | Stokesian Dynamics as shipped: R = (M∞)⁻¹ + Σ_pairs (R₂B,exact − R₂B,∞), FTS far field, lubrication for centre distance < 4 (`cutoff_factor` 2), FTE solve with E∞ = 0 | ~1.5–4 s (numba; 2200 × 2200 inverse + solve) |
| `SD_Minf` | the same with `use_Minfinity_only`: (M∞)⁻¹ alone, i.e. the many-body FTS multipole solution, no lubrication | same |
| `M_rpy` | RPY, the pairwise far field both NeMO and SD's M∞ reduce to | ~4 s CPU op |

`src/sd_ops.py::SDMob` calls the SD package's own matrix builders (`generate_grand_resistance_matrix`,
`fts_to_fte_matrix`, `construct_force_vector_from_fts`, `deconstruct_velocity_vector_for_fts`) with numba on; it
reproduces the package's native FTE path (`tests/test_all.py` two-sphere setups) to 1e-12
(`tests/test_sd_ops.py`), and its numba kernels equal the package's pure-Python path to 5e-16 on a 60-sphere subset.

## 3. Truths

| φ | gravity truth | provenance |
|---|---|---|
| 0.025, 0.05, 0.1, 0.15 | `tmp/nbody_moments_truth/uniform_N200_phi*_seed*_grav.npz`, `acc = broms_Xfine` | cluster, widebvh `sphere_mfs` (`benchmarks/broms_truth.py --forcing gravity`, 2026-09-06) |
| 0.075, 0.125, 0.175, 0.2 | same cache, `acc = batched_fine_triton32` | this laptop, `benchmarks/sd_gravity_truth.py`: `src/mfs_batched.py::BatchedMFS` (fine clouds, exact Oseen sum, GMRES converged on the velocities to 1e-5), positions read from the cell's cached random-forcing truth |

The local solver was gated on re-solving cached truths first (`--validate`, worst seam < 1e-3 or nothing is written):

| cached truth | acc | rel-L2 all | lin | ang |
|---|---|---|---|---|
| φ = 0.1, seed 4423, gravity | broms_Xfine | 2.7e-5 | 2.6e-5 | 2.2e-4 |
| φ = 0.025, seed 1423, gravity | broms_Xfine | 8.8e-6 | 7.9e-6 | 2.0e-4 |
| φ = 0.15, seed 6423, gravity | broms_Xfine | 4.8e-5 | 4.6e-5 | 3.9e-4 |
| φ = 0.1, seed 4423, random wrench | Xfine (GS, L_cut 25) | 2.9e-4 | 2.9e-4 | 3.0e-4 |
| φ = 0.2, seed 8423, random wrench | Xfine (GS, L_cut 25) | 8.4e-4 | 7.6e-4 | 1.1e-3 |

The gravity seams (≤ 5e-5 translational) are 20–100 × below the smallest error scored on the gravity protocol
(SD far field, 0.08 % at φ = 0.025, 0.21 % at φ = 0.075); the random-wrench seams (near-contact relative motion, where the fine clouds and the
GS solver's Oseen truncation both show) do not enter the gravity figure. The Xfine clouds were tried first and
abandoned: the fp32 Triton kernel pads one right-hand side to a 32-column tile, and at 486 × 425 points per sphere a
single N = 200 solve took minutes; the fine clouds solve in 3 s after a one-off 7-minute
autotune. The exact fp64 backend does not fit the 8 GB GPU at N = 200.

## 4. Why SD scores this way (`artifacts/sd_diagnostics.md`)

1. **Two spheres: SD is exact.** Against fp64 Xfine MFS, SD reproduces the pair velocities to ≤ 1e-7 at
   r = 2.5, 3.0, 3.9 for co-moving, opposite and torqued pairs (≤ 5e-4 at r = 2.1, the MFS clouds' own resolution
   limit at gap 0.1), while its far field alone is off by up to 15 % at r = 2.1 (squeezing pair). Beyond the
   lubrication cutoff (r = 4.4) the two variants coincide, 6e-4 from MFS. The wrapper and the two-body tables are
   right.
2. **Compact clusters: the pairwise term already hurts the collective mode.** Cube of 8 at spacing 2.3: random
   wrench SD 3.0 % vs far field 6.5 % (lubrication helps, as it should for relative motion); gravity SD 3.4 % vs far
   field 0.6 % (lubrication hurts). At spacing 3.0: 0.8 % vs 0.1 % under gravity.
3. **Cutoff sweep on the N = 200, φ = 0.1 gravity configuration.** With the lubrication cutoff lowered until no
   pair receives the correction, SD's translational error is 0.26 % (far field only); admitting pairs closer than
   2.5, 3, 4 (default), 4.5 radii gives 5.0, 8.2, 11.8, 12.7 % with the mean settling speed falling from 20.0 (MFS
   19.98) to 17.6 Stokes units. The error grows monotonically with the number of corrected pairs (0.9 → 5.8 → 8.2
   per sphere), and each pair's correction is tiny for co-moving spheres (0.4 % of the Stokes drag at r = 2.5,
   0.08 % at r = 3.9) — it is the sum of many small, mutually inconsistent pair corrections applied to a mode whose
   effective resistance is only 1/20 of a single sphere's that produces the 12 %.
4. **numba on = numba off** to 5e-16, so this is SD's own arithmetic.

The same mechanism, weaker, is visible on the random-wrench protocol (no collective mode, but every sphere has
~6 neighbours within the cutoff at φ = 0.1): translational error 9.4 % vs 2.6 % for the far field alone, while the
rotational block — where the pair correction is physically dominant and reciprocal — improves from 3.7 % to 1.3 %.

## 5. Results — gravity (main figure, `figures/fig_sd_compare_phi.*`; all metrics in `artifacts/sd_comparison_tables.md`)

Translational PRMSE (%), mean ± std over ten seeds:

| operator | φ=0.025 | 0.05 | 0.075 | 0.1 | 0.125 | 0.15 | 0.175 | 0.2 |
|---|---|---|---|---|---|---|---|---|
| SD far field only (FTS) | 0.08 ± 0.01 | 0.13 ± 0.02 | 0.21 ± 0.02 | 0.27 ± 0.02 | 0.37 ± 0.02 | 0.50 ± 0.03 | 0.62 ± 0.05 | 0.77 ± 0.05 |
| NeMO (moments + diagonal) | 0.66 ± 0.05 | 1.08 ± 0.07 | 1.50 ± 0.09 | 1.73 ± 0.06 | 2.11 ± 0.08 | 2.43 ± 0.13 | 2.74 ± 0.12 | 3.19 ± 0.15 |
| RPY | 1.13 ± 0.07 | 2.07 ± 0.09 | 3.21 ± 0.12 | 4.02 ± 0.16 | 5.08 ± 0.15 | 5.97 ± 0.15 | 6.85 ± 0.20 | 7.74 ± 0.26 |
| SD (far field + lubrication) | 2.16 ± 0.22 | 4.88 ± 0.35 | 8.81 ± 0.26 | 11.87 ± 0.49 | 15.25 ± 0.39 | 18.91 ± 0.38 | 22.13 ± 0.47 | 25.18 ± 0.43 |

Error ratio SD / NeMO: 3.3, 4.5, 5.9, 6.9, 7.2, 7.8, 8.1, 7.9; SD far field / NeMO: 0.12–0.24; RPY / NeMO: 1.7–2.5.

Almost all of SD's error is in the collective velocity (`err_mean_pct` 2.1 → 25.1 %, i.e. the cloud settles too
slowly), so it is *not* removed by looking at the fluctuations: fluctuation PRMSE SD 3.6 → 33.2 % vs NeMO 3.9 → 29.3 %,
RPY 5.4 → 48.4 %, far field 0.5 → 5.7 %. Rotational PRMSE: far field 1.1 → 5.0 %, SD 2.9 → 29.8 %, NeMO 6.6 → 25.1 %
(SD's lubrication term beats NeMO on the induced rotations only up to φ = 0.075), RPY 8.5 → 45.7 %. Worst-particle
translational error: far field ≤ 1.5 %, NeMO ≤ 6.6 %, RPY ≤ 13.5 %, SD ≤ 26.7 %. The NeMO and RPY rows at
φ ∈ {0.025, 0.05, 0.1, 0.15} are the rows of the HIGNN comparison (`artifacts/hignn_comparison_report.md`), reused.

Companion panels (`figures/fig_sd_compare_metrics.*`): fluctuation PRMSE, rotational PRMSE, worst-particle
translational error.

## 6. Results — random wrench (the paper's Fig 3 protocol on the same configurations; `figures/fig_sd_compare_random.*`)

PRMSE over all six components (%), mean over ten seeds:

| operator | φ=0.025 | 0.05 | 0.075 | 0.1 | 0.125 | 0.15 | 0.175 | 0.2 |
|---|---|---|---|---|---|---|---|---|
| SD far field only (FTS) | 1.46 | 1.72 | 2.33 | 2.76 | 3.76 | 4.13 | 5.47 | 5.79 |
| NeMO (moments + diagonal) | 1.70 | 2.67 | 3.67 | 4.28 | 5.50 | 6.09 | 8.57 | 7.75 |
| SD (far field + lubrication) | 1.33 | 3.21 | 6.09 | 8.73 | 10.19 | 13.78 | 15.06 | 18.58 |
| RPY | 4.44 | 6.67 | 10.03 | 12.23 | 16.56 | 18.46 | 24.71 | 23.69 |
| *paper's n-body (b1), for reference* | 2.32 | 3.80 | 6.00 | 7.28 | 9.85 | 11.54 | 16.12 | 14.69 |

Rotational block (`prmse_ang`, %): SD 0.26 / 0.49 / 0.99 / 1.30 / 1.64 / 2.34 / 2.63 / 3.25; NeMO 1.04 / 1.69 / 2.67 /
3.22 / 3.75 / 4.49 / 5.24 / 5.74; SD far field 1.54 / 2.21 / 3.11 / 3.70 / 4.56 / 5.56 / 6.76 / 8.00.

## 7. Caveats

- SD is an approximation (multipole far field truncated at the stresslet + pairwise lubrication); MFS with the
  Xfine/fine clouds is the truth here, as in the paper. The two-sphere check (§4.1) shows the truth resolves the
  near field at least as well as SD's exact tables down to gap 0.1.
- Non-periodic unbounded fluid, as in Fig 3. In a *periodic* suspension the mean fluid velocity is zero by
  construction and the collective mode is modest, which is where SD's pairwise lubrication is normally validated;
  the finite-cloud protocol is its unfavourable case. The random-wrench protocol has no collective mode and still
  shows the effect on the translational block.
- The SD far-field variant is not "Stokesian Dynamics" as usually meant; it is reported because it isolates what
  the lubrication term does and because it is the strongest reference available at O(N³).

## 8. Reproduce

```
export TORCH_COMPILE_DISABLE=1
python -m pytest tests/test_sd_ops.py -q
python -u benchmarks/sd_gravity_truth.py --validate --clouds fine                      # GPU; gated on the seams
CUDA_VISIBLE_DEVICES= python benchmarks/paper_accuracy_v2.py --exp fig4g --N 200 --ops SD SD_Minf M_rpy M_mom_v2_kinf_rc8_pc8c_diag --skip-done
CUDA_VISIBLE_DEVICES= python benchmarks/paper_accuracy_v2.py --exp fig4  --N 200 --ops SD SD_Minf --skip-done
python benchmarks/sd_diagnostics.py                                                    # artifacts/sd_diagnostics.md
python figures/fig_sd_compare.py [--with-rpy]                                          # figures + tables
```
