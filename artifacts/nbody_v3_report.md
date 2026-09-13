# Moments n-body pair correction, v3: 5 bands, linear bases, split TR/RT corners

*2026-09-13, branch `moments-v3-main` (from `main`; no FTS-reflection code involved). Published model
`data/models/nbody_moments_v3_nb5lin_tr2_kinf_rc8_pc8c.pt` (+ `.json` sidecar), trained on
`data/multibody_v2_cache_pc8c` with the plain residual labels `R = Mts_sym − M2b`, selection K = ∞, r_c = 8,
pair_cutoff = switch_dist = 8, same recipe as the shipped `nbody_moments_v2_kinf_rc8_pc8c.pt` (100 epochs, seed 411).*

## 1. Summary

The v2 pair model (8 unit-width tent bands, 76 invariants, 93 coefficients on 34 TT + 34 RR + 25 TR bases) was
simplified to **5 bands on knots {0.5, 1.5, 2.5, 3.5, 4.5}** (the last band saturates over the 4.5–8 shell), **linear
bases only** (the quadratic `v_a v_aᵀ` and `(z·v_a)E(v_a)` dropped) and **49 invariants → 60 coefficients**, and one
structural error was fixed: v2 wrote the same 3×3 matrix into both off-diagonal corners of the pair block
(`RT = TR`), which caps the reachable TR/RT residual at 40 % of the label's; v3 adds a second class of TR bases
placed with opposite signs in the two corners (`TR = T1 + T2`, `RT = T1 − T2`). Reciprocity `M_ji = M_ijᵀ` and O(3)
equivariance still hold exactly (tests).

| dataset-v2 validation (1 M held-out pairs, all φ / families; random unit force + torque per pair) | coef | lin % | ang % | block % | TT | TR | RT | RR |
|---|---|---|---|---|---|---|---|---|
| v2 pc8c (shipped) | 93 | 3.87 | 15.53 | 4.60 | 2.89 | 16.65 | 15.91 | 10.40 |
| v3 bands + linear bases, `RT = TR` kept (ablation) | 45 | 3.95 | 15.55 | 4.67 | 2.99 | 16.68 | 15.93 | 10.39 |
| **v3 (published)** | **60** | **3.25** | **8.39** | **3.50** | 3.00 | **8.45** | **8.15** | 10.43 |

The size reduction is free (row 2 vs row 1: within 0.1 on every column); the whole angular gain and most of the
translational one come from the split corners (row 3 vs row 2). TT and RR are single blocks where the old rule is
necessary, and they are unchanged.

## 2. Evidence behind the design (published v2 model, 300k validation rows, post-hoc coefficient zeroing)

| zeroed group | residual ‖pred−R‖/‖R‖ % (full model 28.9) | lin | ang |
|---|---|---|---|
| bands 5 / 6 / 7 / 8 (all types) | 29.9 / 29.4 / 29.2 / 29.1 | 4.04 / 3.95 / 3.91 / 3.89 | 15.7 / 15.6 / 15.6 / 15.5 |
| bands 1 / 2 / 3 / 4 | 30.7 / **38.9** / 34.9 / 32.5 | 4.07 / 5.53 / 4.87 / 4.49 | 16.7 / 17.6 / 16.7 / 16.0 |
| all 24 quadratic bases | 29.8 | 4.02 | 15.65 |
| TT `Alt(z v_aᵀ)` / `S(zzᵀQ_a)` / `Q_a` | 37.7 / 35.0 / 33.7 | 5.49 / 5.00 / 4.76 | |
| constants only (old 5-coefficient structure) | 52.2 | 7.63 | 20.9 |

Bands 5–8 are populated (mean s_a 4.4–4.7) but carry almost nothing; bands 1–4 carry the correction, band 1–2
neighbours are rare (occupancy 0.18 / 0.89) but the strongest screeners. The label's TT block is 31 % antisymmetric,
so `Alt(z v_aᵀ)` stays.

**The corner tie.** In the residual labels `‖TR − RT‖ / ‖TR‖ = 0.80` (uniform 0.72, grown 0.85, lattice 0.66, chain 0.90),
so the best any `RT = TR` model can do is the corner mean, a floor of `‖(TR−RT)/2‖/‖TR‖ = 40 %` of the TR residual;
the v2 model sat at 43 / 45 %. This is why capacity ×3.2, cross-band invariants, l = 2 completion and l = 3 octupoles
(`nbody_diag_report.md` §6–7) all left the angular blocks at ~17 %: the limit was in the assembly. Reciprocity only
needs `TR_ji = RT_ijᵀ`, which class-2 tensors with `T(−z) = −T(z)ᵀ` satisfy when placed as `TR = +T`, `RT = −T`
(`moments_for_nbody.md` §5.4). Per band: `E(v_a)`, `(z·v_a) E(z)`, `[E(z), Q_a]`. The two-body block has no such term
(no vector but z), which is how the tie was inherited from Eq. 17.

## 3. Paper harness (warp-free CPU ops, cached MFS truths, 10 seeds; both stacks with the same `nbody_diag_v2_pc8c.pt`)

Fig 3 protocol (random force + torque on every particle), total relative RMSE % / angular PRMSE %:

| N | φ = 0.025 | 0.05 | 0.075 | 0.10 | 0.125 | 0.15 | 0.175 | 0.20 |
|---|---|---|---|---|---|---|---|---|
| 200, v2 + diag | 1.57 / 0.93 | 2.73 / 1.83 | 3.87 / 2.63 | 4.74 / 3.21 | 5.59 / 3.97 | 6.56 / 4.73 | 7.62 / 5.25 | 7.51 / 6.30 |
| 200, **v3 + diag** | 1.54 / 0.70 | 2.65 / 1.32 | 3.79 / 1.93 | 4.66 / 2.39 | 5.47 / 3.02 | 6.49 / 3.62 | 7.58 / 4.16 | 7.39 / 4.82 |
| 300, v2 + diag | 1.76 / 1.01 | 3.02 / 1.90 | 4.15 / 2.59 | 5.14 / 3.32 | 5.92 / 3.94 | 7.18 / 4.77 | 8.23 / 5.29 | 10.29 / 6.09 |
| 300, **v3 + diag** | 1.72 / 0.76 | 2.96 / 1.40 | 4.06 / 1.92 | 5.04 / 2.49 | 5.81 / 3.03 | 7.17 / 3.73 | 8.23 / 4.35 | 10.36 / 5.29 |

Fig 4 protocol, φ = 0.2, mean over N = 20…200: total 5.77 → **5.37 %**, translational 5.91 → 5.73, angular
5.36 → **3.69**; lower at every N. Gravity (force only), N = 200, translational PRMSE by φ (0.025 … 0.2):
0.658 / 1.083 / 1.503 / 1.725 / 2.115 / 2.427 / 2.739 / 3.191 → 0.663 / 1.110 / 1.547 / 1.790 / 2.193 / 2.449 / 2.693 / **2.987**;
mean settling-velocity error at φ = 0.2 1.11 → **0.56 %**. The translational error at φ = 0.2 is the uncorrected
d > 8 far field (`fig3_n300_investigation.md`) and does not move; the angular velocities improve 13–31 % everywhere.

## 4. Figure 7 (Wilson three-sphere triangle; Ω = base-sphere rotation from the apex force, an RT quantity)

| Ω error vs MFS Xfine, S = | 2.01 | 2.10 | 2.50 | 3.00 | 4.00 | 6.00 |
|---|---|---|---|---|---|---|
| v2 + diag | +78 % | +36 % | +14 % | +8 % | +3 % | +0.5 % |
| **v3 + diag** | +27 % | +2 % | −1 % | +0.2 % | −2 % | −0.1 % |
| Stokesian Dynamics | −18 % | −1.3 % | 0.0 % | −0.5 % | −0.2 % | −0.1 % |

U3 at S = 2.1 +7.5 % → +0.4 %; U1, U2 unchanged. Mean |deviation| from Wilson over S ≥ 2.1: 0.0027 → **0.0012**
(SD 0.0035); NeMO closer to Wilson than SD in 12/24 cells (v2: 10). `figures/fig7_helen_3body.*`, `reproduction.md`.

## 5. Implementation

`src/nbody_moments.py` is parametrised (knot bands via `band_weights_knots`, bitwise-equal to v2 at the v2 knots;
`bases_v3(z, v, Q, use_quadratic, has_tr2)`; `assemble_block_v3(c, tt, tr1, tr2)`); the v2-named functions are
wrappers, so the published v2 models and `.wt` files are unaffected (v2 defaults reproduce the shipped model bitwise,
strict `load_state_dict` verified). Rows are `7 + 13·NB` columns; `unpack_features` reads NB from the width.
`MultiBodyMoments(bands=, bases=, invariants=)` keeps the knots in a non-persistent buffer `band_knots` and plain
attributes, readable after `torch.jit.load` (`nbody_moments.layout_of_model`, `knots_of`); **rows are always built
with the model's own knots** (`nbody_features.moment_features(knots=)`, the operator's `_build_rows`, the trainer's
`V2Cache(knots=)`). A `.wt` needs its layout (sidecar next to it, written per run as `<out>/model.json`, or
`Mob_Op_Nbody_Moments(nbody_layout=)`); a `.pt` is self-describing. Trainer flags `--bands`, `--bases
{v2,linear,linear_tr2,v2_tr2}`, `--invariants {full,reduced}`; sidecar keys `bands/bases/invariants/layout/n_in/n_coef/x_dim`;
`fmt()` now prints RT. Harness ops `M_mom_v3_nb5lin_tr2_pc8c[_diag]`, sidecar layout asserted by `_assert_layout`.
The GPU path (`src/gpu_nbody_moments.py`) hard-codes the v2 layout and asserts it; the v3 assembly is linear in the
moments, so its warp port reduces to coefficient-weighted moment sums (follow-up). Tests: `tests/test_nbody_moments.py`
§13 (knot partition of unity, v2 defaults unchanged, layout dims, reciprocity with class 2 active, O(3) incl.
reflections, numpy oracle `tests/nbody_moments_ref.py`, TorchScript round trip, operator rows and grand-M symmetry).

Runtime was not the motivation: at N = 1M the assemble+apply kernel is 0.11 s of a 2.97 s step
(`gpu_moments_port_report.md`); NB 8 → 5 and 93 → 60 save ~0.2 s/step once ported.

## 6. Pending / follow-ups

- Grid runs still training when this was written (`experiments/runs_v2/v3_*`, log `v3_grid_main.log`): 8 bands linear
  (quadratic drop alone), 4 bands + class 2, 64-wide MLP, reduced invariants, second control seed (noise floor).
- Diagonal model: same audit (its `RT = TRᵀ` is correct for a symmetric block); bands 5–8 of its [2, 8] layout are
  likely prunable the same way.
- GPU port of the v3 layout; `experiments/finetune_config_velocity.py` needs `layout=` for v3 `.wt` files.
- Pre-existing test failures unrelated to this work: `test_v2_split_is_configuration_level`,
  `test_oracle_residual_reproduces_grand_M`, `test_harness_cases_and_metric` (all fail identically on `main`).
