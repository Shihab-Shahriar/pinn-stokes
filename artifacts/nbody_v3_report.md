# Moments n-body pair correction, v3: split TR/RT corners, linear bases

*2026-09-13/14, branch `moments-v3-main` (from `main`; no FTS-reflection code involved). All runs on
`data/multibody_v2_cache_pc8c` with the plain residual labels `R = Mts_sym − M2b`, selection K = ∞, r_c = 8,
pair_cutoff = switch_dist = 8, the recipe of the shipped `nbody_moments_v2_kinf_rc8_pc8c.pt` (100 epochs, seed 411).*

**Decision (2026-09-14).** The adopted v3 pair model keeps two changes and nothing else: the class-2 TR bases that untie
the pair block's TR and RT corners (§2) and the removal of the 24 quadratic bases (free). Bands stay the v2 unit bands
0.5 … 7.5, so the model has v2's size (76 inputs, 93 coefficients). Published as
**`data/models/nbody_moments_v3_nb8lin_tr2_kinf_rc8_pc8c.pt`** (run `experiments/runs_v2/v3_nb8lin_tr2`, `--bases
linear_tr2`; harness ops `M_mom_v3_nb8lin_tr2_pc8c[_diag]`). The 5-knot band merge of §1–5 (the first candidate,
`nbody_moments_v3_nb5lin_tr2_*`, commit e2f401d) and the learned radial bands of §7 were evaluated and not adopted
(§6: the merge costs 0.4–0.6 Fig-3 points; §7: the learned bands gain 0.27–0.48 points, judged not worth a new mechanism).

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
| v3 first candidate: 5 knots 0.5 … 4.5 + tail (superseded, §6) | 60 | 3.25 | 8.39 | 3.50 | 3.00 | 8.45 | 8.15 | 10.43 |
| **v3 adopted: v2 bands, linear bases, class 2** | **93** | **3.15** | **8.29** | **3.40** | 2.90 | **8.36** | **8.04** | 10.46 |

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
| 200, **adopted v3 (v2 bands) + diag** | 1.51 / 0.67 | 2.60 / 1.29 | 3.74 / 1.90 | 4.56 / 2.34 | 5.44 / 2.99 | 6.36 / 3.53 | 7.25 / 3.98 | 7.01 / 4.69 |
| 300, **adopted v3 (v2 bands) + diag** | 1.70 / 0.74 | 2.92 / 1.37 | 4.03 / 1.90 | 5.00 / 2.45 | 5.72 / 2.93 | 7.05 / 3.71 | 7.84 / 4.25 | 9.72 / 5.02 |

(The two "v3 + diag" rows are the 5-knot candidate; the "adopted v3" rows are the published model, better than v2 at
every (N, φ) and than the 5-knot candidate at every φ ≥ 0.1. Fig 4 φ = 0.2 mean 5.77 → 5.17 %, gravity φ = 0.2
lin 3.19 → 2.82 %, settling bias 1.11 → 0.79 %.)

Fig 4 protocol, φ = 0.2, mean over N = 20…200: total 5.77 → **5.37 %**, translational 5.91 → 5.73, angular
5.36 → **3.69**; lower at every N. Gravity (force only), N = 200, translational PRMSE by φ (0.025 … 0.2):
0.658 / 1.083 / 1.503 / 1.725 / 2.115 / 2.427 / 2.739 / 3.191 → 0.663 / 1.110 / 1.547 / 1.790 / 2.193 / 2.449 / 2.693 / **2.987**;
mean settling-velocity error at φ = 0.2 1.11 → **0.56 %**. The translational error at φ = 0.2 is the uncorrected
d > 8 far field (`fig3_n300_investigation.md`) and does not move; the angular velocities improve 13–31 % everywhere.

## 4. Figure 7 (Wilson three-sphere triangle; Ω = base-sphere rotation from the apex force, an RT quantity)

| Ω error vs MFS Xfine, S = | 2.01 | 2.10 | 2.50 | 3.00 | 4.00 | 6.00 |
|---|---|---|---|---|---|---|
| v2 + diag | +78 % | +36 % | +14 % | +8 % | +3 % | +0.5 % |
| v3 5-knot candidate + diag | +27 % | +2 % | −1 % | +0.2 % | −2 % | −0.1 % |
| **adopted v3 (v2 bands) + diag** | +22 % | −2 % | −1 % | +1 % | −2 % | −2 % |
| Stokesian Dynamics | −18 % | −1.3 % | 0.0 % | −0.5 % | −0.2 % | −0.1 % |

U3 at S = 2.1 +7.5 % → +0.4 % (5-knot) / +3.6 % (adopted); U1, U2 unchanged. Mean |deviation| from Wilson over
S ≥ 2.1: 0.0027 → **0.0012** for both v3 models (SD 0.0035); closer to Wilson than SD in 12/24 (5-knot) / 10/24
(adopted) cells (v2: 10). `figures/fig7_helen_3body.*` now carries the adopted model; `reproduction.md`.

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

## 6. Band-width study: are unit bands with a saturating tail ad hoc? (2026-09-13)

The published knots {0.5, 1.5, 2.5, 3.5, 4.5} are unit-spaced with the last band aggregating the 4.5–8 shell. The
alternative with no free choice is NB equal-width bands over the whole [0, 8] range, knots (a + ½)·8/NB, so the
question is how accuracy depends on the band width w at fixed recipe (linear bases + class-2 corners, same training).

| layout (all linear + class 2) | knots | coef | lin % | ang % | TT | TR | RT | RR | Fig 3 N=200 φ=0.2 total / ang | Fig 4 φ=0.2 mean | gravity φ=0.2 lin / settling bias |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **published**: unit to 4.5 + tail | 0.5 … 4.5 | 60 | **3.25** | **8.39** | 3.00 | 8.45 | 8.15 | 10.43 | **7.39 / 4.82** | **5.37** | **2.99 / 0.56** |
| 4 equal, w = 2 | 1, 3, 5, 7 | 49 | 3.80 | 9.80 | 3.53 | 9.68 | 9.36 | 13.32 | 8.32 / 5.47 | 5.85 | 3.71 / 1.65 |
| 5 equal, w = 1.6 | 0.8 … 7.2 | 60 | 3.44 | 8.90 | 3.18 | 9.00 | 8.62 | 11.27 | 7.38 / 4.90 | 5.39 | 3.05 / 1.02 |
| 6 equal, w = 4/3 | 0.67 … 7.33 | 71 | 3.35 | 8.54 | 3.11 | 8.57 | 8.26 | 10.95 | 7.55 / 4.77 | 5.39 | 3.11 / 1.09 |
| **8 equal, w = 1** (= v2 bands + class 2) | 0.5 … 7.5 | 93 | **3.15** | **8.29** | 2.90 | 8.36 | 8.04 | 10.46 | **7.01 / 4.69** | **5.17** | **2.82** / 0.79 |
| v2 control (8 unit, `RT = TR`) | 0.5 … 7.5 | 93 | 3.87 | 15.53 | 2.89 | 16.65 | 15.91 | 10.40 | 7.51 / 6.30 | 5.77 | 3.19 / 1.11 |

(All harness rows with the same `nbody_diag_v2_pc8c.pt`; Fig 4 mean over N = 20 … 200; settling bias = mean
translational error of the gravity protocol at N = 200.)

**4 equal bands (w = 2).** Occupancy is what the equal layout fixes — mean band weights 1.2 / 5.6 / 8.9 / 7.8 against
0.18 / 0.9 / 2.1 / 3.6 / 16.9 for the published knots (200k validation pairs) — but the leverage sits in the sparse
inner bands. The model keeps most of the angular gain (that comes from the split corners, not the bands) and gives
back the whole translational one: TT 3.00 → 3.53 and RR 10.4 → 13.3 (worse than v2), and on the paper protocols it
is worse than the v2 model everywhere translational (Fig 3 N = 200 total 8.32 vs 7.51, N = 300 11.68 vs 10.29, Fig 4
φ = 0.2 5.85 vs 5.77, gravity φ = 0.2 3.71 vs 3.19, settling bias 1.65 vs 1.11 %). By pair distance the loss is at
d = 3–5 (lin 3.1 → 3.9 and 3.4 → 4.6 in the 3–4 and 4–5 bins; ≤ 0.2 elsewhere): that is the range where a third
sphere wedges between the pair at r < 2 from the midpoint, and with knots at 1 and 3 such a neighbour is one saturated
band plus a 4-wide tent, so r = 1.5 and r = 2.5 are barely distinguishable. Both runs are converged (validation flat
over the last 20 epochs); the ~0.55-point lin gap is far above the run-to-run noise (8-band linear vs v2 control:
0.006 lin / 0.008 ang).

**8 equal unit bands (w = 1).** This is the v2 band layout with the 24 quadratic bases swapped for the 24 class-2 corner
bases — the same 76 inputs and 93 coefficients as v2, no band tuning at all. On validation it beats the published
5-knot model by 0.10 lin (TT 3.00 → 2.90: every 8-band run sits at TT 2.9, every 5-knot run at 3.0, so the merged
4.5–8 tail costs a small but systematic amount) and it beats it more clearly on the paper protocols: Fig 3 φ = 0.2
total 7.39 → **7.01** (N = 200) and 10.36 → **9.72** (N = 300, where the 5-knot model had been 0.07 *worse* than v2),
Fig 4 φ = 0.2 mean 5.37 → **5.17** with the gain growing with N (N = 180/200: 8.38 → 7.74, 7.79 → 7.21), gravity
φ = 0.2 lin 2.99 → 2.82 (settling bias 0.56 → 0.79, both far below v2's 1.11). The far-shell moments matter more in
the dense N = 200–300 boxes than in the pair-averaged validation metric. Truncating the bands at 4.5 is therefore not
free: it buys 33 coefficients for ~0.4–0.6 points of Fig 3 error. The uniform 8-band layout also keeps the GPU port's
accumulate and invariant kernels unchanged (NB = 8, 76 inputs, 93 coefficients); only the assemble kernel's basis set
changes.

**Trend and conclusion.** Validation improves monotonically with band resolution (w = 2 → 1.6 → 4/3 → 1: lin 3.80 →
3.44 → 3.35 → 3.15, ang 9.80 → 8.90 → 8.54 → 8.29). On the harness, paired over the 10 seeds (± = standard error of
the mean difference): 8 equal − published 5 knots = **−0.38 ± 0.08** (Fig 3 N = 200), **−0.64 ± 0.08** (N = 300),
**−0.20 ± 0.02** (Fig 4 φ = 0.2, 140 cells) — 4–8 standard errors. 5 equal − 5 knots = −0.01 ± 0.06 / −0.06 ± 0.06
/ +0.02 ± 0.02, and 6 equal − 5 equal = +0.17 ± 0.05 / +0.14 ± 0.05 / +0.01 ± 0.02: with 5–6 bands the harness totals
sit on a plateau and the knot placement is irrelevant to them (the published knots' one advantage over 5 equal bands
is the gravity settling bias, 0.56 vs 1.02 %). Notably the published 5-knot model's *total* gain over v2 on Fig 3 was
marginal (−0.12 ± 0.07 at N = 200, +0.07 ± 0.10 at N = 300; its gain was all angular), whereas 8 equal − v2 =
−0.49 ± 0.05 / −0.58 ± 0.09 is the first real total-error improvement of the corner fix. **Recommendation:** adopt the
uniform unit bands (the v2 layout, no band choice to defend) with linear bases and the class-2 corners as v3 — v2's
own size (76 inputs, 93 coefficients), a one-line description ("v2 with the quadratic bases replaced by the corner
bases"), and an unchanged GPU accumulate/invariant path. **Adopted 2026-09-14** and published as
`nbody_moments_v3_nb8lin_tr2_kinf_rc8_pc8c.pt` (run `experiments/runs_v2/v3_nb8lin_tr2`).

## 7. Learned radial bands: Bessel basis × MLP instead of hand-placed tents (2026-09-13; evaluated, not adopted)

The tent bands are a hand-placed radial basis (linear B-splines on knots), and §6 shows the accuracy depends on where
they sit. The standard remedy in equivariant interatomic potentials (DimeNet, NequIP, MACE) is a *learned* radial
function on a smooth fixed basis, which is now an option of the same model (`MultiBodyMoments(radial="bessel", nb=NB)`;
trainer `--radial bessel --nb NB [--n-radial 8]`): the NB band weights are

    w(r) = MLP( sqrt(2/r_c) sin(k π r / r_c) / r · u(r / r_c) ),  k = 1 … 8,   u(d) = 1 − 28 d⁶ + 48 d⁷ − 21 d⁸,

with r_c = 8 (the neighbour-selection cutoff; u and its first two derivatives vanish there) and a bias-free
8 → 32 → NB SiLU MLP (448 weights, `model_archs.RadialBands`), so every band is a smooth learned function of the
midpoint distance that goes to zero at the cutoff. Nothing downstream changes — the weights still depend on the scalar
distance only, so O(3) invariance, reciprocity and the row layout are as before — but the row is now a function of the
parameters, so the model builds its own rows (`MultiBodyMoments.moment_features`, differentiable; used by the trainer
through `V2Cache.features(..., model)` and by the operator through `nbody_features.moment_features_model`, TorchScript
exported and checked at save time). The tent models are untouched (v2 state dicts identical, all previous paths
bitwise). The only choices left are NB and the basis size.

| layout (linear bases + split TR/RT, same recipe) | coef | val lin % | ang % | TT | TR | RT | RR | Fig 3 φ=0.2 N=200 / 300 total (ang) | Fig 4 φ=0.2 mean | gravity φ=0.2 lin / bias |
|---|---|---|---|---|---|---|---|---|---|---|
| 8 uniform unit tent bands (baseline, §6) | 93 | 3.15 | 8.29 | 2.90 | 8.36 | 8.04 | 10.46 | 7.01 (4.69) / 9.72 (5.02) | 5.17 | 2.82 / 0.79 |
| **8 learned bands** (`v3_bessel_nb8_tr2`) | 93 | **3.04** | **7.98** | **2.81** | **7.99** | **7.74** | **9.93** | **6.74 (4.31) / 9.24 (4.59)** | **5.11** | **2.77 / 0.66** |
| 6 learned bands (`v3_bessel_nb6_tr2`) | 71 | 3.10 | 8.08 | 2.87 | 8.10 | 7.85 | 10.07 | 6.97 (4.43) / 9.61 (4.70) | 5.19 | 3.04 / 0.97 |

Paired over the 10 seeds, 8 learned − 8 uniform: **−0.27 ± 0.08** (Fig 3 N = 200), **−0.48 ± 0.08** (N = 300),
**−0.06 ± 0.02** (Fig 4 φ = 0.2, 140 cells); every block of the validation error improves, RR most (10.46 → 9.93).
Gravity at φ ≤ 0.125 is 0.02–0.06 worse and at φ ≥ 0.175 better; the settling bias at φ = 0.2 improves 0.79 → 0.66.
Six learned bands (71 coefficients) beat the 8 uniform tents on validation (3.10 / 8.08) and match them on the harness
(−0.04 ± 0.07 / −0.11 ± 0.06 / +0.02 ± 0.02) but trail the 8 learned bands (+0.23 ± 0.04 / +0.37 ± 0.04 / +0.08 ± 0.01),
so the band count still matters. **Not adopted (2026-09-14):** the gain over the uniform tents was judged too small for
a new band mechanism; the code path stays as an off-by-default option (`--radial bessel`), the model is not published
and its harness rows are kept under the tag `_bes8`. Figure 7 with it (Wilson three-sphere
Ω error, S = 2.01 / 2.1 / 2.5 / 3 / 4 / 6): +11 / −5 / −2 / −1 / −3 / −2 % against +27 / +2 / −1 / +0 / −2 / −0 % for the
5-knot model and +78 / +36 / +14 / +8 / +3 / +0.5 % for v2 — better at contact, a small uniform under-prediction beyond,
with the mean |deviation| from Wilson over S ≥ 2.1 at 0.0015 (5-knot 0.0012, v2 0.0027, SD 0.0035).

Fig 3 protocol over all φ (registered ops, 10 seeds; total relative RMSE % / angular PRMSE %), learned-band v3 + diag
against the 5-knot v3 + diag of §3:

| N | φ = 0.025 | 0.05 | 0.075 | 0.10 | 0.125 | 0.15 | 0.175 | 0.20 |
|---|---|---|---|---|---|---|---|---|
| 200, 5-knot v3 + diag | 1.54 / 0.70 | 2.65 / 1.32 | 3.79 / 1.93 | 4.66 / 2.39 | 5.47 / 3.02 | 6.49 / 3.62 | 7.58 / 4.16 | 7.39 / 4.82 |
| 200, **learned-band v3 + diag** | 1.53 / 0.67 | 2.67 / 1.29 | 3.82 / 1.89 | 4.60 / 2.29 | 5.47 / 2.91 | 6.33 / 3.43 | 7.12 / 3.79 | 6.74 / 4.31 |
| 300, 5-knot v3 + diag | 1.72 / 0.76 | 2.96 / 1.40 | 4.06 / 1.92 | 5.04 / 2.49 | 5.81 / 3.03 | 7.17 / 3.73 | 8.23 / 4.35 | 10.36 / 5.29 |
| 300, **learned-band v3 + diag** | 1.73 | 2.99 | 4.11 | 5.08 | 5.73 | 6.94 | 7.60 | 9.24 / 4.59 |

Equal within 0.05 at φ ≤ 0.1 (the far-field floor), and better by a margin growing with φ from φ = 0.125 on.

**What it learned** (`figures/nbody_learned_bands.png`): all eight functions live inside r ≈ 4, three of them with
distinct shapes in r < 2 (peaks at 0, 0.85, 1.4–1.8 — the region where a third sphere wedges between the pair), and
nothing beyond r ≈ 5, which is the data-driven version of the §2 ablation with no knot placed by hand. It is not a
partition of unity and the bands change sign; the scale is absorbed by the normalisation buffers. Cost: training
takes ~75 min instead of ~35 (the per-neighbour MLP and its backward pass); at inference the per-neighbour work is an
8 → 32 → NB MLP instead of a tent lookup, and for the GPU port w(r) tabulates into a distance LUT (as the two-body path
already does), so the accumulate kernel is unchanged.

## 8. Pending / follow-ups

- Remaining grid (`experiments/runs_v2/v3_*`, log `v3_grid_main.log`), validation lin / ang and blocks TT TR RT RR:
  - 4 unit bands + class 2, truncated at 3.5 (knots 0.5 … 3.5, 49 coef): **3.69 / 9.15**, 3.44 9.20 8.94 11.01 — one
    radius shorter than the published 4.5 costs another 0.44 lin points (truncation 8 → 4.5 → 3.5: lin 3.15 → 3.25 → 3.69).
  - 5 knots + class 2 with a 64-wide MLP (64 64 64 64, ~4× fewer weights): 3.29 / 8.50, 3.04 8.56 8.24 10.70 — within
    0.05 lin / 0.11 ang of the 128-wide run, so the MLP width is not where the accuracy sits.
  - 5 knots + class 2 with the reduced invariant set (5 per band: s_a, |v_a|², (z·v_a)², zᵀQ_a z, tr Q_a²; 29 inputs
    instead of 49): 3.29 / 8.45, 3.04 8.52 8.21 10.51 — within 0.04 lin / 0.06 ang of the full set; the four dropped
    invariants per band (the mixed v–Q and z–Q–v terms) carry almost nothing.
  - v2 control with seed 412: 3.872 / 15.520, 2.89 16.65 15.90 10.39 against seed 411's 3.871 / 15.525, 2.89 16.65
    15.91 10.40 — the **training noise floor is ≤ 0.01 points** on every column (8.9 M rows, zero train/val gap), so
    every difference quoted in this report above 0.05 is real. Done so far: 8 bands linear (quadratic drop alone) lin 3.877 / ang 15.533, TT 2.90 TR 16.64
  RT 15.91 RR 10.50 = the v2 control within 0.01, so the quadratic bases are free to drop.
- Diagonal model: same audit (its `RT = TRᵀ` is correct for a symmetric block); bands 5–8 of its [2, 8] layout are
  likely prunable the same way.
- GPU port of the v3 layout; `experiments/finetune_config_velocity.py` needs `layout=` for v3 `.wt` files.
- Pre-existing test failures unrelated to this work: `test_v2_split_is_configuration_level`,
  `test_oracle_residual_reproduces_grand_M`, `test_harness_cases_and_metric` (all fail identically on `main`).
