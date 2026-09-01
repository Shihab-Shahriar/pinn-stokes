# Learned per-particle diagonal (self-block) correction — pc8

**2026-08-30.** A per-particle model (`SelfBlockMoments`) that maps band moments of each particle's
neighbourhood to a symmetric, O(3)-equivariant 6×6 correction to `M_tt`, stacked on the pc8 moments
operator (`Mob_Op_Nbody_Moments` + `nbody_moments_v2_kinf_rc8_pc8.pt`, switch_dist = pair_cutoff = 8).
Motivation: on the pc8 operating point the diagonal residual is ~half the remaining exact-correction
floor at φ ≥ 0.15 and ~⅔ of it for tight clusters (`artifacts/nbody_v2_ceiling_pc8.md`: uniform φ=0.2
e_near 2.69 → e_diag 1.46 %; grown δ=0.05 4.94 → 1.63 %) — the pc6-era conclusion "the floor is not
the diagonal" does not survive the pair_cutoff-8 correction.

**Results in one line:** the model removes ~46 % of the diagonal residual (val capture 53.5 %) and
captures 60–73 % of the exact-diagonal headroom on the ceiling A/B (uniform φ=0.2: floor 2.69 →
1.82 %, exact 1.46); on the paper harness it improves every uniform (N, φ) — Fig 4 φ=0.2 mean
6.07 → 5.70 %, up to −0.6 points at small N — but only ~0.1 points on Fig 3 N=200/300, where the
pair model's own ~7 % residual dominates in quadrature (§ 4).  Clustered δ is mixed
(out-of-distribution generator).  No SPD degradation; grand-M symmetry preserved by construction.

## 1. Model

- **Encoder** (`src/nbody_moments.py`, `self_*` functions): particle-centred band moments
  `(s_a, v_a, Q_a)` over 8 tent bands remapped to **[2, 8]** (width 0.75, centres 2.375…7.625,
  saturating end bands; hard spheres put every neighbour at r ≥ 2, so the pair layout's unit bands
  1–2 would be structurally empty). Row = 104 columns `[s_a(8) | v_a(24) | Q_a(72)]`.
- **Invariants** (48 = 6/band): `s_a, |v_a|², tr Q_a², tr Q_a³, v_aᵀQ_a v_a, |Q_a v_a|²` — all true
  rotation scalars, no pseudoscalars, so the MLP input is invariant under the full O(3).
- **Bases**: TT/RR (symmetric true tensors) `{I} ∪ {Q_a, v_a v_aᵀ, Q_a², S(Q_a v_a v_aᵀ)}_a` = 33
  each; TR (pseudotensors) `{E(v_a), E(Q_a v_a), [Q_a, E(v_a)]}_a` = 24, **no constant TR term** —
  an isolated particle reduces to `TT = c₀I, RR = c₃₃I, TR = 0`. 90 coefficients total.
- **Assembly** `top = [TT, TR]`, `bot = [TRᵀ, RR]` (not the pair model's `[TR, RR]`): the block is
  **symmetric by construction**, so the grand mobility stays symmetric — nothing else enforces this
  for a diagonal block.
- **Arch** (`src/model_archs.py::SelfBlockMoments`): MLP 48→128→64→128→64→90 (Tanh), `inv_mean`/
  `inv_std`/`basis_scale` buffers, zero-init head (the correction starts as an exact bit-for-bit
  no-op), TorchScript-scriptable.
- **Selection** (`nbody_features.select_particle_neighbours`): all k ≠ t within **r_c = 8** of the
  particle, ascending index, CSR — the single code path shared by trainer, operator and ceiling
  script (mean 17.6 neighbours/particle over dataset v2).

## 2. Labels and training

Labels already existed: `data/multibody_v2_cache_pc8/Mtt_res.npy` (56048, 64, 36) =
`M_tt − diag(1/6π I, 1/8π I) − Σ_{d≤8} K_s`, symmetrised `0.5(A + Aᵀ)` at load (the model output is
symmetric; the antisymmetric part is MFS `symm_err` noise). Label RMS: TT 6.45e-4, RR 5.04e-4,
TR 2.45e-4 (~2.3 % of the self block). **The label subtracts K_s over exactly the d ≤ 8 pair set, so
the model must run with `pair_cutoff = switch_dist = 8`** — the published sidecar records
`pair_cutoff`/`diag_cutoff`/band layout and both the operator ctor and the harness assert it.

Trainer `experiments/train_diag_v2.py`: rows = all 1,123,936 particles (112,224 val, configuration-
level split `seed % 10 == 0`; zero-neighbour rows kept so the constant coefficients are anchored),
features on the fly on the GPU, L1 over `6π·(predict_mobility − R)`, Adam 1e-3 + per-step cosine,
batch 4096, seed 411, `fit_normalisation` on 262k train rows.

### Sweep (val capture = ‖pred − R‖/‖R‖, lower is better; 100 % = no correction)

| run | epochs | inv_norm | capture | TT | TR | RR | vel lin | vel ang | uniform | grown | lattice | time |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diag_pc8 | 100 | off | **53.7 %** | 51.4 | 68.8 | 49.0 | 54.1 | 53.5 | 45.2 | 56.0 | 38.9 | 4.4 min |
| diag_pc8_invnorm | 100 | on | 54.2 % | 51.4 | 68.7 | 50.6 | 54.0 | 54.7 | 44.7 | 56.7 | 39.0 | 4.4 min |
| **diag_pc8_e200** | 200 | off | **53.5 %** | 51.0 | 68.5 | 49.1 | 53.7 | 53.5 | 44.8 | 55.8 | 37.3 | 8.8 min |

Published: **diag_pc8_e200** as `data/models/nbody_diag_v2_pc8.pt` (+ `.json` sidecar, `.wt` in
`experiments/`) — best on every family; inv_norm and epochs beyond 100 are both nearly flat, so the
model, not the recipe, is the current limit (TR in particular).

## 3. Ceiling A/B (learned vs exact diagonal)

`experiments/nbody_v2_ceiling.py --cache data/multibody_v2_cache_pc8 --pair-cutoff 8 --diag-model
data/models/nbody_diag_v2_pc8.pt` — same 384 validation rows as `artifacts/nbody_v2_ceiling_pc8.md`;
`e_diag_nn` = exact near-pair residuals + the **learned** diagonal, bracketed by `e_near` (no
diagonal) and `e_diag` (exact diagonal).

Highlights (rel-L2 % of the velocity, mean over P and configs; full table in
`artifacts/nbody_v2_ceiling_pc8_diag.md`):

| family | param | e_2b | e_near | **e_diag_nn (learned)** | e_diag (exact) | headroom captured |
|---|---|---:|---:|---:|---:|---:|
| uniform | 0.15 | 11.23 | 2.12 | **1.70** | 1.41 | 59 % |
| uniform | 0.20 | 15.60 | 2.69 | **1.82** | 1.46 | 71 % |
| uniform | 0.25 | 20.03 | 3.45 | **1.99** | 1.46 | 73 % |
| grown | 0.05 | 20.31 | 4.94 | **3.50** | 1.63 | 43 % |
| grown | 0.10 | 16.37 | 3.74 | **2.46** | 1.56 | 59 % |
| grown | 0.20 | 11.94 | 2.30 | **1.57** | 1.11 | 61 % |
| lattice | any | — | — | ≈ e_near | ≈ e_near | (no headroom, as expected) |

The angular error is where the correction bites hardest, exactly as the exact-diagonal ablation
predicted: uniform φ=0.25 ang 4.49 → **1.80** (exact 0.78); grown δ=0.1 ang 4.67 → **2.53** (exact
~1.5).  At low φ (≤ 0.075) e_diag_nn tracks e_near within 0.1 — no regression where there is no
headroom.  The remaining gap to e_diag is largest for grown δ=0.05 (near-contact lubrication-like
environments), consistent with TR being the worst-captured block.

## 4. Paper harness

New op `M_mom_v2_kinf_rc8_pc8_diag` ("moments v2 (pairs≤8) + learned diagonal") in
`benchmarks/paper_accuracy_v2.py`; the diag sidecar is mandatory (`_diag_sidecar_for`) and a zero-init
diag model reproduces the pc8 baseline bit-for-bit (`tests/test_diag_moments.py`).

### Fig 3 (N = 200/300, PRMSE % vs φ)

| N | op | 0.025 | 0.05 | 0.075 | 0.1 | 0.125 | 0.15 | 0.175 | 0.2 |
|---|---|---|---|---|---|---|---|---|---|
| 200 | pc8 | 1.58 | 2.74 | 3.89 | 4.78 | 5.61 | 6.57 | 7.60 | 7.41 |
| 200 | **pc8+diag** | 1.57 | 2.73 | 3.87 | 4.74 | 5.60 | 6.50 | 7.52 | **7.29** |
| 300 | pc8 | 1.78 | 3.04 | 4.16 | 5.14 | 5.92 | 7.21 | 8.10 | 9.89 |
| 300 | **pc8+diag** | 1.76 | 3.03 | 4.15 | 5.14 | 5.91 | 7.19 | 8.05 | **9.86** |

**The gain is real but small (≤ 0.12 points), and the ceiling explains why.**  With *exact* pair
residuals as the base, the learned diagonal cuts the φ=0.2 floor 2.69 → 1.82 % — i.e. it removes an
error component worth √(2.69² − 1.82²) ≈ 2.0 % in quadrature.  In the shipped operator the learned
pair model's own residual (≈ 7 %) dominates, so the same orthogonal component moves the total only
to √(7.41² − 2.0²) ≈ 7.1 % — the observed 7.29 % is consistent with that dilution.  The diagonal
correction is doing its job (60–73 % of its headroom, § 3); its headline value grows as the pair
correction improves, and it is already the difference between a floor that rises with φ and one
pinned near 1.4–2 % once the pair term is good enough.

### Fig 4 (mean over N = 20…200 × 10 seeds)

| op | 0.025 | 0.05 | 0.075 | 0.1 | 0.125 | 0.15 | 0.175 | 0.2 |
|---|---|---|---|---|---|---|---|---|
| pc8 | 1.35 | 2.27 | 2.97 | 3.68 | 4.23 | 4.80 | 5.28 | 6.07 |
| **pc8+diag** | 1.34 | 2.25 | 2.93 | 3.61 | 4.13 | 4.65 | 5.04 | **5.70** |
| Δ | −0.01 | −0.02 | −0.04 | −0.06 | −0.10 | −0.15 | −0.24 | **−0.37** |

The gain grows with φ and is largest at small N (φ=0.2: N=20 4.42 → 3.91, N=30 4.80 → 4.23,
N=60 4.97 → 4.58, N=200 7.74 → 7.53) — exactly where the pair model sits closest to its exact
floor, so the diagonal share of the remaining error is largest.  Improvement is monotone in φ and
never a regression at any (N, φ).

### Clustered (N = 10, δ sweep)

Mixed, unlike the v2-box grown family (which improves sharply in the ceiling A/B):

| δ | pc8 | pc8+diag | | δ | pc8 | pc8+diag |
|---|---|---|---|---|---|---|
| 0.1 | 2.56 | 2.78 | | 1.0 | 0.50 | 0.51 |
| 0.2 | 1.46 | 1.50 | | 2.0 | 0.24 | 0.30 |
| 0.5 | 0.76 | 0.75 | | 3.0 | 0.22 | 0.26 |

(max per-particle: better at δ=0.2/0.5, worse at δ=0.1/2/3.)  The reference clusters
(`tmp/reference_sphere_*.csv`) come from a different generator than the v2 `grown` family, and at
δ ≥ 2 every environment is beyond the training support (all neighbours near or past the band edge),
where the model adds ~0.05-point noise on a 0.2-point baseline.  On in-distribution grown boxes the
ceiling A/B shows the opposite: δ=0.1 velocity error 3.74 → 2.46 %.

## 5. Commands

```bash
# train + publish
python experiments/train_diag_v2.py --out experiments/runs_v2/diag_pc8 --publish
# ceiling A/B
CUDA_VISIBLE_DEVICES= python experiments/nbody_v2_ceiling.py --cache data/multibody_v2_cache_pc8 \
    --pair-cutoff 8 --diag-model data/models/nbody_diag_v2_pc8.pt --out artifacts/nbody_v2_ceiling_pc8_diag.csv
# paper harness (truths cached in tmp/nbody_moments_truth)
TORCH_COMPILE_DISABLE=1 python benchmarks/paper_accuracy_v2.py --exp cluster --ops M_mom_v2_kinf_rc8_pc8_diag
TORCH_COMPILE_DISABLE=1 python benchmarks/paper_accuracy_v2.py --exp fig3 --ops M_mom_v2_kinf_rc8_pc8_diag --workers 8 --skip-done
TORCH_COMPILE_DISABLE=1 python benchmarks/paper_accuracy_v2.py --exp fig4 --ops M_mom_v2_kinf_rc8_pc8_diag --workers 8 --skip-done
TORCH_COMPILE_DISABLE=1 python benchmarks/paper_accuracy_v2.py --summary --figures
```

## 6. Pair-model capacity diagnostic (2026-08-31, follow-up)

Since the pair moments model's ~7 % residual now caps the headline numbers (§ 4), we checked whether
it is data- or capacity-limited before investing further.

**Train-vs-val gap of the published `nbody_moments_v2_kinf_rc8_pc8.pt` (300k rows per split): none.**
Train capture 28.8 % / val 28.9 %; PRMSE identical to the third digit (lin 3.899/3.894, ang
15.695/15.645). More training data buys nothing at this architecture.

**Recipe sweep** (pc8 cache, kinf_rc8, 100 epochs each; baseline = published model):

| run | change | capture | lin | ang | verdict |
|---|---|---|---|---|---|
| baseline | — (40.7k params) | 28.9 % | 3.89 | 15.66 | |
| mom_pc8_wide | `--hidden 256 128 256 128` (130.5k params) | 29.0 % | 3.90 | 15.69 | flat |
| mom_pc8_bwrms | `--block-weights rms` (TR/RR upweighted) | 29.2 % | 3.96 | 15.65 | flat (epoch-99 eval; final export interrupted) |
| mom_pc8_vel | `--loss-form velocity` | 29.1 % | 3.93 | 15.63 | flat |

**Conclusion: the pair model is representation-limited.** Zero generalisation gap *and* a 3.2×
parameter increase changing nothing means the MLP already extracts everything the 76 invariants +
93 fixed bases can express — either the basis set cannot span the residual (TR/RT worst, 17 % vs
TT 2.9 %) or distinct neighbourhoods alias to the same l ≤ 2 midpoint moments.  The productive
lever is richer features, not nets or data: cross-band invariants (`v_a·v_b`, `tr Q_aQ_b`), l = 3
octupole band moments, and the extra TR pseudotensor bases they enable; failing that, end-to-end
fine-tuning of pair + diag on a config-level velocity loss.  (`--hidden` is now a trainer/arch knob;
default arch unchanged and published weights still load.)

## 7. Encoder extension ablation (cross-band / l2c / octupole) — follow-up to §6

§6 concluded the pair model is representation-limited, so we extend the encoder's information
content with three strictly separated blocks, trained one at a time on the pc8 cache (kinf_rc8,
default recipe, seed 411) in an isolated eager-mode module (`src/nbody_moments_x.py` — never
published, deleted after the ablation; the shipped 111-column path is byte-identical throughout,
verified by a prefix-identity test and an empty-spec run that reproduces the baseline digit-for-digit).

Every candidate was checked against the pair reciprocity rule (TT/RR true tensors, TR pseudotensors,
each (z-even ∧ symmetric) ∨ (z-odd ∧ antisymmetric); MLP inputs ε-free swap-even scalars):

- **cross** (21 MLP inputs, no new bases): adjacent-band {v_a·v_{a+1}, tr(Q_aQ_{a+1}), (z·v_a)(z·v_{a+1})}.
- **l2c** (+8 inv, +16 TT, +16 TR → 141 coeffs): inv {|Q_av_a|²}; TT {Q_a², S(Q_av_av_aᵀ)};
  TR {[Q_a,E(v_a)] (even-sym), (z·v_a)E(Q_av_a) (odd-antisym)}. Bare E(v_a)/E(Q_av_a) are
  even-antisym → illegal for the pair block (legal in the diag model only via its RT=TRᵀ assembly).
- **oct** (+32 inv, +8 TT, +8 TR, X_DIM 111→327 → 117 coeffs): l=3 octupole O_a; with u_a = O_a:zz —
  inv {O:O, (O:zzz)², |u_a|², u_a·v_a}; TT {Alt(z u_aᵀ)}; TR {S(z(z×u_a)ᵀ)}. Excluded as
  rule-violating: S(z u_aᵀ), E(u_a), (O:zzz)E(z), (O:z).

Keep bar (user-set): alone, ≥2 pts val capture (≤26.9) or ≥1.5 pts ang PRMSE (≤14.2), lin not worse
by >0.2. Singles only — the single best passing block ships as a hard-coded v3; everything else is
deleted. If all flat: one "oct+" escalation round (legal v-mixed items: TR (z·v_a)E(u_a), TT
(z·v_a)(O_a:z), inv u_a·Q_av_a), then stop.

### Results (baseline: capture 28.9 %, lin 3.89, ang 15.66, TR/RT rel ~17)

| run | spec | capture | lin | ang | TR rel | verdict |
|---|---|---|---|---|---|---|
| momx_cross | cross | 28.7 % | 3.87 | 15.66 | 16.9 | **flat — fails the bar** (epoch-99 eval; final export interrupted) |
| momx_l2c | l2c | 29.0 % | 3.92 | 15.60 | 17.0 | **flat — fails the bar** (epoch-99 eval; final export interrupted) |
| momx_oct | oct | 27.9 % | 3.72 | 15.54 | 16.8 | **fails the bar** — but the only non-noise signal (−1.0 capture, TT 2.87→2.66, lin −0.17; epoch-99 eval) |
| momx_octp | oct + v-mixed (escalation) | 27.7 % | 3.68 | 15.52 | 16.8 | **fails the bar** — matches oct within noise (epoch-99 eval) |

### Conclusion — negative result: the moments-encoder family is saturated

cross and l2c are exactly flat, so the l≤2 midpoint-moment basis is **complete** for this data and
loss — recombining existing information cannot help.  The l=3 octupole carries the only real signal
(capture −1.0…−1.2 pts, lin 3.89 → 3.68, TT 2.90 → 2.61), but it is far below the ≥2-point bar and
the angular bottleneck does not move (ang 15.66 → 15.52, TR/RT rel ~17 throughout) — more moments
of the same neighbourhood do not carry the missing TR/RT information.  Per protocol nothing ships:
the experimental module, oracle, tests and trainer hook were deleted; **zero shipped files
changed**; the paper method remains `nbody_moments_v2_kinf_rc8_pc8` (+ `nbody_diag_v2_pc8`).
(The seed-repeat step was moot — no block passed.)

Together with §6 this pins the pair correction's remaining ~7 % residual as unrecoverable by
(a) capacity, (b) data, (c) loss shaping, or (d) richer O(3)-invariant summaries of the midpoint
neighbourhood up to l = 3.  The failure is angular-specific, which points at the midpoint
band-moment *conditioning* itself aliasing the environments that matter for TR/RT.  Next levers,
in order: end-to-end pair+diag fine-tuning on a config-level velocity loss, or per-neighbour
conditioning (attention/GNN over the neighbour list) — architecture changes, not encoder patches.

## 8. Notes / limitations

- CPU operator only this round; the GPU port lands together with the pending moments-pair port
  (structurally the easiest GPU term: one add where the constant self term lives,
  `gpu_mob_2b.py:465`, and `_per_particle_topk` already exists — but it needs a radius variant).
- SPD is monitored, not enforced (project precedent).  Spot check (densest val boxes): the learned
  correction moves the effective diagonal's λ_min *toward* the truth — uniform φ=0.25 P=64:
  0.0226 → 0.0274 (true 0.0286); grown δ=0.05 P=32: 0.0133 → 0.0167 (true 0.0216) — and cuts the
  diagonal-block Frobenius error 0.0398 → 0.0138 / 0.0349 → 0.0229.  No SPD degradation.
- TR is the hardest block to capture (pseudotensor, smallest labels).
