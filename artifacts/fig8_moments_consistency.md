# Figure 8 (falling particle cloud) with the new moments NeMO — consistency vs the published run

`figures/fig8_single_drop.py`, 2026-09-01. Regenerates the paper's Figure 8 (dynamic sedimentation
of an N = 3071 cloud, Metzger et al. configuration) with the dataset-v2 accuracy stack and compares
it against the published run, whose per-snapshot statistics survive in the executed notebook
(`experiments/single_drop_sedimentation_executed.ipynb`, parsed by `--compare`).

## What changed, what didn't

| | published run | this run |
|---|---|---|
| near-field operator | `Mob_Nbody_Torch` (b1 K=10, switch 6) | `Mob_Nbody_Moments_Torch` (pc8 moments pair + learned diagonal, switch = pair_cutoff = 8) |
| far field | WarpFMM, theta 0.28, leaf 16, cutoff 6 | widebvh bary, mac 0.8, pdeg 7, fp32 level 3, cutoff 8 |
| everything else | identical: drop (R 40, phi 0.048, lattice + 5 % jitter, seed 42), gravity Fz = −6π, adaptive RKF45 (dt ∈ [5e-4, 0.03], rtol 5e-3 / atol 2e-3), T = 500, snapshots every 3 | |

Wall clock: 770 s on the RTX 4060 laptop (16,668 accepted steps, **zero rejections**, dt pinned at
the 0.03 cap throughout — the embedded error never approached the tolerance).

## Quantitative consistency

Pre-run settling-speed validation (paper protocol, mean U_d while the tail fraction < 1 %):

| | U_d | vs Eq. (8) HR = 93.13 | tail crosses 1 % |
|---|---|---|---|
| published | 92.601 | −0.57 % | t = 0.835 |
| new stack | 92.492 | −0.69 % | t = 0.785 |

The two operators differ by **0.12 %** on U_d — an order of magnitude less than either differs from
the Hadamard–Rybczynski estimate.

Snapshot-mean |V|(t), new vs published (interpolated to common times; `figures/fig8_consistency.png`):

| window | median rel dev | max |
|---|---|---|
| t ∈ [0, 100] | 0.65 % | 2.7 % |
| t ∈ [100, 250] | 1.64 % | 3.2 % |
| t ∈ [250, 400] | 3.78 % | 17.2 % |
| t ∈ [400, 500] | 33.2 % | 42.5 % |

The curves are statistically indistinguishable through the linear and early nonlinear stages and
separate only once breakup begins, which arrives earlier on the new stack's clock:

| t at which mean |V| first falls below f × initial | f = 0.8 | 0.6 | 0.4 | 0.25 |
|---|---|---|---|---|
| published | 171 | 390 | 450 | 483 |
| new stack | 153 | 357 | 417 | 444 |

— a uniform ~8 % time-lead. The horizontal extent tells the same story with the same endpoint: the
90th-percentile cylindrical radius goes 35.7 → 42.0 (t = 300) → explosive growth from t ≈ 400 →
**179.3** at t = 500, vs the published diagnostics' 36 → ~42 (t = 300) → knee at t ≈ 425 → **180**
at t = 500. Continuous rear leakage is present throughout (trail fraction beyond 2R above the cloud:
5.4 % at t = 100, 17.7 % at t = 400).

## Morphology (storyboard, `figures/fig8_single_drop_storyboard.*`)

The destabilization sequence of the caption is reproduced stage for stage: rear leakage from the
start, flattening into an oblate shape, torus formation, dumbbell elongation, and breakup into two
secondary clusters joined by a dilute bridge — along an oblique in-plane axis in both runs. Shifted
onto the ~20-t-earlier clock, the published t = 440/460/480 panels correspond to the new t ≈
420/441/459 panels. By t = 480 the new run's daughter clusters have each formed their own
mini-torus and trail — the secondary cascade the paper's text describes; the published run reaches
the same stage just beyond its final panel.

## Reading

This is the outcome expected for a chaotic instability computed with two operators that agree to a
few percent per evaluation: trajectory-level statistics coincide until deep into the nonlinear
stage, the breakup clock shifts by a few percent, and the instability pathway, cluster count,
breakup axis and final spatial extent are preserved. It supports the paper's own framing of this
test — a long-time qualitative validation of the far-field dynamics — and extends it: the new
near-field stack changes none of the physics conclusions while being the more accurate operator on
every static benchmark.

Two presentation notes: the storyboard's side-view axis limits now exclude the far leakage trail
(> 6R above the cloud) before taking percentiles — the published percentile rule only framed its
run because that run had shed few particles by t = 400 — and the published notebook's COM-based
"leakage %" diagnostic saturates at 100 % by t ≈ 60 in both runs (the trail drags the COM out of
the cloud), so the trail fraction above is used instead.

## Reproduce

```sh
bash docker/run_local.sh python figures/fig8_single_drop.py    # simulate (warp; ~13 min on the 4060)
python figures/fig8_single_drop.py --plot-only                 # storyboard from the NPZ
python figures/fig8_single_drop.py --compare                   # these metrics + overlay figure
```
