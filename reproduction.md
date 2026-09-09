# Reproducing the paper figures

All commands run from the repo root. Models ship in `data/models/` (git-tracked).
Accuracy runs want `TORCH_COMPILE_DISABLE=1`; performance runs (Fig 11/12) keep compile ON.
Figures 10, 13 not written up here yet.

## Figure 2 — progressive accuracy of the learned stack

Needs the dataset-v2 shards in `data/multibody_v2/` (gitignored; truth comes from their stored grand-M).

```sh
TORCH_COMPILE_DISABLE=1 python figures/fig2_nbody_acc.py    # eval + figure
python figures/fig2_nbody_acc.py --plot-only                # re-render from its CSV
```

Out: `figures/fig2_nbody_acc.{pdf,png}` (+ per-config errors in `figures/fig2_nbody_acc.csv`).

## Figures 3 & 4 — accuracy vs φ (fixed N) and vs N

Needs the MFS truth cache `tmp/nbody_moments_truth/` (1,280 files). Without it, generate first:
`python benchmarks/paper_accuracy_v2.py --exp fig3 fig4 --truth-only` (CUDA GPU; hours on a laptop —
on the cluster use `sbatch --array=0-7 slurm/paper_truth.sbatch`).

```sh
TORCH_COMPILE_DISABLE=1 python benchmarks/paper_accuracy_v2.py --exp fig3 fig4 --workers 8 --skip-done
python benchmarks/paper_accuracy_v2.py --summary --figures
```

Rows accumulate in `data/paper_accuracy_v2.csv`.
Out: `figures/paper_v2_fig3_P{200,300}.*`, `figures/paper_v2_fig4_*.*`.
Optional: `--gpu-ops` adds mfs_coarse + the paper's GPU n-body operator (needs CUDA + warp).

The latest stack is registered as `M_mom_v2_kinf_rc8_pc8c` (chain-fixed pair model) and
`M_mom_v2_kinf_rc8_pc8c_diag` (+ learned diagonal, the headline operator); both load the published
pc8c models and their sidecars directly. The paper-ready Figure 4 (seed-mean PRMSE vs N over the
full 20–200 grid, φ ∈ {0.05, 0.1, 0.15}, shaded ±1 std over the 10 seeds, original layout/palette)
renders from the accumulated CSV to `figures/fig4_diff_sizes.*` plus drop-in copies at the paper's
include name `figures/M_accuracy_nbody_diff_sizes_avg_rel_rmse.*`:

```sh
python figures/fig4_diff_sizes.py            # --no-band for point means only, --phis to change curves
```

For the record, the previously published Figure 4 data (GPU n-body b1 op) stays in
`data/M_accuracy_nbody_diff_sizes.csv`; the new stack is ~1.8–2.4× below it at every φ.

### Large-N extension (N up to 3000, widebvh Broms MFS truth)

Fig 4 extends to N ∈ {300, 500, 1000, 1500, 2000, 2500, 3000} (appended to `FIG4_N`; tapered
seeds, `FIG4_REPEATS`) — N ≤ 2000 batch at the plotted φ ∈ {0.025, 0.05, 0.1, 0.15}, the
1500/2500/3000 batch at all 8 φ. Truth for these cells comes from the widebvh Broms MFS
(`sphere_mfs`: Xfine clouds, degree-7 barycentric treecode over **all** pairs — no `L_cut=25`
truncation — mac 0.3, KSP rtol 1e-10; validated to 1–2e-6 against the legacy truths where `L_cut`
does not bite, and it quantifies the legacy truncation at 3.8e-3 / 4.9e-3 rel-L2 at N=200/300,
φ=0.025). ~27 s per N=2000 solve on an H200. See `artifacts/fig4_large_n_report.md`.

```sh
# truths (GPU node next to a built widebvh sphere_mfs; ~25 min for all 112 cells)
python benchmarks/broms_truth.py --validate     # cross-solver check first
python benchmarks/broms_truth.py                # or: sbatch --array=0-3 slurm/broms_truth.sbatch
# eval (CPU; ~13 core-h for the full large-N set; ~10 min/apply at N=3000) + figure
TORCH_COMPILE_DISABLE=1 python benchmarks/paper_accuracy_v2.py --exp fig4 \
    --N 300 500 1000 1500 2000 2500 3000 --ops M_mom_v2_kinf_rc8_pc8c_diag --workers 2 --part --skip-done
python benchmarks/paper_accuracy_v2.py --merge
python figures/fig4_diff_sizes.py --max-n 3000 --out-suffix _large
```

Out: `figures/fig4_diff_sizes_large.*` + `figures/M_accuracy_nbody_diff_sizes_avg_rel_rmse_large.*`
(log-x). The default `fig4_diff_sizes.py` invocation still renders the 20–200 paper figure only.

**N up to 10,000 (2026-09-06):** N ∈ {5000, 7500, 10000} (3 seeds each) at the 4 plotted φ,
random-forcing truths from the same Broms sbatch (`FORCING=random NS="5000 7500 10000"`,
array 0,1,3,5) and evaluated — the *whole* 20–10000 range, for one coherent series — with the
GPU adapter op `M_mom_gpu_pc8c_diag` (~6 min in docker; parity vs the CPU op certified at
N=5000). The `_large` figure now spans 20–10000 with that op:

```sh
TORCH_COMPILE_DISABLE=1 bash docker/run_local.sh python benchmarks/paper_accuracy_v2.py \
    --exp fig4 --ops M_mom_gpu_pc8c_diag --gpu-ops --phis 0.025 0.05 0.1 0.15 --part --skip-done
python benchmarks/paper_accuracy_v2.py --merge
python figures/fig4_diff_sizes.py --exp fig4 --op M_mom_gpu_pc8c_diag --max-n 10000 --out-suffix _large
```

NB: the cluster submit plugin currently rejects `--constraint=amd24` GPU jobs beyond
"instant"-length limits (`BadConstraints`); request the GPU by type instead —
`--gres=gpu:h200:1` with no feature constraint (see `/tmp/broms_h200.sbatch` pattern in the
report).

### Gravity-forcing variant (exp `fig4g`)

Same configurations (identical seeds/positions), uniform wrench `F=(0,0,-9.81), T=0`; truths in
`tmp/nbody_moments_truth/*_grav.npz`; evaluated with the **GPU moments operator** behind the
harness op `M_mom_gpu_pc8c_diag` (pc8c pair+diag `.wt` weights, warp backend, fp16 off, chunked
far-field RPY; matches the CPU op to ≤1e-3 rel_rmse points — spot rows in the CSV). Full grid
20–10000 at the 4 plotted φ = 752 cells, ~5 min of GPU eval in docker (N=10000: 5 s/apply,
4.1 GiB peak VRAM). The N ∈ {5000, 7500, 10000} truths (3 seeds each) are gravity-only and need
`--mem=96G` (N=10000 = 4.86M boundary points; restart-200 Krylov basis ≈ 23 GB):

```sh
# truths (cluster): sbatch --array=0,1,3,5 --export=ALL,FORCING=gravity slurm/broms_truth.sbatch
#   N>=5000 top-up: sbatch --array=0,1,3,5 --mem=96G --time=3:59:00 \
#       --export=ALL,FORCING=gravity,NS="5000 7500 10000" slurm/broms_truth.sbatch
TORCH_COMPILE_DISABLE=1 bash docker/run_local.sh python benchmarks/paper_accuracy_v2.py \
    --exp fig4g --ops M_mom_gpu_pc8c_diag --gpu-ops --phis 0.025 0.05 0.1 0.15 --part --skip-done
python benchmarks/paper_accuracy_v2.py --merge
python figures/fig4_diff_sizes.py --exp fig4g --op M_mom_gpu_pc8c_diag --max-n 10000 --out-suffix _grav
```

Out: `figures/fig4_diff_sizes_grav.*` (+ `..._avg_rel_rmse_grav.*`). Errors are ~2.5–3× below the
random-forcing protocol (the collective settling velocity dominates the normalisation); the
angular block alone is much worse in relative terms (`prmse_ang` ~19–28 % at large N) since
gravity drives only disorder-induced rotations.

## HIGNN baseline on the Fig 3 / Fig 4 protocols (gravity forcing) — accuracy comparison

Third-party baseline: the Pan-group HIGNN toolkit (github.com/Pan-Group-UW-Madison/hignn, their shipped weights),
re-evaluated in plain torch by `src/hignn_ops.py` (`HIGNN_ROOT` points at the checkout; default
`/home/shihab/throwaway/hignn`). HIGNN is translational-only and torque-free, so it is scored on the torque-free
gravity truths (`fig4g`: F = (0,0,-9.81), T = 0, the Fig 4 configurations at φ ∈ {0.025, 0.05, 0.1, 0.15},
N = 20…10000) with the translational metrics `prmse_lin` / `prmse_fluct` (new columns of `compute_error_stats`,
also `err_mean_pct`, `max_rel_lin`). Two ops: `HIGNN_2b` (what their C++/H-matrix engine computes, dense) and
`HIGNN_full` (+ their Python 3-body and self corrections, cutoff 5.0). Write-up: `artifacts/hignn_comparison_report.md`.

```sh
python -m pytest tests/test_hignn_ops.py -q                                       # adapter self-tests (seconds)
python src/hignn_ops.py --case 200 0.1 4423 --variant full --eps3-sweep 3 4 5 6 8  # 3-body cutoff sensitivity
TORCH_COMPILE_DISABLE=1 python benchmarks/paper_accuracy_v2.py --exp fig4g --phis 0.025 0.05 0.1 0.15 \
    --ops HIGNN_2b HIGNN_full --gpu-ops --part --skip-done                        # ~15 min on the 4060, all 756 truths
TORCH_COMPILE_DISABLE=1 python benchmarks/paper_accuracy_v2.py --exp fig4g --phis 0.025 0.05 0.1 0.15 \
    --N 20 30 40 50 60 70 80 90 100 120 140 160 180 200 300 \
    --ops M_rpy M_2b M_3b M_nbody_b1 M_mom_v2_kinf_rc8_pc8c_diag --workers 6 --part --skip-done   # ~15 min, CPU pool
TORCH_COMPILE_DISABLE=1 bash docker/run_local.sh python benchmarks/paper_accuracy_v2.py --exp fig4g \
    --phis 0.025 0.05 0.1 0.15 --ops M_mom_gpu_pc8c_diag --gpu-ops --part          # headline op with the new columns
python benchmarks/paper_accuracy_v2.py --merge --summary                            # fig4g tables in the summary md
python figures/fig_hignn_compare.py                                                 # figures + artifacts/hignn_comparison_tables.md
# optional: their C++ engine in their CPU Singularity image (cluster), then compare locally
sbatch slurm/hignn_cpp_check.sbatch && rsync h200:pinn-stokes/tmp/hignn_cpp_check.npz tmp/
python benchmarks/hignn_cpp_check.py --compare --out tmp/hignn_cpp_check.npz
```

Out: `figures/fig_hignn_compare_phi[_fluct].*` (Fig-3 style: translational PRMSE vs φ at N = 200 and 300),
`figures/fig_hignn_compare_N[_fluct].*` (Fig-4 style: vs N, one panel per φ), `figures/fig_hignn_compare.csv`,
`artifacts/hignn_comparison_tables.md`.

Result (2026-09-07, translational PRMSE under gravity at N = 200, φ = 0.025 / 0.05 / 0.1 / 0.15): NeMO 0.66 / 1.08 / 1.73 / 2.43 %, HIGNN full 1.01 / 1.93 / 3.74 / 5.47 %, HIGNN 2-body engine 1.23 / 2.20 / 4.22 / 6.24 %, RPY 1.13 / 2.07 / 4.02 / 5.97 % — NeMO 1.5–2.3× (full) and 1.9–2.6× (engine) more accurate; HIGNN's engine sits slightly above RPY and
HIGNN full on top of NeMO 2-body. The gap closes with N under gravity (1.03–1.07× at N ≥ 10 000): the residual is the
collective far-field term that every pairwise far field shares. Details and caveats in the report.

## Figure 5 — opposing-force KDE deep dive

Truth is MFS Xfine (tol 1e-8) computed on the fly, so the full run needs a CUDA GPU with Triton
(no warp): ~1.5 h on the 4060 laptop, ~8 min per configuration (1× N=300 + 10× N=250, φ=0.10,
seeds 42–51 — the notebook protocol of `experiments/accuracy_deep_dive.ipynb`, operator swapped
to the moments-pc8 + learned-diagonal stack).

```sh
TORCH_COMPILE_DISABLE=1 python figures/fig5_deep_dive.py    # eval + figure
python figures/fig5_deep_dive.py --plot-only                # re-render from its NPZ
```

Out: `figures/fig5_deep_dive.{pdf,png}` (+ collected data in `figures/fig5_deep_dive_data.npz`),
and a drop-in copy at the paper's include name `figures/accuracy_deep_dive_fields_2x2.pdf`
(authored at 6.5×4.7 in — include at `width=\columnwidth` with no height override).
The run also prints the paper's summary number: the 5 %-trimmed mean per-particle relative RMSE
averaged over the 10 configurations (6.19 % for the shipped models).

## Figure 6 — horizontal line-of-spheres drag (Durlofsky et al. 1987)

15 unit spheres on a line at spacing 4.0, unit force perpendicular to the line; drag
coefficient λ = F/(6πμaU) per sphere. Warp-free, CPU + a few minutes of numpy MFS — runs
natively on the laptop.

```sh
TORCH_COMPILE_DISABLE=1 python figures/fig6_chain_drag.py \
    --moments-model data/models/nbody_moments_v2_kinf_rc8_pc8c.pt \
    --diag-model data/models/nbody_diag_v2_pc8c.pt             # MFS truth + 4 operators + figure
python figures/fig6_chain_drag.py --plot-only                  # re-render from its CSV
```

Out: `figures/fig6_chain_drag.{pdf,png,csv}` + a drop-in copy at the paper's include name
`figures/horizontal_chain_drag_3.{pdf,png}`. Truth is `src/mfs.py` at tol 1e-9, recomputed at
both fine and Xfine (they agree to <1e-4 %, and fine reproduces the legacy hard-coded values of
`src/horizontal_chain.py` to 5e-9), errors quoted against Xfine.

λ errors vs MFS (max / mean over the 15 spheres, 2026-09-05):

| curve                                       | max rel % | mean rel % |
|---------------------------------------------|-----------|------------|
| Durlofsky et al. (digitized reference)      | 0.16      | 0.11       |
| 2-body only (switch 6)                      | 0.037     | 0.032      |
| n-body b1 (the published figure's op)       | 0.067     | 0.062      |
| moments pc8**c** + diag (the stack above)   | **0.14**  | **0.11**   |
| — pre-fix moments pc8 + diag, for the record| 1.49      | 1.29       |

**History (2026-09-05):** the original pc8 models regressed here by ~1.3 % — the correction
predicted for the next-nearest pairs at d = 8.0 was spurious, an unconstrained error cliff on
the exactly-collinear manifold, which the uniform/grown/lattice training families never sample
(a 3-sphere probe shows the error collapsing from 0.9 % to 0.28 % under a 0.02-radius transverse
offset of the middle particle). Fixed at the data level: a `chain` family in
`src/create_dataset_multibody_v2.py` (22,528 quasi-1D configs, gaps U[2.1, 8], transverse jitter
σ ∈ {0, 0.02, 0.05, 0.1, 0.3}, triton32 labels validated to 1.4e-6 vs fp64), cache
`data/multibody_v2_cache_pc8c` (10.15 M pairs), and a retrain with the chain family downweighted
to 0.25 (`train_nbody_v2.py --family-weights 1 1 1 0.25` — at full weight the dense Fig 3 tail
paid +0.25…+0.51 pts; at 0.25 the chain fix is intact AND every Fig 3/Fig 4 cell is at-or-better
than the pre-fix models: fig3 mean −0.054, fig4 −0.009, worst single cell +0.002). Published as
`nbody_moments_v2_kinf_rc8_pc8c.pt` + `nbody_diag_v2_pc8c.pt` (diag retrained on the same cache).
Chain-OOD suite (jittered/tighter chains, 3-sphere sweep): `tmp/chain_suite_eval.py`, all
geometries ≤ 0.22 %. Mirror symmetry stays exact to 4e-14.

## Figure 7 — 3-sphere equilateral triangle (Wilson 2013)

Three unit spheres at the vertices of an equilateral triangle of side S (center-to-center,
S ∈ [2.01, 6]); the apex sphere is forced with F = −6π ẑ. Reported vs Wilson's semi-analytic
method \[Wilson 2013\] and Stokesian Dynamics \[Townsend 2017\] (both hard-coded reference
tables from the published figure's script): U1 = apex settling speed, U2/U3 = unforced-sphere
drift components, Ω = unforced-sphere rotation. Warp-free, CPU ops + BatchedMFS truth
(torch64 GMRES — the `src/mfs.py` Gauss–Seidel iteration diverges at the S=2.01 gap of 0.01),
~1 min native on the laptop.

```sh
TORCH_COMPILE_DISABLE=1 python figures/fig7_helen_3body.py    # MFS truth + 4 operators + figure
python figures/fig7_helen_3body.py --plot-only                # re-render from its CSV
```

Out: `figures/fig7_helen_3body.{pdf,png,csv}` + a drop-in copy at the paper's include name
`figures/helens_3body_comparison_nbody.{pdf,png}`. The recomputed MFS (Xfine) confirms
Wilson's values to ~1e-4 for S ≥ 2.1; at S=2.01 the point clouds cannot resolve the 0.01 gap
(fine vs Xfine disagree there), so Wilson is the reference throughout, as in the published
figure.

|deviation| from Wilson (2026-09-07, over the 24 (S, metric) cells / the 20 with S ≥ 2.1):

| curve                                | mean   | max    | mean ≥2.1 | max ≥2.1 |
|--------------------------------------|--------|--------|-----------|----------|
| Stokesian Dynamics                   | 0.0037 | 0.0154 | 0.0035    | 0.0154   |
| 2-body only (switch 6)               | 0.0204 | 0.1250 | 0.0120    | 0.0862   |
| n-body b1 (the published figure's op)| 0.0160 | 0.1122 | 0.0086    | 0.0728   |
| moments pc8c + diag (NeMO)           | **0.0066** | **0.0375** | **0.0027** | **0.0185** |

The published figure's visible artefact — U3 read ~0.117 at S=2.01 against Wilson's 0.005 and
had the wrong shape over the whole sweep — is gone: the new stack tracks Wilson's non-monotone
U3 curve to ≤0.002 for S ≥ 2.1 (26× better than b1 at S=2.1). The learned diagonal is the only
term that moves U1 (the apex velocity is a pure self-block response at N=3) and halves its
near-contact error. Remaining visible gap: Ω at S ≤ 2.5 is still overpredicted (0.075 vs 0.037
at S=2.01 — ~30 % better than b1's 0.090, consistent with the known ~17 % angular TR/RT
residual of the pair model); S=2.01 as a whole sits below the training data's minimum gap of
0.1 and is OOD for every learned term.

## Figure 8 — single-drop sedimentation storyboard

Needs warp (GPU operator + treecode), so the simulation runs in the local docker image; ~13 min on
the 4060. Protocol is `experiments/single_drop_sedimentation_executed.ipynb` verbatim (R = 40,
φ = 0.048, seed 42, adaptive RKF45, T = 500) with the moments-pc8 + diagonal stack at switch 8 and
the widebvh far field (fp32 level 3).

```sh
bash docker/run_local.sh python figures/fig8_single_drop.py    # simulate + storyboard
python figures/fig8_single_drop.py --plot-only                 # re-render from its NPZ
python figures/fig8_single_drop.py --compare                   # consistency vs the published run
```

Out: `figures/fig8_single_drop_storyboard.{png,pdf}` (+ drop-in copies at the paper's include name
`figures/single_drop_storyboard_late.*`), data in `figures/fig8_single_drop_data.npz`,
`figures/fig8_consistency.png`. `--compare` parses the published run's statistics out of the
executed notebook; the write-up is `artifacts/fig8_moments_consistency.md`.

## Figure 9 — dynamic 3 falling spheres (Durlofsky et al. 1987, Fig 5)

Three unit spheres on the x-axis at x = −5, 0, 7, unit gravity F = (0, 0, −1) on each, μ = 1,
classical RK4 on the positions with Δt = 1 for 90 frames × 128 time units (11,520 steps, the
published integrator `benchmarks/rk4-3sphere.py` verbatim — the 2-body and RPY runs reproduce the
published panel trajectories to 1e-7). References are bundled in `data/fig9_rk4_3sphere_refs.npz`:
the MFS run (`MobOpMFS` coarse, same integrator), Stokesian Dynamics (Townsend 2017 code) and the
digitised Durlofsky Fig 5, plus the trajectories behind the published panels (b1 n-body at
switch 6, 2b, RPY). Warp-free, CPU ops; the NeMO run (moments pc8c + diag, switch 8, 24 ms per
apply at N = 3) takes ~8 min native on the laptop, 2b/RPY ~1 min each.

```sh
TORCH_COMPILE_DISABLE=1 python figures/fig9_rk4_3sphere.py --ops nemo 2b rpy   # integrate + figure
python figures/fig9_rk4_3sphere.py --plot-only                                 # re-render
```

Out: the paper panels at their include names `figures/rk4-dynamics.*` (NeMO), `rk4-2b.*`, `rk4-rpy.*`;
the combined `figures/fig9_rk4_3sphere.{pdf,png}` (published b1 trajectory greyed into the NeMO
panel), the deviation diagnostic `figures/fig9_rk4_3sphere_error.png`, trajectories
`figures/fig9_rk4_3sphere_<op>.npy` and metrics `figures/fig9_rk4_3sphere.csv`.

**Result (2026-09-07): the new stack is not better on this test.** Mean over the 3 spheres of
|r − r_ref| (radii), over all 91 frames and over the y > −400 window where the references agree:

| curve                                 | mean vs MFS | max/final vs MFS | mean y>−400 | max y>−400 | mean vs SD | final vs SD |
|---------------------------------------|-------------|------------------|-------------|------------|------------|-------------|
| NeMO (moments pc8c + diag, switch 8)  | 3.27        | 16.1             | 0.226       | 0.602      | 2.87       | 13.2        |
| NeMO b1 (published panel)             | 1.44        | 6.62             | 0.181       | 0.831      | 0.99       | 2.81        |
| 2-body only (switch 6)                | 2.18        | 11.0             | 0.181       | 0.829      | 1.73       | 7.57        |
| RPY                                   | 9.05        | 39.2             | 0.582       | 2.54       | 8.70       | 36.9        |
| SD (Townsend) vs MFS                  | 0.54        | 3.93             | 0.018       | 0.060      | —          | —           |

Visually the NeMO panel tracks the references as well as the published one down to y ≈ −450,
then the rightmost sphere drifts left where every reference (and b1) drifts right.

Why (instantaneous velocity error vs BatchedMFS Xfine on the 91 configurations of the MFS
trajectory, relative L2 over the 3 spheres, mean over frames, y > −400 / y ≤ −400):

| operator                                     | linear            | angular           | max abs. x-velocity error |
|----------------------------------------------|-------------------|-------------------|---------------------------|
| 2-body switch 6 (≡ b1: its correction is <1e-5 here) | 1.00e-3 / 0.89e-3 | 6.9e-3 / 5.8e-3 | 7e-5 / 7e-5           |
| 2-body switch 8 (base swap only)             | **0.56e-3** / 0.84e-3 | 6.9e-3 / 5.7e-3 | 4e-5 / 7e-5           |
| + moments pc8c (switch 8)                    | 0.97e-3 / 0.94e-3 | **2.8e-3** / 5.1e-3 | 4e-5 / 6e-5           |
| + learned diagonal (= NeMO)                  | 1.27e-3 / 1.10e-3 | 2.9e-3 / **4.9e-3** | 6e-5 / 7e-5           |
| RPY                                          | 1.9e-3 / 2.0e-3   | 7.3e-3 / 7.3e-3   | —                         |

- The test is chaotic and dominated by an absolute horizontal-velocity error floor of ~5e-5
  (relative ~1e-3) shared by every learned operator: it integrates to 0.2–0.3 radii by y = −400
  and is then amplified. Below y ≈ −450 the three references disagree by up to 3.9 radii among
  themselves, and operators that differ at the 1e-5 level (b1 vs 2-body) end 5 radii apart, so
  the late-time panel does not rank operators.
- On this dilute geometry (pair distances ≥ 3.3, one neighbour per pair) the true many-body
  residual is ≤ 5e-4 of the velocity, below the ~1e-3 noise floor of the learned corrections: the
  base swap to switch 8 alone is the most accurate operator, the moments term gives back what it
  gained and the diagonal adds a further 0.3e-3. Both corrections do what they were trained for on
  the angular velocities (2.4× better at y > −400), which the figure does not show.
- The published panel's b1 correction is inert here (identical to the 2-body operator to five
  digits): that figure was effectively the 2-body operator with a different chaotic draw.

## Ring-like array of spheres (Jordan & Lockerby 2025) — new multi-sphere validation

A planar regular P-gon of unit spheres at surface separation S, all carrying the same unit force parallel or
perpendicular to the ring plane (Jordan & Lockerby, JCP 520 (2025) 113487, §4.2). The per-sphere response is fixed
by five coefficients (M_∥, M_⊥, M_∘, N_tz, N_zt; Eqs. 59–60) that the paper tabulates exactly (Tables B.9/B.10,
transcribed into `data/ring_array_jordan2025.csv`). Warp-free, CPU ops; the static sweep (P = 3…10 × 8 S,
plus P = 20…200 for the global coefficients, 4 operators) takes ~5 min native; the dynamic runs (truth =
BatchedMFS fine/triton32 on the laptop GPU) ~1 min each. Write-up: `artifacts/ring_array_report.md`.

```sh
TORCH_COMPILE_DISABLE=1 python figures/fig_ring_array.py                     # sweep (accumulates in figures/fig_ring_array.csv)
python figures/fig_ring_array.py --zero-crossing --ops rpy 2b nemo b1_paper  # S* of M_o for P = 7 (exact 1.58 R)
python figures/fig_ring_array.py --plot-only
for P in 4 5 6; do
  TORCH_COMPILE_DISABLE=1 python figures/fig_ring_sedimentation.py --P $P --S 0.5 --ops nemo rpy 2b --truth
done
```

Out: `figures/fig_ring_array_{A,B,C,err}.*` (A = Fig.-20-style velocity/rotation pattern exact / NeMO / RPY),
`figures/fig_ring_sed_P{4,5,6}_S0.5.*` (storyboards + gap / elongation / rotation vs fall distance).

Result (2026-09-07): near field S ≤ 2 R, mean error RPY → NeMO: M_∥ 3.4 → 0.28 %, M_⊥ 0.6 → 0.18 %, deformation
coefficient M_∘ 57 → 18 % of scale (3 % at P = 4–6, S = 0.5), in-plane rotation N_zt 39 → 19 %. RPY forces
N_tz = −N_zt while the exact values differ by up to 90 %. Dynamic: at S = 0.5 the truth ring pinches to gap 0.1 after
falling 12.8 / 17.7 / 24.9 R (P = 4 / 5 / 6); NeMO 12.1 / 17.0 / 21.6, RPY 6.4 / 9.5 / 16.7, 2-body-only 6.8 / 8.5 /
12.5. Caveats: NeMO's perpendicular rotation N_tz is slightly worse than RPY for S ≥ 0.5 R, and its M_∘ is
under-predicted at S = 1–2 R for P ≥ 7 (zero crossing 2.33 R vs exact 1.58 R vs RPY 1.20 R).

## Figure 11 — runtime breakdown on the H200 (two panels)

Measurement needs an H200. From the root of the checkout to measure, on the cluster:

```sh
sbatch slurm/fig11_breakdown.sbatch
# interactive GPU-node equivalent:
python benchmarks/figure11_breakdown.py --backend widebvh --csv data/fig11_breakdown_h200_rerun.csv
```

Keep the `*_rerun.csv` path — the default `data/fig11_breakdown_h200.csv` holds the published rows
and is rewritten in place. Render on the laptop (CSV only, no GPU):

```sh
python figures/grand_M_perf.py                                          # canonical, from the published CSV
python figures/fig11_compare.py . data/fig11_breakdown_h200_rerun.csv   # rerun figure + old-vs-new table
```

Out: `figures/runtime_summary_two_panel*.{png,pdf}`.

## Figure 12 — end-to-end scaling on the H200

Same pattern:

```sh
sbatch slurm/fig12_scaling.sbatch
# interactive GPU-node equivalent:
python benchmarks/figure12_grand_M.py --backend widebvh --csv data/fig12_scaling_h200_rerun.csv
```

Render on the laptop:

```sh
python figures/grand_M_perf.py                                          # canonical
python figures/fig12_compare.py . data/fig12_scaling_h200_rerun.csv     # rerun figure + old-vs-new table
```

Out: `figures/grand_M_scaling_test_h200*.{png,pdf}`.
