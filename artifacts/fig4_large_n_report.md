# Fig 4 at large N (up to 3000) with widebvh Broms-MFS ground truth

2026-09-06. Extends paper Fig 4 (seed-mean rel. RMSE vs N, one curve per φ, op
`M_mom_v2_kinf_rc8_pc8c_diag`) from N ≤ 200 to N ∈ {300, 500, 1000, 2000} (batch 1, φ ∈
{0.025, 0.05, 0.1, 0.15}) and then N ∈ {1500, 2500, 3000} (batch 2, **all 8 φ** of the fig4
protocol). Nothing existing was regenerated: the new cells are appended N values (new truth
files, new CSV rows, a separate `_large` figure).

## 1. Why a new truth solver

The legacy truth path (`benchmarks/cluster.py::generate_uniform_testcase` →
`imp_mfs_mobility_sphere_triton`, Xfine, tol 1e-8) hard-codes **`L_cut = 25`**: pairs farther than
25 radii are dropped from the Oseen sum. Boxes outgrow that with N — side 16–32 at N=200 but 35–70
at N=2000 — so the legacy "truth" acquires an N-dependent truncation bias exactly along the axis
Fig 4 varies. The widebvh Broms MFS (`sphere_mfs` in
`/mnt/ffs24/home/khanmd/programs/widebvh`, PETSc GMRES + degree-7 barycentric-Lagrange treecode)
covers **all pairs with no cutoff**, is full 6-DOF, and shares every convention with this repo
(radius 1, μ=1, Oseen 1/(8πμ), U=F/6π, Ω=T/8π, and it consumes our exact
`data/points/{b,s}_sphere_Xfine.txt` clouds: 486 collocation / 425 source points per sphere).

A small patch (widebvh `src/sphere_mfs.cu`) added `--centers=<raw fp64 P×3>` and
`--wrench=<raw fp64 P×6>` file inputs; `--dump-uom` already wrote all 6P velocities. Built at
widebvh 9f963d2 + patch, `build-rel/sphere_mfs`.

Truth-grade settings (all in `benchmarks/broms_truth.py`): `TC_PATH=split-warpspec` (overriding
sphere_mfs's silent `skel` low-rank default), `TC_BVH_BUILDER=sah`, Xfine clouds, `--tc-mac=0.3`,
`--ksp-rtol=1e-10`, `-ksp_gmres_restart 200`, `-ksp_converged_reason` (asserted CONVERGED on every
solve).

## 2. Validation ladder (all on the H200, 2026-09-06)

1. **Single-sphere gate** (Xfine, brute): max abs err vs U=F/6π, Ω=T/8π = **2.0e-9**.
2. **Two-sphere gate** vs the exact-VSH reference (gap 0.2): max abs err **1.1e-5** — the Xfine
   near-contact discretization level, not a solver error.
3. **Treecode vs brute** (N=200, φ=0.1, mac 0.3): rel-L2 **1.74e-7** — the treecode contributes
   nothing above the fp32-geometry floor; effectively direct-sum quality.
4. **Cross-solver, end-to-end** (`broms_truth.py --validate`: re-solve cached legacy truths from
   their stored config+forces):

   | file | relL2 all | lin | ang |
   |---|---|---|---|
   | N200 φ=0.05 / 0.1 / 0.15 / 0.2 | 1.2–2.1e-6 | — | — |
   | N200 φ=0.025 | 3.8e-3 | 4.6e-3 | 2.5e-4 |
   | N300 φ=0.025 / 0.05 / 0.1 / 0.15 / 0.2 | 4.9e-3 / 4.5e-3 / 1.4e-3 / 6.0e-4 / 1.7e-4 | ~1.2× all | ~10× smaller |

   Where L_cut=25 covers the box (N=200, φ≥0.05, box ≤ 25.6) the two completely independent
   solvers agree to **1–2e-6**: conventions, case I/O and both solvers confirmed. Everywhere else
   the discrepancy is the **legacy truth's L_cut truncation**, with the predicted signature:
   grows with box size, shrinks with φ, almost purely translational (1/r Stokeslet sums). So the
   legacy N=300 fig3 truths carry up to ~0.5 % truth error at low φ; harmless against 2–12 % NN
   errors, but the new cells don't pay it.
5. **Boundary-condition residual** (`--bc-check`, method-independent: max |u − (U + Ω×r)| on 578
   off-collocation points/sphere): N=2000 φ=0.1 → 1.74e-3, vs **3.2e-3 on an N=200 case whose
   velocities agree with the independent legacy solver to 1e-6**. I.e. the max-norm BC residual is
   the Xfine near-contact discretization signature, N-independent, and does not propagate to the
   rigid-body velocities.

## 3. Protocol

- `FIG4_N` **appended** {300, 500, 1000, 2000} then {1500, 2500, 3000} (p_idx 14–20 in append
  order, deliberately non-monotonic — the seed formula `123 + v_idx*1000 + p_idx*100 + run` makes
  insertion-anywhere-else invalidate the whole cache).
- Tapered seeds `FIG4_REPEATS = {1000: 5, 1500: 5, 2000: 3, 2500: 3, 3000: 3}` (10 elsewhere).
  Batch 1: φ ∈ {0.025, 0.05, 0.1, 0.15} → 112 cells; batch 2: all 8 φ → 88 cells.
- Config+forces replicate `generate_uniform_testcase` bit-for-bit without solving
  (`broms_truth.make_case`; verified max|Δconfig| = 0.0 vs a cached legacy truth).
- Truth npz drop into `tmp/nbody_moments_truth/` with the harness schema plus provenance
  (`acc="broms_Xfine"`, solver, mac, rtol, restart, tc_path, widebvh sha, KSP line).
- Eval: the standard harness (`paper_accuracy_v2.py`), CPU op, one apply per config. New
  `NNMob.store_M = False` gate skips the 6N×6N diagnostics matrix (1.15 GB/apply at N=2000);
  velocities unchanged (guarded stores only; residual 2.7e-9 run-to-run GPU fp32 noise).

## 4. Results (`M_mom_v2_kinf_rc8_pc8c_diag`, seed-mean rel. RMSE %)

| φ \ N | 200 | 300 | 500 | 1000 | 1500 | 2000 | 2500 | 3000 |
|---|---|---|---|---|---|---|---|---|
| 0.025 | 1.70 | 1.65 | 1.91 | 2.02 | 2.06 | 2.23 | 2.27 | **2.02** |
| 0.05 | 2.67 | 3.26 | 3.25 | 4.25 | 3.96 | 4.21 | 4.27 | **4.16** |
| 0.1 | 4.28 | 4.82 | 6.15 | 7.29 | 6.65 | 7.46 | 8.78 | **8.83** |
| 0.15 | 6.09 | 7.34 | 8.90 | 11.37 | 10.63 | 11.73 | 10.99 | **12.51** |

Batch 2 also covers the four unplotted φ (N = 1500/2500/3000): φ=0.075 → 5.38/5.25/7.24,
φ=0.125 → 9.22/10.03/11.08, φ=0.175 → 13.30/15.36/14.40, φ=0.2 → **14.00/17.26/17.42**.
(N ≤ 2000 large cells remain 4-φ; backfill is one `broms_truth.py` + eval invocation away.)

- The growth-with-N continues past N=200 and strengthens with φ — exactly the uncorrected d>8 RPY
  far-field floor of `artifacts/fig3_n300_investigation.md` (there: floor ratio e300/e200 =
  1.31…1.60 at φ=0.025…0.2, slope b ≈ 0.17 + 1.2φ). Growth is translational (N=2000 prmse lin/ang:
  2.4/1.1 at φ=0.025 → 12.0/5.9 at φ=0.15).
- The decade slopes 200→2000 (0.12–0.28) are shallower than the N≤300 local slopes (~0.35–0.39),
  and with the batch-2 points the 1000→3000 stretch is nearly flat at φ ≤ 0.05 and clearly
  bending at higher φ (wiggles like 11.37 → 10.63 → 11.73 → 10.99 → 12.51 at φ=0.15 sit inside
  the 3–5-seed ±1σ bands) — consistent with the O(φ) screening-error picture saturating once the
  box is ≫ the correction range, not with unbounded error growth.
- Worst per-particle error (`max_rel_rmse`, N=2000): 8.4 / 18.7 / 28.8 / 43.0 % at
  φ=0.025/0.05/0.1/0.15.
- The N=200→300 seam (legacy truth → Broms truth) is smooth in every curve; the step-4 numbers
  bound any seam artefact at ≤0.5 % relative — invisible at these error levels.

**Extension to N=10,000 (2026-09-06, random forcing, GPU-op series).** Truths: same Broms sbatch
(`FORCING=random NS="5000 7500 10000"`, 36 solves, all CONVERGED). To keep one op across the
whole range, the entire 20–10000 × 4-φ grid (752 cells) was (re-)evaluated with the GPU adapter
`M_mom_gpu_pc8c_diag` (~6 min of docker on the 4060) and the `_large` figure now plots that
series. New tail (seed-mean rel_rmse %, N=5000/7500/10000): φ=0.025 → 2.52/2.64/2.54,
φ=0.05 → 4.93/5.48/5.11, φ=0.1 → 7.80/9.80/10.27, φ=0.15 → 11.50/14.38/13.31. Unlike gravity
(clean plateau ≤4.91 %), the random-forcing curves at φ ≥ 0.1 keep creeping and get **noisy at
3 seeds** (φ=0.15 N=10000 spread 9.7–15.7 %, `max_rel_rmse` up to 130 % — a few near-field
particles dominate); φ ≤ 0.05 is near-flat. CPU spot-check (N=5000 φ=0.1 seed 5223, 37 min on
a 64 GB CPU node): rel_rmse 7.11634 vs the GPU row's 7.11633 — parity holds under random
forcing at the new scale too.
Cluster note: the submit plugin began rejecting `--constraint=amd24` GPU jobs beyond
instant-length limits mid-day (`BadConstraints`, probes isolate it); `--gres=gpu:h200:1` with
no feature constraint schedules fine.

## 4b. Gravity-forcing variant (exp `fig4g`, 2026-09-06)

Same configurations (byte-identical positions, verified), uniform wrench `F=(0,0,-9.81), T=0`
(the repo's sedimentation convention). Truths: `broms_truth.py --forcing gravity` → `*_grav.npz`,
716 cells (full N grid 20–3000, plotted 4 φ), one sbatch array ≈ 20 min. Evaluated with the GPU
moments operator via harness op `M_mom_gpu_pc8c_diag` (`_GpuMomentsAdapter`: pc8c `.wt` weights,
warp backend, **fp16 off**, `far_field_2b=None` + chunked far-pair RPY through
`_rpy_velocity_compiled` — the op's own far path is one dense ~5 GB batch at N=3000). Parity:
random-forcing cells N=300/3000 and gravity cells N=100/500/2000 all match the CPU op to
≤1e-3 rel_rmse points (rows in the CSV). Whole eval: **~4 min** in docker on the 4060 vs ~15–20
core-h for the CPU op.

Results (seed-mean rel_rmse %): φ=0.025: 0.67 (N=200) → 0.79 (N=3000); φ=0.05: 1.11 → 1.55;
φ=0.1: 1.77 → 3.10; φ=0.15: 2.50 → 4.61. Same flattening shape as random forcing but ~2.5–3×
lower: the metric is dominated by the collective settling velocity, which the stack captures
well; the angular block alone is relatively much worse (`prmse_ang` ~19–28 % at large N — gravity
drives only disorder-induced rotations, a small-denominator effect). Seed bands are far tighter
(self-averaging mean flow). Figure: `figures/fig4_diff_sizes_grav.*`.

**Extension to N=10,000 (2026-09-06, gravity only).** `FIG4_N` += {5000, 7500, 10000} (p_idx
21–23, append-only), 3 seeds each, 4 plotted φ = 36 new cells. Truths: same sbatch with
`NS="5000 7500 10000" --mem=96G` (N=10000 = 4.86M boundary points → restart-200 Krylov basis
≈ 23 GB, over the file's 32G; RSA config generation is O(N²), ~81 s at N=10000) — all solves
CONVERGED_RTOL at 26–30 GMRES iterations, the whole 4-task array done in < 1 h. Eval: same GPU
adapter, N=10000 apply = 5.1 s at 4.06 GiB peak VRAM on the 4060 (the chunked far field holds;
mean settling velocity −152.8 at φ=0.15, growing with cluster size as expected). Results — the
error has **plateaued** at every φ: 0.79 / 1.60 / 3.15 / 4.78 (N=5000) → 0.79 / 1.61 / 3.21 /
4.91 (N=10000) for φ = 0.025 / 0.05 / 0.1 / 0.15, i.e. ≤ 0.3 points over the final 3.3× in N,
with seed spreads of ~0.01–0.03 points. CPU spot-check (N=5000 φ=0.1 seed 5223, 37 min on a
64 GB cluster CPU node — the laptop OOMs on the CPU op's dense far-field blocks at this N)
matches the GPU row exactly: rel_rmse 3.1505 both.

**Saturation check at N=20,000/30,000 (2026-09-07, φ=0.1 only, 2 seeds each).** Prompted by
"what's the error at N=1M?": pre-registered predictions from the N≤10,000 tail were 3.32/3.36 %
(saturation, e = e∞ − c·N^(−1/3)) vs 3.34/3.42 % (last-decade power law b=0.0577). Measured:
**3.263 ± 0.004 (N=20k) and 3.275 ± 0.006 (N=30k)** — below both, and ~25σ below the power law.
The log-log slope collapsed 0.056 (1k→10k) → 0.017 (10k→30k); a fixed power law forbids that,
box-size saturation requires it. Refit ceiling e∞ ≈ 3.53 (still overshooting the newest points
slightly), giving an N=1M expectation of **≈3.4–3.5 %**. Truth cost: N=30k solve = 24.8 min on
the H200 (14.6M boundary points, converged; RSA ~17 min extra); eval via the adapter's
row-block far field (one-shot N² nonzero would be 14 GB at N=30k), 48 s/apply, validated to
7e-4 pts on a known N=10k cell (fp32 atomic-add reordering). These cells are diagnostic-only
(φ=0.1), not part of the 4-φ figure grid. No random-forcing truths exist at N ≥ 5000 — a full `--exp fig4` sweep would hit the
missing-truth RuntimeError there; backfill on request.

## 5. Cost, and the N=3000 OOM lesson

- Truth: batch 1's 112 cells in ~25 min on one H200 (`srun` in a salloc); batch 2's 88 cells via
  `sbatch --array=0-7` in ≤ 10 min per φ-task. N=2000 ≈ 27–40 s/solve (33 GMRES its), N=300 ≈ 5 s.
  The treecode makes large-N MFS truth essentially free — the legacy GS path was estimated at
  20–45 min *per config* at N=2000.
- Eval (laptop, CPU): batch 1 in 37 min at `--workers 3`; wall/apply 7 s (N=300), 21 s (500),
  85 s (1000), ~190 s (1500), 338 s (2000), ~530 s (2500), ~650 s (3000) — the O(N²) Python
  two-body loop dominates.
- **OOM at N=3000 on the 15 GB laptop:** the moments operator built ALL pair-feature tensors at
  once (only the MLP call was chunked) — ~5 GB RSS per apply at N=3000, φ=0.15. That OOM-killed
  the 3-worker and 2-worker pools (and mp.Pool then **hangs forever**: the respawned worker never
  re-runs the lost task — kill the pool and rerun the missing cell, `--skip-done` makes it free).
  Fixed properly: `Mob_Op_Nbody_Moments.PAIR_ROW_CHUNK = 25_000` chunks the feature build in
  `get_nbody_velocity` (`_selected_pairs`/`_build_rows` split); parity vs unchunked 6e-9
  (float32 pairwise-summation grouping under the per-chunk padding width — the single-chunk new
  path is bit-identical to the old one), 16/16 moments tests pass.

## 6. Files

- `benchmarks/broms_truth.py` (+ `slurm/broms_truth.sbatch`) — truth generator/validator.
- widebvh `src/sphere_mfs.cu` — `--centers/--wrench` patch (laptop copy `~/envs/nemo-ctx/widebvh`
  and cluster copy both patched; cluster `build-rel/sphere_mfs` rebuilt).
- `benchmarks/paper_accuracy_v2.py` — FIG4_N append, `FIG4_REPEATS`, N>300 guard against the
  legacy generator, `store_M=False` in workers, pdist-based `nearfield_interactions`.
- `src/mob_op_2b_combined.py`, `src/mob_op_3body.py` — `store_M` gate.
- `src/mob_op_nbody_moments.py` — `PAIR_ROW_CHUNK` feature-build chunking (see §5).
- `figures/fig4_diff_sizes.py` — `--max-n` (default 200 keeps the paper figure byte-stable),
  `--out-suffix`, log-x for the large range.
- Outputs: `figures/fig4_diff_sizes_large.{pdf,png}`,
  `figures/M_accuracy_nbody_diff_sizes_avg_rel_rmse_large.{pdf,png}`; rows in
  `data/paper_accuracy_v2.csv` (op `M_mom_v2_kinf_rc8_pc8c_diag`, exp fig4, N ≥ 300); truths in
  `tmp/nbody_moments_truth/` (200 files, laptop + cluster).
