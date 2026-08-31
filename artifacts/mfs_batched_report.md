# Batched multi-RHS MFS mobility solver and the n-body dataset v2 generator

*2026-08-30 — developed and validated on the RTX 4060 laptop (torch 2.6 / Triton 3.2); production run intended
for the H200 cluster (torch 2.9.1 / Triton 3.5). User decisions: Triton instead of raw CUDA, `fine` MFS
accuracy, full grand mobility matrix per configuration, 8 h on the H200, default configuration mix.*

## 1. What was built

| piece | file |
|---|---|
| batched solver `BatchedMFS`: fp64 reference backend, fp32 Triton backend, Jacobi, batched restarted GMRES, `solve`, `solve_mobility_matrix`, `solve_mobility_matrix_batch`, column chunking by a memory budget, multi-configuration batching | `src/mfs_batched.py` |
| fp32 Triton kernel: matrix-free blocked GEMM of the Oseen tensor against R right-hand sides (9 `tl.dot`s per tile, `input_precision="ieee"`, autotuned) | `src/mfs_batched_kernels.py` |
| dataset v2 generator: `uniform` / `grown` / `lattice` families, sharded `.npz`, per-worker manifests, deterministic seeds, resume, time budget, multi-worker | `src/create_dataset_multibody_v2.py` |
| consumer: `load_multibody_v2` (ordered pairs with `M_ts`, `M_st`, `M_tt`, all neighbour positions) and `pair_rows_from_M` (old-convention rows with random forces) | `src/nbody_features.py` |
| validation (16 tests) and benchmark | `tests/test_mfs_batched.py`, `benchmarks/bench_mfs_batched.py` |

The reference solvers (`src/mfs.py`, `src/triton_mfs.py`) are untouched.

## 2. Formulation

Per sphere p the MFS unknowns are `x_p = [f_p (3M strengths); V_p; Ω_p] = B_inv [−W_p; F_p; T_p]`, with
`W_p = Σ_{q≠p} A_pq f_q` the boundary velocity induced by the other spheres (exact Oseen sum, μ = 1) and
`B_inv` the shared pseudo-inverse of the single-sphere matrix. Splitting `B_inv = [K | C]`:

```
f = X0_f − K_f W,   [V;Ω] = X0_v − K_v W,   X0 = C [F;T]        (isolated-sphere solution)
T W = b,   T = I + Ŵ K_f,   b = Ŵ X0_f                            (linear system in boundary-velocity space)
```

`Ŵ` (the off-diagonal Oseen operator) is the only expensive piece: `P(P−1)·3N·3M·R·2` flops per application
(1.3e12 at P=64, `fine`, R=6P=384). It is applied to all R columns of all batched configurations in one launch
(layout `S (3M, P_tot·R)`, `W (3N, P_tot·R)`, systems concatenated along the particle axis, coupled only
inside their own `[sys_start, sys_end)` range). `K_f` is one cuBLAS GEMM per iteration. The system is solved by
batched restarted GMRES(20) — 15–26 Krylov steps instead of 60–120 Gauss–Seidel sweeps of the reference —
with one Arnoldi process per (system, column) running in lockstep.

### Three findings that shaped the numerics (all measured)

1. **The pseudo-inverse must be applied in fp64.** `cond(B) ≈ 2e6`; an fp32 `K_f W` carries a net-force
   error of ~2e-3 per unit force whose Stokeslet field is 0.15 % of the label at r = 8 (test
   `test_b_fp32_gemm_is_unusable`). fp32 rounding of `W` before the fp64 GEMM is harmless.
2. **Convergence must be judged on velocities, not strengths or W residuals.** The MFS strengths are
   ill-determined (|f| up to 64, summing to |W| ≈ 0.09: cancellation ≈ 2e3). The fp32 kernel's W residual
   therefore stalls at 1e-4 while the velocities it produces are converged to ~1e-5; and for fp64 the W
   residual reaches 1e-6 long before the velocities reach 1e-8. The solver evaluates the velocities of the
   current GMRES iterate at every step (small triangular solve + a 6×3N GEMM per basis vector) and stops when
   their per-column relative change is below `tol_v` on two consecutive steps; the W residual is a safety stop.
3. **The fp32 floor is ≈1e-5 in velocity**, set by storing the strengths in fp32 (6e-6 alone) plus the
   kernel's own accumulation (1.4e-4 in W, 9e-6 visible in velocity); `TWO_LEVEL` accumulation and tolerance
   settings do not move it. `tf32x3`/`tf32` `tl.dot` crash the Triton 3.2 compiler on register operands and
   Triton has no fp64 `tl.dot`, so the exact backend is the fp64 batched-einsum operator: for every target
   sphere one dgemm `(3N × 3M·Q) @ (3M·Q × R)` over all its partners, G built elementwise in fp64. On this
   laptop it runs at 73 % of the GPU's dgemm peak (0.13–0.15 of 0.21 TFLOPS); on the H200 (fp64 tensor cores)
   it is the production backend.

Defaults: `torch64`: `tol_v = 1e-8` (residual safety stop 1e-10) → 17–26 steps, error ≤ 5e-10 of max|M|;
`triton32`: `tol_v = 1e-5` (its floor) → 15–18 steps, error 4e-6–1e-5.

## 3. Validation (`tests/test_mfs_batched.py`, all pass)

| check | result |
|---|---|
| (h) isolated sphere `self_mobility` vs `diag(1/6π, 1/8π)` | 2.2e-7 (fine), 3e-8 (Xfine) — MFS discretisation |
| (a) fp64 GMRES / Jacobi vs `src/mfs.py::imp_mfs_mobility_vec` (Gauss–Seidel, tol 1e-9), P = 2…6 uniform and grown | 1.4e-12 – 2e-12 on P=3; ≤ 1e-8 overall (the reference converges in strength space) |
| (g) Jacobi = GMRES fixed point | ≤ 1e-9; GMRES uses ~3× fewer operator applications |
| (b) fp32 kernel vs fp64, P = 8 / 16 / 32, φ = 0.15–0.2 | 4e-6 / 7e-6 / 1e-5 of max|M|; single-level accumulation equal; fp32 GEMM variant fails as documented |
| (c) old dataset rows (`data/multibody`, GS tol 1e-7) | fp64 1e-7 – 1.5e-6 (the rows' own accuracy); fp32 ≤ 4e-5 |
| (d) cached Xfine truths `tmp/testcase_uniform_*_20.csv` (GS tol 1e-8) | 7e-7 – 2.6e-6 |
| (e) `‖M − Mᵀ‖/‖M‖` (MFS discretisation, `fine`) | 7e-5 (uniform φ=0.15), 3e-4 (φ=0.2), 6e-4 (grown δ=0.1), 9e-7 (lattice φ=0.1) |
| (f) batch invariance | 6P columns == single column; 4 identical systems batched == one (bit-identical operator); column chunking identical |
| generator + loader | shard/manifest/resume; `Y = M_ts [F;T]_s` reproduces a fresh single-force solve (2e-7 – 1.5e-5 by backend) |

## 4. Performance on the 4060 (`data/mfs_batched_perf.csv`, `fine`, φ = 0.15, full grand M)

| P | R | fp32 kernel | fp64 GEMM | fp32 grand M | fp64 grand M |
|---:|---:|---:|---:|---:|---:|
| 8 | 48 | 1.0 ms (2.4 TFLOPS) | 2.4 ms | 0.21 s (14 steps) | 0.58 s |
| 16 | 96 | 3.7 ms (5.4 TFLOPS) | 6.7 ms | 0.22 s (16) | 6.9 s |
| 32 | 192 | 35 ms (4.7 TFLOPS) | 31 ms | 1.4 s (16) | 35 s |
| 64 | 384 | 293 ms (4.5 TFLOPS) | 114 ms | ~7 s | ~4 min (fp64 at 1/64 rate) |

Reference: the existing single-RHS Triton solver needs 480–750 s for **one** right-hand side at P=200 Xfine on
this GPU; the batched fp32 path computes all 192 columns of a P=32 grand matrix in 1.4 s. The laptop's fp64
numbers are bound by its 0.21 TFLOPS fp64 rate.

### Measured on the H200 (`data/mfs_batched_perf_h200.csv`, SLURM job 16539272, `fine`, full grand M)

| P | R | fp64 operator | fp64 grand M (production) | fp32 operator | fp32 grand M |
|---:|---:|---:|---:|---:|---:|
| 8 | 48 | 1.4 ms (1.7 TFLOPS) | 0.22 s (22 steps) | – | – |
| 16 | 96 | 4.1 ms (4.9 TFLOPS) | 0.18 s (27) | 1.0 ms (19.8 TFLOPS) | 0.17 s |
| 32 | 192 | 15.8 ms (10.4 TFLOPS) | 0.55 s (26) | 5.7 ms (28.7 TFLOPS) | 0.15 s |
| 64 | 384 | 71.9 ms (18.5 TFLOPS) | 2.43 s (27) | 44.4 ms (30.0 TFLOPS) | 0.99 s |

All 16 tests pass there too (torch 2.8 / Triton 3.4). At P=64 the exact fp64 backend is only 2.5× slower
than the fp32 kernel, so it is the production backend without reservation. Small P is launch-bound
(one dgemm per target per matvec); batching several configurations per solve would help there but was not
needed for the budget.

## 5. Dataset v2

- Families (fixed minimum gaps, as in practice): `uniform` (random sequential addition in a box from φ, gap ≥ 0.1,
  particle 0 at the origin; φ ∈ {0.025…0.25}, P ∈ {16, 32, 48, 64}), `grown` (each sphere at exactly gap δ from a
  random existing one, all pairs ≥ 2+δ; δ ∈ {0.05…1.0}, P ∈ {8…32}), `lattice` (jittered cubic drop patch,
  φ ∈ {0.05, 0.1, 0.15}, jitter 5–20 %, P ∈ {32, 64}). Global gap ≥ 0.05. Rounds interleave the families so the
  mix by ordered pair blocks is ≈ 66 / 23 / 12 %.
- Per configuration: positions (P,3) fp64, **M (6P×6P)** (fp32 storage by default; the fp64 solve is accurate to
  ~1e-9, so fp32's 6e-8 is lossless), seed, Krylov steps, residual, velocity change, symmetry error, wall time,
  and metadata (family, params, acc, backend, tolerances, solver version, git hash). Shards of 64–1024
  configurations (~40 MB) under `data/multibody_v2/{family}/{tag}/shard_k.npz`; `manifest_worker{i}.csv`,
  `failures_worker{i}.csv`.
- What M buys for training: every ordered pair's 6×6 block **and** its transpose partner and the self block,
  the complete neighbourhood (all particle positions), random-force augmentation every epoch
  (`pair_rows_from_M`), and a 36-entry block loss for the moments model instead of one 6-vector per solve.
### The generated dataset (SLURM array 16539317, 4 × H200, 55–77 min per task, 30 Aug 2026)

`--plan default --acc fine --backend torch64 --max-rounds 3`: 162 shards, **56,048 configurations, 4.4 GB fp32**,
**12.7 M ordered pairs within 6 radii** (the near-field gate; 19.8 M within 8), zero solver failures
(all residuals ≤ 2e-8, velocity changes ≤ 1e-8). Copied to the laptop (`data/multibody_v2/`, byte-identical).

| family | configurations | ordered pairs ≤ 6 | ≤ 8 | min gap | `‖M − Mᵀ‖/‖M‖` (fine MFS discretisation) |
|---|---:|---:|---:|---:|---|
| uniform (φ 0.025–0.25, P 16–64) | 17,840 | 6.43 M | 10.43 M | 0.100 | 3e-5 (φ=0.025) … 2.6e-4 (φ=0.25) |
| grown (δ 0.05–1.0, P 8–32) | 35,328 | 5.13 M | 7.05 M | = δ | 1.5e-3 (δ=0.05), 6e-4 (0.1), 1.2e-4 (0.2), 3.6e-5 (0.3), 7e-6 (0.5), 1e-6 (1.0) |
| lattice (φ 0.05–0.15, jitter 5–20 %, P 32/64) | 2,880 | 1.15 M | 2.28 M | 0.05–1.9 | 3e-7 … 8e-5 (3.8e-4 max at jitter 0.2, φ=0.15) |

The only rejections (2,320) are random-sequential-addition placement failures at φ=0.25 (a handful at φ=0.2),
so those items are thinner (85–301 configurations instead of 192–1,536); the seeds are in `failures_worker*.csv`.
The symmetry errors are the `fine` discretisation at small gaps (identical to the existing datasets): pairs at gap
0.05 carry ~1.5e-3 relative block asymmetry, gap 0.1 ~6e-4, gap ≥ 0.3 below 4e-5. `--acc Xfine` (4× cost) is the
lever if the near-contact clusters need tighter labels.

For training, iterate configurations lazily (`nbody_features.iter_multibody_v2_configs` +
`pairs_from_config`): the eager `load_multibody_v2` materialises ~2.4 KB per ordered pair and is meant for subsets.

### Cluster recipe

```bash
source ~/warp_env.sh
python benchmarks/bench_mfs_batched.py --backends torch64 --P 8 16 32 64        # confirm the fp64 rate first
for k in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$k nohup python src/create_dataset_multibody_v2.py --plan default --acc fine \
      --backend torch64 --mem-budget-gb 40 --time-budget 28800 --num-workers 4 --worker $k \
      > gen_worker$k.log 2>&1 &
done
```
Add `--max-rounds N` to bound the volume; `--acc Xfine` for higher near-contact fidelity (≈4× the cost; at
`fine` the 6×6 blocks of gap-0.1 pairs are asymmetric by ~1e-3, gap-0.05 worse — that is the MFS discretisation,
identical to the existing datasets).

## 6. Caveats and next steps

- H200 throughput above is a projection; the benchmark script gives the real number. If the fp64 operator turns
  out memory-bound there (G tensor construction + permute copy per target), the next optimisation is building
  G directly in the GEMM layout in a Triton kernel, or an fp64 broadcast-FMA Triton kernel.
- `triton32` is the fast option (≈1e-5 velocity accuracy, below the `fine` discretisation error) if a much larger
  volume is ever needed; a double-single kernel would lift its floor at ~3× the cost.
- Training on v2: `load_multibody_v2` + `pair_rows_from_M` feed the existing trainer unchanged; a 6×6-block
  loss and the neighbourhood-about-the-midpoint sampling (design doc §8) are the natural next steps for the
  moments model.
