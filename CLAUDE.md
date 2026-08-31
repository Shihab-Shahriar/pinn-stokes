# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Physics-Informed Neural Network (PINN) mobility operators for Stokes flow particle simulations. The project replaces expensive numerical solvers (Method of Fundamental Solutions / MFS) with learned neural network approximations for computing hydrodynamic mobility matrices of rigid particles (currently spheres) in viscous flow.

The mobility operator maps forces/torques on particles to their translational/angular velocities. The system builds hierarchical corrections: self-interaction -> two-body -> three-body/n-body, with RPY (Rotne-Prager-Yamakawa) as a far-field analytical fallback beyond a switch distance of ~6 radii.

## Key Conventions
- The code, files resides in an university cluster, we don't have sudo access
- DO NOT try to create any "/tmp" directory as I don't have access.
- MUST do "source ~/warp_env.sh" to load necessary libraries and init conda environment in each terminal session

- For accuracy tests and debugging, always disable torch.compile: `export TORCH_COMPILE_DISABLE=1`. For performance tests, keep it enabled.
- **Local consumer-GPU box (no cluster modules):** run everything inside the Docker image through `bash docker/run_local.sh <cmd>` -- it bind-mounts this repo over `/workspace/pinn-stokes`, the widebvh source (`~/envs/nemo-ctx/widebvh`, git-tracked there) over `/opt/widebvh-src` with `WIDEBVH_BUILD_DIR=/opt/widebvh-src/build-4060`, and a persistent JIT cache. Native runs are not viable (no warp in the base conda; host nvcc 13.1 vs driver 580). See `artifacts/consumer_gpu_far_field_report.md`.
- For VRAM measurements, `torch.cuda.max_memory_allocated/reserved` are **not** the answer: the widebvh treecode allocates its BVH, buckets and pair list with raw `cudaMalloc`, which never passes through PyTorch's caching allocator (~0.7-1.8 GiB unseen at N=1M). Set `NEMO_DEVICE_MEM=1` to add the true process footprint to the `[MobFMM] peak GPU memory` line; leave it off for timing runs, since it forks `nvidia-smi` per step. Use per-PID accounting, not a `mem_get_info` free-memory delta — the cluster GPUs are shared, and the delta reads low (sometimes below the process's own `reserved`) when another tenant frees memory mid-run. Measure one config per process, as with timing.
- Prefer assertions over verbose error checking in production code
- Models are TorchScript-serialized (`.pt` files in `data/models/`)
- Training is done in Jupyter notebooks (`experiments/*.ipynb`), model architectures live in `src/model_archs.py`, and the `.wt` weight files in `experiments/` are converted to TorchScript `.pt` via `model_archs.py`'s `__main__` block
- Particle config format: `(N, 7)` array — columns `[x, y, z, qw, qx, qy, qz]` (position + quaternion). For spheres, quaternion is identity
- Force format: `(N, 6)` array — `[Fx, Fy, Fz, Tx, Ty, Tz]`
- Velocity format: `(N, 6)` array — `[Ux, Uy, Uz, Ox, Oy, Oz]`

## Architecture

### Mobility Operator Hierarchy (src/)

Each operator has an `apply(config, force, viscosity) -> velocity` interface:

1. **`mob_op_2b_combined.py` → `NNMob`**: CPU grand mobility operator. Combines self-interaction NN + two-body NN (separate M_s and M_t models via `TwoBodyCombined`). Falls back to RPY for pairs beyond `switch_dist`. O(N^2) complexity.

2. **`gpu_mob_2b.py` → `NNMobTorch`**: GPU-accelerated version of `NNMob`. Uses Warp hash grid (`hashgrid_neighbors.py`) for neighbor search instead of brute-force. Sphere-only, avoids quaternion rotations.

   The two-body pair path is chunked at `two_body_chunk_size` (default 2M pairs) in `_two_body_velocity`, under the same two invariants as the n-body loop below. Until this was chunked it ran the whole edge list in one call and **set peak VRAM for the entire stack** — 10.5 GiB of a 15.4 GiB total at N=1M, which also meant `pair_chunk_size` was capped from below and appeared to do nothing. Chunk both or neither: alone, either one is masked by the other.

3. **`mob_op_nbody.py` → `Mob_Op_Nbody`** (extends `NNMob`): Adds learned n-body correction on top of two-body. For each pair (t,s), consumes up to 10 neighbor positions and predicts a 6-vector velocity correction.

4. **`gpu_nbody_mob.py` → `Mob_Nbody_Torch`** (extends `NNMobTorch`): GPU version of n-body operator. This is the primary operator used in large-scale simulations.

   Two invariants here are load-bearing and were each violated once (see `artifacts/widebvh_far_field_report.md` §5). They apply equally to `NNMobTorch._two_body_velocity`:
   - **The per-particle neighbour table (`_per_particle_topk`) must be built over the complete edge list, never per pair-chunk.** Every pair indexes it with *both* endpoints, but a chunk holds only a contiguous range of targets, so a per-chunk table silently drops the source's neighbours for pairs that straddle a chunk. It is directional, so it breaks symmetry, and it only appears once `num_pairs > pair_chunk_size` (~400k particles at φ=0.1) — small-N tests cannot see it. It cost 1.75e-2 relative error in the velocities at N=1M.
   - **The chunk loop must stay outside the compiled region.** The loop bound is the pair count, which drifts every timestep in a dynamics run. Inside `torch.compile` that means either a guard on its exact value (recompile per step → `config.recompile_limit` → silent permanent eager fallback, ~2.4× slower with no error) or, once marked dynamic, a `range()` over a symbolic int, which dynamo rejects outright. Compile one chunk at a time with the pair dimension marked dynamic.

5. **`treecode_widebvh.py` → `WidebvhFMM`**: **the production far field.** `ctypes` wrapper over the widebvh BaryStokes treecode (`/mnt/ffs24/home/khanmd/programs/widebvh`, built as `libwidebvh_nemo*.so`); degree-7 barycentric-Lagrange expansion on a cuBQL LBVH. Subclasses `WarpFMM` and overrides only `get_far_field_vel`, so the near-field path and the `[MobFMM]` stdout keys are shared. Key parameter is `mac` (default 0.8) — **not** `WarpFMM`'s `theta`; they are different acceptance criteria and passing one for the other is rejected outright. See `artifacts/widebvh_far_field_report.md` and `benchmarks/mac_calibration.py`.

   `policy` selects the multipole expansion: `"bary"` (production, degree-7 barycentric Lagrange, `mac` 0.8) or `"cart"` (the analytic Cartesian Taylor expansion, runtime `order` 1..4, `mac` 0.33 at matched accuracy) — `NEMO_FAR_FIELD=widebvh-cart` anywhere. They truncate different series, so **no `mac` transfers between them**, and each needs its own bucket granularity: at its 2.4x tighter `mac` the Cartesian policy carries ~11x the near pairs, and on the engine's automatic cell edge (~1024 cells regardless of N) that costs 286 ms of P2P out of a 346 ms far field at N=1M. `cart_hilbert_q()` sizes it instead. See `artifacts/cartesian_far_field_report.md`.

   `fp32_level` (0..3, default 0 = the fp64 production kernels; env `NEMO_FAR_FP32_LEVEL`) selects widebvh's fp32 fast path, one `.so` per level (`libwidebvh_nemo[_p<N>]_f32l<L>.so`, built with `-DWIDEBVH_NEMO_FP32_LEVELS="1;2;3"`). **On GeForce-class GPUs (fp64 at 1/64 the fp32 rate) use level 3**: on an RTX 4060 laptop at N=1M the far field went 8.05 s -> 0.35 s (M2P 5.9 s -> 0.23, P2P 1.9 s -> 0.09, upward 0.19 s -> 0.02) with the same interaction lists and no measurable accuracy or symmetry change (rel_far 1.133e-6 -> 1.138e-6). Level 0 stays byte-identical to the pre-fp32 engine (SASS-diffed), so H200 numbers are unaffected. `pdeg=5` remains a false economy at matched accuracy on this GPU too (needs mac 0.7, lands at the same far-field time as pdeg 7 / mac 0.8 with 2x the error). Consumer-GPU operating point: `--fp32-level 3` with everything else as production (pdeg 7, mac 0.8, leaf 1024; leaf 512 measures the same within noise). Trajectory A/B vs fp64 over the 100-step Figure-13 run: max 0.035 particle radii, panels 99.997% pixel-identical.

6. **`treecode.py` → `WarpFMM`**: the previous far field, kept as the A/B baseline. Warp BVH-based treecode with a monopole+dipole expansion; depends on a patched Warp checkout. Key parameters: opening angle `theta` (0.3 in production) and leaf size (16). Select it anywhere with `NEMO_FAR_FIELD=warp`.

4b. **`mob_op_nbody_moments.py` → `Mob_Op_Nbody_Moments`** (extends `Mob_Op_Nbody`, CPU): the moments-based n-body correction of `moments_for_nbody.md` — per pair, band moments `(s_a, v_a, Q_a)` of the neighbourhood (8 unit-width radial bands about the pair midpoint; `src/nbody_moments.py`, TorchScript-scriptable) → 72 rotation-invariant swap-even invariants → `MultiBodyMoments` MLP (`model_archs.py`, 76→128→64→128→64→93) → 93 coefficients on 34 TT/RR + 25 TR analytic bases, so `M_ji = M_ij^T` and O(3) equivariance hold by construction (`tests/test_nbody_moments.py`). Model input row is 111 columns `[s_vec(3) | d−mean, d−2, (d−mean)², (d−mean)⁴ | s_a(8) | v_a(24) | Q_a(72)]`, built by `nbody_features.moment_features` for both training and inference. Neighbour *selection* is the baseline's (within `neighbor_cutoff` of the midpoint, top-`max_neighbors` by `d_kt·d_ks`; `max_neighbors=None` = unbounded) because the existing training rows carry ≤10 neighbours; the band weights deliberately do **not** zero beyond r_c = 8 (training rows contain neighbours farther from the midpoint that affect the label). Trained by `experiments/train_nbody_moments.py`; compared against MFS truth by `benchmarks/compare_nbody_moments.py` (warp-free; caches truth in `tmp/nbody_moments_truth/`). See `artifacts/nbody_moments_report.md`.

   **Residual-label convention (load-bearing):** the n-body residual is `Y − v2b` with the two-body term evaluated exactly as the operators do it, `X2b[:, :3] = +s_vec = source − target` (`nbody_features.two_body_velocity`). The saved `branch1_multibody_pinn.ipynb` feeds `−s_vec` in `predict_two_body_from_triplet`, which flips the RT/TR blocks (L3 is odd) — its "2-body only" 38 % / 199 % validation errors are that artefact (true values 6.8 % / 16.4 %), and a model trained on those labels is catastrophically wrong inside the operator (28 % vs 4 % at φ=0.05).

4c. **Dataset-v2 retraining + paper accuracy harness (2026-08-30, `artifacts/nbody_v2_retrain_report.md`).** Training rows now come from `data/multibody_v2/` through *the operators' own* selection: `nbody_features.select_pair_neighbours` (unordered near pairs t<s, d ≤ 6; neighbours within `neighbor_cutoff` of the pair midpoint, top-`max_neighbors` by d_kt·d_ks, index-sorted) is the single code path used by `Mob_Op_Nbody_Moments._pair_rows` and by the cache builder. Labels are full residual blocks `R = 0.5(M_ts + M_st^T) − M2b_ts` with `nbody_features.two_body_blocks` (`predict_mobility(X2b)[1]`, +s_vec, **median 5.01** = the operator's constant); both architectures are reciprocal by construction, so one unordered row per pair suffices. Pipeline: `experiments/build_nbody_v2_cache.py` (geometry + labels, 2.7 GB in `data/multibody_v2_cache/`, features built on the fly; `--stats` for coverage/asymmetry/diagonal-residual tables) → `experiments/train_nbody_v2.py --model {moments,baseline} --variant {k10_rc6,kinf_rc6,kinf_rc8} --publish` (L1 over the 36 block entries × 6π, 100 epochs ≈ 10 min on the 4060, configuration-level split `seed % 10 == 0`) → `benchmarks/paper_accuracy_v2.py` (warp-free Fig 3 / Fig 4 / clustered protocols of `accuracy_grand_M.py` with the same truth generator and seeds, truths cached in `tmp/nbody_moments_truth/` (1,280 files, also on the cluster), CPU worker pool, `--gpu-ops` for `mfs_coarse` + the paper's GPU operator, `--part/--merge/--summary/--figures`; SLURM `slurm/paper_truth.sbatch` (GPU, array over φ) then `slurm/paper_eval.sbatch` (32 CPU workers, ~1 min per φ)). **Published models carry a `.json` sidecar with the selection they were trained on; run them with exactly that selection** (`Mob_Op_Nbody_Moments(max_neighbors=…, neighbor_cutoff=…)`): `nbody_moments_v2_k10_rc6.pt` (10, 6), `nbody_moments_v2_kinf_rc6.pt` (None, 6), `nbody_moments_v2_kinf_rc8.pt` (None, 8; best), `nbody_pinn_b1_v2.pt` (K=10, r_c 6, drop-in for `Mob_Op_Nbody`). Results: v2 validation PRMSE lin/ang 15.9/37.5 % (2-body) → 8.97/22.1 (shipped b1) → 6.87/19.8 (b1 on v2) → 3.98/15.4 (moments v2, r_c 8); paper Fig 3 at N=200, φ=0.2: 15.0 % (paper n-body) → 11.2 % (b1 on v2) → **9.1 %** (moments v2 r_c 8), beating the 3-body summation everywhere. `experiments/nbody_v2_ceiling.py` shows the floor of *any* pairwise near-field correction on v2 boxes is 2.9 % (φ 0.1) / 4.1 % (φ 0.2), almost entirely the far-field term (pairs beyond the switch distance 6, RPY, never corrected), not the diagonal block. GPU port of the moments model still pending.

   **Pair-cutoff 8 ablation (2026-08-30, `artifacts/pair_cutoff_ablation.md`):** raising the corrected-pair range to d ≤ 8 attacks that far-field floor directly. `nbody_moments_v2_kinf_rc8_pc8.pt` (sidecar: None, r_c 8, **pair_cutoff 8**) is trained on `data/multibody_v2_cache_pc8` (`build_nbody_v2_cache.py --pair-cutoff 8 --variants kinf_rc8`, 9.88 M pairs, 3.9 GB — build on scratch/laptop, it blew the cluster home quota once) with the identical recipe, and **must run with `pair_cutoff=8, switch_dist=8`**: the 2b NN (trained to d = 8) is the base wherever pairs are corrected, and NN-vs-RPY differs by only ~2e-4 in the 6–8 shell, so the base swap is free. `benchmarks/paper_accuracy_v2.py` passes both from the sidecar (`SELECTION` is now (K, r_c, pair_cutoff) 3-tuples; `_selection_for` asserts the sidecar's pair_cutoff too). Results: Fig 3 φ=0.2 PRMSE 9.08 → **7.41 %** (N=200) and 11.94 → 9.89 % (N=300); Fig 4 −15…−20 % at every (N, φ), steady in N; clustered δ better everywhere; max per-particle error 27.6 → 22.9 % at φ=0.2. The exact-residual floor (`nbody_v2_ceiling.py --pair-cutoff 8`, must match the cache) moves 2.86/4.08 → 1.70/2.69 % at φ=0.1/0.2 — the correction captures roughly half the newly exposed headroom, and what remains is the d > 8 far field plus the diagonal. `benchmarks/pair_cutoff_ablation.py` renders the report and `figures/pair_cutoff_ablation_fig{3,4}.*` from the merged CSV.

7a. **`mfs_batched.py` → `BatchedMFS`**: the **batched multi-right-hand-side MFS solver** (Triton + torch; the reference solvers below are single-RHS Gauss–Seidel). Solves R right-hand sides of n_sys equal-size configurations at once in boundary-velocity space (`T W = b`, `T = I + Ŵ K_f`) with batched restarted GMRES; `solve_mobility_matrix(positions)` returns the full grand mobility matrix M (6P×6P) from all 6P unit force/torque columns. Backends: `torch64` (exact fp64, one cuBLAS dgemm per target over all partners — **production on the H200**, fp64 at half the fp32 rate there; 50–70× slower on GeForce) and `triton32` (fp32 `tl.dot` kernel in `mfs_batched_kernels.py`, 4–5 TFLOPS on the 4060, ~1e-5 relative accuracy: the MFS strengths cancel by ~1e3, so fp32 storage of the strengths alone costs 6e-6). Two things are load-bearing: **the pseudo-inverse is always applied in fp64** (an fp32 GEMM carries a 2e-3 net-force error into the labels), and **convergence is judged on the velocities** (`tol_v`, per column, checked at every Krylov step), never on strengths or on the W residual — the fp32 kernel's W residual stalls at 1e-4 while its velocities are converged. Defaults `tol_v` = 1e-7 (torch64) / 1e-5 (triton32); 17–26 Krylov steps. Validated by `tests/test_mfs_batched.py` against `src/mfs.py` (2e-12 in fp64), the old dataset rows and the cached Xfine truths; `benchmarks/bench_mfs_batched.py` measures throughput. `tf32x3`/`tf32` `tl.dot` crash Triton 3.2 on register operands; fp64 `tl.dot` is unsupported.

   **Dataset v2 generator** `src/create_dataset_multibody_v2.py`: full M per configuration for `uniform` (RSA box, gap ≥ 0.1), `grown` (cluster growth at gap δ) and `lattice` (jittered cubic drop patch) families, sharded `.npz` under `data/multibody_v2/` (gitignored) with per-worker manifests, deterministic seeds, resume, `--time-budget`, `--worker/--num-workers`. Cluster recipe: `source ~/warp_env.sh`, then per GPU `CUDA_VISIBLE_DEVICES=k python src/create_dataset_multibody_v2.py --plan default --acc fine --backend torch64 --mem-budget-gb 40 --time-budget 28800 --num-workers 4 --worker k`. Consumer: `nbody_features.iter_multibody_v2_configs` + `pairs_from_config` (lazy, for full passes), `load_multibody_v2` (eager ordered-pair table with `M_ts`, `M_st`, `M_tt`, all-neighbour positions — ~2.4 KB per pair, subsets only) and `pair_rows_from_M` (old-convention rows with random forces, `Y = M_ts [F;T]_s`, `+s_vec`). **Generated 2026-08-30** (SLURM array 16539317, `slurm/gen_multibody_v2.sbatch`, 4 × H200 ≈ 4.5 GPU-h): 56,048 configurations, 12.7 M ordered pairs within 6 radii, 4.4 GB, in `data/multibody_v2/` on the cluster and the laptop; see `artifacts/mfs_batched_report.md` §5.

7. **`mfs.py` → `MobOpMFS`**: Reference MFS solver (ground truth). Iterative multi-particle solver using Oseen tensor. `triton_mfs.py` → `MobMFSTriton` is the Triton-accelerated variant.

### Two-Body NN Decomposition

The two-body model predicts 5 scalar coefficients that are assembled into a 6x6 mobility kernel using geometric tensor bases: `L1` (parallel/outer product), `L2` (perpendicular), `L3` (cross-product/angular coupling). The kernel has TT (translation-translation), RT (rotation-translation), and RR blocks. M_s is symmetric; M_t is not.

### Neighbor Search

`hashgrid_neighbors.py` wraps Warp's hash grid for GPU neighbor queries. Used by GPU mobility operators to find pairs within cutoff distance, returning COO-format edge indices.

### Reference Data Generation

`benchmarks/cluster.py` generates test configurations (random sphere clusters, uniform packings) and computes ground-truth velocities via MFS for validation.

## Running

```bash
# Accuracy benchmarks (compare NN operators against MFS ground truth)
TORCH_COMPILE_DISABLE=1 python benchmarks/accuracy_grand_M.py

# Performance benchmarks
python benchmarks/performance_grand_M.py

# Large-scale simulation (two suspension drops, ~1M particles)
python benchmarks/two_suspensions_1M.py

# Run a specific mobility operator standalone (most src files have __main__ blocks)
python src/mob_op_nbody.py
python src/mfs.py
```

## Dependencies

Core: PyTorch, Warp (NVIDIA, for GPU hash grid + treecode BVH), NumPy, SciPy. Optional: Triton (for MFS GPU kernels), tiny-cuda-nn. Requires CUDA GPU for GPU operators.
