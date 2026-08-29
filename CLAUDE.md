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
