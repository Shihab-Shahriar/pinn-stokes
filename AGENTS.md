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

3. **`mob_op_nbody.py` → `Mob_Op_Nbody`** (extends `NNMob`): Adds learned n-body correction on top of two-body. For each pair (t,s), consumes up to 10 neighbor positions and predicts a 6-vector velocity correction.

4. **`gpu_nbody_mob.py` → `Mob_Nbody_Torch`** (extends `NNMobTorch`): GPU version of n-body operator. This is the primary operator used in large-scale simulations.

5. **`treecode.py` → `WarpFMM`**: Warp BVH-based treecode (simple FMM) for far-field interactions. Replaces CPU-based STKFMM. Key parameters: opening angle (0.25) and leaf size (16).

6. **`mfs.py` → `MobOpMFS`**: Reference MFS solver (ground truth). Iterative multi-particle solver using Oseen tensor. `triton_mfs.py` → `MobMFSTriton` is the Triton-accelerated variant.

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
