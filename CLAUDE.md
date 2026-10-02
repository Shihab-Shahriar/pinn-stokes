# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This code will be used in a paper called NeMO. Everything specific to the author's machines (the laptop, the MSU HPCC cluster, sibling folders, the paper's LaTeX source) is in the last section, **Author's setup**; everything above it applies on any machine.

## Project Overview

Physics-Informed Neural Network (PINN) mobility operators for Stokes flow particle simulations. The project replaces expensive numerical solvers (Method of Fundamental Solutions / MFS) with learned neural network approximations for computing hydrodynamic mobility matrices of rigid particles (currently spheres) in viscous flow.

The mobility operator maps forces/torques on particles to their translational/angular velocities. The system builds hierarchical corrections: self-interaction -> two-body -> three-body/n-body, with RPY (Rotne-Prager-Yamakawa) as a far-field analytical fallback beyond a switch distance: 6 radii for the 2-body/baseline n-body operators, 8 for the moments models (pc8/pc8c/v3 + diag, which require `switch_dist = 8`).

Unless otherwise stated, we shall always use the moments model from now on for every accuracy/performance measurements in the paper.

## Where to run

- **Warp-free** (torch + triton): the CPU operators (`NNMob`, `Mob_Op_Nbody`, `Mob_Op_Nbody_Moments`), `benchmarks/paper_accuracy_v2.py` without `--gpu-ops`, `experiments/train_*_v2.py`, `BatchedMFS` / Triton MFS truth (N = 200 ≈ 8 min per config on an RTX 4060), pytest. Run from the repo root with `export PYTHONPATH=$PWD`.
- **Warp + widebvh** — anything that imports them (the `gpu_*` operators, `treecode*`, `hashgrid_neighbors`, `accuracy_grand_M.py`, `paper_accuracy_v2.py --gpu-ops`, `fig8_single_drop.py`). The Docker image has both (`docker/README.md`). `bash docker/run_local.sh <cmd>` runs a command in it and bind-mounts:
  - this repo over `/workspace/pinn-stokes`;
  - a widebvh source tree (`WIDEBVH_SRC`) over `/opt/widebvh-src`, with `WIDEBVH_BUILD_DIR=/opt/widebvh-src/build-$WIDEBVH_BUILD_TAG`;
  - a persistent JIT cache (`NEMO_CACHE`).
  Its defaults are the author's laptop. On GeForce cards the far field's fp32 default (level 3) is what makes it usable, since their fp64 runs at 1/64 rate (`artifacts/consumer_gpu_far_field_report.md`).
- **External paths the code reads.** The defaults are the author's machines, so set these elsewhere:
  - `WIDEBVH_BUILD_DIR`: a widebvh build holding `libwidebvh_nemo_f32l3.so` (§5).
  - `HIGNN_ROOT`, `SD_ROOT`: checkouts of the HIGNN and Stokesian Dynamics baselines.
  - `BROMS_SPHERE_MFS`: widebvh's `sphere_mfs` binary, for `benchmarks/broms_truth.py`.
  `benchmarks/mac_calibration.py`'s `WIDEBVH_DISTROS` is hardcoded.
- **Performance runs** belong on a datacenter GPU; the paper's numbers are H200.
  - The `slurm/` scripts' `#SBATCH` headers (account, constraint) and `slurm/h200_v3/env.sh` paths are the author's cluster; adapt them elsewhere.
  - `slurm/h200_v3/` is the current recipe. `env.sh` sets: the widebvh build; `NEMO_FAR_FP32_LEVEL` default 3, with output names tagged `f32l<L>`; torch.compile on; `NEMO_DEVICE_MEM` off; a per-run manifest into `artifacts/logs/`.
  - Pass `--near-op moments-v3`: every perf script defaults to `baseline`, the published K=10 operator.
  Pitfalls:
  - one configuration per process for timing;
  - at most 8 sizes or operators per process — the 9th hits `recompile_limit` and silently runs eager;
  - the third solver built in one process can report a ~10x far field;
  - GeForce idle clocks need a long `--warmup` (400 at 50k on the 5090);
  - above ~52M particles use `--empty-cache`;
  - capacity stops at 340³ because `hashgrid_neighbors.py` counts pairs in int32.
  Current numbers: `artifacts/h200_v3_timing_report.md` (2026-09-23, v3 + fp32 level 2: N=1M apply 559 ms with 63 ms far field; two-drop 0.44 s/step; level 3 takes ~4 ms off at 1M).

## Key Conventions
- For accuracy tests and debugging, always disable torch.compile: `export TORCH_COMPILE_DISABLE=1`. For performance tests, keep it enabled.
- For VRAM measurements, `torch.cuda.max_memory_allocated/reserved` are **not** the answer: the widebvh treecode allocates its BVH, buckets and pair list with raw `cudaMalloc`, which never passes through PyTorch's caching allocator (~0.7-1.8 GiB unseen at N=1M). Set `NEMO_DEVICE_MEM=1` to add the true process footprint to the `[MobFMM] peak GPU memory` line; leave it off for timing runs, since it forks `nvidia-smi` per step. Use per-PID accounting, not a `mem_get_info` free-memory delta — on a shared GPU (a cluster node) the delta reads low (sometimes below the process's own `reserved`) when another tenant frees memory mid-run. Measure one config per process, as with timing.
- Prefer assertions over verbose error checking in production code
- Models are TorchScript-serialized (`.pt` files in `data/models/`)
- Training is done in Jupyter notebooks (`experiments/*.ipynb`), model architectures live in `src/model_archs.py`, and the `.wt` weight files in `experiments/` are converted to TorchScript `.pt` via `model_archs.py`'s `__main__` block
- Particle config format: `(N, 7)` array — columns `[x, y, z, qx, qy, qz, qw]` (position + **scalar-last** quaternion, `Rotation.from_quat(..., scalar_first=False)` in `mob_op_2b_combined.py`). For spheres, quaternion is identity = `[0, 0, 0, 1]` (set column 6 to 1, as in `figures/fig2_nbody_acc.py`). A scalar-first identity lands a 180° rotation in the pair frames and silently corrupts the kernel (~60 % velocity error at Fig-6 spacing)
- Force format: `(N, 6)` array — `[Fx, Fy, Fz, Tx, Ty, Tz]`
- Velocity format: `(N, 6)` array — `[Ux, Uy, Uz, Ox, Oy, Oz]`

## Architecture

### Mobility Operator Hierarchy (src/)

CPU operators have `apply(config, force, viscosity) -> velocity`; GPU operators take tensors, `apply(positions, orientations, force, viscosity, t_idx=None, s_idx=None)` (`WarpFMM`/`WidebvhFMM`: `apply(positions, orientations, forces, vis_arr)`):

1. **`mob_op_2b_combined.py` → `NNMob`**: CPU grand mobility operator. Combines self-interaction NN + two-body NN (separate M_s and M_t models via `TwoBodyCombined`). Falls back to RPY for pairs beyond `switch_dist`. O(N^2) complexity.

2. **`gpu_mob_2b.py` → `NNMobTorch`**: GPU-accelerated version of `NNMob`. Uses Warp hash grid (`hashgrid_neighbors.py`) for neighbor search instead of brute-force. Sphere-only, avoids quaternion rotations.

   The two-body pair path is chunked at `two_body_chunk_size` (default 4M pairs; the n-body `pair_chunk_size` default is 2M) in `_two_body_velocity`, under the same two invariants as the n-body loop below. Until this was chunked it ran the whole edge list in one call and **set peak VRAM for the entire stack** — 10.5 GiB of a 15.4 GiB total at N=1M, which also meant `pair_chunk_size` was capped from below and appeared to do nothing. Chunk both or neither: alone, either one is masked by the other.

3. **`mob_op_nbody.py` → `Mob_Op_Nbody`** (extends `NNMob`): Adds learned n-body correction on top of two-body. For each pair (t,s), consumes up to 10 neighbor positions and predicts a 6-vector velocity correction.

4. **`gpu_nbody_mob.py` → `Mob_Nbody_Torch`** (extends `NNMobTorch`): GPU version of n-body operator. This is the primary operator used in large-scale simulations.

   Two invariants here are load-bearing and were each violated once (see `artifacts/widebvh_far_field_report.md` §5). They apply equally to `NNMobTorch._two_body_velocity`:
   - **The per-particle neighbour table (`_per_particle_topk`) must be built over the complete edge list, never per pair-chunk.** Every pair indexes it with *both* endpoints, but a chunk holds only a contiguous range of targets, so a per-chunk table silently drops the source's neighbours for pairs that straddle a chunk. It is directional, so it breaks symmetry, and it only appears once `num_pairs > pair_chunk_size` (~400k particles at φ=0.1) — small-N tests cannot see it. It cost 1.75e-2 relative error in the velocities at N=1M.
   - **The chunk loop must stay outside the compiled region.** The loop bound is the pair count, which drifts every timestep in a dynamics run. Inside `torch.compile` that means either a guard on its exact value (recompile per step → `config.recompile_limit` → silent permanent eager fallback, ~2.4× slower with no error) or, once marked dynamic, a `range()` over a symbolic int, which dynamo rejects outright. Compile one chunk at a time with the pair dimension marked dynamic.

5. **`treecode_widebvh.py` → `WidebvhFMM`**: **the far field — always use it; fp32 approximations are fine, performance comes first.** `ctypes` wrapper over the widebvh BaryStokes treecode (C ABI `wbnemo_*` in widebvh's `src/nemo_capi.cu`; one `libwidebvh_nemo*.so` per build variant, looked up in `WIDEBVH_BUILD_DIR`). Degree-7 barycentric-Lagrange expansion on a **binary, SAH-built cuBQL BVH** (`TC_BVH_BUILDER=sah`; "wide" is only the project name). The kernel is RPY for unit spheres (μ = 1) and maps **forces to translational velocities only**: torques get no far-field contribution. It evaluates exactly the pairs with r ≥ `near_field_cutoff` — the complement of the NN neighbour list, so nothing is double counted — and that cutoff must equal the near operator's switch distance (6 baseline, 8 moments). Subclasses `WarpFMM` and overrides only the far field (`get_far_field_vel`), so the near-field pass (Warp hash grid) and the `[MobFMM]` stdout keys are shared; far and near run one after the other. Operating point: `mac` 0.8, `max_leaf` 1024, `pdeg` 7, `TC_PATH=split-warpspec-atomic` (the fastest path, and the only one with an fp32 P2P). The constructor writes `TC_PATH`/`TC_BVH_BUILDER`/`TC_QUIET`/`TC_HILBERT_Q` into the environment itself, so setting them outside has no effect; `TC_PAIR_BUDGET_GB` is honoured if set (read once per process), else ~6 % of VRAM clamped to 1–12 GB. `mac` is **not** `WarpFMM`'s `theta`; they are different acceptance criteria and passing one for the other is rejected outright. See `artifacts/widebvh_far_field_report.md` and `benchmarks/mac_calibration.py`.

   Builds.
   - Source: widebvh commit 03efcdb. A pre-fp32 widebvh builds fp64 libraries only, which cannot serve the default level.
   - `WIDEBVH_BUILD_DIR` must hold the library for the level in use (`libwidebvh_nemo_f32l3.so` by default). The code's default path is the author's cluster build (Author's setup).
   - CMake: `-DWIDEBVH_NEMO_PDEG="5;7" -DWIDEBVH_NEMO_FP32_LEVELS="1;2;3"`.
   - Library names: pdeg 7 keeps the plain name, other degrees get `_p<N>`, fp32 levels append `_f32l<L>`, and the Cartesian policy is `libwidebvh_nemo_cart.so`.
   - The Docker image builds levels 1–3 by default; `slurm/h200_v3/build_widebvh.sbatch` builds levels 1–3 + cart for sm_90.

   `fp32_level` (0..3) sets how much of the far field runs in fp32: 1 = M2P, 2 = + P2P (only on `split-warpspec-atomic`), 3 = + upward pass (P2M/M2M). **Default: level 3 on every GPU** (`DEFAULT_FP32_LEVEL`). `NEMO_FAR_FP32_LEVEL` overrides it everywhere, `fp32_level=` / `--fp32-level` per run, and 0 selects the fp64 kernels for an A/B. The far field stays truncation-dominated at every level, and level 3 is the fastest everywhere measured:
   - **RTX 4060, N=1M far field:** 8.05 s (level 0) → 0.35 s (level 3). M2P 5.9 → 0.23 s, P2P 1.9 → 0.09 s, upward pass 0.19 → 0.02 s. rel_far 1.133e-6 → 1.138e-6.
   - **H200 far field, level 0 → 2:** 91 → 62 ms at 1M (2026-08-22, cutoff 6).
   - **H200 far field, level 2 → 3:** 61.7 → 58.1 ms at 1M, 124.7 → 118.9 ms at 2M, 252.9 → 242.6 ms at 4M (cutoff 8).
   - **H200 end-to-end, level 3 vs 2:** −3.8 ms at 1M, −5.8 ms at 2M. rel_far and symmetry identical to 4–5 digits (`artifacts/h200_v3_timing_report.md`, 2026-09-26).
   From level 2 up the P2P cutoff test is fp32, so a pair within ~1e-6 of the cutoff can fall on the other side than in fp64. In 200k random particles this was one pair, showing up as 1.1e-4 in a global norm vs level 0 (5e-7 without it); the error vs the exact far field is unchanged (3.878e-4 at every level). The published H200 numbers (2026-08-22 and 2026-09-23) are level 2. Level 0 is byte-identical to the pre-fp32 engine (SASS-diffed). `pdeg=5` is a false economy at matched accuracy (needs mac 0.7, same far-field time as pdeg 7 / mac 0.8 with 2x the error). Leaf 512 measures the same as 1024. Trajectory A/B, level 3 vs fp64, over the 100-step Figure-13 run: max 0.035 particle radii, panels 99.997% pixel-identical. See `artifacts/consumer_gpu_far_field_report.md` and `artifacts/fig12_h200_f32l2_report.md`.

   `policy="cart"` (`NEMO_FAR_FIELD=widebvh-cart`) is an A/B alternative, not production: the analytic Cartesian Taylor expansion, runtime `order` 1..4, `mac` 0.33 at matched accuracy, no fp32 build (it runs the fp64 kernels; an explicit nonzero `fp32_level` is rejected). The two policies truncate different series, so **no `mac` transfers between them**, and each needs its own bucket granularity: at its 2.4x tighter `mac` the Cartesian policy carries ~11x the near pairs, and on the engine's automatic cell edge (~1024 cells regardless of N) that costs 286 ms of P2P out of a 346 ms far field at N=1M. `cart_hilbert_q()` sizes it instead. See `artifacts/cartesian_far_field_report.md`.

6. **`treecode.py` → `WarpFMM`**: the previous far field (Warp BVH, monopole + dipole, published `theta` 0.3, leaf 16). **Retired** — kept only as the base class of `WidebvhFMM` and to reproduce the published baseline, and only on explicit request (`NEMO_FAR_FIELD=warp`, `--backend warp`, `mac_calibration.py --thetas 0.3,...`). Nothing defaults to it; an unknown backend name raises. Only its `get_far_field_vel` needs the patched Warp fork (github.com/Shihab-Shahriar/warp, adds `wp.bvh_mp_query`; `docker/pack_context.sh --with-warp-fork`). `WidebvhFMM` overrides that method and Warp compiles kernels lazily, so the widebvh path runs on stock `warp-lang`.

**Current published n-body stack (2026-09-26):** pair `nbody_moments_v3_nb8lin_tr2_kinf_rc8_pc8c.pt` (4e) + diagonal `nbody_diag_v2_pc8c.pt`, both trained on the chain-fixed cache `data/multibody_v2_cache_pc8c` (4c) and run with `max_neighbors=None`, `neighbor_cutoff = pair_cutoff = switch_dist = 8` (as their sidecars say). CPU `Mob_Op_Nbody_Moments`, GPU `Mob_Nbody_Moments_Torch` (`two_suspensions_1M.py --near-op moments-v3`), harness op `M_mom_v3_nb8lin_tr2_pc8c_diag`. Every other model in `data/models/` (v2 k10/kinf_rc6/kinf_rc8, pc8, v2 pc8c, b1, b1_v2) is a superseded ablation or a reproduction baseline; `two_suspensions_1M.py --near-op moments` still loads the v2 pc8 pair + diag.

4b. **`mob_op_nbody_moments.py` → `Mob_Op_Nbody_Moments`** (extends `Mob_Op_Nbody`, CPU): the moments-based n-body correction of `moments_for_nbody.md` — per pair, band moments `(s_a, v_a, Q_a)` of the neighbourhood (8 unit-width radial bands about the pair midpoint; `src/nbody_moments.py`, TorchScript-scriptable) → 72 rotation-invariant swap-even invariants → `MultiBodyMoments` MLP (`model_archs.py`, 76→128→64→128→64→93) → 93 coefficients on 34 TT/RR + 25 TR analytic bases, so `M_ji = M_ij^T` and O(3) equivariance hold by construction (`tests/test_nbody_moments.py`). Model input row is 111 columns `[s_vec(3) | d−mean, d−2, (d−mean)², (d−mean)⁴ | s_a(8) | v_a(24) | Q_a(72)]`, built by `nbody_features.moment_features` for both training and inference. Neighbour *selection* is the baseline's (within `neighbor_cutoff` of the midpoint, top-`max_neighbors` by `d_kt·d_ks`; `max_neighbors=None` = unbounded) because the existing training rows carry ≤10 neighbours; the band weights deliberately do **not** zero beyond r_c = 8 (training rows contain neighbours farther from the midpoint that affect the label). Trained by `experiments/train_nbody_moments.py`; compared against MFS truth by `benchmarks/compare_nbody_moments.py` (warp-free; caches truth in `tmp/nbody_moments_truth/`). See `artifacts/nbody_moments_report.md`.

   **Residual-label convention (load-bearing):** the n-body residual is `Y − v2b` with the two-body term evaluated exactly as the operators do it, `X2b[:, :3] = +s_vec = source − target` (`nbody_features.two_body_velocity`). The saved `branch1_multibody_pinn.ipynb` feeds `−s_vec` in `predict_two_body_from_triplet`, which flips the RT/TR blocks (L3 is odd) — its "2-body only" 38 % / 199 % validation errors are that artefact (true values 6.8 % / 16.4 %), and a model trained on those labels is catastrophically wrong inside the operator (28 % vs 4 % at φ=0.05).

4c. **Dataset-v2 retraining + paper accuracy harness (2026-08-30, `artifacts/nbody_v2_retrain_report.md`).** Training rows now come from `data/multibody_v2/` through *the operators' own* selection: `nbody_features.select_pair_neighbours` (unordered near pairs t<s, d ≤ 6; neighbours within `neighbor_cutoff` of the pair midpoint, top-`max_neighbors` by d_kt·d_ks, index-sorted) is the single code path used by `Mob_Op_Nbody_Moments._pair_rows` and by the cache builder. Labels are full residual blocks `R = 0.5(M_ts + M_st^T) − M2b_ts` with `nbody_features.two_body_blocks` (`predict_mobility(X2b)[1]`, +s_vec, **median 5.01** = the operator's constant); both architectures are reciprocal by construction, so one unordered row per pair suffices. Pipeline: `experiments/build_nbody_v2_cache.py` (geometry + labels, 2.7 GB in `data/multibody_v2_cache/`, features built on the fly; `--stats` for coverage/asymmetry/diagonal-residual tables) → `experiments/train_nbody_v2.py --model {moments,baseline} --variant {k10_rc6,kinf_rc6,kinf_rc8} --publish` (L1 over the 36 block entries × 6π, 100 epochs ≈ 10 min on the 4060, configuration-level split `seed % 10 == 0`) → `benchmarks/paper_accuracy_v2.py` (warp-free Fig 3 / Fig 4 / clustered protocols of `accuracy_grand_M.py` with the same truth generator and seeds, truths cached in `tmp/nbody_moments_truth/` (1,280 files), CPU worker pool, `--gpu-ops` for `mfs_coarse` + the paper's GPU operator, `--part/--merge/--summary/--figures`; SLURM `slurm/paper_truth.sbatch` (GPU, array over φ) then `slurm/paper_eval.sbatch` (32 CPU workers, ~1 min per φ)). **Published models carry a `.json` sidecar with the selection they were trained on; run them with exactly that selection** (`Mob_Op_Nbody_Moments(max_neighbors=…, neighbor_cutoff=…)`): `nbody_moments_v2_k10_rc6.pt` (10, 6), `nbody_moments_v2_kinf_rc6.pt` (None, 6), `nbody_moments_v2_kinf_rc8.pt` (None, 8; best of that round), `nbody_pinn_b1_v2.pt` (K=10, r_c 6, drop-in for `Mob_Op_Nbody`). Results: v2 validation PRMSE lin/ang 15.9/37.5 % (2-body) → 8.97/22.1 (shipped b1) → 6.87/19.8 (b1 on v2) → 3.98/15.4 (moments v2, r_c 8); paper Fig 3 at N=200, φ=0.2: 15.0 % (paper n-body) → 11.2 % (b1 on v2) → **9.1 %** (moments v2 r_c 8), beating the 3-body summation everywhere. `experiments/nbody_v2_ceiling.py` shows the floor of *any* pairwise near-field correction on v2 boxes is 2.9 % (φ 0.1) / 4.1 % (φ 0.2), almost entirely the far-field term (pairs beyond the switch distance 6, RPY, never corrected), not the diagonal block. (GPU port since done: `gpu_nbody_moments.py`, see 4e.)

   **Pair-cutoff 8 ablation (2026-08-30, `artifacts/pair_cutoff_ablation.md`):** raising the corrected-pair range to d ≤ 8 attacks that far-field floor directly. `nbody_moments_v2_kinf_rc8_pc8.pt` (sidecar: None, r_c 8, **pair_cutoff 8**; superseded by the pc8c retrain below) is trained on `data/multibody_v2_cache_pc8` (`build_nbody_v2_cache.py --pair-cutoff 8 --variants kinf_rc8`, 9.88 M pairs, 3.9 GB) with the identical recipe, and **must run with `pair_cutoff=8, switch_dist=8`**: the 2b NN (trained to d = 8) is the base wherever pairs are corrected, and NN-vs-RPY differs by only ~2e-4 in the 6–8 shell, so the base swap is free. `benchmarks/paper_accuracy_v2.py` passes both from the sidecar (`SELECTION` is now (K, r_c, pair_cutoff) 3-tuples; `_selection_for` asserts the sidecar's pair_cutoff too). Results: Fig 3 φ=0.2 PRMSE 9.08 → **7.41 %** (N=200) and 11.94 → 9.89 % (N=300); Fig 4 −15…−20 % at every (N, φ), steady in N; clustered δ better everywhere; max per-particle error 27.6 → 22.9 % at φ=0.2. The exact-residual floor (`nbody_v2_ceiling.py --pair-cutoff 8`, must match the cache) moves 2.86/4.08 → 1.70/2.69 % at φ=0.1/0.2 — the correction captures roughly half the newly exposed headroom, and what remains is the d > 8 far field plus the diagonal. `benchmarks/pair_cutoff_ablation.py` renders the report and `figures/pair_cutoff_ablation_fig{3,4}.*` from the merged CSV.

   **Learned per-particle diagonal (2026-08-31, `artifacts/nbody_diag_report.md`):** `SelfBlockMoments` (`model_archs.py`) maps particle-centred band moments (`src/nbody_moments.py` `self_*` functions: 8 tent bands remapped to [2, 8], width 0.75; 48 true-scalar invariants; 33 TT + 33 RR + 24 TR bases with assembly `bot = [TRᵀ, RR]`, so the block is **symmetric by construction** — the pair `assemble_block`'s `[TR, RR]` corner is wrong here) to a 6×6 correction to M_tt, added as a third velocity term in `Mob_Op_Nbody_Moments` (`diag_nn_path`/`diag_cutoff`, `get_diag_velocity`, `v /= μ`). Selection is `nbody_features.select_particle_neighbours` (all k≠t within r_c = 8, one code path for trainer/operator/ceiling). Labels existed in the cache: `Mtt_res` symmetrised, trained by `experiments/train_diag_v2.py` (1.12 M particle rows, zero-neighbour rows kept, ~9 min on the 4060; inv_norm and 200-vs-100 epochs both nearly flat). **Published `nbody_diag_v2_pc8.pt` (now `nbody_diag_v2_pc8c.pt`, same recipe on the pc8c cache) must run with pair_cutoff = switch_dist = 8** — the labels subtract K_s over d ≤ 8, so any other switch double-counts; the ctor asserts pair_cutoff == switch_dist and the harness' `_diag_sidecar_for` asserts the sidecar. Results: val capture 53.5 % (≈ half the diagonal residual removed); ceiling A/B (`nbody_v2_ceiling.py --diag-model`, `artifacts/nbody_v2_ceiling_pc8_diag.md`) captures 60–73 % of the exact-diagonal headroom (uniform φ=0.2 e_near 2.69 → 1.82, exact 1.46; ang φ=0.25 4.49 → 1.80); paper harness op `M_mom_v2_kinf_rc8_pc8_diag` improves every uniform (N, φ) — Fig 4 φ=0.2 mean 6.07 → 5.70 %, −0.4…−0.6 at N ≤ 60 — but only ~0.1 points at Fig 3 N=200 (7.41 → 7.29): the pair model's own ~7 % residual dominates in quadrature, so the diagonal's headline value grows as the pair term improves. Clustered δ (different generator, δ ≥ 2 outside band support) is mixed. λ_min moves toward the truth (no SPD loss). GPU port done (in `gpu_nbody_moments.py`). Encoder ablations on the v2 bases (report §6–7) all left the angular TR/RT blocks at ~17 %; the cause was the `RT = TR` tying, not the encoder — superseded by 4e.

   **Chain family → pc8c models (2026-09-05, `reproduction.md` §Figure 6):** the pc8 stack overpredicted Durlofsky's 15-sphere chain drag (Fig 6) by up to 1.49 %: an unconstrained error cliff on the exactly collinear manifold, which the box families never sample. Fixed in the data only: a `chain` family (7a) in `data/multibody_v2/`, cache `data/multibody_v2_cache_pc8c` (62,448 configs, 9.98 M unordered pairs d ≤ 8, ~1 % of them chain), then the **default recipe with no `--family-weights`** (runs `experiments/runs_v2/{mom_kinf_rc8_pc8c,diag_pc8c}_final`) → `nbody_moments_v2_kinf_rc8_pc8c.pt` + `nbody_diag_v2_pc8c.pt`; v3 (4e) was trained on the same cache. Of the 22,528 chain configs generated, only a quarter is kept (6,400: shard 0 of P = 12 and 16 at every jitter), the user's call to keep the paper story to one dataset with no weighting knob; the other 16,128 are archived in `data/multibody_v2_extra_chain/` (gitignored, invisible to the cache builder — moving them back changes the training data). Effect vs pc8 (pair + diag): Fig 6 max error 1.49 → 0.24 % (the digitised reference itself is 0.16 %), Fig 4 unchanged, dense Fig 3 cells +0.1…0.4 points (N = 200 φ = 0.2 7.29 → 7.51 %, N = 300 9.86 → 10.29). A `--family-weights 1 1 1 0.25` run on the full chain set (`_w025`, slightly better on Fig 3) was published for a few hours and replaced; `reproduction.md`'s Fig 6 history and `figures/fig6_chain_drag.csv` (rendered before the final publish) still describe it.

   **Ring-array validation (2026-09-07, `artifacts/ring_array_report.md`):** Jordan & Lockerby's ring of P spheres (JCP 520 (2025) 113487, exact Tables B.9/B.10 in `data/ring_array_jordan2025.csv`) is the multi-sphere NeMO-vs-RPY showcase: `figures/fig_ring_array.py` fits the five symmetry coefficients (M_∥, M_⊥, M_∘, N_tz, N_zt) from `apply` and `figures/fig_ring_sedimentation.py` integrates the ring falling parallel to its plane (truth = BatchedMFS fine/triton32, 0.1 s/solve, reproduces the tables to 5 digits at t = 0). Near field S ≤ 2 R: M_∥ error 3.4 → 0.28 %, deformation coefficient 57 → 18 % of scale, N_zt 39 → 19 % (RPY → NeMO); RPY pinches the ring in half the truth's fall distance at P = 4, 5, NeMO within 4–13 %. Known limits: NeMO's N_tz is worse than RPY for S ≥ 0.5 R and M_∘ is under-predicted at S = 1–2 R for P ≥ 7 (wrong zero crossing). Runs stop at surface gap 0.1 R (training range / MFS resolution).

   **HIGNN baseline (2026-09-07, `artifacts/hignn_comparison_report.md`):** the Pan-group HIGNN toolkit (a checkout at `HIGNN_ROOT`; AGPL, referenced not vendored) re-evaluated in plain torch by `src/hignn_ops.py` as harness ops `HIGNN_2b` (their C++/H-matrix engine's pairwise kernel, dense = its best case) and `HIGNN_full` (+ their Python 3-body/self terms, cutoff 5.0, verified optimal). HIGNN is translational-only and torque-free, so it is scored on the gravity truths (exp `fig4g`) with the translational columns `prmse_lin` / `prmse_fluct` / `err_mean_pct` / `max_rel_lin` (added to `compute_error_stats`; `rel_rmse`/`prmse_ang` are meaningless for HIGNN rows). N = 200: NeMO 0.66/1.08/1.73/2.43 % vs HIGNN full 1.01/1.93/3.74/5.47 % vs HIGNN engine 1.23/2.20/4.22/6.24 % (φ 0.025..0.15); HIGNN engine ≈ RPY + 5 %, HIGNN full ≈ NeMO 2-body; the gap closes to 1.03–1.07× at N ≥ 10 000 because the large-N gravity error is the shared pairwise-far-field (back-flow) bias. Figures `figures/fig_hignn_compare_{phi,N}[_fluct].*`, tests `tests/test_hignn_ops.py`, their engine cross-checked in their CPU Singularity image (`slurm/hignn_cpp_check.sbatch`, `benchmarks/hignn_cpp_check.py`).

   **Stokesian Dynamics baseline (2026-09-08, `artifacts/sd_comparison_report.md`):** Townsend's Stokesian Dynamics
   (github.com/Pecnut/stokesian-dynamics, a checkout at `SD_ROOT`; referenced, never edited) runs as harness ops `SD` (FTS far field + pairwise lubrication, as shipped) and `SD_Minf`
   (its far field alone) through `src/sd_ops.py::SDMob` — one static FTE mobility solve per apply, non-periodic, numba on,
   ~2–4 s at N = 200 (11N dense inverse; O(N³)). Its two import-time habits are worked around there (`settings.py` parses
   `sys.argv` and sys.exit()s on non-int args; it also turns numba off at decoration time). Tests `tests/test_sd_ops.py`
   (exact vs the package's own two-sphere path). On the Fig-3-style gravity protocol at N = 200 (`figures/fig_sd_compare.py`,
   exp `fig4g`, all 8 φ — the 4 cells the cluster never solved were filled by `benchmarks/sd_gravity_truth.py`, batched
   fine-cloud MFS validated to 3e-5 against the broms truths) **SD as shipped is 3.3–8.1 × less accurate than NeMO**
   (translational PRMSE 2.2 → 25.2 % vs 0.66 → 3.19 %): the pairwise-additive R2B,exact − R2B,∞ term over-resists the
   collective settling of a finite cloud (Ichiki, JFM 452, 2002; `benchmarks/sd_diagnostics.py` — exact for pairs, error
   grows monotonically with the number of corrected pairs). SD's far field alone is the most accurate operator on this
   protocol (0.08 → 0.77 %). Same ordering for the translational block of the random-wrench protocol (φ = 0.1: far field
   2.8, NeMO 4.3, SD 8.7, RPY 12.2 % PRMSE); SD's lubrication does make its rotational velocities the best there.

4e. **Moments v3: split TR/RT corners, linear bases (2026-09-13/14, branch `moments-v3-main`, `artifacts/nbody_v3_report.md`).**
   The bands are fixed: 8 unit-width overlapping tents (centres 0.5..7.5), so rows are always 111 columns and the same for
   every layout (`nbm.moment_features` / `nf.moment_features`). The overlap is what makes the features, and so the
   velocities, continuous as a neighbour crosses a band edge; it predates the chain fix, which was data-only (the `chain`
   family). `src/nbody_moments.py` parametrises only the bases and invariants: `bases_v3(z, v, Q, use_quadratic, has_tr2)` and
   `assemble_block_v3(c, tt, tr1, tr2)` with `TR = T1 + T2`, `RT = T1 − T2`; the v2-named functions are wrappers.
   `MultiBodyMoments(bases=, invariants=)` (defaults = v2); `nbody_moments.layout_of_model` reads any model incl. TorchScript
   (legacy v3 exports carry a `band_knots` buffer, asserted to be the standard bands). A `.wt` needs its layout (sidecar
   `.json` next to it, written per run as `<out>/model.json`, or `Mob_Op_Nbody_Moments(nbody_layout=...)`), a `.pt` is
   self-describing. Trainer: `--bases {v2,linear,linear_tr2,v2_tr2} --invariants {full,reduced}` (non-default layouts need
   `--publish-name`; the harness' `_assert_layout` checks the sidecar). GPU path (`gpu_nbody_moments.py`) takes v2 and v3 `linear_tr2` with full invariants (asserted); a `.wt` needs its sidecar (next to it or in `data/models/`) because v2/v3 weights have identical shapes.
   **Removed 2026-09-24** (evaluated, not adopted; see the report): configurable band knots (`--bands`, the 5-knot model
   `nbody_moments_v3_nb5lin_tr2_*` and its harness ops) and learned Bessel radial bands (`--radial bessel`, `RadialBands`).
   **Why (300k pc8c validation rows of the published v2 model):** the label's TR and RT blocks differ by 80 % (`‖TR−RT‖/‖TR‖`),
   so writing the same matrix in both corners (`RT = TR`, inherited from Eq. 17) leaves a **40 % floor** on the TR/RT residual
   that the model sat on (43/45 %) — the reason every encoder ablation left the angular blocks at ~17 %. Reciprocity only
   needs `TR_ji = RT_ijᵀ`, which admits a second basis class with `T(−z) = −T(z)ᵀ` placed as `TR = +T, RT = −T`: `E(v_a)`,
   `(z·v_a)E(z)`, `[E(z), Q_a]` per band (`moments_for_nbody.md` §5.4). The 24 quadratic bases are free to drop (an 8-band
   linear refit equals the v2 control within 0.01; a second training seed shows the noise floor is ≤ 0.01).
   **Adopted v3 = the v2 unit bands + linear bases + class-2 corners** (`--bases linear_tr2`, default bands): v2's size
   (76 inputs, 93 coefficients), the quadratic bases replaced by the corner bases. **Published
   `nbody_moments_v3_nb8lin_tr2_kinf_rc8_pc8c.pt`** (sidecar None/8/8; run `experiments/runs_v2/v3_nb8lin_tr2`), harness ops
   `M_mom_v3_nb8lin_tr2_pc8c[_diag]` (same `nbody_diag_v2_pc8c.pt`): validation lin/ang 3.87/15.53 → 3.15/8.29 %, TR/RT
   16.7/15.9 → 8.4/8.0; Fig 3 φ=0.2 N=200 total 7.51 → 7.01 % (ang 6.30 → 4.69), N=300 10.29 → 9.72 (ang 6.09 → 5.02); Fig 4
   φ=0.2 mean 5.77 → 5.17 (ang 5.36 → 3.59); gravity φ=0.2 lin 3.19 → 2.82, settling bias 1.11 → 0.79; Fig 7 Ω error at S = 2.1 +36 → −2 % (`reproduction.md`).
   **GPU port done (2026-09-22):** `pair_assemble_apply_v3_kernel` (TR = T1 + T2, RT = T1 − T2, in registers); parity vs the
   CPU op ~1e-6 (`benchmarks/compare_gpu_moments.py --model v3`, compiled and eager); 1M two-drop pair stage 2338 → 2363 ms on
   the 4060 (assemble 117 → 128 ms); run it with `two_suspensions_1M.py --near-op moments-v3` (+ pc8c diag;
   `profile_moments_stage.py 0 v3`). `two_suspensions_1M.py --near-op moments` still loads v2 pc8 (the Fig 11/12 scripts offer only `baseline`, `moments-v3`, `moments-v3-pc6`).
   **Evaluated and not adopted (report §6–7; code removed 2026-09-24):** truncating the bands (5 knots 0.5..4.5 + saturating tail, 60 coefficients —
   the first candidate `nbody_moments_v3_nb5lin_tr2_kinf_rc8_pc8c.pt`) costs 0.4–0.6 Fig-3 points
   although post-hoc zeroing of bands 5–8 said ≤ 0.17 each; coarser equal-width bands are worse than v2 in translation
   (leverage sits at r < 2 from the midpoint); learned radial bands (DimeNet Bessel basis × cutoff envelope through a small
   MLP) gain 0.27/0.48 Fig-3 points over the adopted model and were judged not worth a new mechanism.
   A 64-wide MLP and the reduced 5-invariant set each cost ~0.05 (on the 5-knot layout, untested in combination).
   Pre-existing test failures unrelated to this: `test_v2_split_is_configuration_level` (0.1201 > 0.12 with the chain shards),
   `test_oracle_residual_reproduces_grand_M` (the first shards are now `chain`), `test_harness_cases_and_metric` (Fig 4 has 1536 cells).

7a. **`mfs_batched.py` → `BatchedMFS`**: the **batched multi-right-hand-side MFS solver** (Triton + torch; the reference solvers below are single-RHS Gauss–Seidel). Solves R right-hand sides of n_sys equal-size configurations at once in boundary-velocity space (`T W = b`, `T = I + Ŵ K_f`) with batched restarted GMRES; `solve_mobility_matrix(positions)` returns the full grand mobility matrix M (6P×6P) from all 6P unit force/torque columns. Backends: `torch64` (exact fp64, one cuBLAS dgemm per target over all partners — **production on the H200**, fp64 at half the fp32 rate there; 50–70× slower on GeForce) and `triton32` (fp32 `tl.dot` kernel in `mfs_batched_kernels.py`, 4–5 TFLOPS on the 4060, ~1e-5 relative accuracy: the MFS strengths cancel by ~1e3, so fp32 storage of the strengths alone costs 6e-6). Two things are load-bearing: **the pseudo-inverse is always applied in fp64** (an fp32 GEMM carries a 2e-3 net-force error into the labels), and **convergence is judged on the velocities** (`tol_v`, per column, checked at every Krylov step), never on strengths or on the W residual — the fp32 kernel's W residual stalls at 1e-4 while its velocities are converged. Defaults `tol_v` = 1e-7 (torch64) / 1e-5 (triton32); 17–26 Krylov steps. Validated by `tests/test_mfs_batched.py` against `src/mfs.py` (2e-12 in fp64), the old dataset rows and the cached Xfine truths; `benchmarks/bench_mfs_batched.py` measures throughput. `tf32x3`/`tf32` `tl.dot` crash Triton 3.2 on register operands; fp64 `tl.dot` is unsupported.

   **Dataset v2 generator** `src/create_dataset_multibody_v2.py`: full M per configuration for `uniform` (RSA box, gap ≥ 0.1), `grown` (cluster growth at gap δ), `lattice` (jittered cubic drop patch) and `chain` (quasi-1D line, gaps U[2.1, 8], transverse jitter σ ∈ {0, 0.02, 0.05, 0.1, 0.3}; not in `--plan default`, generate with `--family chain`) families, sharded `.npz` under `data/multibody_v2/` (gitignored) with per-worker manifests, deterministic seeds, resume, `--time-budget`, `--worker/--num-workers`. Multi-GPU recipe, one worker per GPU: `CUDA_VISIBLE_DEVICES=k python src/create_dataset_multibody_v2.py --plan default --acc fine --backend torch64 --mem-budget-gb 40 --time-budget 28800 --num-workers 4 --worker k`. Consumer: `nbody_features.iter_multibody_v2_configs` + `pairs_from_config` (lazy, for full passes), `load_multibody_v2` (eager ordered-pair table with `M_ts`, `M_st`, `M_tt`, all-neighbour positions — ~2.4 KB per pair, subsets only) and `pair_rows_from_M` (old-convention rows with random forces, `Y = M_ts [F;T]_s`, `+s_vec`). **Generated 2026-08-30** (SLURM array 16539317, `slurm/gen_multibody_v2.sbatch`, 4 × H200 ≈ 4.5 GPU-h): 56,048 configurations of the three box families, 12.7 M ordered pairs within 6 radii, 4.4 GB, in `data/multibody_v2/`; see `artifacts/mfs_batched_report.md` §5. **Chain added 2026-09-05** (triton32 on an RTX 4060, labels within 1.4e-6 of fp64): 6,400 configs kept in `data/multibody_v2/chain/` (P = 12, 16; 16,128 more archived, see 4c), so the directory now holds 62,448 configurations, 12.8 M ordered pairs within 6 radii, 4.5 GB.

7. **`mfs.py` → `MobOpMFS`**: Reference MFS solver (ground truth). Iterative multi-particle solver using Oseen tensor. `triton_mfs.py` → `MobMFSTriton` is the Triton-accelerated variant.

### Two-Body NN Decomposition

The two-body model predicts 5 scalar coefficients that are assembled into a 6x6 mobility kernel using geometric tensor bases: `L1` (parallel/outer product), `L2` (perpendicular), `L3` (cross-product/angular coupling). The kernel has TT (translation-translation), RT (rotation-translation), and RR blocks. M_s is symmetric; M_t is not.

### Neighbor Search

`hashgrid_neighbors.py` wraps Warp's hash grid for GPU neighbor queries. Used by GPU mobility operators to find pairs within cutoff distance, returning COO-format edge indices.

### Reference Data Generation

`benchmarks/cluster.py` generates test configurations (random sphere clusters, uniform packings) and computes ground-truth velocities via MFS for validation.

## Running

```bash
# Accuracy, warp-free (CPU operators vs cached MFS truth)
export PYTHONPATH=$PWD TORCH_COMPILE_DISABLE=1
python3 benchmarks/paper_accuracy_v2.py --help
# Accuracy, GPU operators (warp + widebvh, here via Docker; env vars listed in docker/run_local.sh pass through)
TORCH_COMPILE_DISABLE=1 bash docker/run_local.sh python benchmarks/accuracy_grand_M.py

# Performance (H200-class GPU): sbatch slurm/h200_v3/{fig11,fig12,twodrop}.sbatch; by hand
# (fp32 level 3 by default; NEMO_FAR_FP32_LEVEL=0 for an fp64 A/B):
python benchmarks/figure12_grand_M.py --near-op moments-v3
python benchmarks/two_suspensions_1M.py --near-op moments-v3
# (benchmarks/performance_grand_M.py run directly is only a max-size probe of the baseline operator)

# Run a specific mobility operator standalone (most src files have __main__ blocks)
python src/mob_op_nbody.py
python src/mfs.py
```

## Dependencies

Core: PyTorch, Warp (NVIDIA, stock `warp-lang`: the hash-grid neighbour search and the fused near-field kernels in `gpu_mob_2b.py` / `gpu_nbody_moments.py`; the far-field BVH is widebvh's own, and the patched Warp fork is needed only for the `WarpFMM` A/B baseline), widebvh (`libwidebvh_nemo*.so`, production far field), NumPy, SciPy. Optional: Triton (for MFS GPU kernels), tiny-cuda-nn. Requires CUDA GPU for GPU operators.

## Author's setup

Applies only on the author's machines: Linux user `shihab` (the laptop) or `khanmd` (MSU HPCC). Anyone else can ignore this section — its paths and hosts don't exist elsewhere.

- **Where to run.** Default to the **laptop** (RTX 4060 Laptop, 8 GB): all accuracy tests and harnesses, MFS truth up to N ≈ 300, training, pytest, figures. Use the **cluster** (MSU HPCC, H200) via `sbatch` for:
  - performance/timing runs;
  - anything that would take too long on the laptop: large-N truths, dataset generation, multi-hour sweeps.
- **Laptop, native.** `/home/shihab/miniconda3/bin/python3` (torch 2.6+cu124, triton 3.2, pytest; **no warp**) runs everything in the warp-free list.
  - `export PYTHONPATH=$PWD` is mandatory. The login shell sets `PYTHONPATH=/home/shihab/repo:`, and `~/repo/src` (an older checkout) otherwise shadows this repo's `src` in any script that does not reorder `sys.path` itself.
  - `~/warp_env.sh` does not exist here, `/tmp` is writable, and the cluster rules below do not apply.
- **Laptop, Docker.** Everything that needs warp or widebvh runs through `bash docker/run_local.sh <cmd>`, whose defaults are this laptop:
  - image `sskhan39/nemo:2.0`;
  - `WIDEBVH_SRC=~/envs/nemo-ctx/widebvh` (git-tracked there, commit 03efcdb);
  - build dir `build-4060`: sm_89, pdeg 5 and 7 × fp32 levels 0–3, no Cartesian library.
  These cannot run natively: no warp in the base conda, and host nvcc 13.1 vs driver 580.
- **Sibling folders.**
  - `~/nemo`: the paper (next bullet).
  - `~/throwaway/nemo-src`: the core-only public release copy of this code (git). Mirror core changes there only on request.
  - `~/throwaway/hignn`, `~/throwaway/stokesian-dynamics`: the `HIGNN_ROOT` / `SD_ROOT` defaults.
- **Paper: `/home/shihab/nemo`** (git). LaTeX entry `nemo.tex`, prose in `sections/0N-*.tex`, figures in `figs/`, and its own `AGENTS.md`. Build with `latexmk -pdf -interaction=nonstopmode -halt-on-error nemo.tex`. Figures made here are periodically copied there so the paper can be updated.
  - **Sync only when asked.** After regenerating a paper figure, say which `~/nemo/figs` files it makes stale (`cmp`).
  - **Include names are the source of truth** (`grep includegraphics ~/nemo/sections/*.tex`). `reproduction.md` says which repo output is each figure's paper version and under what name. The names can differ: `figures/fig4_n_acc_random_n1000_nemo_linx.pdf` → `figs/fig4_n_acc.pdf`.
  - **This repo may hold many variants of a figure; the paper holds exactly one.**
    - Copy only the included variant, PDF only, under its include name. Never copy the other variants.
    - When the paper switches to another variant, remove the old file from `figs/` (`git rm` if tracked).
    - Leave older, unrelated files in `figs/` alone.
  - **Never edit the `.tex`.** List the caption and prose numbers a new figure makes stale (mostly in `sections/03-experiments.tex`) and leave the edit to the author.
- **Cluster.** `ssh h200` (ProxyJump msu, user khanmd).
  - The home checkout `/mnt/home/khanmd/pinn-stokes` is on branch `performance` and dirty: only add files there, and test other code from a scratch clone.
  - No sudo; do not create `/tmp` directories.
  - Every shell and sbatch script starts with `source ~/warp_env.sh` (modules + conda; over ssh, `module` needs `bash -lc`).
  - `warp_env.sh` pins the home checkout on `PYTHONPATH`. A script run from any other checkout must `export PYTHONPATH="$(pwd -P):${PYTHONPATH:-}"` after `cd`, or it silently imports the home code.
  - Home quota is tight. Bulk data (caches, datasets, large outputs) goes to `/mnt/scratch/khanmd/`, which is purged after ~30 days; building the pc8 cache in home blew the quota once. Array tasks that die instantly with ExitCode 0:53 and no log mean home is over quota.
  - The box-family shards of `data/multibody_v2/` and the truth cache `tmp/nbody_moments_truth/` exist here and on the laptop; the chain shards were generated on the laptop.
  - Needs the cluster outright:
    - H200 timing;
    - the widebvh Broms truth: `benchmarks/broms_truth.py`, PETSc CUDA build, `BROMS_SPHERE_MFS` default `/mnt/ffs24/home/khanmd/programs/widebvh/build-rel/sphere_mfs`.
- **sbatch resources** (as in `slurm/*.sbatch`): `--nodes=1 --ntasks=1 --constraint=amd24 --account=hmakmm --gres=gpu:h200:1 --cpus-per-task=32 --mem=64G`, `--output=slurm/logs/%x_%j.out`, and a generous `--time` (overestimating is fine; most scripts use `2:59:00`). **`--gres=gpu:h200:1` is what pins the H200** (neh-*/nfh-*): `amd24` alone also matches the L40S nodes (nel-*), and a plain `gpu:1` job was once scheduled onto one. Only `slurm/h200_v3/` is pinned so far (its `env.sh` also refuses to run on anything but an H200); the older `slurm/*.sbatch` still ask for `gpu:1`. `PENDING (BadConstraints)` right after submission clears on the next scheduling pass. A pending job can be re-pinned in place with `scontrol update jobid=<id> TresPerNode=gres/gpu:h200:1`.
- **widebvh on the cluster.**
  - `/mnt/ffs24/home/khanmd/programs/widebvh-f32/build-nemo` is the code's default `WIDEBVH_BUILD_DIR`: the 03efcdb tree, sm_90, levels 0–3 + cart, built by `slurm/h200_v3/build_widebvh.sbatch`, and the build `slurm/h200_v3/env.sh` points at.
  - The older `programs/widebvh/build-nemo` (9f963d2, a dirty pre-fp32 tree) is fp64-only and cannot serve the default level.
  - The 2026-09-23 H200 timings used a scratch copy of the 03efcdb tree.
  - `mac_calibration.py`'s `WIDEBVH_DISTROS` is `programs/widebvh/distros`.
