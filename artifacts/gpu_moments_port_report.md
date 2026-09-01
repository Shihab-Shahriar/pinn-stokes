# GPU port of the moments n-body stack + 1M two-drop performance A/B (RTX 4060)

**Date:** 2026-08-31/09-01. **Hardware:** RTX 4060 Laptop (8 GB, sm_89), docker
`sskhan39/nemo:2.0` (torch 2.8.0+cu128, warp 1.12), widebvh build-4060.

## 1. What was ported

The dataset-v2 accuracy stack — the moments pair correction
`nbody_moments_v2_kinf_rc8_pc8` (K = ∞, r_c = 8, pair_cutoff = 8) and the learned
per-particle diagonal `nbody_diag_v2_pc8`, both locked to switch_dist = 8 — now
runs on GPU: `src/gpu_nbody_moments.py` → `Mob_Nbody_Moments_Torch(NNMobTorch)`.
The far-field treecode's `nearCutoff` moves to 8 with it (the 2b NN, trained to
d = 8, is the base through the 6–8 shell; the widebvh far field covers exactly
r ≥ 8). `benchmarks/two_suspensions_1M.py --near-op moments` selects the stack.

Design (why it is fast):

* **One model call per unordered pair.** Reciprocity K_st = K_ts^T holds by
  construction (swap-even invariants; pinned by tests/test_nbody_moments.py), so
  each pair is evaluated once and applied twice: v_t += K F_s, v_s += K^T F_t.
  The CPU operator evaluates both directions.
* **No padded neighbour tensors.** With K = ∞ the band moments are accumulated
  as sums; a fused Warp kernel walks a hash grid (cell = r_c/2) at each pair
  *midpoint* (midpoint neighbourhoods are not subsets of endpoint lists) and
  accumulates the 8×10 band-moment table per pair in one traversal.
* **Tensor algebra in registers.** The 76 invariants and the 34+34+25-basis
  block assembly + force products are two more Warp kernels (one read/write per
  pair); only the MLP GEMMs (76→128→64→128→64→93) stay in torch.
* **Diagonal reuses the near edge list.** diag_cutoff == switch_dist means the
  particle neighbourhood is exactly the ordered near-pair list; self band
  moments come from one scatter pass over edges that already exist.
* Same load-bearing invariants as the other GPU paths: chunk loops in Python
  outside `torch.compile`, chunk dims marked dynamic.

Matched CPU semantics: zero-neighbour pairs get no pair correction; every
particle gets the diagonal correction; pair term not divided by viscosity
(labels at μ = 1), diagonal divided by μ.

## 2. Correctness

`benchmarks/compare_gpu_moments.py` (docker) pins the GPU operator against the
CPU reference `Mob_Op_Nbody_Moments` with the published models on random RSA
boxes (N = 200–300, φ = 0.1/0.15/0.2):

| check | worst rel. L2 |
|---|---|
| `.wt` weights vs published TorchScript `.pt` (pair + diag) | 6e-9 max-abs |
| pair moments correction (torch scatter backend) | ~1.2e-6 |
| pair moments correction (fused warp backend) | ~1.2e-6 |
| diagonal correction | ~2.1e-6 |
| full apply (self + 2b + corrections + dense RPY far) | ~4.5e-7 |

Eager and compiled modes both pass (worst 2.1e-6 overall) — fp32-vs-fp64
feature noise, two orders below the model's own ~4 % residual.

Found along the way: the CPU reference never scales its far-field RPY by
viscosity (`grpy_tensors.mu` has no viscosity argument and `NNMob.get_two_vel`
applies the blocks raw). Invisible at the μ = 1 every benchmark and dataset
uses; the GPU path multiplies by 1/μ. Full-apply parity is therefore checked at
μ = 1; the learned components are also pinned at μ = 1.5.

## 3. Performance: 1M two-drop sedimentation (50 steps, fp32 level 3 far field)

`benchmarks/two_suspensions_1M.py`, N = 1,047,968, gravity, dt = 0.01, means
over 50 timed steps (per-step CSVs in `data/twodrop_1M_4060_*.csv`):

| | baseline (switch 6, old n-body) | moments first pass (torch scatter) | moments fused (final) |
|---|---|---|---|
| step wall (s) | **2.50** | 10.09 | **5.07** |
| far field (ms) | 389 | 380 | 376 |
| neighbour search (ms) | 12 | 19 | 19 |
| self + 2-body (ms) | 627 | 1518 | 1492 |
| n-body corrections (ms) | 1475 | 8172 | 3175 |
| peak torch alloc (MB) | 2580 | 3791 | 3208 |

Scale of the near field at switch 8: 51.8M ordered pairs (2.37× the baseline's
21.9M), 27.0M unordered corrected pairs, ~1.26B pair-midpoint neighbour edges
per step at φ = 0.1.

### Stage anatomy of the pair-moments term (27.0M pairs)

| stage | first pass | fused |
|---|---|---|
| midpoint search + edge list + scatter | ~2.5 s | — |
| warp accumulate (fused single pass) | — | ~1.7 s |
| invariants + bases + assembly (torch, (P,8,3,3) einsums) | ~5.25 s | — |
| warp invariants kernel | — | 0.20 s |
| MLP GEMMs (torch) | — | 0.95 s |
| warp assemble+apply kernel | — | 0.11 s |
| diag stage (scatter + model) | 0.39 s | 0.37 s |

### Kernel lessons (measured, 4060)

* The two-pass edge-list path (count + fill traversals, ~100 GB of
  `index_add_` atomics/step) and the pure-torch finish (invariants/bases as
  (P, 8, 3, 3) einsums, hundreds of GB of intermediates) are both several times
  slower than fused Warp kernels doing the same math in registers.
* A register-resident band accumulator with a 7-way branch is **~5× slower**
  than the dynamically-indexed 8×10 local-memory array (7.8 s vs 1.7 s): warp
  divergence on every accepted neighbour beats the L1-backed local array's
  cost. The local array stays.

## 4. Reading

**The accuracy stack costs 2.0× the published operator on this GPU: 5.07 vs
2.50 s/step.** The cost decomposes cleanly into physics and implementation:

* Moving the near/far switch from 6 to 8 alone is ~0.9 s: the 2-body path
  scales linearly with its 2.37× pair count (627 → 1492 ms), and the neighbour
  search grows with it. This is the price of correcting the 6–8 shell with the
  NN instead of RPY, independent of the moments model.
* The learned corrections themselves cost 3.18 s vs the old K=10 n-body's
  1.48 s — but per unit of work they are far cheaper: the old path touches
  ≤10 neighbours of 21.9M ordered pairs (~0.2B pair-neighbour interactions);
  the moments path integrates the full r_c = 8 midpoint neighbourhood of 27M
  unordered pairs (~1.26B interactions) *and* runs a 6× wider MLP, for ~2.2×
  the time.
* The far field is unchanged (376 ms) — moving its nearCutoff to 8 costs
  nothing measurable at mac 0.8.
* VRAM headroom is comfortable: 3.2 GB torch alloc + ~1.3 GB widebvh on the
  8 GB card.

Against the naive torch port (10.09 s/step), the fused Warp kernels are a 2.0×
end-to-end win; the pair-moments term itself went 7.8 → 2.7 s.

Remaining optimization levers, in measured order: the 2-body NN pair path
(~1.5 s at 52M ordered pairs — shared with the baseline, whose 627 ms at 21.9M
pairs scales linearly; fusing its K assembly + bmm into a Warp kernel would cut
most of the non-GEMM traffic), the warp accumulate traversal (~1.7 s), and the
GEMM-bound MLP. The far field is a non-issue either way (0.38 s, fp32 level 3).

## 5. Files

* `src/gpu_nbody_moments.py` — operator + kernels (`moments_backend="warp"`
  production, `"torch"` reference).
* `src/hashgrid_neighbors.py` — neighbour counts uint8 → int32 (radius-8 lists
  brush 255), int32-overflow assert on total pairs.
* `src/gpu_mob_2b.py` — `get_neighbor_pairs` accepts switch 8.
* `benchmarks/two_suspensions_1M.py` — `--near-op {baseline,moments}`; the
  far-field cutoff follows the operator.
* `benchmarks/compare_gpu_moments.py` — CPU↔GPU parity (both backends).
* `benchmarks/profile_moments_stage.py` — stage-level timing at 1M.
* CSVs: `data/twodrop_1M_4060_{baseline,moments,moments_fused}_f32l3.csv`.
