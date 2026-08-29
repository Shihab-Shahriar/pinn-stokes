# Figure 13 on a consumer GPU: the 1M two-drop sedimentation on an 8 GB RTX 4060 laptop, and an fp32 fast path for the widebvh far field

Date: 2026-08-18. GPU: NVIDIA GeForce RTX 4060 Laptop GPU (AD107, sm_89, 24 SMs,
8188 MiB, driver 580.142). Software: the `sskhan39/nemo:1.0` image (torch 2.8.0+cu128,
warp 1.12.0, nvcc 12.8) with this repo and the widebvh source bind-mounted
(`docker/run_local.sh`); widebvh rebuilt for sm_89 only. Repo commit `ac78eb4` +
working tree; widebvh `9f963d2-dirty` as packed on 2026-08-16 + the changes in
`artifacts/widebvh_fp32_far_field.patch`.

## 0. Summary

| | fp64 production (before) | **fp32 level 3** (after; PDEG 7, mac 0.8, leaf 1024 as production) | change |
|---|---|---|---|
| **50-step timing run** (`benchmarks/two_suspensions_1M.py`, N = 1,047,968) | **537.7 s** | **126.0 s** | **4.27x** |
| mean step (wall) | 10.75 s | 2.52 s | 4.27x |
| far field per step | 8.68 s (81 % of step) | 0.39 s (15 %) | **22x** |
| near field per step | 2.07 s | 2.13 s | unchanged (noise) |
| far-field error `rel_total`, t=0 cloud, gravity / random | 1.133e-6 / 3.071e-4 | 1.138e-6 / 3.071e-4 | unchanged |
| same, deformed t=1.0 cloud, gravity / random | 8.72e-6 / 2.81e-4 | 8.73e-6 / 2.81e-4 | unchanged |
| far-field symmetry `rel_asym` (Hutchinson K=8, far field only), t=0 / t=1.0 | 6.03e-4 / 5.95e-4 | 6.03e-4 / 5.95e-4 | unchanged |
| peak VRAM, process total (`NEMO_DEVICE_MEM=1`) | 3.73 GB | 3.72 GB | fits 8 GB, 4.3 GB headroom |
| 100-step Figure-13 run, trajectory divergence at t=1.0 | reference | RMS 4.2e-4, max 0.035 particle radii | panels 99.997 % pixel-identical |

(`--max-leaf 512` measures the same 126.1 s / 2.52 s -- leaf is a wash on this card;
the recommendation stays on the production tree so that the only difference to the
H200 configuration is the arithmetic precision.)

The far field of the production treecode (widebvh BaryStokes, PDEG 7, mac 0.8) is fp64
in both hot inner loops. On an H200 (fp64 at 1/2 the fp32 rate) that is invisible; on
a GeForce card (fp64 at 1/64) it is the whole step: 8.7 of 10.75 s here, and it is why
the A4500 measured 7.1 s/step in `fig13_mac_report.md`. An opt-in compile-time fp32
ladder in widebvh (`WIDEBVH_FP32_LEVEL` 1..3, level 0 = the unchanged production
kernels, verified byte-identical) takes the far field from 8.05 s to 0.35 s at the same
interaction lists, with no measurable change in accuracy or symmetry. `pdeg = 5` is
still a false economy at matched accuracy (sec 5). The remaining step is the learned
near field (2.1 s), which this work did not touch.

Everything below was measured on this laptop; the H200 numbers quoted for context come
from `artifacts/widebvh_far_field_report.md` / `fig13_mac_report.md`.

## 1. Environment, and the 8 GB fit

Native execution is not an option on this box (no warp in the base conda; the host
nvcc is 13.1 against a 580 driver, which is CUDA 13.0), so everything runs in the
image that was built for the A4500/5090 measurements, with three bind mounts:

    bash docker/run_local.sh python benchmarks/two_suspensions_1M.py --fp32-level 3

- `/workspace/pinn-stokes` <- this repo (working tree, logs and figures land on the host);
- `/opt/widebvh-src` <- `~/envs/nemo-ctx/widebvh` (the only complete local widebvh
  tree, incl. the untracked `src/nemo_capi.cu`; now a git repo so the change is a diff),
  built into `build-4060` (`WIDEBVH_BUILD_DIR`) for `-DCMAKE_CUDA_ARCHITECTURES=89`
  in ~1 min per library at `-j4 -t1`;
- `/workspace/.cache` <- `~/envs/nemo-cache`, the inductor/Triton/Warp caches. Without
  it every run pays the 1M-shape `max-autotune` again (minutes on this CPU).

**Chunking.** The two-body NN pair path was already chunked in the working tree
(`src/gpu_mob_2b.py:_two_body_velocity`, `DEFAULT_TWO_BODY_CHUNK = 4M`; n-body
`pair_chunk_size = 2M`) -- the uncommitted change described in `vram_scaling_report.md`
that took the H200 footprint from 15.4 to 4.07 GiB. It holds here: with those defaults
and the wrapper's automatic pair budget (1.0 GB on an 8 GB card, i.e. a 26.8M-pair cap
against the 7.3M pairs this case emits) the run peaks at **2.58 GB torch-allocated /
3.54 GB reserved / 3.73 GB process total** (`artifacts/logs/4060/memcheck_fp64.log`),
identical to the fp32 build (3.72 GB). Nothing had to be lowered; the chunk sizes are
now CLI knobs (`--two-body-chunk`, `--pair-chunk`) in case a smaller card needs them.
The floor is `_per_particle_topk` (~1.1 GB of (N,129) tables + ~1.8 GB of transients),
which must not be chunked (symmetry, see CLAUDE.md).

## 2. Baseline (fp64, PDEG 7, mac 0.8, leaf 1024) on the 4060

`artifacts/logs/4060/baseline_fp64_p7_mac0.8_leaf1024.{log,csv}`, 5 warmup + 50 timed
steps, benchmark mode:

| | mean | median | min | max |
|---|---|---|---|---|
| step wall (s) | 10.75 | 10.78 | 10.18 | 11.08 |
| far field (ms) | 8685 | 8719 | 8066 | 9042 |
| near field (ms) | 2066 | 2051 | 1966 | 2209 |
| of which self+2-body / n-body / neighbour search (ms) | 599 / 1459 / 7.8 | | | |

`Total simulation time: 537.67 s` for 50 steps. Per-stage far field at t=0
(`wbnemo_stats`): **traverse+M2P 5929 ms, P2P 1916 ms, upward pass 195 ms, build 9
ms** -- against 60 / 34 / 4.7 / 7 ms on the H200. Traversal alone (`TC_PATH=
traverse-count`, no arithmetic) is 1.5 ms, so "traverse" is the fp64 barycentric M2P
evaluation: 37.5M accepted (target, node) pairs x 512 proxy points.

## 3. What is fp64 in the treecode, and the fp32 ladder

Precision map of one far-field evaluation (widebvh `treecode.cuh` / `bary_stokes.cuh`
/ `stokes_kernel.cuh`), before this change: bucketing / LBVH / MAC / traversal fp32;
forces widened to fp64; upward pass (P2M/M2M) fp64 in shared memory, moments stored
fp32; **M2P `BaryStokes::m2pWarp` fp64 arithmetic on fp32 inputs** (fp64 Chebyshev
offset table, `rsqrtf` seed + fp64 Newton step, fp64 lane accumulators, fp64 shared
atomics per target); **P2P `p2pAtomicKernel` fp64** (fp64 bucket coordinates,
`cub::WarpReduce<double>`, fp64 global atomics). Ada issues fp64 at 1/64 the fp32
rate, so both hot loops are issue-slot bound on the fp64 pipe.

`WIDEBVH_FP32_LEVEL` (compile-time, `stokes_kernel.cuh`; one `.so` per level,
`libwidebvh_nemo[_p<N>]_f32l<L>.so`, CMake list `WIDEBVH_NEMO_FP32_LEVELS`; reported
by the new `wbnemo_fp32_level()`, ABI 4):

- **1 - fp32 M2P.** `m2pWarp` body swapped for an all-fp32 one: `dx = Tp - center` in
  fp32 (both already fp32; accepted nodes sit at r > halfDiag/mac, so the relative
  geometric error is ~1e-6), a `float4` twin of the Chebyshev offset table
  (`bary::c_chebf4`, one LDG.128, no F2F converts), `rsqrtf` with no Newton step, an
  fp32 overload of `accumStokesFactored`, fp32 lane accumulators and shuffle reduce.
  The per-target accumulator of the warp-specialized traversal (`s_results`) becomes
  fp32 too (`FarAcc`; the fp64 shared `atomicAdd` was a CAS+DADD loop, the fp32 one is
  native) and is widened to fp64 once per target.
- **2 - + fp32 P2P** on the `split-warpspec-atomic` path NeMO uses: `p2pAtomicKernel32`
  over the fp32 bucket coordinates and an fp32 copy of the gathered forces
  (`stokes::p2p32`, `WarpReduce<float>`, native fp32 global atomics into a per-apply
  zeroed scratch `velNear32_`), folded into the fp64 output by
  `scatterCompToInputOrderFoldKernel`. Other paths keep their fp64 P2P (so
  `wbnemo_smoke`, which runs `split-warpspec`, still works).
- **3 - + fp32 upward pass**: `BaryStokesWarpAggregate::Acc = float` (P2M/M2M shared
  accumulators, contractions), `bary::lagrange1d_f` in centred coordinates
  (`y - c` first, so node offsets carry ulp(h) not ulp(|c|)), fp32 `c_chebf/c_barywf`.
  Halves the refit's per-warp shared footprint (17 -> 8.5 KB at PDEG 7).

MAC / acceptance arithmetic is untouched at every level (same interaction lists,
identical `near_pairs`). Level 0 is byte-identical to the pre-change engine: the SASS
of the rebuilt `libwidebvh_nemo.so` and `libwidebvh_nemo_p5.so` diffs empty against
the pre-change binaries (`cuobjdump -sass`), and `wbnemo_smoke` reproduces the H200
checksum `-2.450400935e+07` exactly. Levels 1/2/3 move it by 8e-9 / 8e-9 / 1.6e-8
relative.

## 4. Ablation at the production settings (t=0 two-drop cloud, N=1,047,968)

`benchmarks/mac_calibration.py --case twodrop0 --pdegs 7,5 --fp32-levels 0,1,2,3
--macs 0.8 --leaves 1024 --loadings gravity,random --asym-probes 8` (reference: exact
fp64 RPY-TT at 2048 sampled targets; `rel_asym`: far-field-only Hutchinson probe;
`data/widebvh_4060_sweep.csv`, `artifacts/logs/4060/ablation_twodrop0*.log`):

| pdeg | level | loading | rel_far | rel_total | rel_asym | far ms | M2P | P2P | upward | build |
|---|---|---|---|---|---|---|---|---|---|---|
| 7 | 0 | gravity | 1.133e-06 | 1.133e-06 | 6.032e-04 | **8053** | 5929 | 1916 | 195 | 9 |
| 7 | 1 | gravity | 1.138e-06 | 1.138e-06 | 6.032e-04 | 2373 | **223** | 1944 | 196 | 9 |
| 7 | 2 | gravity | 1.138e-06 | 1.138e-06 | 6.032e-04 | 535 | 220 | **87** | 207 | 11 |
| 7 | 3 | gravity | 1.138e-06 | 1.138e-06 | 6.029e-04 | **354** | 231 | 88 | **18** | 11 |
| 7 | 0 | random | 3.089e-04 | 3.071e-04 | 6.032e-04 | 8100 | 5965 | 1927 | 196 | 9 |
| 7 | 3 | random | 3.089e-04 | 3.071e-04 | 6.029e-04 | 362 | 237 | 89 | 19 | 11 |
| 5 | 0 | gravity | 3.365e-05 | 3.365e-05 | 3.531e-03 | 4875 | 2839 | 1927 | 95 | 9 |
| 5 | 3 | gravity | 3.365e-05 | 3.365e-05 | 3.531e-03 | 275 | 176 | 77 | 9 | 10 |
| 5 | 0 | random | 1.782e-03 | 1.771e-03 | 3.531e-03 | 4876 | 2840 | 1927 | 94 | 9 |
| 5 | 3 | random | 1.782e-03 | 1.771e-03 | 3.531e-03 | 284 | 172 | 85 | 10 | 11 |

(pairs 7,335,886 in every row; the level-1/2 rows of this table were timed with a
compile running on the CPU, so their `far` totals carry ~5 % noise; the level-0 and
level-3 rows were re-timed on an idle machine.)

Reading: M2P 5929 -> 223 ms (27x), P2P 1916 -> 87 ms (22x), upward 195 -> 18 ms
(11x); far field 8053 -> 354 ms (**22.7x**) at PDEG 7. Accuracy: `rel_far` 1.133e-6 ->
1.138e-6 under gravity (the fp32 arithmetic adds ~1e-7 in quadrature; the truncation
error of the expansion dominates by two orders under random loading, 3.089e-4 in every
row); `rel_asym` unchanged to three digits (it is MAC-driven, `asym/trunc = sqrt 2` in
`widebvh_far_field_report.md`, and the interaction lists did not change).

**Where the fp32 loss actually is.** It is not zero -- it is below the truncation error
at the operating point. Pushing the truncation error down (tighter `mac`, level 0 vs
level 3, same cloud, `artifacts/logs/4060/fp32_floor_twodrop0.log`) exposes the fp32
floor at ~2e-8 relative (gravity) / ~2e-7 (random loading):

| mac | rel_far gravity, fp64 -> fp32 L3 | rel_far random, fp64 -> fp32 L3 |
|---|---|---|
| 0.4 | 1.54e-08 -> 2.58e-08 (fp32 dominates) | 4.62e-07 -> 6.27e-07 |
| 0.5 | 1.92e-08 -> 2.39e-08 | 2.98e-06 -> 3.01e-06 |
| 0.6 | 7.67e-08 -> 1.37e-07 | 1.62e-05 -> 1.62e-05 |
| 0.8 (production) | 1.133e-06 -> 1.138e-06 | 3.089e-04 -> 3.089e-04 |

At mac 0.8 the expansion's truncation error is 50x (gravity) to 1000x (random) above
that floor, and the learned near field's error is another two orders above it. The
reason the floor is that low is that the inputs were fp32 already (positions, forces,
stored moments, the 1/(8 pi mu) prefactor) and the output is downcast to fp32 when it
is added to the near field; the fp64 arithmetic in between was operating on fp32-
quantized data, and the fp32 replacements keep every long sum short (per-lane
partials, warp reduce, one accumulate per node/target).

What is left in the 354 ms: 231 ms of M2P (37.5M node evaluations x 512 proxies, i.e.
~1.5 TFLOP/s effective -- the moment loads, 6 KB per accepted node, are now the
limiter, not arithmetic), 88 ms of P2P (7.3M leaf pairs x ~1000 sources), 18 ms
upward, 11 ms build, ~6 ms bucketing/prep.

## 5. Parameter sweep (level 3): PDEG 5 vs 7, mac, maxLeaf

Same protocol, `--pdegs 7,5 --fp32-levels 3 --macs 0.6..1.0 --leaves 256..2048
--pair-budget-gb 2` (the 2 GB budget only widens the pair-list cap so that the small
leaves' pair counts do not fall into the engine's tiled fallback and confound the
timing; production keeps the automatic 1 GB, which the chosen point fits with 3.6x
headroom). Worst case is always the random loading, and `rel_asym` is loading-
independent:

| pdeg | mac | leaf | rel_total gravity | rel_total random | rel_asym | far ms (grav / rand) | M2P | P2P | pairs |
|---|---|---|---|---|---|---|---|---|---|
| 7 | 0.6 | 512 | 1.49e-07 | 1.63e-05 | 3.06e-05 | 669 / 694 | 510 | 121 | 17.0M |
| 7 | 0.7 | 512 | 3.63e-07 | 7.30e-05 | 1.12e-04 | 476 / 494 | 359 | 75 | 10.8M |
| 7 | 0.8 | 256 | 1.10e-06 | 3.25e-04 | 6.10e-04 | 369 / 387 | 296 | 30 | 7.3M |
| 7 | 0.8 | 512 | 1.18e-06 | 3.06e-04 | 6.56e-04 | 355 / 368 | 266 | 49 | 7.4M |
| **7** | **0.8** | **1024** | **1.14e-06** | **3.07e-04** | **6.03e-04** | **364 / 388** | **231** | **90** | **7.3M** |
| 7 | 0.8 | 2048 | 1.63e-06 | 2.92e-04 | 5.86e-04 | 372 / 383 | 202 | 132 | 6.2M |
| 7 | 0.9 | 512 | 3.94e-06 | 1.10e-03 | 2.85e-03 | 285 / 295 | 208 | 44 | 6.6M |
| 7 | 0.9 | 1024 | 3.79e-06 | 1.13e-03 | 2.01e-03 | 285 / 299 | 176 | 69 | 5.9M |
| 7 | 1.0 | 1024 | 1.07e-05 | 2.11e-03 | 4.57e-03 | 272 / 281 | 160 | 67 | 5.9M |
| 5 | 0.6 | 512 | 5.47e-06 | 2.09e-04 | 4.03e-04 | 519 / 533 | 369 | 120 | 17.0M |
| 5 | 0.7 | 256 | 1.43e-05 | 6.95e-04 | 1.40e-03 | 371 / 383 | 291 | 38 | 10.1M |
| 5 | 0.7 | 512 | 1.50e-05 | 6.28e-04 | 1.66e-03 | 367 / 378 | 264 | 74 | 10.8M |
| 5 | 0.7 | 1024 | 1.49e-05 | 6.32e-04 | 1.36e-03 | 394 / 400 | 232 | 134 | 10.8M |
| 5 | 0.8 | 512 | 3.44e-05 | 1.82e-03 | 3.36e-03 | 272 / 281 | 195 | 46 | 7.4M |
| 5 | 0.8 | 1024 | 3.36e-05 | 1.77e-03 | 3.53e-03 | 294 / 299 | 178 | 86 | 7.3M |
| 5 | 0.9 | 512 | 7.66e-05 | 4.50e-03 | 8.27e-03 | 223 / 230 | 149 | 40 | 6.6M |
| 5 | 1.0 | 512 | 1.44e-04 | 6.83e-03 | 1.20e-02 | 209 / 211 | 137 | 40 | 6.6M |

(all 40 configurations x 2 loadings are in the CSV; the mac 0.6/0.7 and 0.9/1.0 rows
for the other leaves behave the same way as the ones shown.)

Selection criteria, chosen before the sweep: worst-case `rel_total <= 1e-3` (H200
production 2.9e-4; the published Figure 13 was made at 4.8e-3) **and** far-only
`rel_asym <= 3x` the fp64 baseline measured here (6.03e-4 -> 1.8e-3); then the lowest
time.

- **PDEG 7 / mac 0.8** passes with the production error; leaf 512 measures 3-5 %
  faster than 1024 in this isolated far-field timing (the P2P halves, the M2P grows by
  ~15 %; leaf 256 is the same within noise), a difference that vanishes in the full
  step (sec 6). mac 0.9 fails both criteria marginally (1.1e-3 /
  2.0e-3-2.9e-3) for 20 % of the far field, i.e. ~3 % of the step -- not worth it.
- **PDEG 5** needs mac 0.7 to pass (6.3e-4 / 1.4e-3-1.7e-3) and then costs the same
  367-400 ms as PDEG 7 at mac 0.8, with 2x the error and 2.5x the asymmetry; at
  mac 0.8 it fails both (1.8e-3 / 3.4e-3). At mac 0.6 it is very accurate (2.1e-4 /
  4.0e-4) but slower (519-576 ms) because of the 2.3x near-pair count. The H200
  finding that lowering the degree is a false economy at matched accuracy therefore
  holds on this GPU as well, for a different reason: on the H200 the extra P2P was
  the cost, here the fp32 P2P is cheap but the M2P per accepted node is now
  bandwidth-bound and PDEG 5's 216-proxy nodes (6.75 iterations per lane, more
  per-node overhead) do not save proportionally.

Confirmation over 50 dynamics steps (`benchmarks/far_field_drift.py`,
`data/far_field_drift_1M_4060.csv`, per-step far / near / total):

| config | mean step | far first 10 -> last 10 | near first 10 -> last 10 |
|---|---|---|---|
| **fp32 L3, PDEG 7, mac 0.8, leaf 512** | **2487.5 ms** | 361 -> 388 ms (+7.3 %) | 2205 -> 2082 ms |
| fp32 L3, PDEG 7, mac 0.8, leaf 1024 | 2509.1 ms | 374 -> 399 ms (+6.4 %) | 2228 -> 2089 ms |
| fp32 L3, PDEG 5, mac 0.7, leaf 512 | 2528.2 ms | 376 -> 399 ms (+6.3 %) | 2242 -> 2105 ms |

The three are within 1.6 % of each other -- the leaf choice is a wash once the P2P is
fp32 (leaf 512's 40 ms of P2P saving is spent on 35 ms more M2P), and PDEG 5 buys
nothing at matched accuracy. Head-to-head in the headline benchmark (sec 6) leaf 512
and leaf 1024 tie at 126.1 vs 126.0 s. **Operating point for GeForce-class cards:
`--fp32-level 3`, everything else as production (PDEG 7, mac 0.8, leaf 1024)** -- the
tree, the interaction lists and the accuracy are then those of the H200 runs, only the
kernel arithmetic differs. Leaf 512 is an equivalent alternative (its far-field
`rel_asym` runs 6.6e-4 at t=0 and 8.0e-4 at t=1.0 against 6.0e-4 / 5.9e-4 for the
production tree; that is a property of the tree, not of the precision).

The far field drifts +7 % over 50 steps as the trailing drop deforms (the H200 measured
+16 % over 150), the near field falls 6 % with the pair count.

## 6. Headline: the 50-step timing run, before and after

`benchmarks/two_suspensions_1M.py` (5 warmup + 50 timed steps), same seed, same
image, one process each (`artifacts/logs/4060/{baseline_fp64_p7_mac0.8_leaf1024,
final_fp32l3_p7_mac0.8_leaf1024,final_fp32l3_p7_mac0.8_leaf512}.{log,csv}`):

| | fp64, PDEG 7, mac 0.8, leaf 1024 | **fp32 L3, same tree** | ratio | fp32 L3, leaf 512 |
|---|---|---|---|---|
| Total simulation time | **537.67 s** | **126.04 s** | **4.27x** | 126.10 s |
| step wall, mean / median | 10.75 / 10.78 s | 2.52 / 2.52 s | 4.27x | 2.52 / 2.52 s |
| far field, mean (min-max) | 8685 (8066-9042) ms | 387 (355-410) ms | 22.4x | 378 (346-398) ms |
| near field, mean | 2066 ms | 2132 ms | 0.97x | 2142 ms |
| self+2-body / n-body / nsearch | 599 / 1459 / 7.8 ms | 630 / 1490 / 11.6 ms | | 633 / 1497 / 11.5 ms |
| far share of step | 81 % | 15 % | | 15 % |
| peak process VRAM | 3.73 GB | 3.72 GB | | 3.72 GB |

For scale: H200 production is 280 ms/step (far 104, near 176). The 4060 laptop is now
9x an H200 on this benchmark instead of 38x, and 2.8x faster per step than the A4500
number in `fig13_mac_report.md` (7.13 s, fp64) despite having 43 % of that card's SMs.
The near field is now 85 % of the step (see sec 8).

## 7. Figure 13 on this GPU

Three 100-step snapshot runs of the figure configuration (`--snapshots --t-final 1.0`,
seed 0, one process each; VTK/npy/png every 0.1):

| run | out dir | 100 steps incl. snapshot I/O | mean step (compute) |
|---|---|---|---|
| fp64 production | `figures/drop_1M_4060_fp64/` | 1640.5 s (*) | 10.87-11.5 s |
| **fp32 L3, leaf 1024** | `figures/drop_1M_4060_leaf1024/` | **283.8 s** | 2.52 s |
| fp32 L3, leaf 512 | `figures/drop_1M_4060/` | 282.8 s | 2.52 s |

(*) the fp64 figure run overlapped with the t=1.0 accuracy check on the same GPU for
~10 of its steps, which inflated those steps to 16-23 s; its per-step numbers are not
the baseline -- sec 2 and 6 are.

Rendered with `figures/drop_1m.py` (the six published panels, t = 0, 0.2, ..., 1.0):
`figures/img_drop_1m_4060_leaf1024.{png,pdf}` (recommended config),
`figures/img_drop_1m_4060.{png,pdf}` (leaf 512), `figures/img_drop_1m_4060_fp64.{png,pdf}`
(fp64 reference). The trailing drop stretches into the column, penetrates the leading
drop and coalesces exactly as in nemo.pdf Fig. 13.

**Trajectory divergence** (`benchmarks/compare_snapshots.py`), fp32 L3 against the fp64
reference, both on this GPU, displacements in particle radii; the H200 mac 0.9 / leaf
512 A/B in `fig13_mac_report.md` measured 0.136 at t = 1.0:

| t | fp32 L3, leaf 1024: RMS / max | fp32 L3, leaf 512: RMS / max | z mean (fp64) | z std (fp64) |
|---|---|---|---|---|
| 0.2 | 4.2e-05 / 2.3e-03 | 2.0e-04 / 2.6e-03 | -319.980 | 232.139 |
| 0.6 | 1.7e-04 / 1.6e-02 | 2.1e-03 / 2.0e-02 | -1405.419 | 222.736 |
| 1.0 | **4.2e-04 / 3.5e-02** | 6.3e-03 / 4.5e-02 | -2416.889 | 310.801 |

z mean and z std of the fp32 runs agree with the reference to every printed digit
(-2416.889 / 310.801 at t = 1.0; the H200 fp64 run of the same configuration reported
-2416.886 / 310.801). Worst case at the end of the run is **0.035 of one particle
radius** with the production tree (0.045 with leaf 512, where the tree itself differs).

**Rendered panels** (`benchmarks/compare_panels.py`, 669 x 1825 px): fp32 L3 leaf 1024
vs fp64 -- **99.997 % of pixels bit-identical, max channel delta 1/255, zero pixels
above 4/255**; leaf 512 vs fp64 -- 99.966 %, max 2/255, zero above 4/255. Against the
published `figures/img_drop_1m.png` (WarpFMM theta 0.3 on the H200) the fp64 4060 run
is 97.4 % identical -- the same qualitative figure with a different far field and
seed history.

## 8. What did not change, and what is next

- **Near field.** 2.1 s/step here (self+two-body 0.63 s, n-body 1.50 s) against 0.18 s
  on the H200 -- 12x, more than the 5-6x fp32 throughput ratio, so it is not simply
  compute-bound; it was outside the scope of this task (the treecode) and is now the
  whole step. Leads, untested: `torch.set_float32_matmul_precision("high")` for the
  MLP matmuls, and profiling `_per_particle_topk` / `get_k_per_pair` (large int64 and
  (P,20,3) transients on a 256 GB/s card).
- **M2P bandwidth.** At level 3 the M2P is limited by moment traffic (6 KB per
  accepted node per target). Evaluating a node once for several targets of the same
  block (the queue already groups targets in Hilbert order) would cut it further;
  not attempted.
- **`WIDEBVH_FP32_M2P_NR`** (one fp32 Newton step on `rsqrtf` in the fp32 M2P) is
  wired but not built or measured -- the accuracy tables show no need.
- **P2P boundary pairs.** The fp32 P2P decides `r >= 6` in fp32; a pair within ~1e-5
  of the cutoff can land on the other side of the split than the fp64 kernel put it.
  This is the same band in which the caller's own fp32 hash-grid neighbour search
  already disagrees with fp64, so the level-2 partition is no less consistent with the
  near-field operator than level 0's; it is not visible in any table above.
- The Cartesian policy (`widebvh_nemo_cart`) has no fp32 path (`#error` if asked).

## 9. Reproduce

    # widebvh (once): sm_89 libraries incl. fp32 levels, into ~/envs/nemo-ctx/widebvh/build-4060
    bash docker/run_local.sh bash -c 'cmake -S /opt/widebvh-src -B $WIDEBVH_BUILD_DIR -G Ninja \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_CUDA_FLAGS=-t1 \
        -DWIDEBVH_NEMO_PDEG="5;7" -DWIDEBVH_NEMO_FP32_LEVELS="1;2;3" -DWIDEBVH_CPU_NATIVE=OFF && \
        cmake --build $WIDEBVH_BUILD_DIR -j4 --target widebvh_nemo widebvh_nemo_p5 widebvh_nemo_f32l3 widebvh_nemo_p5_f32l3'
    bash docker/run_local.sh python docker/smoke_test.py            # level 0 checksum, exact
    # timing runs (one config per process)
    bash docker/run_local.sh python benchmarks/two_suspensions_1M.py --log-csv out/base.csv
    bash docker/run_local.sh python benchmarks/two_suspensions_1M.py --fp32-level 3 --log-csv out/fp32.csv
    # accuracy / symmetry / per-stage timing of the far field alone
    TORCH_COMPILE_DISABLE=1 bash docker/run_local.sh python benchmarks/mac_calibration.py --case twodrop0 \
        --pdegs 7 --fp32-levels 0,3 --macs 0.8 --leaves 512,1024 --loadings gravity,random --thetas "" --asym-probes 8
    # Figure 13
    bash docker/run_local.sh python benchmarks/two_suspensions_1M.py --fp32-level 3 \
        --snapshots --t-final 1.0 --out-dir figures/drop_1M_4060_leaf1024/
    python figures/drop_1m.py --data-dir figures/drop_1M_4060_leaf1024 --out figures/img_drop_1m_4060_leaf1024.png   # host python (PyVista)
    python benchmarks/compare_snapshots.py figures/drop_1M_4060_fp64 figures/drop_1M_4060_leaf1024

Files: `data/widebvh_4060_sweep.csv` (ablation + sweep), `data/far_field_drift_1M_4060.csv`
(per-step confirmation runs), `artifacts/logs/4060/` (every log and per-step CSV),
`artifacts/widebvh_fp32_far_field.patch` (the widebvh change, against `9f963d2-dirty`).

## 10. The `sskhan39/nemo:2.0` image (2026-08-22)

The fp32 far field is now packaged for the A4500 / 5090 runs. Built on this laptop
from a fresh `docker/pack_context.sh` context (`~/envs/nemo-ctx-2.0`, manifest
`pinn-stokes @ ac78eb4-dirty`, `widebvh @ 03efcdb` = the fp32 commit) with

    docker build --build-arg NEMO_GIT_SHA=ac78eb4-dirty \
      --build-arg CUDA_ARCHS="86-real;89-real;90-real;120-real;120-virtual" \
      --build-arg NVCC_THREADS=1 --build-arg NINJA_JOBS=4 \
      --build-arg WIDEBVH_FP32_LEVELS="1;2;3" -t sskhan39/nemo:2.0 .

The only difference from 1.0 is `WIDEBVH_FP32_LEVELS="1;2;3"` (plus the
refreshed working trees): the image ships `libwidebvh_nemo.so` (fp64, still the
default), `libwidebvh_nemo_cart.so` and `libwidebvh_nemo_f32l{1,2,3}.so`, each
with SASS for sm_86 (A4500), sm_89, sm_90 (H200) and sm_120 (5090) plus
compute_120 PTX in the per-TU fatbins, exactly as 1.0. No PDEG-5 variants (the
sweep in section 5 showed they are not worth the 2x build; add
`--build-arg WIDEBVH_EXTRA_PDEG=5` if a degree sweep is wanted). The widebvh
step took 22.5 min; the image is 17.4 GB. Same python stack as 1.0 (torch
2.8.0+cu128, warp-lang 1.12.0, numpy 2.3.3, scipy 1.16.2).

Verified on this 4060 (`artifacts/logs/4060/image2.0/`):

| check | result |
|---|---|
| `docker run --rm --gpus all sskhan39/nemo:2.0` (smoke, level 0) | abi 4, checksum `-2.450400935e+07` = H200 reference exactly; all 3 stages pass |
| `-e NEMO_SMOKE_LIB=libwidebvh_nemo_f32l{1,2,3}.so` | `fp32_level` 1/2/3 reported; checksums `-2.450400915e+07`, `-2.450400915e+07`, `-2.450400896e+07` (8e-9 / 1.6e-8 relative, same as the local build-4060 libraries) |
| `two_suspensions_1M.py --fp32-level 3 --t-final 0.05` from the image, no bind mounts | N=1,047,968, `fp32_level=3`, far field 376-393 ms, 2.76 s/step with cold JIT caches, peak 2.69 GB allocated / 3.54 GB reserved |

On the A4500 / 5090 the command is the one in `docker/README.md` step 7c:
`docker run --rm --gpus all sskhan39/nemo:2.0 bash -c "python benchmarks/two_suspensions_1M.py --fp32-level 3"`;
omit `--fp32-level` (or pass 0) for the fp64 A/B. The image has not been pushed
to Docker Hub from here (`docker push sskhan39/nemo:2.0`).
