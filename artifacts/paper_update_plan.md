# Updating `nemo.pdf` for the widebvh far field and the v3 operator

Status as of 2026-09-30. This started (2026-08) as the list of paper edits forced by the widebvh far field
(`widebvh_far_field_report.md`, `cartesian_far_field_report.md`, the RTX 5090 / RTX A4500 runs). Most of those edits
are now in the paper. What is left is a short list of text fixes (§2) and a larger job: the performance section
still describes the baseline operator, not v3 (§1).

Line numbers are `sections/` lines in `~/nemo` at 714cacd (2026-09-24): `02` = `02-methodology.tex`, `03` =
`03-experiments.tex`, `04` = `04-conclusion.tex`. This file was restructured on 2026-09-30, so the `P§n` references in
`~/nemo/update_plan.md` point at the previous version (`git show 11288e6:artifacts/paper_update_plan.md`).

**Still true:** no treecode enters any accuracy number. `benchmarks/paper_accuracy_v2.py` sums the far field as
direct RPY: on the CPU operators (`far_field_2b='rpy'`), and in `_GpuMomentsAdapter` in bounded chunks. Far-field
changes cannot move Figs 2–9.

## Status at a glance

| item | where | state |
|---|---|---|
| Fig 10: two subfigures, caption incl. (b), pair kernel 1214 / 497 M/s, ~2.4× | `03:333–354` | done |
| Fig 10b text: non-monotone throughput, MAC / asymmetry paragraph | `03:356–358` | done |
| Fig 11: caption, 1.1–6.0 %, 1.8 / 2.0 / 2.2×, far share 65.6 → 33.3 %, nsearch 4.9 → 1.0 % | `03:368–376` | done, two nits open (§2.6, §2.7) |
| Fig 12: 2.3 → 4.0 M/s, 0.38 s at 1.5M, 5090 within 8 %, "despite FP64" argument dropped | `03:390–396` | done, caption precision open (§2.4) |
| §2.4.3: barycentric–Lagrange p = 7, RPY in both terms of Eq. (21), near-cutoff exclusion | `02:546–575` | done |
| "MAC parameter of 0.3" | — | removed |
| §3.4: 1,047,968 particles; 0.280 s, 107 / 172 ms, 42.1 s, +16 / −12 / 2 % | `03:435–447` | done |
| H-HIGNN: 1.12 s, 56 s, < 5 GB, ~26×, 65.4M in 18.8 s | `03:449` | done, precision clause open (§2.4) |
| Old Fig 11 TODO, stale "Left panel" comment | — | removed |
| Barycentric-FMM "order of magnitude slower"; low-order argument | `02:543` | **open** (§2.1, §2.2) |
| "LBVH" — production builds an SAH BVH | `02:76, 110, 546, 550` | **open** (§2.3) |
| Precision: "accumulation in double precision", Fig 12 caption, A4500 sentence | `03:323, 390, 449` | **open** (§2.4) |
| "autotuning set to max" | `03:323` | **open** (§2.5) |
| Additions: modularity, translation-only far field (future work) | `03:468`, `04` | **open** (§2.8) |
| Performance section on the v3 operator | `03:323–458` | **open, new** (§1) |

---

## 1. The performance section still describes the baseline operator

CLAUDE.md: every paper measurement uses the moments model. Every number in §3.3–3.4 except Fig 10a's pair kernel
comes from the published K = 10 baseline n-body operator at switch 6. The v3 H200 re-timing (2026-09-23,
`h200_v3_timing_report.md`, `data/*_h200_v3_f32l2.csv`) has replacements for the H200 items. It used fp32 level 2, the
fp16 moments MLP and switch 8. **No figure has been rendered from those CSVs yet.**

| paper now (baseline) | where | v3, H200, level 2 |
|---|---|---|
| Fig 10b: 4.8 → ~11 → just under 10 M upd/s at 4M, "not monotone", log-factor erosion (fp64, cutoff 6) | `03:356` | 5.0 → 15.7 M/s at 1M, then **flat** to 4M (15.6). The roll-off and erosion story does not survive; per doubling 2.02× / 1.99×. rel_asym 4.54e-4 at 1M |
| ~22 neighbours per particle | `03:372` | ~50 (switch 8: 50.0M ordered pairs at 1M) |
| NeMO-2b 1.1–6.0 % slower than RPY | `03:374` | at switch 8 the 2b NN is **faster** than RPY at every size (75.0 vs 121.9 ms at 1M), because the RPY near pass does not use the distance LUT. The comparison needs redesigning (see decisions below) |
| n-body correction 1.8 / 2.0 / 2.2× the 2b op (100k / 200k / 1M) | `03:374` | 3.8 / 5.0 / 7.5× |
| far share 65.6 → 33.3 %, nsearch 4.9 → 1.0 % (10k → 1M) | `03:376` | 50 → 11 %; 5.0 → 0.8 % |
| Fig 12 H200: 2.3 → 4.0 M/s, then saturates; 0.38 s at 1.5M | `03:394` | 1.46 → 1.85 M/s (500k) → 1.72 at 2M; 0.56 s at 1M, 0.86 s at 1.5M |
| RTX 5090 within 8 % of the H200 | `03:396` | no v3 run |
| two-drop 0.280 s/step; far 107 ms (38 %), near 172 (61 %); 150 steps 42.1 s; far +16 %, near −12 %, total within 2 % | `03:447` | 0.44 s/step (50 steps); far 70, near 370 ms; 150 steps 66.6 s; far +32 %, near −5 %, total +0.6 % |
| A4500 1.12 s/step, ~26× H-HIGNN | `03:449` | no v3 run |
| 65.4M particles in 18.8 s on one H200 | `03:449` | 39.3M (340³), 24.4 s warm. The limit is the int32 pair count in `src/hashgrid_neighbors.py`, not memory (55.7 of 140 GB) |
| RTX 4060 4.4×10⁵ upd/s (vs JFSD) | `03:458` | not measured for v3; the v2 moments stack ran 2.97 s/step ≈ 3.5×10⁵ (`gpu_moments_port_report.md` §4) |
| "NN modules use TF32 … same precisions as the accuracy results" | `03:323` | v3 timing runs the moments MLP in **fp16** (in the v2 port the pair correction moved 1.36e-3 from the CPU reference; `gpu_moments_port_report.md`). The sentence becomes false |

Decisions to make before rendering:
- **pc8 or pc6.** pc8 (the trained operating point) runs the moments correction on all pairs d ≤ 8. pc6
  (`--near-op moments-v3-pc6`) gates it at d ≤ 6: 0.267 s/step two-drop, 40.4 s per 150 steps. pc6 is a timing
  variant, not a trained operating point.
- **Level 2 or level 3.** The v3 runs are level 2, but the code default has been level 3 since 2026-09-26. A level-3
  rerun moves the far field ~5 % and totals < 1 %, and lets every disclosure say "level 3 everywhere".
- **Switch for the Fig 11 two-body comparison operators.** At switch 8 they are not comparable with the published
  switch-6 bars.

## 2. Text edits still open

These are needed whichever operator the performance section ends up on.

**2.1 `02:543` — the barycentric-FMM citation.** *"a GPU-accelerated barycentric FMM implementation
\cite{wilson2021gpu} was about an order of magnitude slower"* now reads as an argument against the method we
adopted. Reword in the fewest words: that implementation is older; we reuse only its barycentric–Lagrange moments,
nothing else of the FMM pipeline.

**2.2 `02:543` — the low-order argument.** *"This pushes the fast-summation method toward low-order approximations
…"* is reversed by the measurements. Remove it.
- At matched error (~7e-4), p = 7 costs 95 ms against p = 3's 310 ms at 1M. The tighter `mac` that low order needs
  takes the near pairs from 10.5M to 62.4M.
- The Cartesian Taylor policy at order 4 confirms it: it needs a 2.4× tighter `mac` and is 1.24× slower end to end
  (`cartesian_far_field_report.md` §7).
- Optional clause for Eq. (21): the old Stokeslet M2P had a 6.8e-3 RPY-mismatch floor that no MAC could lower.

**2.3 `02:76, 110, 546, 550` — "LBVH".** Algorithm 1 and §2.4.3 say Linear BVH and cite Karras' LBVH construction.
The production tree is a binary **SAH-built** cuBQL BVH (`WidebvhFMM(bvh_builder="sah")`, `TC_BVH_BUILDER=sah`;
CLAUDE.md, corrected 2026-09-27). `widebvh_far_field_report.md` still says LBVH. Decide the wording; the
BVH-vs-octree argument itself is unaffected.

**2.4 Precision disclosures.** The paper never says the far field runs in fp32.
- `03:323`: *"Velocity accumulation is performed in double precision"* is false wherever the far field runs in
  fp32: Fig 12's H200 column (level 2), the 5090 and A4500 (level 3), and every v3 run. If the section moves to v3,
  add the fp16 MLP here.
- Fig 12 caption (`03:390`): H200 level 2 (fp32 M2P/P2P, fp64 upward pass), RTX 5090 level 3.
- A4500 sentence (`03:449`): level 3. Check which precision H-HIGNN \cite{hignn25} runs at before calling the
  comparison like-for-like.
- Add the accuracy cost once. Level 3 vs fp64 (two-drop, N = 1.05M, against an exact fp64 sum): rel. L2
  1.13e-6 → 1.14e-6 under gravity, 3.09e-4 unchanged under random loading. Over 100 sedimentation steps the
  trajectories differ by at most 0.035 radii (`consumer_gpu_far_field_report.md`). Truncation dominates at
  `mac` 0.8.

**2.5 `03:323` — "autotuning set to max".** True for the operators: `gpu_mob_2b.py` and `gpu_nbody_moments.py`
compile with `mode="max-autotune"`. Not true for Fig 10a's pair-kernel benchmarks: `benchmarks/figure10_components.py`,
`bench_rpy.py` and `benchmark.py` use the default mode. Either scope the sentence or change the scripts (which moves
Fig 10a).

**2.6 `03:372` — "median of five timed runs".** The caption and the protocol say six. Fix it to six.

**2.7 `03:376` — why the far-field share falls.** The paper attributes it to the O(NK²) many-body term growing
faster than the O(N log N) treecode. At fixed φ, K is constant, so the near field is O(N) and that cannot be the
cause. The measured cause is the far field's near-constant build cost (4.7 ms at 10k, 7.0 ms at 1M, fp64), which is
almost the whole far field at small N and amortises. The same `03:376` line carries a TODO to derive O(NK²) in §2.

**2.8 Additions.**
- **Modularity (`03:468`; extend the sentence, two clauses).** The far field was replaced a second time with no
  retraining: 37× more accurate and up to 4.6× faster (`widebvh_far_field_report.md`). The Fig 11 control shows only
  that segment moving (§3). Over 150 drift steps the near field's mean is the same under both far fields to 0.08 %
  (172.43 vs 172.57 ms).
- **Future work (`04`).** The far field is translation-only: torques get no far-field contribution, and the
  RT/TR/RR blocks are zero beyond the switch distance. At switch 6 this was a 5.8 % gap to the dense-RPY grand
  matrix, against 4e-4 truncation (`treecode_symmetry_report.md` §6). Not re-measured at switch 8.
- **Optional, partly in already.**
  - Symmetry. The paper has 4.6e-4 and "two orders below". Not yet in: the treecode is the sole source of
    asymmetry; asymmetry = √2 × truncation error (measured 1.408–1.414); the negative eigenvalues come from the near
    field (33, with and without the tree).
  - A footnote that "sum everything, then subtract the near pairs" fails at any useful `mac`.
  - The drift contrast with the old Warp far field: +16 % vs +278 % over 150 steps.

---

## 3. Reference: the baseline-operator measurements behind the current text

Keep this section until §1 replaces these numbers. All runs are H200, φ = 0.1, `mac` 0.8 / p 7 / leaf 1024, one
process per (N, operator) (traps: CLAUDE.md "Pitfalls").

**Fig 10** (`benchmarks/figure10_components.py --panel both` → `data/fig10_{pair_kernels,far_field}.csv`, rendered by
`figures/component_performance.py`)
- **Pair kernel** at 2^22: RPY 1214 M/s (published 1023), m_t^(2) 496.5 (published 495). The CSV keeps both
  protocols: `published` (1 warm / 3 timed, compile inside the window) and `matched` (10 / 50, plotted). They agree to
  1 % at 2^22.
- **Far field** 50k…4M, fp64, cutoff 6: rel_far 2.27e-4…5.05e-4; rel_asym 3.80e-4…4.61e-4, flat across the n-body
  chunking boundary; 191 ms at 2M, 403 ms at 4M.
- **Excluded sizes:** 5k and 10k (~7 ms build cost) are left out via `min_n` in `_load_far_field`.
- **Clouds:** 2M and 4M from `cluster.uniform_cluster_generation_large(0.1, N, seed=0)`.
- **rel_asym** = ‖M − Mᵀ‖_F / ‖M‖_F, from `benchmarks/symmetry_treecode.py:hutchinson`.
  - Method: 24 Rademacher probes (seed 2) over all probe pairs, using E[(uᵀMv)²] = ‖M‖_F² and
    E[(uᵀMv − vᵀMu)²] = ‖M − Mᵀ‖_F². Costs 24 applies instead of 6N.
  - Checks: against a dense assembly at N = 150 the ratio is 0.933; standard error ~13 %.

**Fig 11** (`benchmarks/figure11_breakdown.py` → `data/fig11_breakdown_h200.csv`, rendered by
`figures/grand_M_perf.py:runtime_breakdown()`)
- **Rebuilding the published figure:** `--from-logs` re-parses its captures (`figures/runtime_breakdown/*_h200.txt`)
  and reproduces them within 1.9 %. The published figure came from a hand-transcribed dict, kept as
  `figures/plot_runtime_breakdown.py`.
- **Conventions:**
  - Panel (a) is the operator's end-to-end timer. ~0.6 ms of staging is outside the segments (`unaccounted_ms`).
  - Neighbour search is counted once; the old figure summed two timers of one interval.
  - Medians over six timed applies.
- **Backend keys:** `warp-published` (what the paper said), `warp` (A/B control at HEAD), `widebvh`. Use `warp` for
  "what the swap bought".
- **Control at 1M:**
  - far field 226.48 → 91.70 ms (2.47×); by size 1.00 / 0.98 / 1.27 / 2.09 / 2.47× over 10k…1M.
  - self+2b 28.07 / 28.08, n-body 151.65 / 151.64, nsearch 2.61 / 2.65 ms.
  - total 410.09 → 275.10 ms.
  - Under `warp` the n-body overhead stays 1.5–1.7×. Its rise to 2.24× under widebvh is the 2b operator getting
    cheaper, not the n-body term.
- **Cross-checks:** Fig 12 at 200k agrees to 0.5 % (total) and 0.7 % (far). The Fig 10 far-field sweep reads
  0.9–6.2 % below Fig 11's far segment, since its tighter loop leaves out a per-call overhead.

**Fig 12** (`figures/grand_M_perf.py:scaling_test_h200_vs_5090`; the paper's `figs/gpu_scaling_h200_vs_5090.pdf`,
2026-08-23, stops at 1.5M)
- **H200 column:** `data/fig12_scaling_h200_f32l2.csv`, level 2 (`fig12_h200_f32l2_report.md`). 3.97M upd/s from
  750k, 252 ms at 1M, 504 at 2M; fp64 280 / 570 ms; process VRAM 6.2 / 9.0 GB at 1M / 2M.
- **RTX 5090 column:** `data/fig12_scaling_5090.csv`, RunPod, level 3 (`fig12_5090_runpod_report.md`). A flat
  1.07–1.08× off the H200 from 50k to 2M. Its far field is faster from 100k up (101 vs 124 ms at 2M); the remaining
  gap is the TF32 near field (441 vs 380 ms). 2M in 0.54 s. 50k needs `--warmup 400` (GeForce idle clocks).
- **Precision:** fp64 vs level-3 far field on the two-drop:

  | card | far, fp64 | far, level 3 | step |
  |---|---|---|---|
  | RTX 4060 laptop | 8.05 s | 0.35 s | 10.75 → 2.52 s |
  | RTX A4500 | 6.14 s | 0.215 s | 7.13 → 1.12 s |

**§3.4 two-drop, N = 1,047,968** (`benchmarks/far_field_drift.py` → `data/far_field_drift_1M.csv`, 150 steps, same
seeded cloud for both backends, fp64)

| | published | warp, re-measured | widebvh |
|---|---|---|---|
| per step | 0.48 s | 0.626 s | 0.280 s |
| far / near | 283 / 182 ms | 452 / 173 ms | 107 / 172 ms |
| first 150 steps | 74 s | 93.9 s | 42.1 s |
| far field, steps 0–9 → 140–149 | — | 256 → 969 ms (+278 %) | 98.9 → 115.2 ms (+16 %) |

- **The published 74 s does not reproduce.** It matches 150 × the early-step rate.
- **The old 50-step aggregate hid the divergence.** `data/widebvh_perf_nemo_distros.csv` covers steps 0–49, and
  the backends only diverge after step 50.
- **A4500** (`A4500_docker.md`, `nemo:2.0`, level 3, 50 timed steps after 5 warm-up):
  - 1.12 s/step (1.07–1.17): far field 215 ms (19 %), near field 901 ms (self+2b 263, n-body 628).
  - 55.9 s per 50 steps.
  - Memory: 2.58 GB torch peak, plus the treecode's ~0.7–1.8 GiB of raw `cudaMalloc`, hence "under 5 GB".
  - Speedup: 28.8 / 1.118 = 25.8×.
  - The published 2.6 s was a Warp run with no surviving log.
- **Capacity** (`max_particles_h200_f32l2_report.md`, level 2): 65,450,827 (403³) runs sustainably at 18.75 s per
  apply, with `empty_cache()` per step and `TC_PAIR_BUDGET_GB=11`. 404³ runs out of memory in
  `_per_particle_topk`.

---

## 4. Measurements still needed

| # | measurement | for | note |
|---|---|---|---|
| 1 | Fig 10b / 11 / 12 renders from `data/*_h200_v3_f32l2.csv` | §1 | after the §1 decisions |
| 2 | v3 RTX 5090 Fig 12 column (RunPod, level 3) | `03:396`, Fig 12 | `--warmup 400` at 50k |
| 3 | v3 A4500 two-drop, 50 steps, level 3 | `03:449` H-HIGNN | |
| 4 | v3 RTX 4060 two-drop throughput | `03:458` JFSD | |
| 5 | v3 H200 items at level 3 | if the paper states level 3 | far −5 %, total < 1 % |
| 6 | capacity after int64 pair counts in `hashgrid_neighbors.py` | `03:449` capacity | v3 stops at 39.3M on int32 |
| 7 | the baseline A4500 per-step CSV (`/persistent/results/a4500_two_drop_fp32.csv`) | provenance of the current 1.12 s | moot once #3 is done |
