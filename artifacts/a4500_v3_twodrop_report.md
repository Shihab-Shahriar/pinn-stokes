# §3.4 two-drop on an RTX A4500 with the v3 NeMO, pc8 and pc6 (RunPod, 2026-10-04)

This closes measurement #3 of `artifacts/paper_update_plan.md` §4: the 1M two-drop
sedimentation, 50 steps, with the current default NeMO on the GPU that the H-HIGNN
comparison names. The paper's A4500 sentence (`~/nemo/sections/03-experiments.tex:456`:
1.12 s/step, ~56 s per 50 steps, < 5 GB, ~26× H-HIGNN) was measured with the
**baseline** operator (`A4500_docker.md`, nemo:2.0).

## Setup

- **Operator:** `two_suspensions_1M.py --near-op moments-v3 --fp32-level 3 --t-final 0.5`, the same command as `slurm/h200_v3/twodrop.sbatch`.
  - v3 moments pair model + pc8c diagonal, fp16 MLP, switch 8.
  - widebvh bary far field: pdeg 7, mac 0.8, leaf 1024, fp32 level 3, pair budget auto (1.2 GB).
  - torch.compile on; 5 warm-up steps, then 50 timed steps; N = 1,047,968.
- **Code:** pinn-stokes `ff13cda`, uploaded as a `git archive` of HEAD.
  - widebvh was built on the pod from the vendored `extern/widebvh`, for sm_86, `libwidebvh_nemo_f32l3.so` only.
  - The image (`sskhan39/nemo:2.0`, torch 2.8.0+cu128, triton 3.4.0, warp 1.12.0) supplied only the environment. Its baked code and widebvh libraries were not used.
- **Machine:** RunPod community cloud, pod `ajk5730ze5iii3`.
  - NVIDIA RTX A4500, 20 GB, driver 580.159.03, 200 W power limit.
  - AMD Ryzen 7 7700, 8 vCPU / 30 GB for the pod.
- **Procedure:** `cloud/runpod_bench.py run twodrop`, two processes in one pod session.
  - **Timing:** the measured run, nothing else on the GPU.
  - **Memory:** the same 50 steps again, under a whole-device `nvidia-smi` sampler at 2 Hz.
  - Per-PID `NEMO_DEVICE_MEM` cannot see PIDs inside a container. The pod rents the whole GPU, so device `memory.used` minus the 1 MiB idle baseline is the process footprint, including widebvh's raw `cudaMalloc`.
- **Cost:** pod lifetime 11.8 min at $0.19/hr, about $0.04. That covers a 2.6 min boot, 0.5 min to build widebvh, 2.7 min per process, then fetch. The pod terminated itself through its dead-man switch 11 s after the request.

## Results

| | v3 A4500 (this run) | baseline A4500 (paper) | v3 H200 (`twodrop_1M_h200_v3_f32l3.csv`) |
|---|---|---|---|
| s/step, mean (median, min–max) | **3.18** (3.19, 3.08–3.50) | 1.12 (1.12, 1.07–1.17) | 0.435 |
| 50 steps | **159.2 s** | 55.9 s | 21.8 s |
| far field, ms | 202 | 215 | 67 |
| near field, ms | 2981 | 901 | 368 |
| – n-body (moments + diag), ms | 2940 | 628 | 356 |
| – self + 2b, ms | 22 | 263 | 7 |
| – neighbour search, ms | 18 | 9 | 4 |
| near pairs | 49.5 M (switch 8) | 20.2 M (switch 6) | 49.5 M |
| peak memory | **3.58 GiB** device (2.21 GiB torch allocated) | 2.58 GB torch allocated | 1.98 GB torch allocated |

The memory process reproduces the timing process to 0.2 % (3.189 vs 3.184 s/step).
Steps 0–9 average 3.30 s and steps 40–49 average 3.08 s, because the near-pair
count relaxes from 54.1 M to 48.5 M as the jittered lattice settles. Under load
the SM clock held at ~1680 MHz (max 2100), power-limited at 199–200 W. That is
the A4500's normal sustained state, not an artefact of the pod.

**H-HIGNN comparison.** H-HIGNN reports 28.8 s/step on four A4500s, about 24 min
for 50 steps. With the v3 NeMO on one A4500 the ratio is **28.8 / 3.18 ≈ 9.0×**
(1440 / 159 s), down from the baseline's ~26×. It is still on one quarter of the
hardware, and memory stays under 4 GB.

## Where the time goes

The v3 n-body stage is 92 % of the A4500 step:
- **A4500 vs H200:** 8.3× slower in that stage (2940 vs 356 ms), against 3.0× in the far field.
- **Against bandwidth:** the HBM bandwidth ratio of the two cards is ~7.5× (4.8 TB/s vs 640 GB/s). That suggests the moments stage is memory-bound.
- **Against the 4060 laptop:** the stage runs *faster* there. On steps 0–2, with identical trajectories and pair counts (54.1 → 50.1 M) in both runs, it takes 2666 ms on the 4060 (`--local --quick` rehearsal) vs 3131 ms on the A4500. The whole step is 3.08 vs 3.37 s; the A4500's far field is faster (189 vs 344 ms). The laptop has 256 GB/s of bandwidth but a 32 MB L2, against the A4500's 6 MB.
- **Untested:** whether it is L2 capacity that matters was not checked with a profiler; it is a hypothesis.

The far field is not the issue: fp32 level 3 holds it at 202 ms, slightly below the
baseline's 215 even though the cutoff went from 6 to 8.

## Paper numbers this makes stale (`03-experiments.tex:456`, not edited)

- "1.12 s per step on one RTX A4500": 3.18 s with v3.
- "approximately 56 s for 50 steps": 159 s.
- "less than 5 GB of memory": still true at 3.6 GiB measured device footprint ("less than 4 GB" would also hold).
- "roughly 26× wall-clock speedup": about 9×.
- The sentence on "one neighborhood-based network pass rather than the O(K) triplet evaluations" still describes v3.

Whether the paper quotes v3 or keeps the baseline for this comparison is the
author's call. `--near-op moments-v3-pc6` is a timing variant, not a trained
operating point, but it cut the H200 moments stage 285 → 119 ms. It would be the
next thing to time here if a faster v3 configuration is wanted.

## pc6: the moments pair correction restricted to d ≤ 6 (same day, same host)

Same job, but with `--near-op moments-v3-pc6` (`cloud/runpod_bench.py run twodrop_pc6`):
- The v3 moments pair correction is applied only to near pairs with d ≤ 6.
- The 2b NN, the learned diagonal, and the far field's near cutoff all stay at 8, so the far field and the diagonal are unchanged.
- This is the H200 `twodrop_pc6` variant (2026-09-23): a timing variant, **not a trained operating point**, and its accuracy has not been measured anywhere (see below).

It ran on pod `wovo0nd5ysb8ts`, the same RunPod host (213.144.200.206) and code (`ff13cda`) as the pc8 run above. The pod lived 8.2 min (the image was already cached there), about $0.03.

| | pc8 (`moments-v3`) | **pc6 (`moments-v3-pc6`)** | pc6 / pc8 | H200 pc8 → pc6 |
|---|---|---|---|---|
| s/step, mean (median, min–max) | 3.18 (3.19, 3.08–3.50) | **1.61** (1.57, 1.52–1.74) | 0.51 | 0.435 → 0.266 (0.61) |
| 50 steps | 159.2 s | **80.5 s** | | 21.8 → 13.3 s |
| far field, ms | 202 | 200 | 1.00 | 67 → 67 |
| near field, ms | 2981 | 1409 | 0.47 | 368 → 199 |
| – n-body (moments + diag), ms | 2940 | 1368 | 0.47 | 356 → 187 |
| – self + 2b, ms | 22 | 22 | | 7 → 7 |
| near pairs (2b list, d ≤ 8) | 54.1 → 48.5 M | 54.1 → 48.6 M | | |
| peak memory, device | 3.58 GiB | **3.44 GiB** | | |
| peak torch allocated | 2.21 GiB | 2.36 GiB | | 1.94 → 2.38 GiB |

- **Repeatability:** the memory process reproduces the timing process to 0.4 % (1.616 vs 1.610 s/step).
- **Drift:** steps 0–9 average 1.71 s and steps 40–49 average 1.54 s, the same pair-count relaxation as at pc8.
- **Clocks:** the clock and power state matched pc8 (median 1695 MHz, 196 W).

**Reading it:**
- **Where the saving comes from:** pc6 halves the n-body stage on the A4500 (−1.57 s/step), slightly more than on the H200 (−47 %). That fits the stage scaling with the number of corrected pairs, which drops from all d ≤ 8 to d ≤ 6 (24.3 M → 10.1 M on the H200).
- **What is left:** the step is still 85 % n-body. The diagonal, which still sees every neighbour within 8, sits inside that 1.37 s, as does the per-particle neighbour table. Their split was not measured here.
- **Memory:** torch's own allocation goes *up* by ~150 MB (the d ≤ 6 gate adds a mask and index buffers, as on the H200). The device footprint goes down slightly, because the moments chunks are smaller.
- **H-HIGNN:** 28.8 / 1.61 ≈ **17.9×** on one A4500 (24 min vs 80.5 s per 50 steps), against 9.0× for pc8 and the paper's ~26× (baseline operator).

**Caveat.** pc6 changes the operator, not just its cost. Pairs at 6 < d ≤ 8 get the 2b NN with no many-body correction, while the diagonal was trained on labels with the pair correction out to 8. No accuracy number exists for this combination:
- `paper_accuracy_v2.py` has no pc6 op;
- the pair_cutoff ablation (`pair_cutoff_ablation.md`) compared *trained* pc6 and pc8 models, not a pc8 model gated at 6.

Quoting 1.61 s/step in the paper would need that measurement first, e.g. a harness op `M_mom_v3_pc8c_diag` with `pair_cutoff=6` on the Fig 3/4 protocols.

## Files

- `data/twodrop_1M_a4500_v3_f32l3.csv`: the 50 per-step rows of the timing process.
- `data/twodrop_1M_a4500_v3_pc6_f32l3.csv` and `artifacts/logs/a4500_v3_pc6_f32l3/twodrop_pc6_20261004T064712Z/`: the same for pc6.
- `artifacts/logs/a4500_v3_f32l3/twodrop_20261004T062031Z/` holds:
  - `driver.log`, `pod.json`;
  - the pod's `bootstrap.log` / `watchdog.log`;
  - `results/`: manifest (versions, md5s, `nvidia-smi -q`), both process logs and CSVs, `*_mem_gpu.csv` (2 Hz samples), `*_mem_summary.txt`, `widebvh_build.log`.
- The driver and job: `cloud/runpod_bench.py`, `cloud/jobs/twodrop.sh` (see `cloud/README.md`).
