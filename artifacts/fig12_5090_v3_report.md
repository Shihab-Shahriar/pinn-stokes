# Figure 12 with the v3 NeMO: RTX 5090 vs H200, 50k–2M (RunPod, 2026-10-04)

This closes measurement #2 of `artifacts/paper_update_plan.md` §4: the RTX 5090 column of
Figure 12 (`figs/gpu_scaling_h200_vs_5090.pdf`, `03-experiments.tex:396–403`) for the
current NeMO, next to the H200 sweep with the same operator. The published figure and
its prose come from the **baseline** operator (`fig12_5090_runpod_report.md`, 2026-08-23).

## Setup

- **Operator:** `benchmarks/figure12_grand_M.py --backend widebvh --near-op moments-v3`.
  - v3 moments pair + pc8c diagonal, fp16 MLP, switch 8.
  - widebvh bary far field: pdeg 7, mac 0.8, leaf 1024, **fp32 level 3**.
  - torch.compile on; 6 warmup + 6 timed applications, sorted, with the outer ones dropped.
  - Random forces, seed 2024.
- **One process per size**, each writing its own CSV, merged afterwards (`cloud/jobs/fig12.sh`).
- **Warmup:** `--warmup 400` at N ≤ 200k for the GeForce idle-clock trap (the pod GPU idles at ~180 MHz and `nvidia-smi -lgc` is not allowed); 6 above that.
- **H200 column:** `data/fig12_scaling_h200_v3_f32l3.csv` (2026-09-26, git `11288e6`, the same operator, fp32 level and protocol, warmup 6 everywhere).
- **Inputs:** identical clouds on both cards.
  - The ten `tmp/uniform_large_0.1_<N>.csv` files are byte-identical on the laptop and the cluster (`/mnt/scratch/khanmd/nemo_v3_timing/tmp`); md5s in the manifest.
  - On the pod, 50k–750k came from the image (same md5s) and 1M–2M were uploaded; the job verified all ten before timing.
- **Code:** pinn-stokes `ff13cda` (`git archive` of HEAD). widebvh was built on the pod from the vendored `extern/widebvh` for sm_120, `libwidebvh_nemo_f32l3.so` only. Environment: `sskhan39/nemo:2.0` (torch 2.8.0+cu128, triton 3.4.0, warp 1.12.0).
- **Machine:** RunPod secure cloud, pod `8inrtgq2zujl4w`, NVIDIA GeForce RTX 5090 32 GB, driver 580.159.03.
  - The host was shared: 256 threads, load average ~109 during the run.
  - Shared CPUs stretched the widebvh compile to 7 min (40 s on the laptop). The timings themselves stayed tight (σ ≤ 0.72 ms).
- **Cost:** 19.6 min at $0.99/hr ≈ **$0.32**: 1.9 min boot, 3.4 min of uploads (the configs alone 168 s), 7 min building widebvh, 6 min sweeping. The pod terminated itself through its dead-man switch.

## Results (`data/fig12_scaling_5090_v3_f32l3.csv`)

| N | H200 ms | 5090 ms (±) | 5090/H200 | far H200 / 5090 ms | near H200 / 5090 ms | M upd/s H200 / 5090 | 5090 VRAM GB |
|---|---|---|---|---|---|---|---|
| 50k | 33.5 | 36.9 (0.32) | 1.10 | 9.8 / 10.1 | 23.6 / 26.8 | 1.49 / 1.36 | 0.85 |
| 100k | 56.9 | 61.8 (0.08) | 1.08 | 12.1 / 12.3 | 44.9 / 49.5 | 1.76 / 1.62 | 0.90 |
| 200k | 106.2 | 110.5 (0.22) | 1.04 | 17.4 / 17.5 | 88.8 / 93.0 | 1.88 / 1.81 | 1.00 |
| 500k | 263.9 | 257.9 (0.24) | 0.98 | 31.8 / 30.2 | 232.1 / 227.7 | 1.90 / 1.94 | 1.41 |
| 750k | 403.3 | 382.2 (0.41) | 0.95 | 44.5 / 38.9 | 358.8 / 343.4 | 1.86 / 1.96 | 1.62 |
| 1M | 547.6 | 502.8 (0.12) | 0.92 | 58.5 / 49.3 | 489.1 / 453.5 | 1.83 / 1.99 | 1.82 |
| 1.25M | 693.2 | 632.7 (0.65) | 0.91 | 74.1 / 62.4 | 619.2 / 570.3 | 1.80 / 1.98 | 2.06 |
| 1.5M | 843.4 | 763.2 (0.66) | 0.90 | 89.9 / 78.9 | 753.5 / 684.4 | 1.78 / 1.97 | 2.32 |
| 1.75M | 1000.2 | 892.7 (0.49) | 0.89 | 105.8 / 88.9 | 894.3 / 803.8 | 1.75 / 1.96 | 2.57 |
| 2M | 1145.0 | 1027.7 (0.72) | 0.90 | 119.0 / 101.9 | 1025.9 / 925.8 | 1.75 / 1.95 | 2.83 |

(VRAM is the torch allocator's peak, as in every Fig 12 CSV; it misses widebvh's raw `cudaMalloc`.)

**Reading it:**
- **Where the cards cross:** the 5090 is 4–10 % *slower* up to 200k, and the cards cross at ~500k. From 750k up the 5090 is **5–11 % faster** (0.89–0.95×).
- **Throughput:** the H200 peaks at 1.90 M updates/s at 500k and sags to 1.75 at 2M. The 5090 climbs to 1.99 at 1M and holds ~1.95.
- **Far field:** the 5090's is faster from 500k up (101.9 vs 119.0 ms at 2M), as it was with the baseline operator (101 vs 124).
- **Near field, the change from the baseline:** with the baseline operator the H200's near field was the faster one (380 vs 441 ms at 2M). With v3 the 5090 also wins the near field from 500k up (926 vs 1026 ms at 2M), even though the H200 has ~2.7× its memory bandwidth.
  - This fits the A4500 observation (`a4500_v3_twodrop_report.md`) that the v3 moments stage favours a large L2: 96 MB on the 5090 vs 50 MB on the H200, and a 4060 laptop (32 MB) beating an A4500 (6 MB).
  - Higher clocks may also contribute. Neither has been profiled.
- **Small N:** the 5090's deficit at small N is in the near field (26.8 vs 23.6 ms at 50k). A busy host CPU can inflate launch overhead there. The spread is tight, but a quiet host might close part of the 4–10 %.

## Figure

`figures/gpu_scaling_h200_vs_5090_v3.pdf`, from `figures/grand_M_perf.py:scaling_test_h200_vs_5090_v3()`:
- the published figure's function, now parametrized by CSVs and output name;
- the same five drawn sizes (50k, 200k, 500k, 1M, 1.5M), PDF only.

It is **not** synced to `~/nemo/figs/` (it differs from the included `gpu_scaling_h200_vs_5090.pdf`). Switching the paper to v3 is the author's call.

## Paper text this makes stale (`03-experiments.tex:397–403`, not edited)

- **Caption:** "from 50k to 1.5M particles … MAC 0.8, p = 7, leaf 1024" still holds. Under v3 both cards run fp32 level 3, so the precision note in plan §2.4 simplifies to one level.
- **"throughput rises from 2.3 to about 4.0 million particle updates per second and then saturates":** H200 v3 goes 1.49 → 1.90 M/s (500k), then 1.78 at 1.5M.
- **"one mobility application takes approximately 0.38 s" at 1.5M:** 0.84 s on the H200, 0.76 s on the 5090.
- **"the RTX 5090 remains within 8 % of the H200 across the entire range":** within 10 % everywhere, and *faster* from 500k up (by up to 11 %).
- **"the RTX 5090 is faster in the far field, while the H200 is faster in the TF32 near field":** no longer true. The 5090 wins both terms from 500k up, the H200 wins both below 200k, and the near-field MLP is fp16, not TF32 (plan §1, row `03:323`).

## Files

- `data/fig12_scaling_5090_v3_f32l3.csv`: the 10 merged rows.
- `artifacts/logs/5090_v3_f32l3/fig12_20261004T185902Z/` holds:
  - `driver.log`, `pod.json`;
  - the pod's bootstrap and watchdog logs;
  - `results/`: per-size CSVs and logs, manifest with config md5s, `widebvh_build.log`.
- The driver and job: `cloud/runpod_bench.py run fig12`, `cloud/jobs/fig12.sh`.
