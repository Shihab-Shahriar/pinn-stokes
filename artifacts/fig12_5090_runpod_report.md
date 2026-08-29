# Figure-12 RTX 5090 column, re-measured and extended to 2M (RunPod, 2026-08-23)

Replaces the hand-transcribed 50k..750k rows in `data/fig12_scaling_5090.csv`
(kept at `artifacts/logs/fig12_5090_runpod/fig12_scaling_5090_handtranscribed_oldbox.csv`;
they came from a different 5090 box via `artifacts/5090_docker.md`) with a
machine-written column, and extends it to the H200 grid's 1M..2M so
`figures/gpu_scaling_h200_vs_5090.pdf` has both cards at every size. This closes
measurement #8 of `artifacts/paper_update_plan.md`.

## Machine

RunPod pod, NVIDIA GeForce RTX 5090 32 GB, driver 590.48.01, torch 2.8.0+cu128 —
the nemo/widebvh-fp32 image (pinn-stokes @ `ac78eb4-dirty`, widebvh @ `03efcdb`,
`libwidebvh_nemo*_f32l{1,2,3}.so`, `NEMO_FAR_FIELD=widebvh`). The image runs no
sshd, so the direct-TCP address is connection-refused (as on the H200 pods); all
driving went through the RunPod ssh proxy (`-tt`, commands on stdin, CR-stripped),
files in and out as base64 over the PTY, md5-verified.

## Protocol

`benchmarks/figure12_grand_M.py --backend widebvh`, `NEMO_FAR_FP32_LEVEL=3` (the
consumer-GPU operating point; the H200 column is fp32 level 2), warmup 6 / runs 6,
pdeg 7, mac 0.8, leaf 1024, random forces seed 2024 — i.e. the
`fig12_h200_f32l2_report.md` protocol, **one process per size** (the >8-sizes
recompile_limit trap), each writing its own CSV (the script rewrites all rows of
a backend in whatever `--csv` it gets), merged afterwards. The five large
configurations were generated on the pod with
`cluster.uniform_cluster_generation_large(0.1, N, seed=0)` (38..75 s each), the
documented recipe for the pre-existing 2M/4M files; 50k..750k configs ship in the
image. Whole sweep: 11 min wall.

**The 50k row needed `--warmup 400`.** GeForce idle SM clock on this pod is
180 MHz (max 3090) and `nvidia-smi -lgc` is not permitted inside the container.
`far_ms` comes from a single post-timing apply, so at N=50k — where six ~20 ms
warmup steps cannot hold boost clocks — the sweep's row carried far 45.7 ms
against total 22.8 ms (negative near). A fresh-process rerun from an idle GPU was
noisy throughout (total 35.4 ± 4.5 ms). 400 discarded warmup steps (~9 s of
continuous work) hold the clock up through the timed window and the stats apply:
total 23.42 ± 0.33 ms, far 10.99 ms, matching the shape of the H200's split.
Sizes ≥100k ramp the clock inside the standard 6 warmup steps and were stable
(std ≤ 0.5 ms) as measured. Anyone re-measuring small N on GeForce from an idle
GPU should bump `--warmup` likewise.

## Results (now in `data/fig12_scaling_5090.csv`)

| N | H200 ms | 5090 ms | 5090/H200 | 5090 far ms | 5090 Mu/s | old-box ms |
|---|---|---|---|---|---|---|
| 50k | 21.77 | 23.42 | 1.08 | 10.99 | 2.135 | 32.88 |
| 100k | 32.97 | 35.36 | 1.07 | 12.35 | 2.828 | 44.80 |
| 200k | 56.97 | 60.79 | 1.07 | 17.15 | 3.290 | 70.79 |
| 500k | 128.59 | 137.32 | 1.07 | 29.08 | 3.641 | 148.03 |
| 750k | 189.36 | 201.99 | 1.07 | 39.04 | 3.713 | 211.12 |
| 1M | 252.03 | 269.04 | 1.07 | 50.32 | 3.717 | — |
| 1.25M | 315.83 | 338.87 | 1.07 | 64.32 | 3.689 | — |
| 1.5M | 379.51 | 406.63 | 1.07 | 78.82 | 3.689 | — |
| 1.75M | 440.90 | 476.38 | 1.08 | 88.71 | 3.674 | — |
| 2M | 503.58 | 542.63 | 1.08 | 101.22 | 3.686 | — |

The consumer card holds a flat **1.07–1.08×** of the H200 total across the whole
range — the near field dominates both (at 2M: 441 of 543 ms on the 5090) and the
fp32-level-3 far field keeps the GeForce fp64 penalty out of the picture. Peak
torch-allocator VRAM at 2M is 4.17 GB (same as the H200 — same allocation
pattern), nowhere near the 32 GB. This pod's numbers are faster than the old
5090 box at every overlapping size (0.71×..0.96×, converging with N — the
old box's small-N rows were likely clock-sagged, see above), so the columns were
replaced wholesale rather than mixed across systems.

Raw logs, per-size CSVs and the driver script: `artifacts/logs/fig12_5090_runpod/`.
