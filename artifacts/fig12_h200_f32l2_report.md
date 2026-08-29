# Figure 12 on H200, widebvh far field at fp32 level 2, N = 50k .. 2M

Measured 2026-08-22 on a RunPod H200 (143 GB, driver 580.159.04) from the
`sskhan39/nemo:2.0` image (pinn-stokes @ ac78eb4-dirty, widebvh @ 03efcdb,
torch 2.8.0+cu128, warp 1.12.0). Operator: full NeMO (analytic self + two-body
NN + n-body NN inside r=6, widebvh BaryStokes treecode beyond; pdeg 7, mac 0.8,
leaf 1024), `NEMO_FAR_FP32_LEVEL=2` (fp32 M2P + P2P, fp64 upward pass).
Protocol is the published one (`benchmarks/figure12_grand_M.py`: 6 warmup,
6 timed, sorted, drop fastest and last two; torch.compile on). Uniform
suspensions at phi = 0.1, `tmp/uniform_large_0.1_<N>.csv`, random forces seed 2024.

Data: `data/fig12_scaling_h200_f32l2.csv`. Raw logs: `artifacts/logs/fig12_h200_f32l2/`.

## Results (fp32 level 2)

|        N | total ms | far ms | near ms | M updates/s | torch peak GB |
|---------:|---------:|-------:|--------:|------------:|--------------:|
|   50,000 |    21.77 |  10.65 |   11.11 |       2.297 |          1.01 |
|  100,000 |    32.97 |  12.59 |   20.38 |       3.033 |          1.94 |
|  200,000 |    56.97 |  18.18 |   38.79 |       3.510 |          2.02 |
|  500,000 |   128.59 |  33.25 |   95.34 |       3.888 |          2.13 |
|  750,000 |   189.36 |  47.10 |  142.26 |       3.961 |          2.22 |
| 1,000,000 |  252.03 |  61.54 |  190.50 |       3.968 |          2.31 |
| 1,250,000 |  315.83 |  77.45 |  238.38 |       3.958 |          2.61 |
| 1,500,000 |  379.51 |  93.40 |  286.11 |       3.952 |          3.13 |
| 1,750,000 |  440.90 | 109.06 |  331.84 |       3.969 |          3.65 |
| 2,000,000 |  503.58 | 123.75 |  379.82 |       3.972 |          4.17 |

Scaling is linear from 500k on: the update rate saturates at ~3.96-3.97 M
particle updates/s and holds flat to 2M (252 ms per 1M particles). The far
field is ~24-25% of the step at every size >= 500k; the near field (two-body +
n-body NN) is the remaining ~75%.

## fp64 reference (level 0) at the new sizes, same node, same protocol

|        N | L0 total | L0 far | L2 total | L2 far | near (both) | far speedup | step speedup |
|---------:|---------:|-------:|---------:|-------:|------------:|------------:|-------------:|
| 1,000,000 |  280.16 |  91.02 |   252.03 |  61.54 |     ~186-190 |       1.48x |        1.11x |
| 2,000,000 |  570.13 | 190.31 |   503.58 | 123.75 |     ~374-380 |       1.54x |        1.13x |

The near-field operator time is identical between levels (185.9 vs 186.0 ms at
1M, 374.1 vs 374.2 ms at 2M; fp32 level only touches the treecode), so the
whole 11-13% step gain is the far field. On the H200 (full-rate fp64) level 2
buys ~1.5x on the far field, compared with ~23x on the RTX 4060 -- as expected.
For comparison, the existing fp64 H200 rows in `data/fig12_scaling_h200.csv`
(50k-750k, same SHA) give 3.60-3.64 M updates/s at 500k-750k vs 3.89-3.96 M
here at L2.

## VRAM

`peak_vram_gb` in the CSV is `torch.cuda.max_memory_allocated`, which does not
see widebvh's raw `cudaMalloc` (BVH, buckets, pair list). The pod had the GPU
exclusively (0 MiB used at idle), so a 200 ms `nvidia-smi --query-gpu=memory.used`
sampler during a dedicated L2 run gives the true process footprint:

|        N | torch allocated | torch reserved | nvidia-smi peak (true) | non-torch |
|---------:|----------------:|---------------:|-----------------------:|----------:|
| 1,000,000 |        2.37 GB |        4.44 GB |                 6.2 GB |   ~1.8 GB |
| 2,000,000 |        4.27 GB |        6.35 GB |                 9.0 GB |   ~2.7 GB |

So a 2M-particle step of the full operator fits comfortably in 12 GB.

## Gotcha: one config per process

The first pass ran all ten sizes in one process (as `figure12_grand_M.py` does
by default). Every size builds a fresh operator instance, and the guards in the
compiled `forward`s of `gpu_mob_2b.py:73` and `gpu_nbody_mob.py:110` are on the
instance (`___check_obj_id(self._buffers['median'])`, `___check_type_id(self)`),
so each size is one recompile. The 9th size (1.75M) hit
`torch._dynamo hit config.recompile_limit (8)` and dynamo silently fell back
to eager: 1.75M read 1319 ms and 2M read 1506 ms (1.33 M updates/s), a 3.5x
jump while the far field stayed linear (the log is
`artifacts/logs/fig12_h200_f32l2/sweep_50k-2M_single_process.log`, warning at
line ~1999). Those two points were re-measured in their own processes
(`rerun_*.log`) and are the values in the CSV; the first eight were compiled
and are kept from the single-process sweep. With more than 8 sizes per run,
either split the sizes across processes or raise
`torch._dynamo.config.recompile_limit` (the CLAUDE.md note about this limit and
the silent eager fallback applies to the benchmark driver as well as to the
chunk loop).
