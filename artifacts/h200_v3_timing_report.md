# H200 timing of the current NeMO (v3 moments + learned diagonal, fp16 MLP, fp32 L2 far field)

Measured 2026-09-23 on the MSU HPCC H200 nodes (neh-001 / nfh-001), SLURM jobs 17620384 (smoke), 17620385 (Fig 11),
17620386 (Fig 12), 17620387 (two-drop), 17620388 (capacity), 17620389 (Fig 10b); widebvh fp32 build 17620383.

**Operator.** `Mob_Nbody_Moments_Torch`: self NN + two-body NN (distance LUT) + v3 moments pair correction
(`nbody_moments_v3_nb8lin_tr2_kinf_rc8_pc8c`, fp16 MLP, fused warp kernels) + learned diagonal (`nbody_diag_v2_pc8c`),
switch = pair cutoff = 8. Far field `WidebvhFMM` bary, pdeg 7, mac 0.8, leaf 1024, **fp32 level 2**, near cutoff 8.
pinn-stokes `bc4b227` (branch `moments-v3-main`), widebvh `03efcdb` built for sm_90 with its own toolchain
(`slurm/h200_v3/build_widebvh.sbatch`). torch.compile on (Fig 10b off by design). Protocols are the published ones:
Fig 11 one process per (N, operator), 6 warm + 6 timed, median; Fig 12 6 + 6, sorted, drop fastest and last two, 50k..1.5M
in one process and 1.75M / 2M one each. Inputs `tmp/uniform_large_0.1_<N>.csv` (md5-identical to the laptop copies; 4M
generated in-job with `uniform_cluster_generation_large(0.1, 4e6, seed=0)`, md5s in the fig10b log dir).

Data: `data/{fig10_far_field,fig11_breakdown,fig12_scaling,twodrop_1M,far_field_drift_1M,max_particles}_h200_v3_f32l2.csv`.
Raw data (every process log, per-job manifest with shas / nvidia-smi / conda list, slurm .out):
`artifacts/logs/h200_v3_f32l2/<job>_<jobid>/`. Jobs: `slurm/h200_v3/` (README there).

## Figure 12: end-to-end scaling (baseline = published fp32-L2 curve, `data/fig12_scaling_h200_f32l2.csv`)

| N | total ms | far ms | near ms | M upd/s | torch peak GB | baseline total ms | ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 50,000 | 34.2 | 10.5 | 23.7 | 1.46 | 0.88 | 21.8 | 1.57 |
| 100,000 | 58.2 | 13.1 | 45.1 | 1.719 | 1.27 | 33.0 | 1.76 |
| 200,000 | 109.0 | 19.0 | 90.0 | 1.835 | 1.03 | 57.0 | 1.91 |
| 500,000 | 270.3 | 34.5 | 235.8 | 1.85 | 1.82 | 128.6 | 2.1 |
| 750,000 | 412.1 | 48.3 | 363.8 | 1.82 | 2.51 | 189.4 | 2.18 |
| 1,000,000 | 558.8 | 63.0 | 495.8 | 1.79 | 1.88 | 252.0 | 2.22 |
| 1,250,000 | 707.4 | 79.0 | 628.4 | 1.767 | 2.78 | 315.8 | 2.24 |
| 1,500,000 | 861.0 | 94.7 | 766.4 | 1.742 | 2.41 | 379.5 | 2.27 |
| 1,750,000 | 1005.3 | 111.3 | 894.0 | 1.741 | 2.68 | 440.9 | 2.28 |
| 2,000,000 | 1161.8 | 126.1 | 1035.7 | 1.722 | 2.94 | 503.6 | 2.31 |

The far field is unchanged by the switch 6 -> 8 (63.0 vs 61.5 ms at 1M): the treecode's cost is dominated by the
multipole work, not the near complement. The near field is 2.6-2.7x the baseline's: switch 8 carries 2.4x the near
pairs (50.0 M vs 20.9 M ordered pairs at 1M, `data/fig11_breakdown_h200_rerun.csv`) and the moments correction evaluates a per-pair neighbourhood of all
particles within 8 of the pair midpoint. Throughput saturates at ~1.72-1.85 M updates/s (baseline 3.97).

## Figure 11: runtime breakdown (all three operators at switch 8, ms)

| N | operator | far | self+2b | n-body | nsearch | total | near pairs |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | FMM_2body_RPY | 7.27 | 1.11 | 0.0 | 0.72 | 9.62 | 475506 |
| 10,000 | FMM_2body_NN | 7.25 | 0.64 | 0.0 | 0.72 | 9.13 | 475506 |
| 10,000 | FMM_Nbody_Moments | 7.28 | 0.51 | 5.22 | 0.73 | 14.46 | 475506 |
| 50,000 | FMM_2body_RPY | 9.95 | 2.82 | 0.0 | 0.78 | 14.08 | 2449228 |
| 50,000 | FMM_2body_NN | 9.82 | 0.93 | 0.0 | 0.77 | 12.05 | 2449228 |
| 50,000 | FMM_Nbody_Moments | 10.01 | 0.8 | 21.06 | 0.78 | 33.39 | 2449228 |
| 100,000 | FMM_2body_RPY | 12.44 | 5.12 | 0.0 | 0.99 | 19.1 | 4929204 |
| 100,000 | FMM_2body_NN | 12.35 | 1.3 | 0.0 | 0.97 | 15.18 | 4929204 |
| 100,000 | FMM_Nbody_Moments | 12.45 | 1.16 | 41.95 | 0.99 | 57.32 | 4929204 |
| 200,000 | FMM_2body_RPY | 17.82 | 9.82 | 0.0 | 1.29 | 29.55 | 9906820 |
| 200,000 | FMM_2body_NN | 17.74 | 2.08 | 0.0 | 1.26 | 21.6 | 9906820 |
| 200,000 | FMM_Nbody_Moments | 17.82 | 1.93 | 85.87 | 1.34 | 107.75 | 9906820 |
| 1,000,000 | FMM_2body_RPY | 61.69 | 55.27 | 0.0 | 4.3 | 121.94 | 49995572 |
| 1,000,000 | FMM_2body_NN | 61.7 | 8.38 | 0.0 | 4.3 | 75.02 | 49995572 |
| 1,000,000 | FMM_Nbody_Moments | 61.84 | 8.21 | 485.73 | 4.36 | 561.28 | 49995572 |

At 1M the n-body correction (pair moments + learned diagonal) is 486 ms of the 561 ms step (87 %); the far field is
11 %. The 2b comparison operators run at switch 8 here, so their bars are NOT comparable with the published switch-6
bars (the RPY near pass is 55 ms at 1M vs the NN's 8.4 ms: the 2b NN goes through the distance LUT, the analytic RPY
does not). Fig 11 and Fig 12 agree at 1M (561.3 vs 558.8 ms total).

## §3.4 / Figure 13: 1M two-drop sedimentation (N = 1,047,968)

50-step aggregate (`two_suspensions_1M.py --t-final 0.5`, `data/twodrop_1M_h200_v3_f32l2.csv`): **0.44 s/step** wall
incl. Euler update (22.0 s for 50 steps); far 70.0 ms, near 369.5 ms (n-body 357.8, self+2b 7.3, neighbour search 4.1),
torch peak 2.0 GB. The paper's baseline figures: 0.48 s/step, 283 ms far / 182 ms near (WarpFMM far field). The
like-for-like reference is the baseline operator on the widebvh far field, 0.280 s/step with 104 ms far (fp64, switch 6;
`artifacts/widebvh_far_field_report.md` §2b): against that v3 costs +57 % per step -- the far field is 34 ms cheaper
(fp32 level 2), the near field ~2.1x dearer (switch 8 carries ~2.4x the pairs, plus the moments correction).

150-step per-step run (`far_field_drift.py`, `data/far_field_drift_1M_h200_v3_f32l2.csv`): mean step 443.7 ms,
**150 steps in 66.6 s** of GPU time (paper: 74 s); far field first/last 10 steps 67.5 -> 89.3 ms (+32 %, the drops
become non-uniform), near field 381 -> 363 ms (-5 %, near pairs 54.1 M -> 45.6 M), total +0.6 %.

## Figure 10b: far field vs N (near cutoff 8, random loading, mac 0.8)

| N | far ms | M upd/s | rel_far | rel_asym |
|---:|---:|---:|---:|---:|
| 5,000 | 7.1 | 0.7 | 3.10e-04 | 1.83e-04 |
| 10,000 | 7.13 | 1.4 | 1.94e-04 | 2.39e-04 |
| 50,000 | 9.97 | 5.02 | 3.09e-04 | 3.66e-04 |
| 100,000 | 12.53 | 7.98 | 3.55e-04 | 3.85e-04 |
| 200,000 | 18.39 | 10.87 | 4.90e-04 | 3.63e-04 |
| 300,000 | 23.67 | 12.68 | 3.63e-04 | 3.86e-04 |
| 400,000 | 28.87 | 13.85 | 2.52e-04 | 4.22e-04 |
| 500,000 | 34.43 | 14.52 | 3.30e-04 | 4.53e-04 |
| 750,000 | 48.58 | 15.44 | 3.29e-04 | 4.18e-04 |
| 1,000,000 | 63.63 | 15.72 | 2.26e-04 | 4.54e-04 |
| 2,000,000 | 128.62 | 15.55 | 2.30e-04 | 3.95e-04 |
| 4,000,000 | 256.34 | 15.6 | 3.73e-04 | 4.21e-04 |

## Capacity (lattice at phi = 0.1, cold + 3 warm applies with empty_cache, default pair budget)

| n_side | N | ok | warm s | proc GB |
|---:|---:|---:|---:|---:|
| 200 | 8,000,000 | 1 | 3.81 | 12.0 |
| 250 | 15,625,000 | 1 | 8.38 | 22.9 |
| 290 | 24,389,000 | 1 | 13.97 | 35.0 |
| 320 | 32,768,000 | 1 | 19.72 | 46.7 |
| 335 | 37,595,375 | 1 | 23.09 | 53.3 |
| 338 | 38,614,472 | 1 | 23.90 | 54.7 |
| 340 | 39,304,000 | 1 | 24.37 | 55.7 |
| 341 | 39,651,821 | 0 | - | 6.5 |
| 342 | 40,001,688 | 0 | - | 6.6 |
| 350 | 42,875,000 | 0 | - | 7.0 |

**The ceiling at 340^3 = 39.3 M is not memory.** At 341^3 the neighbour search's ordered near-pair count passes 2^31
(`Total near-field pairs found: -2139711726`): `src/hashgrid_neighbors.py` counts and offsets pairs in int32, and the
edge buffer allocation then fails (`AttributeError: 'NoneType' object has no attribute 'dtype'` in `wp.from_torch`).
The process footprint at 340^3 is 55.7 GB of 140. Switch 8 has ~54 ordered pairs per particle on this lattice (50 on the random Fig 11/12 clouds; ~21 at switch 6),
so the int32 limit binds at ~39.7 M particles, before memory. The baseline's published 65.45 M (403^3) was memory-bound.
Warm apply at 340^3: 24.4 s.

## fp32 level 3 on the H200 (2026-09-26)

Every run above is fp32 level 2, because the cluster build only had level 2. Level 3 (+ fp32 upward pass, P2M/M2M)
was built for sm_90 (`build_widebvh.sbatch`, levels 1-3) and A/B'd against level 2 in one job on one node
(`slurm/h200_v3/fp32_levels.sbatch`, job 17859221, sha f79f8c4, same v3 operator; raw data
`artifacts/logs/h200_v3_f32l2/h200v3_fp32lv_17859221/`, CSVs `data/fp32_levels_fig{12,10b}_h200.csv`).

Fig 12 protocol, end-to-end apply, torch.compile on, one process per level in ABBA order:

| N | level 2 total (far), ms | level 3 total (far), ms | delta |
|---|---|---|---|
| 1M | 554.2 (62.1), 553.7 (61.9) | 550.2 (58.6), 550.0 (58.6) | -3.8 ms (-0.7 %) |
| 2M | 1158.9 (125.1), 1157.9 (124.7) | 1152.2 (119.0), 1152.9 (119.2) | -5.8 ms (-0.5 %) |

Fig 10b protocol, far field alone (random loading, near cutoff 8, torch.compile off):

| N | far ms L2 -> L3 | upward ms L2 -> L3 | rel_far L2 / L3 | rel_asym L2 / L3 |
|---|---|---|---|---|
| 1M | 61.71 -> 58.09 | 7.06 -> 3.49 | 2.26321e-4 / 2.26318e-4 | 4.5447e-4 / 4.5448e-4 |
| 2M | 124.69 -> 118.88 | 11.19 -> 5.38 | 2.29967e-4 / 2.29973e-4 | 3.9495e-4 / 3.9495e-4 |
| 4M | 252.87 -> 242.57 | 18.42 -> 7.96 | 3.73435e-4 / 3.73439e-4 | 4.2125e-4 / 4.2115e-4 |

Level 3 halves the upward pass and is faster at every size, with accuracy and symmetry unchanged to 4-5 digits
(truncation dominates at mac 0.8, as on the 4060). **Level 3 is therefore the default on every GPU from 2026-09-26**
(`treecode_widebvh.DEFAULT_FP32_LEVEL`, and `slurm/h200_v3/env.sh`). The numbers above this section stay level 2; a
level-3 rerun would move the far field by ~5 % and the totals by < 1 %.

## Notes
* The five jobs ran concurrently on two nodes (one GPU each; other tenants may share the node). Timed std is
  0.04-0.23 ms at every Fig 12 size, so contention is not visible.
* Inductor prints `No valid triton configs ... out of resource: triton_mm` during max-autotune of the moments MLP; it
  drops that candidate and uses another (no fallback to eager, no recompile-limit warnings).
* First build attempt (job 17620153) failed on a missing `cmake` in the python environment; its log is kept.
