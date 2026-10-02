# H200 timing runs for the current NeMO (v3 moments + pc8c diagonal)

Operator: `Mob_Nbody_Moments_Torch` with `experiments/nbody_moments_v3_nb8lin_tr2_kinf_rc8_pc8c.wt` +
`experiments/nbody_diag_v2_pc8c.wt` (fp16 MLP, warp backend, switch 8), inside `WidebvhFMM` (bary, pdeg 7,
mac 0.8, leaf 1024) at the fp32 level in `env.sh` -- **3** since 2026-09-26 (the 2026-09-23 runs and their
`*_f32l2` CSVs are level 2; `fp32_levels.sbatch` showed level 3 is faster at the same accuracy) -- near cutoff 8. torch.compile enabled (except Fig 10b, an
accuracy/symmetry panel that disables it by design).

Setup (once): a checkout of this repo on scratch, `tmp/uniform_large_0.1_<N>.csv` for N = 5k..2M, and the
widebvh fp32 build: widebvh 03efcdb (laptop `~/envs/nemo-ctx/widebvh`, shipped as a git bundle) with cuBQL
e82f1dc at `/mnt/ffs24/home/khanmd/programs/widebvh-f32`, built by `build_widebvh.sbatch` into `build-nemo` (levels
0-3 + cart; this is also `WidebvhFMM`'s default `WIDEBVH_BUILD_DIR`). The 2026-09-23 runs used the same tree on scratch
(`/mnt/scratch/khanmd/widebvh-f32/build-h200-f32`). `env.sh` holds the common environment and writes a per-job
manifest; output CSVs and log directories are named by level (`$F32` = `f32l<L>`).

    sbatch slurm/h200_v3/build_widebvh.sbatch
    sbatch --dependency=afterok:<build> slurm/h200_v3/smoke.sbatch
    for j in fig11 fig12 twodrop capacity fig10b; do sbatch --dependency=afterok:<smoke> slurm/h200_v3/$j.sbatch; done

| job | output CSV | what |
|---|---|---|
| fig11 | `data/fig11_breakdown_h200_v3_f32l2.csv` | breakdown, N 10k..1M, 2b RPY / 2b NN / NeMO at switch 8 |
| fig12 | `data/fig12_scaling_h200_v3_f32l2.csv` | scaling, N 50k..2M |
| twodrop | `data/twodrop_1M_h200_v3_f32l2.csv`, `data/far_field_drift_1M_h200_v3_f32l2.csv` | §3.4 1M two-drop: 50-step aggregate + 150-step per-step |
| capacity | `data/max_particles_h200_v3_f32l2.csv` | max N (lattice, phi 0.1), cold + 3 warm with empty_cache |
| fig10b | `data/fig10_far_field_h200_v3_f32l2.csv` | far-field throughput / rel_far / rel_asym vs N, 5k..4M |

Raw data: `artifacts/logs/h200_v3_f32l2/<job>_<jobid>/` (every process log, manifest, slurm .out, CSV copy).

`twodrop_pc6` (2026-09-23): the two-drop job with `--near-op moments-v3-pc6` -- the same v3 stack, but the moments pair
correction only on pairs with d <= 6 (2b NN, diagonal and the far field's near cutoff stay at 8). A timing variant for
the HIGNN comparison, not a trained operating point. CSVs `data/{twodrop,far_field_drift}_1M_h200_v3_pc6_f32l2.csv`.

`fig11_pc6`, `fig12_pc6` (2026-09-30): Fig 11 / Fig 12 with `--near-op moments-v3-pc6` (the pc6 variant above; CSVs
`data/{fig11_breakdown,fig12_scaling}_h200_v3_pc6_f32l<L>.csv`). Fig 10b has no pc6 variant: it times the far field
alone, which depends only on the near cutoff (8 for both).

`fp32_levels` (2026-09-26): level 2 vs 3 A/B -- Fig 12 at 1M/2M (ABBA) and Fig 10b at 1M/2M/4M; results in
`artifacts/h200_v3_timing_report.md` ("fp32 level 3 on the H200"). CSVs `data/fp32_levels_fig{12,10b}_h200.csv`.
