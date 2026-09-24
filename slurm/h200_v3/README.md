# H200 timing runs for the current NeMO (v3 moments + pc8c diagonal)

Operator: `Mob_Nbody_Moments_Torch` with `experiments/nbody_moments_v3_nb8lin_tr2_kinf_rc8_pc8c.wt` +
`experiments/nbody_diag_v2_pc8c.wt` (fp16 MLP, warp backend, switch 8), inside `WidebvhFMM` (bary, pdeg 7,
mac 0.8, leaf 1024) at **fp32 level 2**, near cutoff 8. torch.compile enabled (except Fig 10b, an
accuracy/symmetry panel that disables it by design).

Setup (once): a checkout of this repo on scratch, `tmp/uniform_large_0.1_<N>.csv` for N = 5k..2M, and the
widebvh fp32 build: widebvh 03efcdb (laptop `~/envs/nemo-ctx/widebvh`, shipped as a git bundle) at
`/mnt/scratch/khanmd/widebvh-f32` with cuBQL e82f1dc, built by `build_widebvh.sbatch` into `build-h200-f32`.
`env.sh` holds the common environment and writes a per-job manifest.

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
