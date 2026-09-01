# Reproducing the paper figures

All commands run from the repo root. Models ship in `data/models/` (git-tracked).
Accuracy runs want `TORCH_COMPILE_DISABLE=1`; performance runs (Fig 11/12) keep compile ON.
Figures 6, 7, 9, 10, 13 not written up here yet.

## Figure 2 — progressive accuracy of the learned stack

Needs the dataset-v2 shards in `data/multibody_v2/` (gitignored; truth comes from their stored grand-M).

```sh
TORCH_COMPILE_DISABLE=1 python figures/fig2_nbody_acc.py    # eval + figure
python figures/fig2_nbody_acc.py --plot-only                # re-render from its CSV
```

Out: `figures/fig2_nbody_acc.{pdf,png}` (+ per-config errors in `figures/fig2_nbody_acc.csv`).

## Figures 3 & 4 — accuracy vs φ (fixed N) and vs N

Needs the MFS truth cache `tmp/nbody_moments_truth/` (1,280 files). Without it, generate first:
`python benchmarks/paper_accuracy_v2.py --exp fig3 fig4 --truth-only` (CUDA GPU; hours on a laptop —
on the cluster use `sbatch --array=0-7 slurm/paper_truth.sbatch`).

```sh
TORCH_COMPILE_DISABLE=1 python benchmarks/paper_accuracy_v2.py --exp fig3 fig4 --workers 8 --skip-done
python benchmarks/paper_accuracy_v2.py --summary --figures
```

Rows accumulate in `data/paper_accuracy_v2.csv`.
Out: `figures/paper_v2_fig3_P{200,300}.*`, `figures/paper_v2_fig4_*.*`.
Optional: `--gpu-ops` adds mfs_coarse + the paper's GPU n-body operator (needs CUDA + warp).

## Figure 5 — opposing-force KDE deep dive

Truth is MFS Xfine (tol 1e-8) computed on the fly, so the full run needs a CUDA GPU with Triton
(no warp): ~1.5 h on the 4060 laptop, ~8 min per configuration (1× N=300 + 10× N=250, φ=0.10,
seeds 42–51 — the notebook protocol of `experiments/accuracy_deep_dive.ipynb`, operator swapped
to the moments-pc8 + learned-diagonal stack).

```sh
TORCH_COMPILE_DISABLE=1 python figures/fig5_deep_dive.py    # eval + figure
python figures/fig5_deep_dive.py --plot-only                # re-render from its NPZ
```

Out: `figures/fig5_deep_dive.{pdf,png}` (+ collected data in `figures/fig5_deep_dive_data.npz`),
and a drop-in copy at the paper's include name `figures/accuracy_deep_dive_fields_2x2.pdf`
(authored at 6.5×4.7 in — include at `width=\columnwidth` with no height override).
The run also prints the paper's summary number: the 5 %-trimmed mean per-particle relative RMSE
averaged over the 10 configurations (6.19 % for the shipped models).

## Figure 8 — falling particle cloud (N ≈ 3000 sedimentation storyboard)

Needs warp (GPU operator + treecode), so the simulation runs in the local docker image; ~13 min on
the 4060. Protocol is `experiments/single_drop_sedimentation_executed.ipynb` verbatim (R = 40,
φ = 0.048, seed 42, adaptive RKF45, T = 500) with the moments-pc8 + diagonal stack at switch 8 and
the widebvh far field (fp32 level 3).

```sh
bash docker/run_local.sh python figures/fig8_single_drop.py    # simulate + storyboard
python figures/fig8_single_drop.py --plot-only                 # re-render from its NPZ
python figures/fig8_single_drop.py --compare                   # consistency vs the published run
```

Out: `figures/fig8_single_drop_storyboard.{png,pdf}` (+ drop-in copies at the paper's include name
`figures/single_drop_storyboard_late.*`), data in `figures/fig8_single_drop_data.npz`,
`figures/fig8_consistency.png`. `--compare` parses the published run's statistics out of the
executed notebook; the write-up is `artifacts/fig8_moments_consistency.md`.

## Figure 11 — runtime breakdown on the H200 (two panels)

Measurement needs an H200. From the root of the checkout to measure, on the cluster:

```sh
sbatch slurm/fig11_breakdown.sbatch
# interactive GPU-node equivalent:
python benchmarks/figure11_breakdown.py --backend widebvh --csv data/fig11_breakdown_h200_rerun.csv
```

Keep the `*_rerun.csv` path — the default `data/fig11_breakdown_h200.csv` holds the published rows
and is rewritten in place. Render on the laptop (CSV only, no GPU):

```sh
python figures/grand_M_perf.py                                          # canonical, from the published CSV
python figures/fig11_compare.py . data/fig11_breakdown_h200_rerun.csv   # rerun figure + old-vs-new table
```

Out: `figures/runtime_summary_two_panel*.{png,pdf}`.

## Figure 12 — end-to-end scaling on the H200

Same pattern:

```sh
sbatch slurm/fig12_scaling.sbatch
# interactive GPU-node equivalent:
python benchmarks/figure12_grand_M.py --backend widebvh --csv data/fig12_scaling_h200_rerun.csv
```

Render on the laptop:

```sh
python figures/grand_M_perf.py                                          # canonical
python figures/fig12_compare.py . data/fig12_scaling_h200_rerun.csv     # rerun figure + old-vs-new table
```

Out: `figures/grand_M_scaling_test_h200*.{png,pdf}`.
