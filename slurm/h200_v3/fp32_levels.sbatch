#!/bin/bash
# H200 A/B of the widebvh far-field fp32 level, 2 (the level every H200 run used so far) vs 3 (+ fp32
# upward pass, never measured on the H200 before 2026-09-26). Current NeMO (v3 moments + pc8c diagonal,
# switch 8). One process per (script, level):
#   * Fig 12 end-to-end apply at 1M and 2M, torch.compile on, levels in ABBA order (shared nodes drift);
#   * Fig 10b far-field time / rel_far / rel_asym at 1M, 2M, 4M.
#   sbatch --dependency=afterok:<build> slurm/h200_v3/fp32_levels.sbatch      (from the repo root)
#SBATCH --job-name=h200v3_fp32lv
#SBATCH --account=hmakmm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=amd24
#SBATCH --time=2:59:00
#SBATCH --output=slurm/logs/%x_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repo root}"
source slurm/h200_v3/env.sh
trap 'cp -f "slurm/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" "$RUN_DIR/" 2>/dev/null || true' EXIT
write_manifest
for L in 2 3; do test -f "$WIDEBVH_BUILD_DIR/libwidebvh_nemo_f32l$L.so"; done
echo "host $(hostname) job ${SLURM_JOB_ID} sha $(git rev-parse --short HEAD) raw data -> $RUN_DIR"
F12=data/fp32_levels_fig12_h200.csv
F10=data/fp32_levels_fig10b_h200.csv
i=0
for L in 2 3 3 2; do
  i=$((i + 1))
  NEMO_FAR_FP32_LEVEL=$L python benchmarks/figure12_grand_M.py --backend widebvh --near-op moments-v3 \
      --sizes 1000000,2000000 --csv "$F12" 2>&1 | tee "$RUN_DIR/fig12_${i}_l$L.log"
done
for L in 2 3; do
  NEMO_FAR_FP32_LEVEL=$L python benchmarks/figure10_components.py --panel right --near-op moments-v3 \
      --sizes 1000000,2000000,4000000 --far-csv "$F10" 2>&1 | tee "$RUN_DIR/fig10b_l$L.log"
done
cp "$F12" "$F10" "$RUN_DIR/"
echo "finished $(date)"
