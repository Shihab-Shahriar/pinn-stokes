#!/bin/bash
# H200 timing, current NeMO (v3 moments pair + pc8c diagonal, fp16 MLP, switch 8),
# widebvh far field at the fp32 level from env.sh. Figure 12: end-to-end scaling, N = 50k..2M. 50k..1.5M in one process (published protocol); 1.75M and 2M one process each (torch.compile recompile limit, artifacts/fig12_h200_f32l2_report.md).
#   sbatch slurm/h200_v3/fig12.sbatch      (from the repo root; see slurm/h200_v3/README.md)
#SBATCH --job-name=h200v3_fig12
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
# keep the slurm .out with the rest of the raw data, whatever happens
trap 'cp -f "slurm/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" "$RUN_DIR/" 2>/dev/null || true' EXIT
write_manifest
test -f "$WIDEBVH_BUILD_DIR/libwidebvh_nemo_f32l${NEMO_FAR_FP32_LEVEL}.so"
echo "host $(hostname) job ${SLURM_JOB_ID} sha $(git rev-parse --short HEAD) raw data -> $RUN_DIR"
CSV=data/fig12_scaling_h200_v3_${F32}.csv
for SIZES in 50000,100000,200000,500000,750000,1000000,1250000,1500000 1750000 2000000; do
  python benchmarks/figure12_grand_M.py --backend widebvh --near-op moments-v3 \
      --sizes "$SIZES" --csv "$CSV" 2>&1 | tee "$RUN_DIR/fig12_${SIZES//,/_}.log"
done
cp "$CSV" "$RUN_DIR/"
echo "finished $(date)"
