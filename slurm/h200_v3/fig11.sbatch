#!/bin/bash
# H200 timing, current NeMO (v3 moments pair + pc8c diagonal, fp16 MLP, switch 8),
# widebvh far field at fp32 level 2. Figure 11: runtime breakdown, N = 10k..1M, operators 2b RPY / 2b NN / NeMO, all at switch 8, one subprocess per (N, operator).
#   sbatch slurm/h200_v3/fig11.sbatch      (from the repo root; see slurm/h200_v3/README.md)
#SBATCH --job-name=h200v3_fig11
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
test -f "$WIDEBVH_BUILD_DIR/libwidebvh_nemo_f32l2.so"
echo "host $(hostname) job ${SLURM_JOB_ID} sha $(git rev-parse --short HEAD) raw data -> $RUN_DIR"
python benchmarks/figure11_breakdown.py --backend widebvh --near-op moments-v3 \
    --csv data/fig11_breakdown_h200_v3_f32l2.csv 2>&1 | tee "$RUN_DIR/fig11.log"
cp data/fig11_breakdown_h200_v3_f32l2.csv "$RUN_DIR/"
echo "finished $(date)"
