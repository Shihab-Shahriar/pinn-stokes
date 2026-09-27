#!/bin/bash
# H200 timing, current NeMO (v3 moments pair + pc8c diagonal, fp16 MLP, switch 8),
# Variant: moments pair correction only on pairs d <= 6 (2b NN + diagonal still to 8).
# widebvh far field at the fp32 level from env.sh. Section 3.4 / Fig 13: the 1M two-drop sedimentation run -- 50-step aggregate (two_suspensions_1M.py) and 150-step per-step drift (far_field_drift.py).
#   sbatch slurm/h200_v3/twodrop_pc6.sbatch      (from the repo root; see slurm/h200_v3/README.md)
#SBATCH --job-name=h200v3_twodrop_pc6
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
python benchmarks/two_suspensions_1M.py --near-op moments-v3-pc6 --fp32-level "$NEMO_FAR_FP32_LEVEL" \
    --t-final 0.5 --log-csv data/twodrop_1M_h200_v3_pc6_${F32}.csv 2>&1 | tee "$RUN_DIR/twodrop_50steps.log"
python benchmarks/far_field_drift.py --near-op moments-v3-pc6 --fp32-level "$NEMO_FAR_FP32_LEVEL" --steps 150 \
    --label widebvh-v3-pc6-${F32} --csv data/far_field_drift_1M_h200_v3_pc6_${F32}.csv \
    2>&1 | tee "$RUN_DIR/drift_150steps.log"
cp data/twodrop_1M_h200_v3_pc6_${F32}.csv data/far_field_drift_1M_h200_v3_pc6_${F32}.csv "$RUN_DIR/"
echo "finished $(date)"
