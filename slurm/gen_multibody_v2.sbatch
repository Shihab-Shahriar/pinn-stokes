#!/bin/bash
# n-body dataset v2 generation on the HPCC H200 nodes (one GPU per array task = one generator worker).
#   sbatch --array=0-3 slurm/gen_multibody_v2.sbatch                # 4 workers, 2.5 h budget each
#   BUDGET=3600 ROUNDS=1 sbatch --array=0-1 slurm/gen_multibody_v2.sbatch
# Resumable: re-submitting skips finished shards (per-worker manifests under data/multibody_v2/).
#SBATCH --job-name=mbv2gen
#SBATCH --account=hmakmm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=amd24
#SBATCH --time=2:59:00
#SBATCH --output=slurm/logs/%x_%A_%a.out
set -euo pipefail
source ~/warp_env.sh
cd /mnt/home/khanmd/pinn-stokes
export PYTHONUNBUFFERED=1
NUM_WORKERS=${NUM_WORKERS:-${SLURM_ARRAY_TASK_COUNT:-1}}
WORKER=${SLURM_ARRAY_TASK_ID:-0}
BUDGET=${BUDGET:-9000}          # seconds; leaves ~30 min of the 2:59 wall time for the last shard
ROUNDS=${ROUNDS:-2}             # rounds of the default plan (1 round ~ 27k configs ~ 2.3 GB fp32)
ACC=${ACC:-fine}
echo "host $(hostname) job ${SLURM_JOB_ID:-?} worker $WORKER/$NUM_WORKERS budget ${BUDGET}s rounds $ROUNDS acc $ACC"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python src/create_dataset_multibody_v2.py --plan default --acc "$ACC" --backend torch64 --mem-budget-gb 40 \
    --time-budget "$BUDGET" --max-rounds "$ROUNDS" --num-workers "$NUM_WORKERS" --worker "$WORKER" \
    --out data/multibody_v2
echo "worker $WORKER finished $(date)"
