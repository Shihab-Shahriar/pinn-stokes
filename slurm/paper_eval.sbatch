#!/bin/bash
# Paper accuracy protocols: the CPU operators (paper + retrained) on the cached truths, one volume fraction per
# array task, 32 worker processes each (the GPU is requested only to keep the known-good request shape and is
# hidden from the workers).
#   sbatch --array=0-7 --dependency=afterok:<truth job id> slurm/paper_eval.sbatch
#   OPS="M_2b M_mom_v2_k10_rc6" sbatch --array=0-7 slurm/paper_eval.sbatch
#SBATCH --job-name=pav2eval
#SBATCH --account=hmakmm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=amd24
#SBATCH --time=2:59:00
#SBATCH --output=slurm/logs/%x_%A_%a.out
set -euo pipefail
source ~/warp_env.sh
cd /mnt/home/khanmd/pinn-stokes
export PYTHONUNBUFFERED=1 TORCH_COMPILE_DISABLE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PHIS=(0.025 0.05 0.075 0.1 0.125 0.15 0.175 0.2)
PHI=${PHIS[${SLURM_ARRAY_TASK_ID:-0}]}
EXPS=${EXPS:-"fig3 fig4"}
WORKERS=${WORKERS:-${SLURM_CPUS_PER_TASK:-32}}
OPS_ARG=""
if [ -n "${OPS:-}" ]; then OPS_ARG="--ops $OPS"; fi
echo "host $(hostname) job ${SLURM_JOB_ID:-?} task ${SLURM_ARRAY_TASK_ID:-0} phi $PHI exps $EXPS workers $WORKERS ops ${OPS:-default}"
CUDA_VISIBLE_DEVICES="" python benchmarks/paper_accuracy_v2.py --exp $EXPS --phis "$PHI" --workers "$WORKERS" \
    --part --skip-done $OPS_ARG
echo "task ${SLURM_ARRAY_TASK_ID:-0} finished $(date)"
