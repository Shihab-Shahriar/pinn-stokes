#!/bin/bash
# Paper accuracy protocols (Fig 3 + Fig 4): MFS truths (Xfine, tol 1e-8) + the GPU yardsticks (mfs_coarse,
# the paper's GPU n-body operator), one volume fraction per array task.
#   sbatch --array=0-7 slurm/paper_truth.sbatch
# Truths are cached in tmp/nbody_moments_truth/ (resumable); rows go to data/paper_accuracy_v2/parts/.
#SBATCH --job-name=pav2truth
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
export PYTHONUNBUFFERED=1 TORCH_COMPILE_DISABLE=1
PHIS=(0.025 0.05 0.075 0.1 0.125 0.15 0.175 0.2)
PHI=${PHIS[${SLURM_ARRAY_TASK_ID:-0}]}
EXPS=${EXPS:-"fig3 fig4"}
echo "host $(hostname) job ${SLURM_JOB_ID:-?} task ${SLURM_ARRAY_TASK_ID:-0} phi $PHI exps $EXPS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python benchmarks/paper_accuracy_v2.py --exp $EXPS --phis "$PHI" --truth-only --gpu-ops --part --skip-done
echo "task ${SLURM_ARRAY_TASK_ID:-0} finished $(date)"
