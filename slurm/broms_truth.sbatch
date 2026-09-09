#!/bin/bash
# Large-N Fig 4 truths via the widebvh Broms MFS (sphere_mfs, treecode, no L_cut truncation),
# one volume fraction per array task, expensive cells first (resumable; existing files skipped).
#   sbatch --array=0-7 slurm/broms_truth.sbatch          # optionally NS="1500 2500 3000" etc.
# Needs a built sphere_mfs (default /mnt/ffs24/home/khanmd/programs/widebvh/build-rel/sphere_mfs;
# override with BROMS_SPHERE_MFS). Truths land in tmp/nbody_moments_truth/.
#SBATCH --job-name=bromstruth
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
PHIS=(0.025 0.05 0.075 0.1 0.125 0.15 0.175 0.2)
PHI=${PHIS[${SLURM_ARRAY_TASK_ID:-0}]}
NS=${NS:-""}                     # empty = the script's default N set for the chosen forcing
FORCING=${FORCING:-random}       # FORCING=gravity: full FIG4_N grid, _grav truth files (array 0,1,3,5)
echo "host $(hostname) job ${SLURM_JOB_ID:-?} task ${SLURM_ARRAY_TASK_ID:-0} phi $PHI forcing $FORCING N ${NS:-default}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python benchmarks/broms_truth.py --phis "$PHI" --forcing "$FORCING" ${NS:+--N $NS}
echo "task ${SLURM_ARRAY_TASK_ID:-0} finished $(date)"
