#!/bin/bash
# Figure 12 reproduction: end-to-end scaling of the full NeMO operator on the H200,
# widebvh far field, N = 50k..750k. benchmarks/figure12_grand_M.py runs all sizes in
# one process (5 shapes stays under torch.compile's recompile limit; this matches the
# published protocol exactly).
#   sbatch slurm/fig12_scaling.sbatch
# Submit from the root of the checkout to measure; the job runs in $SLURM_SUBMIT_DIR,
# so the measured sha is exactly what the CSV's git_sha column says.
#SBATCH --job-name=fig12
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
source ~/warp_env.sh
cd "${SLURM_SUBMIT_DIR:?submit from the repo root}"
# If the environment (e.g. warp_env.sh) pins another checkout on PYTHONPATH, an
# empty entry (= cwd) there makes the scripts' own sys.path.insert a no-op, and
# `src`/`benchmarks` silently bind to that other checkout. Prepend this one.
export PYTHONPATH="$(pwd -P):${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
# Performance run: torch.compile stays ENABLED (no TORCH_COMPILE_DISABLE).
echo "host $(hostname) job ${SLURM_JOB_ID:-?} sha $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python benchmarks/figure12_grand_M.py --backend widebvh \
    --csv data/fig12_scaling_h200_rerun.csv
echo "finished $(date)"
