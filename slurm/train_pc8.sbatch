#!/bin/bash
# pair_cutoff-8 moments training on an H200: build the pc8 cache from the local v2 shards (fast), then
# train + publish nbody_moments_v2_kinf_rc8_pc8.  sbatch slurm/train_pc8.sbatch
#SBATCH --job-name=pc8train
#SBATCH --account=hmakmm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint=amd24
#SBATCH --time=1:59:00
#SBATCH --output=slurm/logs/%x_%j.out
set -euo pipefail
source ~/warp_env.sh
cd /mnt/home/khanmd/pinn-stokes
export PYTHONUNBUFFERED=1 TORCH_COMPILE_DISABLE=1
CACHE=data/multibody_v2_cache_pc8
if [ ! -f "$CACHE/meta.json" ]; then
    python experiments/build_nbody_v2_cache.py --out "$CACHE" --pair-cutoff 8 --variants kinf_rc8 --workers 8
fi
python experiments/train_nbody_v2.py --model moments --variant kinf_rc8 --cache "$CACHE" \
    --out experiments/runs_v2/mom_kinf_rc8_pc8 --publish --publish-name nbody_moments_v2_kinf_rc8_pc8
echo "done $(date)"
