#!/bin/bash
# Verification job for the batched MFS solver + dataset v2 generator on an H200:
# tests against the fp64 references, throughput benchmark, and a short generation run of every family.
#SBATCH --job-name=mbv2verify
#SBATCH --account=hmakmm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=amd24
#SBATCH --time=0:45:00
#SBATCH --output=slurm/logs/%x_%j.out
set -uo pipefail
source ~/warp_env.sh
cd /mnt/home/khanmd/pinn-stokes
export PYTHONUNBUFFERED=1
echo "host $(hostname) job ${SLURM_JOB_ID:-?} $(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c "import torch, triton; print('torch', torch.__version__, 'triton', triton.__version__)"
echo "=== tests"
python -m pytest tests/test_mfs_batched.py -q -x --timeout=1500 2>/dev/null || python -m pytest tests/test_mfs_batched.py -q -x
echo "=== benchmark (torch64 production backend, then triton32)"
python benchmarks/bench_mfs_batched.py --backends torch64 --P 8 16 32 64 --mem-budget-gb 40 --csv data/mfs_batched_perf_h200.csv
python benchmarks/bench_mfs_batched.py --backends triton32 --P 16 32 64 --mem-budget-gb 40 --csv data/mfs_batched_perf_h200.csv
echo "=== short generation run (all families, small)"
rm -rf data/multibody_v2_verify
python src/create_dataset_multibody_v2.py --family grown --delta 0.1 --P 8 --n-configs 32 --backend torch64 --mem-budget-gb 40 --out data/multibody_v2_verify
python src/create_dataset_multibody_v2.py --family uniform --phi 0.1 --P 64 --n-configs 4 --backend torch64 --mem-budget-gb 40 --out data/multibody_v2_verify
python src/create_dataset_multibody_v2.py --family lattice --phi 0.1 --jitter 0.1 --P 32 --n-configs 4 --backend torch64 --mem-budget-gb 40 --out data/multibody_v2_verify
python - <<'PY'
import sys; sys.path.insert(0, '.')
from src.nbody_features import load_multibody_v2, pair_rows_from_M
import numpy as np
d = load_multibody_v2(roots=("data/multibody_v2_verify",), check=True)
rows = pair_rows_from_M(d, np.random.default_rng(0))
print("verify dataset: configs", len(d["positions"]), "ordered pairs", len(d["dist"]), "Y shape", rows["Y"].shape)
PY
cat data/multibody_v2_verify/manifest_worker0.csv
echo "=== done $(date)"
