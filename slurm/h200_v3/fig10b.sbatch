#!/bin/bash
# H200 timing, current NeMO (v3 moments pair + pc8c diagonal, fp16 MLP, switch 8),
# widebvh far field at fp32 level 2. Figure 10b: far-field throughput / accuracy / symmetry vs N (5k..4M), near cutoff 8. Runs with TORCH_COMPILE_DISABLE=1 by design (the script sets it; it is an accuracy+symmetry panel). Generates the 4M config first if missing (recorded recipe, seed 0).
#   sbatch slurm/h200_v3/fig10b.sbatch      (from the repo root; see slurm/h200_v3/README.md)
#SBATCH --job-name=h200v3_fig10b
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
if [ ! -f tmp/uniform_large_0.1_4000000.csv ]; then
  python -c "import sys; sys.path.insert(0, '.'); from benchmarks.cluster import uniform_cluster_generation_large as g; g(0.1, 4_000_000, seed=0)" \
      2>&1 | tee "$RUN_DIR/gen_4M.log"
fi
md5sum tmp/uniform_large_0.1_*.csv > "$RUN_DIR/config_md5.txt"
python benchmarks/figure10_components.py --panel right --near-op moments-v3 \
    --far-csv data/fig10_far_field_h200_v3_f32l2.csv 2>&1 | tee "$RUN_DIR/fig10b.log"
cp data/fig10_far_field_h200_v3_f32l2.csv "$RUN_DIR/"
echo "finished $(date)"
