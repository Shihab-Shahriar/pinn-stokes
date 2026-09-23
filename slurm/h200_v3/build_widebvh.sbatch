#!/bin/bash
# Build the widebvh NeMO far-field libraries with the fp32 fast path (level 2)
# for the H200 (sm_90), from the 03efcdb tree at /mnt/scratch/khanmd/widebvh-f32.
#   sbatch slurm/h200_v3/build_widebvh.sbatch      (from the repo root)
#SBATCH --job-name=h200v3_build
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
write_manifest
# Build with widebvh's own toolchain (its env.sh: GCC 14.3, CUDA 12.9.1, CMake,
# Ninja -- what the production build-nemo libraries were built with), in a
# subshell so the python environment above is untouched.
(
  cd "$WIDEBVH_SRC"
  set +u; source ./env.sh; set -u
  cmake -S . -B "$WIDEBVH_BUILD_DIR" -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=90 -DWIDEBVH_NEMO_FP32_LEVELS="2"
  cmake --build "$WIDEBVH_BUILD_DIR" -j 32 --target widebvh_nemo widebvh_nemo_f32l2
) 2>&1 | tee "$RUN_DIR/build.log"
ls -la "$WIDEBVH_BUILD_DIR"/libwidebvh_nemo*.so
echo "finished $(date)"
