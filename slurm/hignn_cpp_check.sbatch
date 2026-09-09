#!/bin/bash
# HIGNN C++/Kokkos engine cross-check inside the authors' CPU Singularity image (best-effort; the numbers go
# into artifacts/hignn_comparison_report.md). Waits for the image pull + recursive clone started by hand on the
# dev node (/mnt/scratch/khanmd/hignn_setup.log), builds their python module with their CI recipe, then runs
# benchmarks/hignn_cpp_check.py --run on a handful of our gravity truths.
#   sbatch slurm/hignn_cpp_check.sbatch          # then: rsync tmp/hignn_cpp_check.npz back and --compare locally
#SBATCH --job-name=hignn_cpp
#SBATCH --account=hmakmm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=1:30:00
#SBATCH --output=slurm/logs/%x_%j.out
set -euo pipefail
SCR=/mnt/scratch/khanmd
SIF=$SCR/hignn_cpu.sif
SRC=$SCR/hignn
TRUTH=/mnt/home/khanmd/pinn-stokes/tmp/nbody_moments_truth
cd /mnt/home/khanmd/pinn-stokes
echo "host $(hostname) job ${SLURM_JOB_ID:-?} $(date)"
for i in $(seq 1 100); do
    if [ -f "$SIF" ] && [ -f "$SRC/extern/pybind11/CMakeLists.txt" ] && grep -q "^cloned" $SCR/hignn_setup.log 2>/dev/null; then break; fi
    echo "waiting for image / clone ($i)"; sleep 30
done
[ -f "$SIF" ] || { echo "no image at $SIF"; exit 1; }
cp benchmarks/hignn_cpp_check.py "$SRC/"
export TERM=xterm
if [ ! -f "$SRC/python/hignn.so" ]; then
    echo "--- build (their CI recipe) $(date)"
    singularity exec --pwd /workspace -B "$SRC:/workspace" "$SIF" python3 python/init.py --rebuild 2>&1 | tail -60
fi
ls -la "$SRC/python/hignn.so"
FILES=""
for f in uniform_N200_phi0.1_seed4423_grav uniform_N200_phi0.1_seed4424_grav uniform_N200_phi0.025_seed1423_grav \
         uniform_N1000_phi0.05_seed2723_grav uniform_N3000_phi0.1_seed5123_grav; do
    FILES="$FILES /truth/$f.npz"
done
# --cleanenv: the image's OpenMPI (no SLURM PMI support) must not see SLURM_*/PMI* and must init as a singleton
RUN="singularity exec --cleanenv --env OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16},TERM=xterm,OMPI_MCA_ess_singleton_isolated=1 \
    --pwd /workspace -B $SRC:/workspace -B $TRUTH:/truth $SIF python3 hignn_cpp_check.py --run --hignn-root /workspace"
echo "--- run N <= 3000 $(date)"
$RUN --out /workspace/cpp_check.npz $FILES || echo "N<=3000 run exited $? (partial results saved per case)"
cp "$SRC/cpp_check.npz" tmp/hignn_cpp_check.npz
echo "--- run N = 10000 (their engine segfaulted here on the first attempt; separate invocation) $(date)"
$RUN --out /workspace/cpp_check_10k.npz /truth/uniform_N10000_phi0.1_seed5423_grav.npz || echo "N=10000 run exited $?"
[ -f "$SRC/cpp_check_10k.npz" ] && cp "$SRC/cpp_check_10k.npz" tmp/hignn_cpp_check_10k.npz
echo "done $(date)"
