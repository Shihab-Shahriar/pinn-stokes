# Common environment for the H200 v3 timing jobs (slurm/h200_v3/*.sbatch).
# Sourced from the repo root inside a job. Operator under test: NeMO v3 moments
# pair model + pc8c learned diagonal (fp16 MLP, switch 8), widebvh far field at
# fp32 level $NEMO_FAR_FP32_LEVEL (pdeg 7, mac 0.8, leaf 1024). The 2026-09-23 runs used
# level 2; level 3 is faster on the H200 at the same accuracy (2026-09-26,
# fp32_levels.sbatch), so it is the default from then on. Output names carry $F32.
source ~/warp_env.sh
# warp_env.sh pins the home checkout on PYTHONPATH (plus an empty entry = cwd):
# prepend this checkout or src/benchmarks silently import the other tree.
export PYTHONPATH="$(pwd -P):${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
# widebvh 03efcdb (the fp32 engine), permanent copy; the 2026-09-23 runs used the same
# tree on scratch (/mnt/scratch/khanmd/widebvh-f32/build-h200-f32).
export WIDEBVH_SRC=/mnt/ffs24/home/khanmd/programs/widebvh-f32
export WIDEBVH_BUILD_DIR=$WIDEBVH_SRC/build-nemo
export NEMO_FAR_FP32_LEVEL=${NEMO_FAR_FP32_LEVEL:-3}
F32=f32l${NEMO_FAR_FP32_LEVEL}
# Performance runs: torch.compile ENABLED, no per-step nvidia-smi forks.
unset TORCH_COMPILE_DISABLE NEMO_DEVICE_MEM

# Raw-data directory for this job: every process log, the manifest and the
# slurm .out end up here (artifacts/logs is git-tracked).
RUN_DIR=artifacts/logs/h200_v3_${F32}/${SLURM_JOB_NAME:-job}_${SLURM_JOB_ID:-local}
mkdir -p "$RUN_DIR"

write_manifest() {
  {
    echo "date        $(date -Is)"
    echo "host        $(hostname)"
    echo "job         ${SLURM_JOB_ID:-?} ${SLURM_JOB_NAME:-?}"
    echo "pinn-stokes $(git rev-parse HEAD) $(git status --porcelain --untracked-files=no | wc -l) modified tracked files"
    echo "widebvh     $(git -C "$WIDEBVH_SRC" rev-parse HEAD)  build $WIDEBVH_BUILD_DIR"
    echo "fp32 level  $NEMO_FAR_FP32_LEVEL"
    echo "python      $(which python)"
    python -c "import torch, warp; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'warp', warp.__version__)" 2>/dev/null
    echo "---- nvidia-smi"; nvidia-smi
    echo "---- env"; env | grep -E '^(NEMO|WIDEBVH|TC_|TORCH|PYTORCH|CUDA|SLURM_JOB|PYTHONPATH)' | sort
    echo "---- conda list"; conda list 2>/dev/null
  } > "$RUN_DIR/manifest.txt" 2>&1
}
