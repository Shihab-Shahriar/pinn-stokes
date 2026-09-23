# Common environment for the H200 v3 timing jobs (slurm/h200_v3/*.sbatch).
# Sourced from the repo root inside a job. Operator under test: NeMO v3 moments
# pair model + pc8c learned diagonal (fp16 MLP, switch 8), widebvh far field at
# fp32 level 2 (pdeg 7, mac 0.8, leaf 1024).
source ~/warp_env.sh
# warp_env.sh pins the home checkout on PYTHONPATH (plus an empty entry = cwd):
# prepend this checkout or src/benchmarks silently import the other tree.
export PYTHONPATH="$(pwd -P):${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export WIDEBVH_SRC=/mnt/scratch/khanmd/widebvh-f32
export WIDEBVH_BUILD_DIR=$WIDEBVH_SRC/build-h200-f32
export NEMO_FAR_FP32_LEVEL=2
# Performance runs: torch.compile ENABLED, no per-step nvidia-smi forks.
unset TORCH_COMPILE_DISABLE NEMO_DEVICE_MEM

# Raw-data directory for this job: every process log, the manifest and the
# slurm .out end up here (artifacts/logs is git-tracked).
RUN_DIR=artifacts/logs/h200_v3_f32l2/${SLURM_JOB_NAME:-job}_${SLURM_JOB_ID:-local}
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
