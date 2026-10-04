# Common environment for the RunPod jobs (cloud/jobs/*.sh), sourced by each.
# Runs inside the nemo image on the pod, in the code the driver uploaded (a
# `git archive` of HEAD at /workspace/pinn-stokes; this cloud/ tree comes from
# the driver's working copy). The analogue of slurm/h200_v3/env.sh.
#
# The driver sets NEMO_GPU_TAG (output names), NEMO_GIT_SHA and, optionally,
# NEMO_FAR_FP32_LEVEL (default 3, as everywhere).

# sshd does not hand the image's ENV (PATH to /opt/venv, the JIT cache dirs,
# HOME=/workspace) to its sessions; the bootstrap saved it. What the driver
# passed wins: the image's ENV carries its own baked NEMO_GIT_SHA (ac78eb4).
if [[ -f /workspace/.container_env ]]; then
    _driver_env="$(export -p | grep -E '^declare -x (NEMO_|TWODROP_)')"
    source /workspace/.container_env
    eval "$_driver_env"
    unset _driver_env
fi
cd /workspace/pinn-stokes
[[ -f GIT_SHA ]] && export NEMO_GIT_SHA="$(cat GIT_SHA)"
export PYTHONPATH="$PWD" PYTHONUNBUFFERED=1
# Performance runs: torch.compile ENABLED, no per-step nvidia-smi forks.
unset TORCH_COMPILE_DISABLE NEMO_DEVICE_MEM
export NEMO_FAR_FIELD=widebvh
export NEMO_FAR_FP32_LEVEL=${NEMO_FAR_FP32_LEVEL:-3}
F32=f32l${NEMO_FAR_FP32_LEVEL}
TAG=${NEMO_GPU_TAG:?the driver sets NEMO_GPU_TAG}
RESULTS=/workspace/results
mkdir -p "$RESULTS"

stage() { echo "=== stage $* $(date -Is)"; }

# widebvh from the vendored tree, for this GPU, only the library this level
# loads (one nemo_capi.cu compile, ~1-2 min). The image's own prebuilt
# /opt/widebvh/build-nemo is deliberately not used.
build_widebvh() {
    local cc lib=widebvh_nemo
    cc="$(nvidia-smi -i 0 --query-gpu=compute_cap --format=csv,noheader | tr -d '. ')"
    [[ $NEMO_FAR_FP32_LEVEL == 0 ]] || lib+=_f32l$NEMO_FAR_FP32_LEVEL
    CUDA_ARCHS=$cc WIDEBVH_TARGETS=$lib NINJA_JOBS=${NINJA_JOBS:-4} \
        bash extern/widebvh/build_nemo.sh > "$RESULTS/widebvh_build.log" 2>&1
    export WIDEBVH_BUILD_DIR="$PWD/extern/widebvh/build-sm$cc"
    test -f "$WIDEBVH_BUILD_DIR/lib$lib.so"
    echo "widebvh: $WIDEBVH_BUILD_DIR/lib$lib.so"
}

write_manifest() {
    {
        echo "date        $(date -Is)"
        echo "host        $(hostname)  pod ${RUNPOD_POD_ID:-local}  dc ${RUNPOD_DC_ID:-?}"
        echo "pinn-stokes ${NEMO_GIT_SHA:-?} (git archive of HEAD; cloud/ from the driver's working tree)"
        echo "widebvh     vendored extern/widebvh, build $WIDEBVH_BUILD_DIR"
        echo "fp32 level  $NEMO_FAR_FP32_LEVEL"
        echo "image       ${NEMO_IMAGE:-?} (baked NEMO_GIT_SHA of the image itself is not used)"
        echo "---- md5"
        md5sum "$WIDEBVH_BUILD_DIR"/lib*.so /workspace/cloud/pod_bootstrap.sh /workspace/cloud/jobs/*.sh
        echo "python      $(command -v python)"
        python -c "import torch, warp, triton; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'warp', warp.__version__, 'triton', triton.__version__)"
        echo "---- nvidia-smi"; nvidia-smi
        echo "---- nvidia-smi -q (clocks, power, performance)"; nvidia-smi -q -d CLOCK,POWER,PERFORMANCE
        echo "---- cpu / memory"; lscpu | grep -E '^(Model name|CPU\(s\))'; free -g
        echo "---- env"
        env | grep -E '^(NEMO|WIDEBVH|TC_|TORCH|PYTORCH|CUDA_VERSION|NVIDIA_DRIVER|RUNPOD_(POD_ID|GPU_COUNT|CPU_COUNT|DC_ID))' \
            | grep -viE 'KEY|TOKEN|SECRET|B64' | sort
        echo "---- pip freeze"; python -m pip freeze
    } > "$RESULTS/manifest.txt" 2>&1
}

# Whole-device memory sampler. The pod rents the whole GPU, so device-wide
# memory.used is this process plus its CUDA context -- the true footprint,
# including widebvh's raw cudaMalloc that torch's allocator statistics miss.
# (Per-PID accounting, NEMO_DEVICE_MEM=1, cannot match PIDs inside a container.)
# One long-lived nvidia-smi, not a fork per step; still only in its own process.
gpu_mem_idle_mb() { nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' '; }
mem_sampler_start() {
    nvidia-smi -i 0 --query-gpu=timestamp,memory.used,memory.total,utilization.gpu,clocks.sm,power.draw \
        --format=csv,nounits -lms 500 > "$1" 2>&1 &
    MEM_SAMPLER_PID=$!
}
mem_sampler_stop() { kill "$MEM_SAMPLER_PID" 2>/dev/null; wait "$MEM_SAMPLER_PID" 2>/dev/null || true; }
