# NeMO — reproducible image for the two remaining paper measurements.
#
#   RTX 5090 (sm_120)  Figure 12    benchmarks/figure12_grand_M.py --backend widebvh
#   RTX A4500 (sm_86)  Figure 13    benchmarks/two_suspensions_1M.py
#
# Build context is produced by docker/pack_context.sh on the cluster; it copies
# the widebvh and pinn-stokes WORKING TREES, because neither is reproducible
# from its git remote (see that script's header).
#
#   docker build --build-arg NEMO_GIT_SHA=$(cat GIT_SHA) -t <user>/nemo:1.0 .   # fp64 only
#   docker build --build-arg NEMO_GIT_SHA=$(cat GIT_SHA) \
#       --build-arg WIDEBVH_FP32_LEVELS="1;2;3" -t <user>/nemo:2.0 .        # + fp32 far field
#
# CUDA 12.8 is not incidental: sm_120 (Blackwell / RTX 5090) does not exist
# before it, and warp-lang's official wheel for this release is built with 12.8,
# so it already carries sm_120. The container ships its own toolkit, so it does
# not matter that the cluster's module is 12.6 — only the host driver matters
# (>= 570 for a 5090, >= 525 elsewhere via CUDA 12.x minor-version compat).

FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG NEMO_GIT_SHA=unknown

# Ampere (A4500=86) .. Hopper (H200=90) .. Blackwell (5090=120), plus 120-virtual
# PTX so a future architecture can JIT rather than fail to load.
ARG CUDA_ARCHS="75-real;80-real;86-real;89-real;90-real;120-real;120-virtual"

# 1 rebuilds the patched Warp fork so `--backend warp` works. Off by default:
# see docker/pack_context.sh --with-warp-fork.
ARG WITH_WARP_FORK=0

# nvcc compiles its -gencode targets SEQUENTIALLY within one invocation, so the
# 7-architecture list above costs ~7x a single-arch build (measured: 6m26s for
# sm_120 alone, 3 TUs, -j16). `-t` parallelizes them. Peak process count is
# roughly (ninja -j) x NVCC_THREADS, so lower this on a small build machine and
# raise it toward 0 (= one per CPU) on a big one.
ARG NVCC_THREADS=4

# Concurrent nvcc invocations. Peak cicc process count is roughly NINJA_JOBS x
# NVCC_THREADS; on a 15 GB host that product must stay small or the host OOMs
# during nemo_capi.cu. Empty = $(nproc), the original behaviour.
ARG NINJA_JOBS=

# fp32 fast-path levels of the widebvh far field to build IN ADDITION to the fp64
# production library (widebvh WIDEBVH_NEMO_FP32_LEVELS; "1;2;3" gives
# libwidebvh_nemo_f32l{1,2,3}.so, selected at runtime with WidebvhFMM(fp32_level=)
# / two_suspensions_1M.py --fp32-level). Needed on GeForce-class GPUs, where fp64
# issues at 1/64 the fp32 rate and the fp64 far field is ~20x slower than it needs
# to be (artifacts/consumer_gpu_far_field_report.md). Empty = fp64 only, the
# image that produced the H200 numbers. Each level is another compile of
# nemo_capi.cu per architecture, so keep CUDA_ARCHS short when you set it.
ARG WIDEBVH_FP32_LEVELS=""
# Extra PDEG variants ("5" adds libwidebvh_nemo_p5*.so for the pdeg sweep).
ARG WIDEBVH_EXTRA_PDEG=""

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev python3-pip \
        build-essential pkg-config git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------------- python env --
# A venv rather than conda: ~40% of the cluster's conda env is vtk/qt6/pyside6/
# mesalib, and the production import chain here is only numpy, pandas, scipy,
# matplotlib, torch, triton and warp.
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# CMake from pip, not apt: Ubuntu 24.04 ships 3.28 and widebvh is developed
# against 4.0.3.
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
 && pip install --no-cache-dir "cmake>=4.0" ninja

# Pinned to what actually produced the H200 numbers in data/fig12_scaling_h200.csv
# and data/far_field_drift_1M.csv. Note numpy/scipy are the versions the cluster
# interpreter really imports (a ~/.local user-site shadow), not the newer ones
# `conda env export` reports there.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.8.0
# seaborn is a dead import (src/analysis_utils.py:3 never calls sns.*), but it sits
# on the import chain of BOTH deliverables: figure12_grand_M.py -> performance_grand_M
# -> src/mob_op_2b_combined.py -> src/analysis_utils.py. Installed in this same
# command so the resolver honours the pins above rather than upgrading them.
RUN pip install --no-cache-dir \
        warp-lang==1.12.0 \
        numpy==2.3.3 scipy==1.16.2 pandas==2.3.3 matplotlib==3.10.8 \
        seaborn==0.13.2

# ------------------------------------------------------------------ widebvh --
# Rebuilt here rather than shipped prebuilt: the .so on the cluster has an RPATH
# baked to /mnt/ffs24/.../build-nemo/cuBQL/cuBQL, which would dangle, and it
# carries only sm_80/sm_90 — no A4500 and no 5090.
#
# WIDEBVH_CPU_NATIVE defaults ON (-march=native) and would bake this build host's
# CPU into the image. PETSc/OpenBLAS are absent, which downgrades to a CMake
# warning and skips the MFS executables only; the widebvh_nemo* targets are
# unaffected. p3/p5 exist for degree sweeps neither experiment runs, and each
# extra target is another 7 architecture compilations.
COPY widebvh/ /opt/widebvh/
RUN cmake -S /opt/widebvh -B /opt/widebvh/build-nemo -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DWIDEBVH_NEMO_PDEG="7${WIDEBVH_EXTRA_PDEG:+;$WIDEBVH_EXTRA_PDEG}" \
        -DWIDEBVH_NEMO_FP32_LEVELS="${WIDEBVH_FP32_LEVELS}" \
        -DWIDEBVH_CPU_NATIVE=OFF \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
        -DCMAKE_CUDA_FLAGS="-t${NVCC_THREADS}" \
 && targets="widebvh_nemo widebvh_nemo_cart" \
 && pdegs="$(echo "$WIDEBVH_EXTRA_PDEG" | tr ';' ' ')" \
 && levels="$(echo "$WIDEBVH_FP32_LEVELS" | tr ';' ' ')" \
 && for p in $pdegs; do targets="$targets widebvh_nemo_p$p"; done \
 && for l in $levels; do targets="$targets widebvh_nemo_f32l$l"; \
        for p in $pdegs; do targets="$targets widebvh_nemo_p${p}_f32l$l"; done; done \
 && echo "widebvh targets: $targets" \
 && cmake --build /opt/widebvh/build-nemo -j"${NINJA_JOBS:-$(nproc)}" \
        --target $targets \
 && find /opt/widebvh/build-nemo -name '*.o' -delete \
 && ls -la /opt/widebvh/build-nemo/*.so /opt/widebvh/build-nemo/cuBQL/cuBQL/*.so

# --------------------------------------------------- optional: patched Warp --
# pack_context.sh always creates warp/ (a .keep file when the fork was not
# requested) so this COPY never fails.
COPY warp/ /opt/warp/
RUN if [ "$WITH_WARP_FORK" = "1" ]; then \
        set -eux; \
        cd /opt/warp; \
        python build_lib.py --cuda_path=/usr/local/cuda --no_libmathdx -j"$(nproc)"; \
        pip install --no-cache-dir --no-deps -e /opt/warp; \
    else \
        rm -rf /opt/warp; \
    fi

# --------------------------------------------------------------------- repo --
COPY pinn-stokes/ /workspace/pinn-stokes/
RUN install -m 0755 /workspace/pinn-stokes/docker/entrypoint.sh /usr/local/bin/entrypoint.sh

ENV WIDEBVH_BUILD_DIR=/opt/widebvh/build-nemo \
    NEMO_FAR_FIELD=widebvh \
    NEMO_GIT_SHA=${NEMO_GIT_SHA} \
    PYTHONPATH=/workspace/pinn-stokes \
    PYTHONNOUSERSITE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    HOME=/workspace \
    TMPDIR=/workspace/.cache/tmp \
    XDG_CACHE_HOME=/workspace/.cache \
    MPLCONFIGDIR=/workspace/.cache/matplotlib \
    TORCHINDUCTOR_CACHE_DIR=/workspace/.cache/inductor \
    TRITON_CACHE_DIR=/workspace/.cache/triton \
    WARP_CACHE_PATH=/workspace/.cache/warp

WORKDIR /workspace/pinn-stokes

# No import check here, deliberately: src/treecode.py runs wp.get_stream("cuda:0")
# at module scope, so nothing in src/ can be imported without a GPU, and
# `docker build` has none. Validation happens in `docker run --gpus all`.
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["python", "docker/smoke_test.py"]
