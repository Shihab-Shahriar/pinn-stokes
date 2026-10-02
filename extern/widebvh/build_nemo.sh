#!/usr/bin/env bash
#
# Build the NeMO far-field libraries (libwidebvh_nemo*.so) from this tree for the
# GPU in this machine. Each GPU architecture gets its own build dir,
# build-sm<compute capability> (build-sm89 for an RTX 40xx, build-sm90 for an
# H100/H200, ...), and src/treecode_widebvh.py loads the one matching the GPU it
# runs on -- so on a new kind of GPU, run this once more; machines with
# different GPUs can share one checkout.
#
#   bash extern/widebvh/build_nemo.sh                              # natively
#   bash docker/run_local.sh bash extern/widebvh/build_nemo.sh     # inside the NeMO image
#   CUDA_ARCHS=90 bash extern/widebvh/build_nemo.sh                # no GPU visible (login node)
#   CUDA_ARCHS=89 bash extern/widebvh/build_nemo.sh <dir>          # elsewhere; then export WIDEBVH_BUILD_DIR=<dir>
#
# Natively it needs CUDA 12.x (12.8+ for sm_120), CMake >= 3.18, a C++17 compiler
# with OpenMP, and pkg-config; the image has all of them.
#
# Default targets: the fp64 library, the fp32 levels 1-3 (level 3 is what
# WidebvhFMM runs by default) and the Cartesian A/B policy, all at PDEG 7.
# About 5 min per architecture on a laptop.
#
# Environment knobs:
#   CUDA_ARCHS                 compute capabilities to build, e.g. "90" or "86;89"
#                              (default: those of the GPUs nvidia-smi lists), one
#                              build dir each
#   WIDEBVH_NEMO_PDEG          e.g. "5;7" adds the libwidebvh_nemo_p5* variants (default "7")
#   WIDEBVH_NEMO_FP32_LEVELS   default "1;2;3"; "" builds the fp64 library only
#   WIDEBVH_TARGETS            explicit target list, overriding the two above
#   NINJA_JOBS                 parallel compiles (default 3). Each nemo_capi.cu compile
#                              takes several GB; 3 fits a 16 GB host.
#
# PETSc is optional: without it CMake warns and skips only the MFS executables
# (sphere_mfs, for benchmarks/broms_truth.py), never the NeMO libraries.

set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${CUDA_ARCHS:-}" ]]; then
    command -v nvidia-smi >/dev/null || { echo "no nvidia-smi here: name the architecture, e.g. CUDA_ARCHS=90" >&2; exit 1; }
    CUDA_ARCHS="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d '. ' | sort -u | paste -sd';')"
    [[ -n "$CUDA_ARCHS" ]] || { echo "nvidia-smi lists no GPU: set CUDA_ARCHS" >&2; exit 1; }
fi
read -ra archs <<<"${CUDA_ARCHS//;/ }"
for a in "${archs[@]}"; do
    [[ $a =~ ^[0-9]+$ ]] || { echo "CUDA_ARCHS takes compute capabilities like 89 or 90, not '$a'" >&2; exit 1; }
done
if [[ $# -ge 1 && ${#archs[@]} -ne 1 ]]; then
    echo "an explicit build dir takes exactly one architecture (CUDA_ARCHS=...)" >&2; exit 1
fi
PDEG="${WIDEBVH_NEMO_PDEG:-7}"
LEVELS="${WIDEBVH_NEMO_FP32_LEVELS-1;2;3}"

# Target names mirror add_widebvh_nemo_library in CMakeLists.txt (and
# library_name in src/treecode_widebvh.py): PDEG 7 keeps the plain name, other
# degrees get _p<N>, fp32 levels append _f32l<L>.
if [[ -z "${WIDEBVH_TARGETS:-}" ]]; then
    WIDEBVH_TARGETS="widebvh_nemo_cart"
    for p in ${PDEG//;/ }; do
        base=widebvh_nemo; [[ $p == 7 ]] || base=widebvh_nemo_p$p
        WIDEBVH_TARGETS+=" $base"
        for l in ${LEVELS//;/ }; do WIDEBVH_TARGETS+=" ${base}_f32l$l"; done
    done
fi

generator=()
command -v ninja >/dev/null && generator=(-G Ninja)

for arch in "${archs[@]}"; do
    BUILD="${1:-$SRC/build-sm$arch}"
    echo "widebvh: sm_$arch -> $BUILD"
    echo "         targets: $WIDEBVH_TARGETS"
    # The same dir is /workspace/pinn-stokes/... in the image and the checkout
    # path on the host, and CMake refuses a cache made under the other one.
    if [[ -f "$BUILD/CMakeCache.txt" ]] &&
       ! grep -qxF "CMAKE_CACHEFILE_DIR:INTERNAL=$BUILD" "$BUILD/CMakeCache.txt"; then
        echo "         configured under another path (image vs host): reconfiguring from scratch"
        rm -rf "$BUILD/CMakeCache.txt" "$BUILD/CMakeFiles"
    fi
    # WIDEBVH_CPU_NATIVE=OFF: -march=native only matters for the CPU executables,
    # and would tie a build dir shared with the image to the host that made it.
    # RPATH_USE_ORIGIN: the libraries load libcuBQL_cuda_float3.so through their
    # RUNPATH, otherwise the absolute build path -- so a build made in the image
    # would not load on the host, or vice versa.
    cmake -S "$SRC" -B "$BUILD" "${generator[@]}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON \
        -DCMAKE_CUDA_ARCHITECTURES="$arch" \
        -DWIDEBVH_NEMO_PDEG="$PDEG" \
        -DWIDEBVH_NEMO_FP32_LEVELS="$LEVELS" \
        -DWIDEBVH_CPU_NATIVE=OFF
    # shellcheck disable=SC2086
    cmake --build "$BUILD" -j"${NINJA_JOBS:-3}" --target $WIDEBVH_TARGETS
    ls -la "$BUILD"/libwidebvh_nemo*.so 2>/dev/null || true
done
if [[ $# -ge 1 ]]; then
    echo "not the default location: export WIDEBVH_BUILD_DIR=$1"
fi
