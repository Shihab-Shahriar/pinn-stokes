#!/usr/bin/env bash
# Source this on the cluster before configuring/building:
#   source env.sh

if ! command -v module >/dev/null 2>&1; then
  if [ -f /etc/profile.d/modules.sh ]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh
  elif [ -f /usr/share/Modules/init/bash ]; then
    # shellcheck disable=SC1091
    source /usr/share/Modules/init/bash
  fi
fi

if ! command -v module >/dev/null 2>&1; then
  echo "env.sh: environment modules are not available in this shell" >&2
  return 1 2>/dev/null || exit 1
fi

if [ "${WIDEBVH_SKIP_MODULE_PURGE:-0}" != "1" ]; then
  module purge
fi

module load GCC/14.3.0
module load CUDA/12.9.1
module load OpenBLAS/0.3.30-GCC-14.3.0
module load CMake/4.0.3-GCCcore-14.3.0
module load Ninja/1.13.0-GCCcore-14.3.0 2>/dev/null || true

export PETSC_DIR="${PETSC_DIR:-/mnt/ffs24/home/khanmd/programs/petsc/install-cuda-nompi}"
export PETSC_ARCH="${PETSC_ARCH:-}"
export PKG_CONFIG_PETSC_PETSC_DIR="$PETSC_DIR"

if [ -n "$PETSC_ARCH" ] && [ -d "$PETSC_DIR/$PETSC_ARCH/lib/pkgconfig" ]; then
  _widebvh_petsc_pc="$PETSC_DIR/$PETSC_ARCH/lib/pkgconfig"
else
  _widebvh_petsc_pc="$PETSC_DIR/lib/pkgconfig"
fi

case ":${PKG_CONFIG_PATH:-}:" in
  *":$_widebvh_petsc_pc:"*) ;;
  *) export PKG_CONFIG_PATH="$_widebvh_petsc_pc${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}" ;;
esac
unset _widebvh_petsc_pc

# A100 = sm_80, H200/H100 = sm_90. CMake also reads CUDAARCHS during configure.
export WIDEBVH_CUDA_ARCHITECTURES="${WIDEBVH_CUDA_ARCHITECTURES:-80;90}"
export CUDAARCHS="${CUDAARCHS:-$WIDEBVH_CUDA_ARCHITECTURES}"

if command -v nvcc >/dev/null 2>&1; then
  export CUDACXX="$(command -v nvcc)"
fi

echo "widebvh env: CUDAARCHS=$CUDAARCHS PETSC_DIR=$PETSC_DIR PETSC_ARCH=${PETSC_ARCH:-<empty>}"
