#!/usr/bin/env bash
#
# Run a command inside the NeMO image on a LOCAL GPU box, with this repo and the
# widebvh source tree bind-mounted over the image's baked copies.
#
#   bash docker/run_local.sh python benchmarks/two_suspensions_1M.py --help
#   bash docker/run_local.sh bash -c 'cmake --build $WIDEBVH_BUILD_DIR -j3'
#
# Why a wrapper: the image bakes /workspace/pinn-stokes and /opt/widebvh at
# build time; iterating on either means rebuilding a 17 GB image. Mounting the
# host trees instead makes a source edit visible on the next `docker run`.
#
#   /workspace/pinn-stokes  <- this repo (so `python benchmarks/...` runs the
#                              working tree, and logs/figures land on the host)
#   /opt/widebvh-src        <- $WIDEBVH_SRC (default ~/envs/nemo-ctx/widebvh),
#                              built for the local sm into
#                              $WIDEBVH_BUILD_DIR=/opt/widebvh-src/build-<tag>
#   /workspace/.cache       <- $NEMO_CACHE (default ~/envs/nemo-cache): the
#                              inductor / triton / warp JIT caches. Persisting
#                              them across runs is what keeps the 1M warmup at
#                              seconds instead of the minutes a cold autotune
#                              costs on a laptop.
#
# Runs as the calling user (-u) so everything written to the mounts is owned by
# you, not root; HOME is pointed into the cache mount because /workspace itself
# is root-owned in the image.
#
# Env overrides: IMG, WIDEBVH_SRC, WIDEBVH_BUILD_TAG (default 4060), NEMO_CACHE,
# and any extra `docker run` flags in RUN_LOCAL_DOCKER_ARGS (e.g. -e VAR=1).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMG="${IMG:-sskhan39/nemo:2.0}"
WIDEBVH_SRC="${WIDEBVH_SRC:-$HOME/envs/nemo-ctx/widebvh}"
WIDEBVH_BUILD_TAG="${WIDEBVH_BUILD_TAG:-4060}"
NEMO_CACHE="${NEMO_CACHE:-$HOME/envs/nemo-cache}"

mkdir -p "$NEMO_CACHE/home" "$WIDEBVH_SRC/build-$WIDEBVH_BUILD_TAG"

sha="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"

# Pass through the knobs the benchmarks read, when set in the caller's shell.
passthru=()
for v in NEMO_FAR_FIELD NEMO_DEVICE_MEM NEMO_MAC NEMO_CART_ORDER NEMO_FAR_FP32_LEVEL \
         NEMO_MID_CELL_SCALE \
         TORCH_COMPILE_DISABLE TC_PAIR_BUDGET_GB TC_PATH TC_QUIET TC_HILBERT_Q \
         TORCH_LOGS CUDA_LAUNCH_BLOCKING; do
    if [[ -n "${!v:-}" ]]; then passthru+=(-e "$v=${!v}"); fi
done

exec docker run --rm --gpus all \
    -u "$(id -u):$(id -g)" \
    -v "$ROOT:/workspace/pinn-stokes" \
    -v "$WIDEBVH_SRC:/opt/widebvh-src" \
    -v "$NEMO_CACHE:/workspace/.cache" \
    -e HOME=/workspace/.cache/home \
    -e WIDEBVH_BUILD_DIR="/opt/widebvh-src/build-$WIDEBVH_BUILD_TAG" \
    -e NEMO_GIT_SHA="$sha" \
    "${passthru[@]}" \
    ${RUN_LOCAL_DOCKER_ARGS:-} \
    -w /workspace/pinn-stokes \
    "$IMG" "$@"
