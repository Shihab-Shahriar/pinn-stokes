#!/usr/bin/env bash
#
# Assemble a self-contained Docker build context for the NeMO container.
#
# This script exists because neither of the two things the image needs can be
# obtained with `git clone`:
#
#   * widebvh (github.com/Shihab-Shahriar/treecode) has 9 modified tracked files
#     and, critically, `src/nemo_capi.cu` -- the entire wbnemo_* C ABI that
#     src/treecode_widebvh.py dlopens -- is UNTRACKED. `src/cpu/` is untracked
#     too and is referenced unconditionally by add_executable(two_ball_cpu ...),
#     so a clone fails at CMake configure time. `cuBQL/` is a gitlink with no
#     .gitmodules, so --recursive leaves it empty.
#   * this repo has src/treecode_widebvh.py, both production .wt models and 8
#     benchmarks/figure*.py untracked, plus ~43 modified tracked files.
#
# So: copy the working trees. Run this on the cluster, scp the tarball to a
# machine with Docker, and build there.
#
#   source ~/warp_env.sh
#   bash docker/pack_context.sh                    # -> nemo-ctx.tar.gz, 42 MB
#   bash docker/pack_context.sh --with-warp-fork   # + the patched Warp fork
#
# --with-warp-fork is only needed to run the OLD far field (`--backend warp` /
# NEMO_FAR_FIELD=warp) for an A/B. The widebvh path runs on stock warp-lang:
# every patched builtin lives inside WarpFMM.get_far_field_vel, which
# WidebvhFMM overrides, and Warp compiles a module's kernels lazily on first
# launch. Warp itself is still required either way -- the near-field neighbour
# search is Warp's hash grid.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WIDEBVH_SRC="${WIDEBVH_SRC:-$HOME/programs/widebvh}"
WARP_SRC="${WARP_SRC:-$HOME/programs/warp}"
OUT="${OUT:-$ROOT/nemo-ctx.tar.gz}"
CTX="$(mktemp -d "${TMPDIR:-/tmp}/nemo-ctx.XXXXXX")"
trap 'rm -rf "$CTX"' EXIT

WITH_WARP_FORK=0
for arg in "$@"; do
    case "$arg" in
        --with-warp-fork) WITH_WARP_FORK=1 ;;
        -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "unknown argument: $arg" >&2; exit 2 ;;
    esac
done

# The five sizes benchmarks/figure12_grand_M.py:47 DEFAULT_SIZES sweeps, and
# nothing else. The 1M/2M/4M configs (376 MB) are not used by either of the two
# experiments, and the sedimentation run needs no configuration file at all --
# generate_suspension_drop() builds its particles procedurally.
FIG12_SIZES=(50000 100000 200000 500000 750000)

say() { printf '\033[1m==>\033[0m %s\n' "$*"; }

# ---------------------------------------------------------------- preflight --
[[ -d "$WIDEBVH_SRC" ]] || { echo "no widebvh source at $WIDEBVH_SRC (set WIDEBVH_SRC=)" >&2; exit 1; }
[[ -f "$WIDEBVH_SRC/CMakeLists.txt" ]] || { echo "$WIDEBVH_SRC has no CMakeLists.txt" >&2; exit 1; }
# The two files a clone would miss. If these are gone the image cannot build.
[[ -f "$WIDEBVH_SRC/src/nemo_capi.cu" ]] || { echo "MISSING $WIDEBVH_SRC/src/nemo_capi.cu -- the C ABI" >&2; exit 1; }
[[ -d "$WIDEBVH_SRC/cuBQL/cuBQL" ]] || { echo "MISSING $WIDEBVH_SRC/cuBQL (gitlink, not a submodule)" >&2; exit 1; }
[[ -f "$ROOT/src/treecode_widebvh.py" ]] || { echo "MISSING src/treecode_widebvh.py" >&2; exit 1; }

for m in self_interaction_model.pt combined_2body.wt nbody_cross_tmp.wt; do
    [[ -f "$ROOT/data/models/$m" ]] || { echo "MISSING data/models/$m" >&2; exit 1; }
done
for n in "${FIG12_SIZES[@]}"; do
    f="$ROOT/tmp/uniform_large_0.1_${n}.csv"
    [[ -f "$f" ]] || { echo "MISSING $f -- regenerate with benchmarks/cluster.py:uniform_cluster_generation_large" >&2; exit 1; }
done

if [[ $WITH_WARP_FORK == 1 ]]; then
    [[ -d "$WARP_SRC/warp" ]] || { echo "no Warp fork at $WARP_SRC (set WARP_SRC=)" >&2; exit 1; }
fi

# ------------------------------------------------------------------- widebvh --
# A WHITELIST, not an exclude list. The checkout is 2.6 GB of build dirs,
# profiles and case data (mfs_ellipsoid_case alone is 755 MB); everything CMake
# actually references is under src/ and cuBQL/ -- verified by grepping every
# add_executable/add_library/add_subdirectory in CMakeLists.txt. An exclude list
# silently grows stale as new result directories appear.
say "widebvh source  <- $WIDEBVH_SRC"
mkdir -p "$CTX/widebvh"
tar -C "$WIDEBVH_SRC" -cf - \
    --exclude='__pycache__' --exclude='*.pyc' --exclude='*.o' --exclude='*.so' \
    CMakeLists.txt env.sh src cuBQL \
    CLAUDE.md AGENTS.md .gitignore \
    | tar -C "$CTX/widebvh" -xf -

# cuBQL is a gitlink with no .gitmodules, so `clone --recursive` leaves it empty
# and a missing/empty copy fails at add_subdirectory(cuBQL) rather than at link.
[[ -f "$CTX/widebvh/cuBQL/CMakeLists.txt" ]] || { echo "cuBQL did not copy" >&2; exit 1; }
[[ -f "$CTX/widebvh/src/nemo_capi.cu" ]] || { echo "nemo_capi.cu did not copy" >&2; exit 1; }
wb_mb=$(du -sm "$CTX/widebvh" | cut -f1)
(( wb_mb < 50 )) || { echo "widebvh context is ${wb_mb} MB, expected ~3 -- the whitelist picked up a build or data directory" >&2; exit 1; }

# ---------------------------------------------------------------------- warp --
# Always create the directory: the Dockerfile COPYs it unconditionally and
# decides what to do with it from the WITH_WARP_FORK build arg.
mkdir -p "$CTX/warp"
if [[ $WITH_WARP_FORK == 1 ]]; then
    say "Warp fork       <- $WARP_SRC  (patched, for --backend warp)"
    tar -C "$WARP_SRC" -cf - \
        --exclude='./.git' --exclude='./_build' --exclude='./warp/bin/*.so' \
        --exclude='__pycache__' --exclude='*.pyc' \
        . | tar -C "$CTX/warp" -xf -
else
    : > "$CTX/warp/.keep"
    say "Warp fork       -- skipped (stock warp-lang; --backend warp unavailable)"
fi

# ---------------------------------------------------------------------- repo --
say "repo subset     <- $ROOT"
mkdir -p "$CTX/pinn-stokes"
tar -C "$ROOT" -cf - \
    --exclude='__pycache__' --exclude='*.pyc' --exclude='.ipynb_checkpoints' \
    src benchmarks utils docker __init__.py CLAUDE.md README.md \
    | tar -C "$CTX/pinn-stokes" -xf -

say "models + points"
mkdir -p "$CTX/pinn-stokes/data/models" "$CTX/pinn-stokes/data/points"
cp "$ROOT"/data/models/* "$CTX/pinn-stokes/data/models/"
# Only reached by the MFS ground-truth path, which neither experiment runs, but
# benchmarks/cluster.py is on the production import chain and it is 128 KB.
cp "$ROOT"/data/points/* "$CTX/pinn-stokes/data/points/" 2>/dev/null || true

say "Figure 12 configs: ${FIG12_SIZES[*]}"
mkdir -p "$CTX/pinn-stokes/tmp"
for n in "${FIG12_SIZES[@]}"; do
    cp "$ROOT/tmp/uniform_large_0.1_${n}.csv" "$CTX/pinn-stokes/tmp/"
done

# ------------------------------------------------------------ build metadata --
cp "$ROOT/docker/Dockerfile"   "$CTX/Dockerfile"
cp "$ROOT/docker/dockerignore" "$CTX/.dockerignore"

SHA="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
if ! git -C "$ROOT" diff --quiet 2>/dev/null; then SHA="${SHA}-dirty"; fi
printf '%s\n' "$SHA" > "$CTX/GIT_SHA"

{
    echo "pinn-stokes    $ROOT  @ $SHA"
    echo "widebvh        $WIDEBVH_SRC  @ $(git -C "$WIDEBVH_SRC" rev-parse --short HEAD 2>/dev/null || echo unknown)$(git -C "$WIDEBVH_SRC" diff --quiet 2>/dev/null || echo -dirty)"
    echo "warp fork      $([[ $WITH_WARP_FORK == 1 ]] && echo "$WARP_SRC" || echo '(not included -- stock warp-lang)')"
    echo "fig12 configs  ${FIG12_SIZES[*]}"
    echo "packed         $(date -u +%Y-%m-%dT%H:%M:%SZ) on $(hostname)"
} > "$CTX/MANIFEST.txt"

# ---------------------------------------------------------------------- tar ---
say "sizes"
du -sh "$CTX"/* | sed 's/^/    /'

tar -C "$CTX" -czf "$OUT" .
say "wrote $OUT  ($(du -h "$OUT" | cut -f1))"
cat <<EOF

Next, on a machine with Docker:

    scp $OUT you@docker-box:
    mkdir nemo-ctx && tar -C nemo-ctx -xzf $(basename "$OUT") && cd nemo-ctx
    docker build --build-arg NEMO_GIT_SHA=$SHA --build-arg WIDEBVH_FP32_LEVELS="1;2;3" $([[ $WITH_WARP_FORK == 1 ]] && echo '--build-arg WITH_WARP_FORK=1 ') -t <dockerhub-user>/nemo:2.0 .
    docker run --rm --gpus all <dockerhub-user>/nemo:2.0        # smoke test
    docker push <dockerhub-user>/nemo:2.0

See docker/README.md for the run and verification commands.
EOF
