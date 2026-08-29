#!/usr/bin/env bash
#
# Make the cache directories exist and be writable before anything runs.
#
# torch.compile (inductor), Triton and Warp all JIT at runtime and all need a
# writable cache; matplotlib needs MPLCONFIGDIR. They are pointed under
# /workspace/.cache by the Dockerfile. chmod 777 rather than chown so the image
# also works under `docker run --user`, and under Singularity, where the
# process keeps the host UID and would otherwise be unable to write anywhere.

set -euo pipefail

for d in "${TMPDIR:-}" "${XDG_CACHE_HOME:-}" "${MPLCONFIGDIR:-}" \
         "${TORCHINDUCTOR_CACHE_DIR:-}" "${TRITON_CACHE_DIR:-}" \
         "${WARP_CACHE_PATH:-}"; do
    [[ -n "$d" ]] || continue
    mkdir -p "$d" 2>/dev/null || true
    chmod 0777 "$d" 2>/dev/null || true
done

# Under Singularity the image's own /workspace may be read-only or shadowed by a
# bind mount; fall back to somewhere the caller can definitely write rather than
# failing several minutes into a compile.
if ! touch "${XDG_CACHE_HOME:-/workspace/.cache}/.writable" 2>/dev/null; then
    fallback="$(mktemp -d)"
    echo "[entrypoint] ${XDG_CACHE_HOME} is not writable; caching under $fallback" >&2
    export XDG_CACHE_HOME="$fallback" TMPDIR="$fallback/tmp" \
           MPLCONFIGDIR="$fallback/matplotlib" \
           TORCHINDUCTOR_CACHE_DIR="$fallback/inductor" \
           TRITON_CACHE_DIR="$fallback/triton" WARP_CACHE_PATH="$fallback/warp"
    mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$TORCHINDUCTOR_CACHE_DIR" \
             "$TRITON_CACHE_DIR" "$WARP_CACHE_PATH"
fi

exec "$@"
