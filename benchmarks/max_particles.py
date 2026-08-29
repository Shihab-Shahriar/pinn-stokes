"""How many particles fit on one GPU: a single-N capacity probe.

Reproduces the protocol of artifacts/vram_scaling_report.md section 5 so the
numbers are comparable with the fp64 figures there (65.45M single shot,
52.31M repeatable on an H200):

  * uniform suspension at phi = 0.1 on a cubic lattice, spacing
    a = ((4/3) pi / phi)^(1/3), +/-5% uniform jitter, built on the GPU
    (an N=65M float64 meshgrid would cost several GB of host RAM);
  * the production stack: WidebvhFMM far field over Mob_Nbody_Torch, fp32
    level from NEMO_FAR_FP32_LEVEL, everything else default;
  * PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, as in the 1M dynamics run;
  * one cold apply() (includes torch.compile), then --warm repeated applies on
    the same positions. "Single shot" = the cold apply completes; "repeatable"
    = every warm apply completes too. Past the edge the failure is usually the
    engine's cudaMalloc (widebvh bad_alloc), not a torch OOM, because the
    caching allocator never hands blocks back to the driver.

ONE N PER PROCESS. Run it from a driver loop, e.g.

    for s in 374 380 390; do NEMO_FAR_FP32_LEVEL=2 \
        python benchmarks/max_particles.py --n-side $s --warm 3; done

The last line is machine-readable:
    RESULT n_side=<s> N=<n> cold=<ok|fail> warm_ok=<k>/<w> ...
Exit status is 0 when every apply succeeded, 2 otherwise, so a driver can
bisect on it. Memory: torch's allocated/reserved high-water marks plus the
process footprint from nvidia-smi per-PID accounting when the container lets us
see it (NEMO_DEVICE_MEM is not needed -- this script samples once at the end).
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.performance_grand_M import build_far_field, build_near_field  # noqa: E402

PHI = 0.1
SPACING = ((4.0 / 3.0) * math.pi / PHI) ** (1.0 / 3.0)   # 3.472931 radii
JITTER = 0.05


def lattice_on_gpu(n_side: int, seed: int, device) -> torch.Tensor:
    n = n_side ** 3
    g = torch.Generator(device=device).manual_seed(seed)
    idx = torch.arange(n, device=device, dtype=torch.int64)
    i = idx // (n_side * n_side)
    j = (idx // n_side) % n_side
    k = idx % n_side
    pos = torch.stack((i, j, k), dim=1).to(torch.float32)
    pos -= (n_side - 1) / 2.0
    pos *= SPACING
    pos += (torch.rand((n, 3), generator=g, device=device, dtype=torch.float32)
            - 0.5) * (2.0 * JITTER * SPACING)
    del idx, i, j, k
    return pos


def process_device_mb() -> float | None:
    """This process's device memory from nvidia-smi, or None if the container
    hides PIDs (then use an external sampler of memory.used on an exclusive
    GPU)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return None
    for line in out.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 2 and parts[0] == str(os.getpid()):
            try:
                return float(parts[1])
            except ValueError:
                return None
    return None


def gpu_used_mb() -> float | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout
        return float(out.strip().splitlines()[0])
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-side", type=int, required=True)
    ap.add_argument("--warm", type=int, default=3,
                    help="warm applies after the cold one (0 = single shot)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--mem-fraction", type=float, default=None,
                    help="torch.cuda.set_per_process_memory_fraction: cap the "
                         "caching allocator so it recycles its cache instead of "
                         "hoarding the room the treecode's own cudaMalloc needs")
    ap.add_argument("--empty-cache", action="store_true",
                    help="torch.cuda.empty_cache() before every apply (the "
                         "report's 'repeatable + empty_cache' variant)")
    args = ap.parse_args()

    device = torch.device("cuda")
    if args.mem_fraction is not None:
        torch.cuda.set_per_process_memory_fraction(args.mem_fraction, 0)
    n = args.n_side ** 3
    level = int(os.environ.get("NEMO_FAR_FP32_LEVEL", "0"))
    print(f"n_side={args.n_side}  N={n:,}  phi={PHI}  a={SPACING:.6f}  "
          f"fp32_level={level}  mem_fraction={args.mem_fraction}  expandable_segments="
          f"{'expandable_segments:True' in os.environ['PYTORCH_CUDA_ALLOC_CONF']}",
          flush=True)
    print(f"gpu={torch.cuda.get_device_name(device)}  "
          f"total={torch.cuda.mem_get_info()[1] / 1024**3:.2f} GiB", flush=True)

    t0 = time.perf_counter()
    positions = lattice_on_gpu(args.n_side, args.seed, device)
    orientations = torch.zeros((n, 4), dtype=torch.float32, device=device)
    orientations[:, 0] = 1.0
    forces = torch.zeros((n, 6), dtype=torch.float32, device=device)
    forces[:, 2] = -1.0                       # uniform gravity
    vis = torch.ones((n,), dtype=torch.float32, device=device)
    torch.cuda.synchronize()
    print(f"lattice built in {time.perf_counter() - t0:.2f} s, "
          f"extent {positions.min().item():.1f} .. {positions.max().item():.1f}",
          flush=True)

    op = build_far_field(build_near_field("nbody", "sphere", 6.0), "widebvh", 6.0)
    torch.cuda.reset_peak_memory_stats(device)

    cold_ok = False
    warm_ok = 0
    warm_times = []
    fail = ""
    vel = None
    try:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        vel = op.apply(positions, orientations, forces, vis)
        torch.cuda.synchronize()
        cold_s = time.perf_counter() - t0
        cold_ok = True
        print(f"COLD apply: {cold_s:.2f} s", flush=True)
        for r in range(args.warm):
            if args.empty_cache:
                del vel
                vel = None
                torch.cuda.empty_cache()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            vel = op.apply(positions, orientations, forces, vis)
            torch.cuda.synchronize()
            warm_times.append(time.perf_counter() - t0)
            warm_ok += 1
            print(f"WARM apply {r + 1}/{args.warm}: {warm_times[-1]:.2f} s",
                  flush=True)
    except BaseException as e:          # torch OOM, widebvh RuntimeError, ...
        fail = f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"
        traceback.print_exc()
        print(f"FAILED after cold_ok={cold_ok} warm_ok={warm_ok}: {fail}",
              flush=True)

    alloc = torch.cuda.max_memory_allocated(device) / 1024 ** 3
    resv = torch.cuda.max_memory_reserved(device) / 1024 ** 3
    proc = process_device_mb()
    used = gpu_used_mb()

    sanity = ""
    if vel is not None and torch.isfinite(vel).all():
        net = vel[:, :3].sum(0)
        ratio = (net[:2].norm() / net[2].abs().clamp_min(1e-30)).item()
        sanity = f"finite=1 transverse/axial={ratio:.2e}"
    elif vel is not None:
        sanity = "finite=0"

    print(f"RESULT n_side={args.n_side} N={n} cold={'ok' if cold_ok else 'fail'} "
          f"warm_ok={warm_ok}/{args.warm} "
          f"warm_s={','.join(f'{t:.2f}' for t in warm_times) or '-'} "
          f"torch_alloc_gib={alloc:.2f} torch_reserved_gib={resv:.2f} "
          f"proc_mb={proc if proc is not None else 'na'} "
          f"gpu_used_mb_now={used if used is not None else 'na'} "
          f"fp32_level={level} mem_fraction={args.mem_fraction} "
          f"empty_cache={int(args.empty_cache)} {sanity} fail=\"{fail}\"", flush=True)
    return 0 if (cold_ok and warm_ok == args.warm) else 2


if __name__ == "__main__":
    sys.exit(main())
