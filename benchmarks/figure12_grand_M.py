"""Figure 12: end-to-end scaling of the grand mobility operator.

One application of the full NeMO operator (analytic self + two-body NN + n-body
NN inside r=6, treecode far field beyond) on uniform suspensions at 10% volume
fraction, N = 50k ... 750k. Bars are total runtime, the line is the corresponding
particle-update rate.

The published figure was produced with the Warp far field; `--backend widebvh`
re-measures it with the widebvh treecode, `--backend widebvh-cart` with the same
engine's Cartesian Taylor expansion, and `--backend warp` reproduces the baseline
on the same node for a like-for-like comparison. All are written to the same CSV
so the plot script can draw any of them.

Timing protocol is deliberately identical to the published one
(benchmarks/performance_grand_M.py:benchmark_apply): 6 warmup applications so
torch.compile can specialize, 6 timed, sorted, first and last two discarded.
torch.compile stays ENABLED -- this is a performance run.

Usage:
    source ~/warp_env.sh
    python benchmarks/figure12_grand_M.py --backend widebvh
    python benchmarks/figure12_grand_M.py --backend warp
    python figures/grand_M_perf.py            # renders from the CSV
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.performance_grand_M import (  # noqa: E402
    BenchmarkConfig, build_far_field, build_near_field, build_forces,
    load_configuration, TMP_DIR,
)

DEFAULT_SIZES = (50_000, 100_000, 200_000, 500_000, 750_000)
CSV_PATH = ROOT / "data" / "fig12_scaling_h200.csv"

FIELDS = ("backend", "n", "total_ms", "std_ms", "min_ms", "far_ms", "near_ms",
          "updates_per_sec", "peak_vram_gb", "mac", "theta", "pdeg", "order",
          "max_leaf", "gpu", "git_sha")


def git_sha() -> str:
    # NEMO_GIT_SHA first: the Docker image ships the working tree without .git
    # (docker/pack_context.sh stamps the host SHA into the environment), and
    # the 5090 column is precisely the row whose provenance has to be legible.
    stamped = os.environ.get("NEMO_GIT_SHA", "").strip()
    if stamped:
        return stamped
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def measure(n: int, backend: str, bench_cfg: BenchmarkConfig, **kwargs) -> dict:
    device = torch.device("cuda")
    config = load_configuration(TMP_DIR / f"uniform_large_0.1_{n}.csv")
    assert config.shape[0] == n, f"{n} requested, CSV has {config.shape[0]}"
    forces = build_forces(n, seed=2024)

    positions = torch.as_tensor(config[:, :3], dtype=torch.float32, device=device)
    orientations = torch.as_tensor(config[:, 3:], dtype=torch.float32, device=device)
    forces_t = torch.as_tensor(forces, dtype=torch.float32, device=device)
    vis_arr = torch.full((n,), bench_cfg.viscosity, dtype=torch.float32,
                         device=device)

    if backend == "widebvh-cart":
        # These clouds fill their box, so fill=1. Without this the Cartesian
        # policy runs at bary's bucket granularity, which is far too coarse for
        # its much denser near-pair set -- see cart_hilbert_q.
        from src.treecode_widebvh import cart_hilbert_q
        kwargs.setdefault("hilbert_q", cart_hilbert_q(n))

    op = build_far_field(build_near_field("nbody", "sphere", 6.0), backend, 6.0,
                         **kwargs)

    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(bench_cfg.warmup_runs):
        op.apply(positions, orientations, forces_t, vis_arr)
        torch.cuda.synchronize(device)

    timings = []
    for _ in range(bench_cfg.timed_runs):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        op.apply(positions, orientations, forces_t, vis_arr)
        torch.cuda.synchronize(device)
        timings.append(time.perf_counter() - t0)
        time.sleep(0.1)          # let the clocks settle between samples

    peak_vram = torch.cuda.max_memory_allocated(device) / 1024 ** 3

    # Same trim as the published protocol: drop the fastest sample and the last
    # two, which are the ones that pick up recompilation / clock artifacts.
    t = np.array(sorted(timings), dtype=np.float64)[1:-2] * 1000.0

    # WidebvhFMM records the far field's event time on every call; WarpFMM only
    # prints it, so time it directly with the same warm/repeat discipline.
    far_ms = float(getattr(op, "last_stats", {}).get("far_ms", float("nan")))
    if far_ms != far_ms:
        f3 = forces_t[:, :3].contiguous()
        op.get_far_field_vel(positions, f3)
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        for _ in range(3):
            op.get_far_field_vel(positions, f3)
        torch.cuda.synchronize(device)
        far_ms = (time.perf_counter() - t0) * 1000.0 / 3

    total_ms = float(t.mean())

    row = dict(
        backend=backend, n=n, total_ms=total_ms,
        std_ms=float(t.std(ddof=1)) if t.size > 1 else float("nan"),
        min_ms=float(t.min()), far_ms=far_ms,
        near_ms=total_ms - far_ms if far_ms == far_ms else float("nan"),
        updates_per_sec=n / (total_ms * 1e-3),
        peak_vram_gb=peak_vram,
        mac=getattr(op, "mac", ""), theta=getattr(op, "theta", ""),
        pdeg=getattr(op, "pdeg", ""), order=getattr(op, "order", ""),
        max_leaf=getattr(op, "max_leaf", ""),
        gpu=torch.cuda.get_device_name(device), git_sha=git_sha(),
    )

    if hasattr(op, "close"):
        op.close()
    del op
    torch.cuda.empty_cache()
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="widebvh",
                    choices=("widebvh", "widebvh-cart", "warp"))
    ap.add_argument("--sizes", default=",".join(str(s) for s in DEFAULT_SIZES))
    ap.add_argument("--warmup", type=int, default=6)
    ap.add_argument("--runs", type=int, default=6)
    ap.add_argument("--csv", default=str(CSV_PATH))
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",") if s]
    bench_cfg = BenchmarkConfig(warmup_runs=args.warmup, timed_runs=args.runs)

    rows = []
    for n in sizes:
        print(f"\n########## {args.backend}  N={n:,} ##########", flush=True)
        row = measure(n, args.backend, bench_cfg)
        rows.append(row)
        print(f"@@ N={n:,}  total={row['total_ms']:.2f} ms "
              f"(+-{row['std_ms']:.2f})  far={row['far_ms']:.2f} ms  "
              f"{row['updates_per_sec'] / 1e6:.3f} M updates/s  "
              f"VRAM {row['peak_vram_gb']:.2f} GB", flush=True)

    out = Path(args.csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Rewrite this backend's rows, keep the other backend's.
    existing = []
    if out.exists():
        with out.open() as fh:
            existing = [r for r in csv.DictReader(fh)
                        if r["backend"] != args.backend]
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(existing)
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows -> {out}")

    print("\n N          total ms    far ms   M updates/s   VRAM GB")
    for r in rows:
        print(f" {r['n']:>9,}  {r['total_ms']:>8.2f}  {r['far_ms']:>8.2f}  "
              f"{r['updates_per_sec'] / 1e6:>11.3f}  {r['peak_vram_gb']:>8.2f}")


if __name__ == "__main__":
    main()
