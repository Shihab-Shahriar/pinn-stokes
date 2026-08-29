"""Per-step far-field cost over the two-drop sedimentation run.

Measurement #4 in `artifacts/paper_update_plan.md` §4. The paper claims at lines
711-713 that "as the simulation progresses, the particle distribution becomes
more non-uniform, which leads to a slight increase in the far-field runtime due
to the treecode traversal." `data/widebvh_perf_nemo_distros.csv` holds only
50-step *aggregates* (one row per backend), so the claim cannot be checked by
reading it -- the drift is exactly what an aggregate averages away. This records
every component per step.

The physical setup is copied from `benchmarks/two_suspensions_1M.py` verbatim
(R_drop 175, phi 0.10, gap 100, dt 0.01, F_z = -9.81) so the numbers are
comparable with that script's aggregates. Two deliberate differences:

  * the cloud is seeded, so the widebvh and warp runs see identical particles
    and the A/B is exact rather than statistical;
  * stdout is captured per step instead of per run.

One process per backend -- the far-field CUDA-event bracket is contaminated by
allocator churn when several solvers are built in one process (see the header of
`benchmarks/figure11_breakdown.py`).

    python benchmarks/far_field_drift.py --backend widebvh --steps 150
    python benchmarks/far_field_drift.py --backend warp --steps 150
"""

import argparse
import contextlib
import csv
import io
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# Match two_suspensions_1M.py's allocator and inductor settings exactly -- these
# must be set before torch initialises CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch._inductor.config as inductor_config

inductor_config.triton.cudagraph_skip_dynamic_graphs = True
inductor_config.triton.cudagraphs = False
inductor_config.freezing = True

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.figure11_breakdown import parse_components  # noqa: E402
from benchmarks.two_suspensions_1M import generate_suspension_drop  # noqa: E402
from src.gpu_nbody_mob import Mob_Nbody_Torch  # noqa: E402
from src.treecode import WarpFMM  # noqa: E402
from src.treecode_widebvh import (  # noqa: E402
    DEFAULT_CART_MAC, DEFAULT_CART_ORDER, DEFAULT_MAC, DEFAULT_MAX_LEAF,
    WidebvhFMM)

CSV_PATH = ROOT / "data" / "far_field_drift_1M.csv"

FIELDS = ("backend", "n", "step", "t", "far_ms", "nsearch_ms", "self2b_ms",
          "nbody_ms", "overall_near_ms", "total_gpu_ms", "near_pairs",
          "wall_ms", "z_std", "bbox_vol", "gpu", "git_sha",
          # configuration columns, added with the consumer-GPU sweep. `label`
          # is the row-replacement key (defaults to the backend name, which is
          # what the H200 rows carry implicitly).
          "label", "mac", "max_leaf", "pdeg", "fp32_level", "hilbert_q",
          "pair_budget_gb")

# From two_suspensions_1M.main().
R_DROP = 175.0
PHI = 0.10
GAP = 100.0
DT = 0.01
VISCOSITY = 1.0
WARMUP_STEPS = 5


def git_sha() -> str:
    # NEMO_GIT_SHA first: the Docker image ships the working tree without .git
    # (docker/pack_context.sh stamps the host SHA into the environment).
    stamped = os.environ.get("NEMO_GIT_SHA", "").strip()
    if stamped:
        return stamped
    out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                         capture_output=True, text=True)
    return out.stdout.strip() or "unknown"


def build_solver(backend: str, mob, mac=None, max_leaf=DEFAULT_MAX_LEAF,
                 hilbert_q=None, pdeg=7, fp32_level=0, pair_budget_gb=None):
    """The same construction two_suspensions_1M.py performs, per backend."""
    if backend.startswith("widebvh"):
        cart = backend == "widebvh-cart"
        # None means "this policy's calibrated default": no mac transfers
        # between the two expansions.
        if mac is None:
            mac = DEFAULT_CART_MAC if cart else DEFAULT_MAC
        # Tuned for this two-ball geometry, not the uniform case -- see the
        # comment at two_suspensions_1M.py:213.
        if hilbert_q is None:
            hilbert_q = 26.0 if cart else None
        elif hilbert_q <= 0.0:
            hilbert_q = None          # explicit request for the engine's auto
        solver = WidebvhFMM(
            near_field_operator=mob,
            near_field_cutoff=6.0,
            device="cuda",
            policy="cart" if cart else "bary",
            mac=mac,
            max_leaf=max_leaf,
            order=DEFAULT_CART_ORDER if cart else None,
            hilbert_q=hilbert_q,
            pdeg=pdeg,
            fp32_level=fp32_level,
            pair_budget_gb=pair_budget_gb,
        )
        print(f"Far field: widebvh, policy={solver.policy}, mac={solver.mac}, "
              f"PDEG={solver.pdeg}, order={solver.order}, "
              f"maxLeaf={solver.max_leaf}, fp32_level={solver.fp32_level}, "
              f"pair_budget_gb={solver.env['TC_PAIR_BUDGET_GB']}")
    else:
        solver = WarpFMM(near_field_operator=mob, theta=0.3, leaf_size=16,
                         near_field_cutoff=6.0, device="cuda", block_dim=256)
        print("Far field: Warp treecode, theta=0.3")
    return solver


def last(raw: dict, key: str, default=""):
    """The value a key emitted during this step, or `default` if it was silent.

    `NNMobTorch` never emits the `[Mob_Nbody]` keys; nothing else here is
    optional, so a missing value is a real absence rather than a parse failure.
    """
    values = raw.get(key) or []
    return values[-1] if values else default


def run(backend: str, steps: int, csv_path: Path, mac=None,
        max_leaf=DEFAULT_MAX_LEAF, hilbert_q=None, pdeg=7, fp32_level=0,
        pair_budget_gb=None, two_body_chunk=None, pair_chunk=None,
        label=None) -> list:
    label = label or backend
    np.random.seed(0)
    print("Generating Drop 1...")
    drop1 = generate_suspension_drop((0, 0, 0.0), R_DROP)
    print("Generating Drop 2...")
    drop2 = generate_suspension_drop((0, 0, 2 * R_DROP + GAP), R_DROP)
    all_particles = np.vstack([drop1, drop2])
    n = len(all_particles)
    print(f"Total particles: {n}")

    mob = Mob_Nbody_Torch(
        shape="sphere",
        self_nn_path=str(ROOT / "data/models/self_interaction_model.pt"),
        two_nn_path=str(ROOT / "data/models/combined_2body.wt"),
        nbody_nn_path=str(ROOT / "data/models/nbody_cross_tmp.wt"),
        near_field_2b="nn",
        far_field_2b=None,
        near_far_switch=6.0,
        **({"two_body_chunk_size": two_body_chunk} if two_body_chunk else {}),
        **({"pair_chunk_size": pair_chunk} if pair_chunk else {}),
    )
    solver = build_solver(backend, mob, mac=mac, max_leaf=max_leaf,
                          hilbert_q=hilbert_q, pdeg=pdeg, fp32_level=fp32_level,
                          pair_budget_gb=pair_budget_gb)
    cfg = dict(label=label, mac=getattr(solver, "mac", ""),
               max_leaf=getattr(solver, "max_leaf", ""),
               pdeg=getattr(solver, "pdeg", ""),
               fp32_level=getattr(solver, "fp32_level", ""),
               hilbert_q="" if hilbert_q is None else hilbert_q,
               pair_budget_gb=getattr(solver, "env", {}).get("TC_PAIR_BUDGET_GB", ""))

    device = torch.device("cuda")
    positions = torch.from_numpy(all_particles.astype(np.float32)).to(device)
    initial_positions = positions.clone()
    orientations = torch.zeros((n, 4), dtype=torch.float32, device=device)
    orientations[:, 3] = 1.0
    forces = torch.zeros((n, 6), dtype=torch.float32, device=device)
    forces[:, 2] = -9.81
    vis_arr = torch.full((n,), VISCOSITY, dtype=torch.float32, device=device)

    print(f"Warmup ({WARMUP_STEPS} steps, compilation)...")
    with torch.no_grad():
        for _ in range(WARMUP_STEPS):
            vel = solver.apply(positions, orientations, forces, vis_arr)
            positions += vel[:, :3] * DT
    positions = initial_positions.clone()

    gpu = torch.cuda.get_device_name(0)
    sha = git_sha()
    rows = []

    print(f"Timing {steps} steps...")
    with torch.no_grad():
        for step in range(steps):
            # The cloud shape at the *start* of this step, which is what the
            # far field this step actually sees.
            z = positions[:, 2]
            z_std = float(z.std())
            extent = (positions.max(dim=0).values -
                      positions.min(dim=0).values)
            bbox_vol = float(extent.prod())

            buf = io.StringIO()
            torch.cuda.synchronize()
            wall0 = time.perf_counter()
            with contextlib.redirect_stdout(buf):
                vel = solver.apply(positions, orientations, forces, vis_arr)
            torch.cuda.synchronize()
            wall_ms = (time.perf_counter() - wall0) * 1e3

            raw = parse_components(buf.getvalue())
            rows.append(dict(
                backend=backend, n=n, step=step, t=round(step * DT, 4),
                far_ms=last(raw, "far_ms"),
                nsearch_ms=last(raw, "nsearch_ms"),
                self2b_ms=last(raw, "self2b_ms"),
                nbody_ms=last(raw, "nbody_ms"),
                overall_near_ms=last(raw, "overall_near_ms"),
                total_gpu_ms=last(raw, "total_gpu_ms"),
                near_pairs=int(last(raw, "near_pairs", 0)),
                wall_ms=round(wall_ms, 3),
                z_std=round(z_std, 3), bbox_vol=round(bbox_vol, 1),
                gpu=gpu, git_sha=sha, **cfg,
            ))
            assert rows[-1]["far_ms"] != "", (
                f"step {step}: no far-field timing in captured stdout")

            positions += vel[:, :3] * DT
            if step % 25 == 0 or step == steps - 1:
                r = rows[-1]
                print(f"  step {step:3d}  far {r['far_ms']:7.3f}  "
                      f"total {r['total_gpu_ms']:8.3f}  "
                      f"pairs {r['near_pairs']:,}", flush=True)

    write_rows(csv_path, rows, label)
    summarize(rows, label)
    return rows


def write_rows(path: Path, rows: list, label: str) -> None:
    """Replace this label's rows, leaving any other label's in place.

    Rows written before the `label` column existed are keyed by their
    backend name, which is what `label` defaults to."""
    existing = []
    if path.exists():
        with open(path, newline="") as fh:
            existing = [r for r in csv.DictReader(fh)
                        if (r.get("label") or r.get("backend")) != label]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(existing)
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows for {label} -> {path}")


def summarize(rows: list, backend: str) -> None:
    far = [float(r["far_ms"]) for r in rows]
    tot = [float(r["total_gpu_ms"]) for r in rows]
    near = [float(r["overall_near_ms"]) for r in rows]
    k = min(10, len(rows))

    def mean(xs):
        return sum(xs) / len(xs)

    first, last_ = mean(far[:k]), mean(far[-k:])
    print(f"\n@@ {backend}  n={rows[0]['n']}  steps={len(rows)}")
    print(f"@@ far-field   first {k}: {first:7.3f} ms   "
          f"last {k}: {last_:7.3f} ms   drift {100*(last_/first - 1):+6.2f}%")
    print(f"@@ near-field  first {k}: {mean(near[:k]):7.3f} ms   "
          f"last {k}: {mean(near[-k:]):7.3f} ms   "
          f"drift {100*(mean(near[-k:])/mean(near[:k]) - 1):+6.2f}%")
    print(f"@@ total       first {k}: {mean(tot[:k]):7.3f} ms   "
          f"last {k}: {mean(tot[-k:]):7.3f} ms   "
          f"drift {100*(mean(tot[-k:])/mean(tot[:k]) - 1):+6.2f}%")
    print(f"@@ mean step {mean(tot):.1f} ms, far share {mean(far)/mean(tot):.3f}"
          f", {len(rows)} steps in {sum(tot)/1e3:.1f} s of GPU time")
    pairs = [r["near_pairs"] for r in rows]
    print(f"@@ near pairs  {pairs[0]:,} -> {pairs[-1]:,} "
          f"({100*(pairs[-1]/pairs[0] - 1):+.2f}%)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", default="widebvh",
                    choices=["widebvh", "widebvh-cart", "warp"])
    ap.add_argument("--steps", type=int, default=150,
                    help="timed steps; the paper quotes the first 150")
    ap.add_argument("--csv", type=Path, default=CSV_PATH)
    ap.add_argument("--mac", type=float, default=None,
                    help="widebvh acceptance criterion; default is the "
                         "policy's calibrated value. Saturates at 1.0.")
    ap.add_argument("--max-leaf", type=int, default=DEFAULT_MAX_LEAF,
                    help="widebvh leaf size (default %(default)s); trades "
                         "traversal against fp64 P2P at constant accuracy")
    ap.add_argument("--hilbert-q", type=float, default=None,
                    help="source-bucket cells per axis; 0 forces engine auto")
    ap.add_argument("--pdeg", type=int, default=7,
                    help="widebvh barycentric degree (compile-time .so)")
    ap.add_argument("--fp32-level", type=int, default=0,
                    help="widebvh fp32 fast-path level (0 fp64, 1 fp32 M2P, "
                         "2 + fp32 P2P, 3 + fp32 upward pass)")
    ap.add_argument("--pair-budget-gb", type=float, default=None,
                    help="widebvh pair-list budget override")
    ap.add_argument("--two-body-chunk", type=int, default=None,
                    help="two-body NN chunk size (default: operator default)")
    ap.add_argument("--pair-chunk", type=int, default=None,
                    help="n-body NN chunk size (default: operator default)")
    ap.add_argument("--label", default=None,
                    help="row key in the CSV (default: the backend name); "
                         "give each configuration its own label so runs do "
                         "not overwrite each other")
    args = ap.parse_args()
    run(args.backend, args.steps, args.csv, mac=args.mac,
        max_leaf=args.max_leaf, hilbert_q=args.hilbert_q, pdeg=args.pdeg,
        fp32_level=args.fp32_level, pair_budget_gb=args.pair_budget_gb,
        two_body_chunk=args.two_body_chunk, pair_chunk=args.pair_chunk,
        label=args.label)


if __name__ == "__main__":
    main()
