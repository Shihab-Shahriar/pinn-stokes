#!/usr/bin/env python3
"""Gravity (fig4g) MFS truths for the N = 200 cells the cluster never solved, computed locally.

The `fig4g` protocol (uniform F = (0, 0, -9.81), T = 0 on the Fig 4 configurations) was solved on the cluster
with the widebvh sphere_mfs binary (benchmarks/broms_truth.py --forcing gravity) for phi in {0.025, 0.05, 0.1,
0.15} only. The Fig-3-style NeMO-vs-Stokesian-Dynamics figure (figures/fig_sd_compare.py) needs all eight
volume fractions at N = 200, so the remaining four are solved here with this repo's batched MFS
(src/mfs_batched.py: the same Xfine boundary/source clouds, exact Oseen sum with no L_cut truncation,
GMRES converged on the velocities). The configurations are read from the cached random-forcing truth of the
same (N, phi, seed) cell -- identical positions by construction -- so nothing is regenerated.

Truths land in the harness cache (tmp/nbody_moments_truth/uniform_N{N}_phi{phi:g}_seed{S}_grav.npz) with
`acc`/`solver` provenance keys; existing files are never overwritten (resumable; the broms files stay).

    python -u benchmarks/sd_gravity_truth.py --validate         # re-solve cached broms/Xfine truths (rel-L2 seams), then fill
                                                                # the missing N=200 gravity cells if every seam < --seam-tol
    python benchmarks/sd_gravity_truth.py --validate --dry-run  # seams only
    python benchmarks/sd_gravity_truth.py --backend torch64     # exact fp64 operator (needs > 8 GB VRAM at N=200 Xfine)

The Triton kernel autotunes once per process (minutes at Xfine) and again for every distinct column count, so
validation and generation run in one process with one right-hand side per solve.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from benchmarks.paper_accuracy_v2 import PHIS, cases, truth_path  # noqa: E402

GRAVITY_FT = np.array([0.0, 0.0, -9.81, 0.0, 0.0, 0.0])  # the repo's sedimentation wrench (T = 0), as broms_truth.py
DEFAULT_N = [200]
VALIDATE_PICKS = [(200, 0.1, 4423, "gravity"), (200, 0.025, 1423, "gravity"), (200, 0.15, 6423, "gravity"),
                  (200, 0.1, 4423, "random"), (200, 0.2, 8423, "random")]


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def make_solver(args):
    from src.mfs_batched import BatchedMFS
    return BatchedMFS(acc=args.clouds, backend=args.backend, tol_v=args.tol_v, mem_budget_gb=args.mem_gb,
                      max_iter=args.max_iter, max_restarts=args.max_restarts, verbose=args.verbose)


def solve(solver, config: np.ndarray, forces: np.ndarray):
    N = config.shape[0]
    vel, info = solver.solve(np.ascontiguousarray(config[:, :3], dtype=np.float64), forces.reshape(N, 6, 1))
    assert info.converged, info
    return vel[:, :, 0].cpu().numpy().astype(np.float64), info


def provenance(args, info, sibling: Path | None) -> dict:
    md5 = None
    if sibling is not None and sibling.exists():
        d = np.load(sibling)
        md5 = str(d["cluster_md5"]) if "cluster_md5" in d.files else None
    if md5 is None:
        md5 = hashlib.md5(open(ROOT / "benchmarks" / "cluster.py", "rb").read()).hexdigest()
    return dict(acc=f"batched_{args.clouds}_{args.backend}", solver="src/mfs_batched.py BatchedMFS gmres",
                tol_v=float(info.tol_v), iters=int(info.iters), n_matvec=int(info.n_matvec),
                max_rel_dv=float(info.max_rel_dv), cluster_md5=md5)


def validate(args, solver) -> float:
    """Re-solve cached truths from their stored config + forces; returns the worst rel-L2 seam."""
    picks = [truth_path(N, phi, seed, forcing) for N, phi, seed, forcing in VALIDATE_PICKS]
    picks = [p for p in picks if p.exists()]
    print(f"[validate] {len(picks)} cached truths, BatchedMFS acc={args.clouds} backend={args.backend} tol_v={solver.tol_v:g}",
          flush=True)
    worst = 0.0
    for p in picks:
        d = np.load(p)
        t0 = time.time()
        v_new, info = solve(solver, np.array(d["config"]), np.array(d["forces"]))
        v_ref = d["velocity"]
        seam = rel_l2(v_new, v_ref)
        worst = max(worst, seam)
        print(f"  {p.name:<44} acc={str(d['acc']):<14} relL2 all={seam:.3e}  "
              f"lin={rel_l2(v_new[:, :3], v_ref[:, :3]):.3e}  ang={rel_l2(v_new[:, 3:], v_ref[:, 3:]):.3e}  "
              f"iters={info.iters} matvecs={info.n_matvec} ({time.time() - t0:.1f} s)", flush=True)
    print(f"[validate] worst seam {worst:.3e}", flush=True)
    return worst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--N", type=int, nargs="+", default=DEFAULT_N)
    ap.add_argument("--phis", type=float, nargs="+", default=None, help="default: every phi with a missing gravity truth")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--clouds", default="Xfine", choices=["coarse", "medium", "fine", "Xfine"])
    ap.add_argument("--backend", default="triton32", choices=["triton32", "torch64"])
    ap.add_argument("--tol-v", type=float, default=None, help="velocity tolerance (default: solver default per backend)")
    ap.add_argument("--max-iter", type=int, default=60)
    ap.add_argument("--max-restarts", type=int, default=3)
    ap.add_argument("--mem-gb", type=float, default=4.0)
    ap.add_argument("--validate", action="store_true", help="re-solve cached truths first; generate only if seams pass")
    ap.add_argument("--seam-tol", type=float, default=1e-3, help="max rel-L2 vs cached truths allowed before generating")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    solver = None
    if args.validate:
        solver = make_solver(args)
        worst = validate(args, solver)
        assert worst < args.seam_tol, f"seam {worst:.3e} >= {args.seam_tol:g}: not generating truths with this solver"

    todo = []
    for c in cases("fig4g", Ns=args.N, phis=args.phis, seeds=args.seeds):
        out = truth_path(c["N"], c["phi"], c["seed"], "gravity")
        src = truth_path(c["N"], c["phi"], c["seed"], "random")
        if out.exists():
            continue
        assert src.exists(), f"no cached configuration for {c}: {src}"
        todo.append((c, src, out))
    print(f"[cases] {len(todo)} gravity truths to compute: "
          + ", ".join(f"phi={phi:g}:{sum(np.isclose(c['phi'], phi) for c, _, _ in todo)}" for phi in PHIS
                      if any(np.isclose(c["phi"], phi) for c, _, _ in todo)))
    if args.dry_run or not todo:
        return
    solver = solver or make_solver(args)
    t_start = time.time()
    for i, (c, src, out) in enumerate(todo):
        d = np.load(src)
        config = np.array(d["config"], dtype=np.float64)
        forces = np.tile(GRAVITY_FT, (config.shape[0], 1))
        t0 = time.time()
        vel, info = solve(solver, config, forces)
        wall = time.time() - t0
        np.savez(out, config=config, forces=forces, velocity=vel, forcing="gravity", phi=c["phi"], N=c["N"],
                 seed=c["seed"], wall=wall, **provenance(args, info, src))
        print(f"[truth {i + 1}/{len(todo)}] N={c['N']} phi={c['phi']:g} seed={c['seed']}: iters={info.iters} "
              f"maxdv={info.max_rel_dv:.1e} {wall:.1f} s -> {out.name}  ({time.time() - t_start:.0f} s total)", flush=True)


if __name__ == "__main__":
    main()
