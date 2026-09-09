#!/usr/bin/env python
"""Large-N Fig 4 ground truth via the widebvh Broms MFS solver (sphere_mfs).

The legacy truth path (benchmarks/cluster.py -> imp_mfs_mobility_sphere_triton)
truncates the Oseen sum at L_cut=25 radii, an N-dependent bias once the box
outgrows it (N=2000 boxes are 35-70 radii). The widebvh MFS covers all pairs
with a degree-7 barycentric treecode (no cutoff) and matches this repo's
conventions exactly: radius 1, mu=1, Oseen 1/(8 pi mu), U=F/6pi, Omega=T/8pi.

Configurations and forces replicate generate_uniform_testcase bit-for-bit
(same RNG calls in the same order); only the solver differs. Truths land in
the harness cache (tmp/nbody_moments_truth/uniform_N{N}_phi{phi:g}_seed{S}.npz)
with acc="broms_Xfine" plus solver provenance, and are drop-in for
benchmarks/paper_accuracy_v2.py.

Needs a GPU node and a built sphere_mfs (widebvh, PETSc CUDA build):
  python benchmarks/broms_truth.py --validate              # solver-vs-cached-truth seam check first
  python benchmarks/broms_truth.py                         # generate the large-N fig4 cells
  python benchmarks/broms_truth.py --phis 0.05 --N 2000    # one cell subset
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from benchmarks.paper_accuracy_v2 import FIG4_N, PHIS, TRUTH_DIR, cases, truth_path  # noqa: E402

DEFAULT_N = [300, 500, 1000, 1500, 2000, 2500, 3000]
DEFAULT_PHIS = list(PHIS)  # all 8; the run is resumable (existing truth files are skipped)
GRAV_PHIS = [0.025, 0.05, 0.1, 0.15]  # gravity cells cover the plotted set over the full N grid
GRAVITY_FT = np.array([0.0, 0.0, -9.81, 0.0, 0.0, 0.0])  # the repo's sedimentation wrench (T=0)
DEFAULT_BIN = os.environ.get("BROMS_SPHERE_MFS",
                             "/mnt/ffs24/home/khanmd/programs/widebvh/build-rel/sphere_mfs")
# Truth-grade treecode env. sphere_mfs's own setenv(...,0) defaults silently pick the extra
# `skel` low-rank approximation; split-warpspec is the plain (whole-sum) treecode path.
SOLVER_ENV = {"TC_PATH": "split-warpspec", "TC_BVH_BUILDER": "sah", "MFS_TIMING": "0"}


def make_case(N: int, phi: float, seed: int):
    """(config (N,7), forces (N,6)) exactly as generate_uniform_testcase draws them (no solve)."""
    from benchmarks.cluster import uniform_sphere_cluster

    np.random.seed(seed)  # legacy global stream: consumed by the force draws only
    centers, orients = uniform_sphere_cluster(phi, N, seed=seed)
    F = [np.random.uniform(-1, 1, 3).astype(np.float64) for _ in centers]
    T = [np.random.uniform(-1, 1, 3).astype(np.float64) for _ in centers]
    F = [f / np.linalg.norm(f) for f in F]
    T = [t / np.linalg.norm(t) for t in T]
    quats = np.stack([o.as_quat(scalar_first=False) for o in orients])
    config = np.concatenate([centers, quats], axis=1).astype(np.float64)
    forces = np.concatenate([np.stack(F), np.stack(T)], axis=1).astype(np.float64)
    return config, forces


def make_case_forcing(N: int, phi: float, seed: int, forcing: str):
    """Case for the requested forcing. Gravity reuses the random-forcing truth's config when cached
    (identical positions by construction, and it skips the O(N^2) RSA re-generation)."""
    if forcing == "random":
        return make_case(N, phi, seed)
    base = truth_path(N, phi, seed)  # the random-forcing file for the same cell
    config = np.array(np.load(base)["config"]) if base.exists() else make_case(N, phi, seed)[0]
    return config, np.tile(GRAVITY_FT, (N, 1))


def solve(config: np.ndarray, forces: np.ndarray, args) -> tuple[np.ndarray, str]:
    """One sphere_mfs mobility solve: (N,6) velocities + the KSP convergence line."""
    N = config.shape[0]
    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)
    centers_f, wrench_f, out_f = work / "centers.f64", work / "wrench.f64", work / "uom.f64"
    config[:, :3].astype("<f8").tofile(centers_f)
    forces.astype("<f8").tofile(wrench_f)
    out_f.unlink(missing_ok=True)

    cmd = [args.sphere_mfs,
           f"--centers={centers_f}", f"--wrench={wrench_f}",
           f"--collocation=data/points/b_sphere_{args.clouds}.txt",
           f"--source=data/points/s_sphere_{args.clouds}.txt",
           "--treecode", f"--tc-mac={args.mac}", f"--ksp-rtol={args.rtol}",
           f"--dump-uom={out_f}",
           "-ksp_gmres_restart", str(args.restart), "-ksp_converged_reason"]
    if args.check:
        cmd.append("--check")
    if args.bc_check is not None:
        cmd.append("--bc-check" if args.bc_check == 0 else f"--bc-check={args.bc_check}")
    env = {**os.environ, **SOLVER_ENV}
    if args.solver_env:  # the binary needs widebvh's module set (GCC 14.3 ABI, PETSc); env.sh module-purges,
        cmd = ["bash", "-c",  # so it runs in the solver's subshell only, never in this python's environment
               f"source {shlex.quote(args.solver_env)} >/dev/null 2>&1; exec {shlex.join(cmd)}"]

    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if args.verbose or r.returncode != 0:
        print(r.stdout[-4000:])
        print(r.stderr[-2000:], file=sys.stderr)
    assert r.returncode == 0, f"sphere_mfs failed (exit {r.returncode})"
    ksp = [l for l in r.stdout.splitlines() if "Linear solve" in l or "converged" in l.lower()]
    assert not any("DIVERGED" in l for l in ksp), f"KSP diverged: {ksp}"
    for line in r.stdout.splitlines():
        if "check]" in line or "gate]" in line or "bc-check" in line:
            print(f"    {line.strip()}")
    vel = np.fromfile(out_f, dtype="<f8")
    assert vel.size == 6 * N, f"dump-uom has {vel.size} doubles, expected {6 * N}"
    return vel.reshape(N, 6), (ksp[-1].strip() if ksp else "")


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def provenance(args) -> dict:
    sha = ""
    try:
        d = Path(args.sphere_mfs).resolve().parents[1]
        sha = subprocess.run(["git", "-C", str(d), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
    except OSError:
        pass
    return dict(acc=f"broms_{args.clouds}", solver="widebvh sphere_mfs", tc_mac=args.mac,
                ksp_rtol=args.rtol, restart=args.restart, tc_path=SOLVER_ENV["TC_PATH"],
                widebvh_sha=sha,
                cluster_md5=hashlib.md5(open(ROOT / "benchmarks" / "cluster.py", "rb").read()).hexdigest())


def validate(args) -> None:
    """Re-solve existing cached truths from their stored config+forces; prints the solver seam.

    A convention bug shows as O(1); the legacy L_cut=25 truncation shows as up to ~1e-2 at
    low phi (box side > 25) and shrinks as phi grows.
    """
    picks = [Path(p) for p in args.validate] if args.validate else []
    if not picks:
        for N in (200, 300):
            for phi in ("0.025", "0.05", "0.1", "0.15", "0.2"):
                found = sorted(TRUTH_DIR.glob(f"uniform_N{N}_phi{phi}_seed*.npz"))
                picks.extend(found[:1])
    print(f"[validate] {len(picks)} cached truths, solver={args.sphere_mfs}")
    for p in picks:
        d = np.load(p)
        config, forces, v_ref = d["config"], d["forces"], d["velocity"]
        t0 = time.time()
        v_new, ksp = solve(np.array(config), np.array(forces), args)
        print(f"  {p.name:<44} relL2 all={rel_l2(v_new, v_ref):.3e}  "
              f"lin={rel_l2(v_new[:, :3], v_ref[:, :3]):.3e}  "
              f"ang={rel_l2(v_new[:, 3:], v_ref[:, 3:]):.3e}  ({time.time() - t0:.1f} s)  {ksp}",
              flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--N", type=int, nargs="+", default=None,
                    help="default: the large-N set (random forcing) / the full FIG4_N grid (gravity)")
    ap.add_argument("--phis", type=float, nargs="+", default=None,
                    help="default: all 8 (random forcing) / the 4 plotted (gravity)")
    ap.add_argument("--forcing", choices=["random", "gravity"], default="random",
                    help="gravity = F=(0,0,-9.81), T=0 on every particle; same configs, _grav truth files")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--sphere-mfs", default=DEFAULT_BIN)
    ap.add_argument("--solver-env", default=None,
                    help="env.sh sourced in the solver subshell (default: <sphere_mfs>/../../env.sh; '' disables)")
    ap.add_argument("--clouds", default="Xfine", choices=["coarse", "medium", "fine", "Xfine"])
    ap.add_argument("--mac", type=float, default=0.3)
    ap.add_argument("--rtol", type=float, default=1e-10)
    ap.add_argument("--restart", type=int, default=200)
    ap.add_argument("--check", action="store_true", help="also run brute; O(N^2 M^2), small N only")
    ap.add_argument("--bc-check", type=int, nargs="?", const=0, default=None,
                    help="boundary-condition residual on a fine off-collocation grid")
    ap.add_argument("--validate", nargs="*", default=None,
                    help="re-solve cached truths (paths, or auto-pick N=200/300) instead of generating")
    ap.add_argument("--workdir", default=str(ROOT / "tmp" / f"broms_truth_work_{os.getpid()}"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    if args.N is None:
        args.N = list(FIG4_N) if args.forcing == "gravity" else DEFAULT_N
    if args.phis is None:
        args.phis = GRAV_PHIS if args.forcing == "gravity" else DEFAULT_PHIS
    assert Path(args.sphere_mfs).exists(), f"sphere_mfs binary not found: {args.sphere_mfs}"
    if args.solver_env is None:
        cand = Path(args.sphere_mfs).resolve().parents[1] / "env.sh"
        args.solver_env = str(cand) if cand.exists() else ""

    if args.validate is not None:
        validate(args)
        return

    todo = [c for c in cases("fig4", Ns=args.N, phis=args.phis, seeds=args.seeds)
            if not truth_path(c["N"], c["phi"], c["seed"], args.forcing).exists()]
    todo.sort(key=lambda c: (-c["N"], c["phi"], c["seed"]))  # expensive cells first (job-timeout friendly)
    print(f"[broms_truth] {len(todo)} truths to generate -> {TRUTH_DIR}")
    if args.dry_run:
        for c in todo:
            print(f"  N={c['N']} phi={c['phi']:g} seed={c['seed']}")
        return

    TRUTH_DIR.mkdir(parents=True, exist_ok=True)
    prov = provenance(args)
    for i, c in enumerate(todo):
        t0 = time.time()
        config, forces = make_case_forcing(c["N"], c["phi"], c["seed"], args.forcing)
        velocity, ksp = solve(config, forces, args)
        wall = time.time() - t0
        p = truth_path(c["N"], c["phi"], c["seed"], args.forcing)
        np.savez(p, config=config, forces=forces, velocity=velocity, forcing=args.forcing,
                 phi=c["phi"], N=c["N"], seed=c["seed"], wall=wall, ksp=ksp, **prov)
        print(f"[truth {i + 1}/{len(todo)}] N={c['N']} phi={c['phi']:g} seed={c['seed']}: "
              f"{wall:.1f} s  {ksp} -> {p}", flush=True)


if __name__ == "__main__":
    main()
