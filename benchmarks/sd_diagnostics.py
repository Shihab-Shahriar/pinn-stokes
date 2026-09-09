#!/usr/bin/env python3
"""Why Stokesian Dynamics scores the way it does on the Fig 3 protocol: the evidence behind artifacts/sd_comparison_report.md.

Four checks on src/sd_ops.py (Townsend's Stokesian Dynamics as a harness operator), each against this repo's MFS:
  pairs     two unit spheres, SD (far field + lubrication) vs its far field alone vs exact MFS (BatchedMFS, fp64,
            Xfine): SD reproduces the exact pair mobility at every separation inside its lubrication table, which
            rules out a wrapper or convention error.
  clusters  compact 3- and 8-sphere clusters, random wrench and gravity: the pairwise-additive lubrication term
            already costs a few per cent on the collective (gravity) mode of an 8-sphere cube while the far-field
            part alone is within 0.1-0.6 %.
  cutoff    the N = 200, phi = 0.1 gravity truth with SD's lubrication cutoff swept from "no pairs" (1.05) to the
            default (2): the many-body error grows monotonically with the number of pairs that receive the
            pairwise correction -- the finite-cluster inconsistency of R2B,exact - R2B,inf (Ichiki, JFM 452, 2002).
  jit       the numba kernels equal SD's pure-Python path (SD_DISABLE_JIT=1) on a 60-sphere subset.

    TORCH_COMPILE_DISABLE=1 python benchmarks/sd_diagnostics.py [--checks pairs clusters cutoff jit]
Writes artifacts/sd_diagnostics.md (needs the GPU for the small MFS solves; SD itself is CPU).
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from benchmarks.compare_nbody_moments import compute_error_stats  # noqa: E402
from benchmarks.paper_accuracy_v2 import truth_path  # noqa: E402
from src.sd_ops import SDMob  # noqa: E402

U0 = 1.0 / (6.0 * math.pi)
OUT = ROOT / "artifacts" / "sd_diagnostics.md"


def cfg(pos):
    pos = np.asarray(pos, dtype=np.float64)
    return np.hstack([pos, np.tile([0.0, 0.0, 0.0, 1.0], (len(pos), 1))])


def mfs_solver():
    from src.mfs_batched import BatchedMFS
    return BatchedMFS(acc="Xfine", backend="torch64")


def mfs_apply(m, pos, F):
    vel, info = m.solve(np.asarray(pos, dtype=np.float64), np.asarray(F, dtype=np.float64).reshape(len(pos), 6, 1))
    assert info.converged, info
    return vel[:, :, 0].cpu().numpy()


def check_pairs(sd, sdm, m) -> list[str]:
    lines = ["## Two spheres: SD vs exact (velocity of sphere 1 in units of F / 6 pi mu a)", "",
             "| r | forcing | MFS | SD | SD far field | SD - MFS | far field - MFS |", "|---|---|---|---|---|---|---|"]
    cases = [("same F along line", [[1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]], 0),
             ("same F perpendicular", [[0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0]], 2),
             ("opposite F along line", [[1, 0, 0, 0, 0, 0], [-1, 0, 0, 0, 0, 0]], 0),
             ("torque on sphere 1", [[0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0]], 5)]
    for r in [2.1, 2.5, 3.0, 3.9, 4.4]:
        pos = np.array([[0, 0, 0], [r, 0, 0]], dtype=np.float64)
        for label, F, comp in cases:
            F = np.array(F, dtype=np.float64)
            vm, vs, vf = mfs_apply(m, pos, F), sd.apply(cfg(pos), F), sdm.apply(cfg(pos), F)
            scale = U0 if comp < 3 else 1.0 / (8 * math.pi)
            a, b, c = vm[0, comp] / scale, vs[0, comp] / scale, vf[0, comp] / scale
            lines.append(f"| {r} | {label} | {a:.6f} | {b:.6f} | {c:.6f} | {b - a:+.1e} | {c - a:+.1e} |")
    return lines + [""]


def check_clusters(sd, sdm, m) -> list[str]:
    lines = ["## Compact clusters: relative error vs MFS (%), SD vs its far field alone", "",
             "| cluster | forcing | SD PRMSE | SD lin | SD ang | far-field PRMSE | far-field lin | far-field ang |",
             "|---|---|---|---|---|---|---|---|"]
    rng = np.random.default_rng(1)
    geoms = []
    for d in [2.2, 2.5, 3.0]:
        geoms.append((f"triangle, side {d}", np.array([[0, 0, 0], [d, 0, 0], [d / 2, d * math.sqrt(3) / 2, 0]], float)))
    for d in [2.3, 3.0]:
        geoms.append((f"cube of 8, spacing {d}", np.array([[i * d, j * d, k * d] for i in (0, 1) for j in (0, 1) for k in (0, 1)], float)))
    for name, pos in geoms:
        for forcing, F in [("random wrench", rng.normal(size=(len(pos), 6))),
                           ("gravity", np.tile([0, 0, -1.0, 0, 0, 0], (len(pos), 1)))]:
            vm = mfs_apply(m, pos, F)
            s1 = compute_error_stats(sd.apply(cfg(pos), F), vm)
            s2 = compute_error_stats(sdm.apply(cfg(pos), F), vm)
            lines.append(f"| {name} | {forcing} | {s1['rel_rmse']:.2f} | {s1['prmse_lin']:.2f} | {s1['prmse_ang']:.2f} "
                         f"| {s2['rel_rmse']:.2f} | {s2['prmse_lin']:.2f} | {s2['prmse_ang']:.2f} |")
    return lines + [""]


def check_cutoff() -> list[str]:
    p = truth_path(200, 0.1, 4423, "gravity")
    d = np.load(p)
    u0 = np.linalg.norm(d["forces"][0, :3]) / (6 * math.pi)
    lines = [f"## Lubrication cutoff sweep on `{p.name}` (N = 200, phi = 0.1, gravity)", "",
             "cutoff_factor = r* / (a1 + a2): pairs closer than r* receive R2B,exact - R2B,inf; 1.05 admits none of "
             "this configuration's pairs (min gap 0.1), 2 is SD's default.", "",
             "| cutoff_factor | pairs corrected (per sphere) | translational PRMSE | fluctuation PRMSE | rotational | mean settling speed / U_stokes (MFS) |",
             "|---|---|---|---|---|---|"]
    from scipy.spatial.distance import pdist
    dist = pdist(d["config"][:, :3])
    ref = np.linalg.norm(d["velocity"][:, :3], axis=1).mean() / u0
    for cf in [1.05, 1.25, 1.5, 2.0, 2.25]:
        op = SDMob(cutoff_factor=cf)
        v = op.apply(d["config"], d["forces"])
        s = compute_error_stats(v, d["velocity"])
        npairs = 2 * np.sum(dist < 2 * cf) / len(d["config"])
        lines.append(f"| {cf} | {npairs:.2f} | {s['prmse_lin']:.2f} | {s['prmse_fluct']:.2f} | {s['prmse_ang']:.2f} "
                     f"| {np.linalg.norm(v[:, :3], axis=1).mean() / u0:.2f} ({ref:.2f}) |")
    return lines + [""]


def check_jit() -> list[str]:
    d = np.load(truth_path(200, 0.1, 4423, "gravity"))
    sub = ROOT / "tmp" / "sd_jit_subset.npz"
    np.savez(sub, config=d["config"][:60], forces=d["forces"][:60])
    code = ("import sys, numpy as np; sys.path.insert(0, '.'); from src.sd_ops import SDMob; "
            "d = np.load(sys.argv[1]); np.save(sys.argv[2], SDMob().apply(d['config'], d['forces']))")
    outs = {}
    for flag in ("0", "1"):
        out = ROOT / "tmp" / f"sd_jit_{flag}.npy"
        t0 = time.time()
        subprocess.run([sys.executable, "-c", code, str(sub), str(out)], check=True,
                       env={**os.environ, "SD_DISABLE_JIT": flag, "CUDA_VISIBLE_DEVICES": ""}, capture_output=True)
        outs[flag] = (np.load(out), time.time() - t0)
    a, b = outs["0"][0], outs["1"][0]
    return ["## numba kernels vs SD's pure-Python path (60-sphere subset of the phi = 0.1 configuration)", "",
            f"max |difference| {np.abs(a - b).max():.2e}, relative L2 {np.linalg.norm(a - b) / np.linalg.norm(b):.2e} "
            f"(process wall incl. imports: numba {outs['0'][1]:.0f} s, pure Python {outs['1'][1]:.0f} s)", ""]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checks", nargs="+", default=["pairs", "clusters", "cutoff", "jit"])
    args = ap.parse_args()
    sd, sdm = SDMob(), SDMob(minfinity_only=True)
    lines = [f"# Stokesian Dynamics operator diagnostics (benchmarks/sd_diagnostics.py; SD checkout {sd.version})", ""]
    m = mfs_solver() if ({"pairs", "clusters"} & set(args.checks)) else None
    if "pairs" in args.checks:
        lines += check_pairs(sd, sdm, m)
    if "clusters" in args.checks:
        lines += check_clusters(sd, sdm, m)
    if "cutoff" in args.checks:
        lines += check_cutoff()
    if "jit" in args.checks:
        lines += check_jit()
    text = "\n".join(lines)
    print(text)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text + "\n")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
