#!/usr/bin/env python3
"""Stokesian Dynamics (Townsend, JOSS 9(94) 6011, 2024; github.com/Pecnut/stokesian-dynamics) as a NeMO
harness mobility operator.

One static mobility solve per `apply`: the SD grand resistance matrix R = (M_inf)^-1 + R_2B,exact of
Durlofsky, Brady & Bossis (1987) -- a multipole far field truncated at the stresslet (FTS) level plus the
tabulated exact two-sphere lubrication resistance for pairs closer than `cutoff_factor` (a1 + a2) --
converted to FTE form (E_inf = 0) and solved for (U, Omega) from the given (F, T). Non-periodic (unbounded
fluid), monodisperse spheres of radius 1, mu = `viscosity`: the same conventions as this repo's MFS truths
(U = F / 6 pi mu a for an isolated sphere), so the (N,7) config / (N,6) wrench / (N,6) velocity layouts pass
straight through. Dense O(N^2) assembly (numba) and an O(N^3) inverse + solve on 11N unknowns; N = 200 is
seconds, N = 300 tens of seconds.

The SD checkout is referenced, never copied or edited: `SD_ROOT` (default /home/shihab/throwaway/stokesian-dynamics)
is put on sys.path lazily. Two of its import-time habits are worked around here: `settings.py` parses
sys.argv[1:] as (setup, input, timestep, frames) and sys.exit()s on anything else, and it turns numba off
(`config.DISABLE_JIT = True`, read when the kernels are decorated, so it is switched back on before the
`functions.*` modules import). Nothing from `setups.positions` is imported (it runs a setup + checkpoint glob).

    python src/sd_ops.py --case 200 0.1 4423      # one cached truth: SD vs MFS, timing breakdown
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SD_ROOT = Path(os.environ.get("SD_ROOT", "/home/shihab/throwaway/stokesian-dynamics"))
DEFAULT_CUTOFF = 2  # SD's default: R2Bexact for centre distance < cutoff_factor * (a1 + a2)
_SD = None


def sd_package_dir(root=None) -> Path:
    return Path(root or SD_ROOT) / "stokesian_dynamics"


def sd_version(root=None) -> str:
    try:
        return subprocess.run(["git", "-C", str(root or SD_ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _import_sd(root=None):
    """Import the SD solver pieces once (module-level cache): argv-safe, numba on."""
    global _SD
    if _SD is not None:
        return _SD
    pkg = sd_package_dir(root)
    assert (pkg / "settings.py").exists(), f"Stokesian Dynamics checkout not found at {pkg} (set SD_ROOT)"
    if str(pkg) not in sys.path:
        sys.path.insert(0, str(pkg))
    argv = sys.argv
    sys.argv = argv[:1]
    try:
        import settings  # noqa: F401  (parses sys.argv; sets numba DISABLE_JIT = True)
    finally:
        sys.argv = argv
    from numba import config as nb_config
    nb_config.DISABLE_JIT = os.environ.get("SD_DISABLE_JIT", "0") == "1"  # pure-Python kernels: debugging only
    from functions.generate_grand_resistance_matrix import generate_grand_resistance_matrix
    from functions.shared import add_sphere_rotations_to_positions
    from functions.simulation_tools import (construct_force_vector_from_fts, deconstruct_velocity_vector_for_fts,
                                            fts_to_fte_matrix)
    _SD = SimpleNamespace(generate_grand_resistance_matrix=generate_grand_resistance_matrix,
                          add_sphere_rotations_to_positions=add_sphere_rotations_to_positions,
                          construct_force_vector_from_fts=construct_force_vector_from_fts,
                          deconstruct_velocity_vector_for_fts=deconstruct_velocity_vector_for_fts,
                          fts_to_fte_matrix=fts_to_fte_matrix, package_dir=pkg)
    return _SD


class SDMob:
    """`apply(config (N,7), forces (N,6), viscosity) -> (N,6)` = [U, Omega], the harness contract.

    cutoff_factor: lubrication switch-on distance in units of (a1 + a2); SD default 2.
    minfinity_only: drop R2Bexact (far-field FTS multipole only) -- a diagnostic variant, not the SD baseline.
    """

    def __init__(self, cutoff_factor: float = DEFAULT_CUTOFF, minfinity_only: bool = False, radius: float = 1.0,
                 sd_root=None):
        self.cutoff_factor = cutoff_factor
        self.minfinity_only = bool(minfinity_only)
        self.radius = float(radius)
        self.sd_root = Path(sd_root or SD_ROOT)
        self.version = sd_version(self.sd_root)
        self.last_info: dict = {}
        _import_sd(self.sd_root)

    @property
    def name(self) -> str:
        return "SD_Minf" if self.minfinity_only else "SD"

    def posdata(self, positions: np.ndarray):
        sd = _import_sd(self.sd_root)
        pos = np.ascontiguousarray(positions, dtype=np.float64)
        sizes = np.full(pos.shape[0], self.radius, dtype=np.float64)
        rot = sd.add_sphere_rotations_to_positions(pos, sizes, np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))
        return (sizes, pos, rot, np.array([]), np.empty([0, 3]), np.empty([0, 3]))

    def resistance_matrix_fte(self, positions: np.ndarray, viscosity: float = 1.0):
        """The FTE-form grand resistance matrix (11N x 11N) and SD's per-stage timings."""
        sd = _import_sd(self.sd_root)
        posdata = self.posdata(positions)
        R, _heading, _minf_inv, gen_times = sd.generate_grand_resistance_matrix(
            posdata, [], regenerate_Minfinity=True, cutoff_factor=self.cutoff_factor, printout=0,
            use_drag_Minfinity=False, use_Minfinity_only=self.minfinity_only, frameno=0, mu=viscosity)
        t0 = time.time()
        R = sd.fts_to_fte_matrix(posdata, R)
        return posdata, R, {"minfinity_s": gen_times[0], "minfinity_inv_s": gen_times[1], "r2bexact_s": gen_times[2],
                            "fts_to_fte_s": time.time() - t0}

    def apply(self, config, forces, viscosity: float = 1.0):
        sd = _import_sd(self.sd_root)
        config = np.asarray(config, dtype=np.float64)
        forces = np.asarray(forces, dtype=np.float64)
        N = config.shape[0]
        assert config.shape[1] >= 3 and forces.shape == (N, 6), (config.shape, forces.shape)
        t0 = time.time()
        posdata, R, info = self.resistance_matrix_fte(config[:, :3], viscosity)
        t1 = time.time()
        fv = sd.construct_force_vector_from_fts(posdata, forces[:, :3].tolist(), forces[:, 3:].tolist(),
                                                np.zeros((N, 3, 3)), [], [])
        vv = np.linalg.solve(R, np.asarray(fv, dtype=np.float64))
        U, O, _E, _Ub, _dUb = sd.deconstruct_velocity_vector_for_fts(posdata, vv)
        info.update({"solve_s": time.time() - t1, "wall_s": time.time() - t0, "N": N})
        self.last_info = info
        return np.column_stack([np.asarray(U), np.asarray(O)]).astype(np.float64)


def _main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", nargs=3, metavar=("N", "PHI", "SEED"), default=["200", "0.1", "4423"])
    ap.add_argument("--forcing", choices=["gravity", "random"], default="gravity")
    ap.add_argument("--cutoff", type=float, default=DEFAULT_CUTOFF)
    ap.add_argument("--minfinity-only", action="store_true")
    args = ap.parse_args()
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    from benchmarks.compare_nbody_moments import compute_error_stats
    from benchmarks.paper_accuracy_v2 import truth_path

    N, phi, seed = int(args.case[0]), float(args.case[1]), int(args.case[2])
    p = truth_path(N, phi, seed, args.forcing)
    d = np.load(p)
    op = SDMob(cutoff_factor=args.cutoff, minfinity_only=args.minfinity_only)
    print(f"[sd] checkout {op.sd_root} @ {op.version}; truth {p.name} (acc={d['acc']})")
    v = op.apply(d["config"], d["forces"], 1.0)  # first call includes the numba JIT
    v = op.apply(d["config"], d["forces"], 1.0)
    s = compute_error_stats(v, d["velocity"])
    print(f"[sd] {op.name} vs MFS: rel_rmse={s['rel_rmse']:.3f}%  lin={s['prmse_lin']:.3f}%  ang={s['prmse_ang']:.3f}%  "
          f"fluct={s['prmse_fluct']:.3f}%  max_lin={s['max_rel_lin']:.2f}%")
    print("[sd] timings " + "  ".join(f"{k}={t:.2f}" for k, t in op.last_info.items() if k.endswith("_s")))
    u0 = np.linalg.norm(d["forces"][0, :3]) / (6 * math.pi)
    print(f"[sd] mean |U| / U_stokes = {np.linalg.norm(v[:, :3], axis=1).mean() / u0:.3f} (MFS "
          f"{np.linalg.norm(d['velocity'][:, :3], axis=1).mean() / u0:.3f})")


if __name__ == "__main__":
    _main()
