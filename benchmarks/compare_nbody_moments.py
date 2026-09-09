"""Grand-mobility accuracy: moments-based n-body correction vs the baseline, against MFS truth.

Warp-free (CPU operators + Triton MFS truth only), so it runs natively on a box without
warp as well as inside ``bash docker/run_local.sh``.  Truth is generated once per
(N, phi, seed) with ``benchmarks.cluster.generate_uniform_testcase`` (Xfine MFS, the same
generator/seeds as ``accuracy_grand_M.run_experiment_fixed_size_diff_operators``) and cached
in ``tmp/nbody_moments_truth/`` so operator variants can be re-run cheaply.

    python benchmarks/compare_nbody_moments.py --only-truth            # generate/cache truth
    python benchmarks/compare_nbody_moments.py                         # all ops, all configs
    python benchmarks/compare_nbody_moments.py --ops M_2b M_mom_k10_rc6 --phis 0.1 --seeds 123
    python benchmarks/compare_nbody_moments.py --summary               # re-print table from CSV
    python benchmarks/compare_nbody_moments.py --symmetry              # reciprocity checks only

Run with TORCH_COMPILE_DISABLE=1 (accuracy work).  Results are appended to
``data/nbody_moments_compare.csv`` (long format; one row per N/phi/seed/op, later runs replace
earlier rows with the same key).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path = [p for p in sys.path if p != str(ROOT)]
sys.path.insert(0, str(ROOT))  # this repo's src/ must shadow any other `src` package on the path
os.chdir(ROOT)

SHAPE = "sphere"
SELF_PATH = "data/models/self_interaction_model.pt"
TWO_BODY_PATH = "data/models/two_body_combined_model.pt"
NBODY_B1_PATH = "data/models/nbody_pinn_b1.pt"
NBODY_B1_RETRAINED_PATH = "data/models/nbody_pinn_b1_retrained.pt"
NBODY_MOMENTS_PATH = "data/models/nbody_moments.pt"

TRUTH_DIR = ROOT / "tmp" / "nbody_moments_truth"
DEFAULT_CSV = ROOT / "data" / "nbody_moments_compare.csv"

# Historic CPU n-body numbers at N=200 (data/grand_M_acc_uniform_fixed_N.csv), for orientation.
HISTORIC_REL_RMSE = {"M_2b": {0.05: 5.17, 0.10: 10.16, 0.15: 15.72, 0.20: 21.10},
                     "M_nbody": {0.05: 3.94, 0.10: 7.48, 0.15: 11.64, 0.20: 16.13}}

CONFIG_COLS = ["x", "y", "z", "q_x", "q_y", "q_z", "q_w"]
FORCE_COLS = ["f_x", "f_y", "f_z", "t_x", "t_y", "t_z"]
VEL_COLS = ["v_x", "v_y", "v_z", "w_x", "w_y", "w_z"]


# ----------------------------------------------------------------------------- truth
def truth_path(N: int, phi: float, seed: int) -> Path:
    return TRUTH_DIR / f"uniform_N{N}_phi{phi:g}_seed{seed}.npz"


def get_truth(N: int, phi: float, seed: int, regen: bool = False):
    p = truth_path(N, phi, seed)
    if p.exists() and not regen:
        d = np.load(p)
        return d["config"], d["forces"], d["velocity"]
    from benchmarks.cluster import generate_uniform_testcase

    t0 = time.time()
    df = generate_uniform_testcase(shape=SHAPE, volume_fraction=phi, numParticles=N,
                                   seed=seed, save_to_file=False)
    wall = time.time() - t0
    config = df[CONFIG_COLS].values.astype(np.float64)
    forces = df[FORCE_COLS].values.astype(np.float64)
    velocity = df[VEL_COLS].values.astype(np.float64)
    TRUTH_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(p, config=config, forces=forces, velocity=velocity, phi=phi, N=N, seed=seed,
             acc="Xfine", wall=wall)
    print(f"[truth] N={N} phi={phi} seed={seed}: {wall:.1f} s -> {p}", flush=True)
    return config, forces, velocity


# ----------------------------------------------------------------------------- metrics
def compute_error_stats(predicted, velocity):
    """Numpy branch of accuracy_grand_M._compute_error_stats (copied: that module imports warp)."""
    diff = velocity - predicted
    rmse = np.sqrt(np.mean(np.mean(diff ** 2, axis=1)))
    ref_rms = np.sqrt(np.mean(np.mean(velocity ** 2, axis=1)))
    rel_rmse = 0.0 if ref_rms < 1e-12 else (rmse / ref_rms) * 100
    mae = np.mean(np.linalg.norm(diff, axis=1))
    v_norms = np.linalg.norm(velocity, axis=1)
    mean_v_norm = np.mean(v_norms)
    rel_mae = 0.0 if mean_v_norm < 1e-12 else (mae / mean_v_norm) * 100
    with np.errstate(divide="ignore", invalid="ignore"):
        element_rel = np.nan_to_num(np.linalg.norm(diff, axis=1) / v_norms)
    max_rel_rmse = np.max(element_rel) * 100
    # per-block relative L2 (translational / rotational), not in the original harness
    lin = np.linalg.norm(diff[:, :3]) / max(np.linalg.norm(velocity[:, :3]), 1e-12) * 100
    ang = np.linalg.norm(diff[:, 3:]) / max(np.linalg.norm(velocity[:, 3:]), 1e-12) * 100
    # translational-only extras for the torque-free / gravity protocols (HIGNN predicts no angular velocity):
    # error of the mean (collective) velocity, error of the fluctuations about it, per-particle max
    dl, vl = diff[:, :3], velocity[:, :3]
    d_mean, v_mean = dl.mean(axis=0), vl.mean(axis=0)
    err_mean_pct = np.linalg.norm(d_mean) / max(np.linalg.norm(v_mean), 1e-12) * 100
    prmse_fluct = np.linalg.norm(dl - d_mean) / max(np.linalg.norm(vl - v_mean), 1e-12) * 100
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_lin = np.nan_to_num(np.linalg.norm(dl, axis=1) / np.linalg.norm(vl, axis=1))
    max_rel_lin = np.max(rel_lin) * 100
    return {"rmse": rmse, "rel_rmse": rel_rmse, "mae": mae, "rel_mae": rel_mae,
            "max_rel_rmse": max_rel_rmse, "prmse_lin": lin, "prmse_ang": ang,
            "err_mean_pct": err_mean_pct, "prmse_fluct": prmse_fluct, "max_rel_lin": max_rel_lin}


# ----------------------------------------------------------------------------- operators
def _two_body():
    from src.mob_op_2b_combined import NNMob
    return NNMob(SHAPE, SELF_PATH, TWO_BODY_PATH, nn_only=False, rpy_only=False)


def _nbody(path):
    from src.mob_op_nbody import Mob_Op_Nbody
    return Mob_Op_Nbody(shape=SHAPE, self_nn_path=SELF_PATH, two_nn_path=TWO_BODY_PATH,
                        nbody_nn_path=path, nn_only=False, rpy_only=False, switch_dist=6.0)


# NOTE: hardcoded pair_cutoff/switch_dist 6.0 -- pc6 models only. The pc8 model (nbody_moments_v2_kinf_rc8_pc8)
# must be run via benchmarks/paper_accuracy_v2.py, which passes pair_cutoff=8, switch_dist=8 from its sidecar.
def _moments(path, max_neighbors, neighbor_cutoff):
    from src.mob_op_nbody_moments import Mob_Op_Nbody_Moments
    return Mob_Op_Nbody_Moments(shape=SHAPE, self_nn_path=SELF_PATH, two_nn_path=TWO_BODY_PATH,
                                nbody_nn_path=path, nn_only=False, rpy_only=False, switch_dist=6.0,
                                pair_cutoff=6.0, neighbor_cutoff=neighbor_cutoff,
                                max_neighbors=max_neighbors)


PATHS = {"b1": NBODY_B1_PATH, "b1_retrained": NBODY_B1_RETRAINED_PATH, "moments": NBODY_MOMENTS_PATH}

OPS = {
    "M_2b": lambda: _two_body(),
    "M_nbody_b1": lambda: _nbody(PATHS["b1"]),
    "M_nbody_b1_retrained": lambda: _nbody(PATHS["b1_retrained"]),
    "M_mom_k10_rc6": lambda: _moments(PATHS["moments"], 10, 6.0),      # baseline-matched (primary)
    "M_mom_kinf_rc6": lambda: _moments(PATHS["moments"], None, 6.0),
    "M_mom_kinf_rc8": lambda: _moments(PATHS["moments"], None, 8.0),
}
MOMENT_OPS = [k for k in OPS if k.startswith("M_mom")]


# ----------------------------------------------------------------------------- csv
def append_rows(csv_path: Path, rows: list[dict]) -> pd.DataFrame:
    new = pd.DataFrame(rows)
    if csv_path.exists():
        old = pd.read_csv(csv_path)
        key = ["N", "phi", "seed", "op"]
        merged = pd.concat([old, new], ignore_index=True)
        merged = merged.drop_duplicates(subset=key, keep="last")
    else:
        merged = new
    merged = merged.sort_values(["N", "phi", "op", "seed"]).reset_index(drop=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(csv_path, index=False, float_format="%.6g")
    return merged


def print_summary(df: pd.DataFrame, ops: list[str] | None = None):
    if df.empty:
        print("(no rows)")
        return
    ops = ops or list(dict.fromkeys(df["op"]))
    for N in sorted(df["N"].unique()):
        sub = df[df["N"] == N]
        phis = sorted(sub["phi"].unique())
        print(f"\n=== N={N}: rel_rmse % (mean ± std over seeds; n = seeds) ===")
        header = f"{'op':<22}" + "".join(f"{('phi=%g' % p):>18}" for p in phis)
        print(header)
        for op in ops:
            line = f"{op:<22}"
            for p in phis:
                r = sub[(sub["op"] == op) & (sub["phi"] == p)]["rel_rmse"]
                line += f"{'-':>18}" if r.empty else f"{r.mean():8.3f} ± {r.std(ddof=0):5.3f} ({len(r)})".rjust(18)
            print(line)
        for name, hist in HISTORIC_REL_RMSE.items():
            line = f"{('historic ' + name):<22}"
            for p in phis:
                line += f"{hist[round(p, 3)]:>18.2f}" if round(p, 3) in hist else f"{'-':>18}"
            print(line)
        for col, label in [("prmse_lin", "translational rel-L2 %"), ("prmse_ang", "rotational rel-L2 %"),
                           ("max_rel_rmse", "max per-particle rel err %")]:
            print(f"\n--- N={N}: {label} (mean over seeds) ---")
            for op in ops:
                line = f"{op:<22}"
                for p in phis:
                    r = sub[(sub["op"] == op) & (sub["phi"] == p)][col]
                    line += f"{'-':>18}" if r.empty else f"{r.mean():>18.3f}"
                print(line)


# ----------------------------------------------------------------------------- symmetry
def symmetry_checks(op_names: list[str], N: int, phis: list[float], seed: int):
    """Reciprocity of the moments correction: per-pair blocks and the assembled n-body grand M."""
    from benchmarks.cluster import uniform_sphere_cluster

    for name in op_names:
        op = OPS[name]()
        print(f"\n[symmetry] {name}")
        for phi in phis:
            config, _, _ = get_truth(N, phi, seed)
            pos = config[:, :3]
            pairs, K = op.nbody_pair_blocks(pos)
            if len(pairs) == 0:
                print(f"  phi={phi}: no near pairs")
                continue
            P = len(pairs) // 2
            asym = np.abs(K[:P] - np.transpose(K[P:], (0, 2, 1))).max()
            print(f"  phi={phi}: {2 * P} ordered pairs, max|K_ts - K_st^T| / max|K| = {asym / np.abs(K).max():.3e}")
        # small-N: full grand M of the n-body term by unit forces
        centers, _ = uniform_sphere_cluster(0.2, 12, seed=0)
        n = centers.shape[0]
        M = np.zeros((6 * n, 6 * n))
        for j in range(6 * n):
            F = np.zeros((n, 6))
            F[j // 6, j % 6] = 1.0
            M[:, j] = op.get_nbody_velocity(centers, F, 1.0).reshape(-1)
        rel = np.linalg.norm(M - M.T) / max(np.linalg.norm(M), 1e-300)
        print(f"  N=12 grand M of the n-body term: ||M - M^T|| / ||M|| = {rel:.3e}")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--N", type=int, default=200)
    ap.add_argument("--phis", type=float, nargs="+", default=[0.05, 0.1, 0.15, 0.2])
    ap.add_argument("--seeds", type=int, nargs="+", default=[123, 124, 125])
    ap.add_argument("--ops", nargs="+", default=list(OPS), choices=list(OPS))
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--only-truth", action="store_true")
    ap.add_argument("--regen-truth", action="store_true")
    ap.add_argument("--summary", action="store_true", help="print the table from the CSV and exit")
    ap.add_argument("--symmetry", action="store_true", help="run reciprocity checks (moments ops) and exit")
    ap.add_argument("--moments-path", default=None, help="override the moments model file (.pt/.wt)")
    ap.add_argument("--b1-retrained-path", default=None, help="override the retrained-baseline model file")
    ap.add_argument("--tag", default="", help="suffix appended to op names in the CSV (for model variants)")
    args = ap.parse_args()
    if args.moments_path:
        PATHS["moments"] = args.moments_path
    if args.b1_retrained_path:
        PATHS["b1_retrained"] = args.b1_retrained_path

    if args.summary:
        print_summary(pd.read_csv(args.csv) if args.csv.exists() else pd.DataFrame())
        return
    if args.symmetry:
        symmetry_checks([o for o in args.ops if o in MOMENT_OPS], args.N, args.phis, args.seeds[0])
        return

    for phi in args.phis:
        for seed in args.seeds:
            get_truth(args.N, phi, seed, regen=args.regen_truth)
    if args.only_truth:
        return

    ops = {}
    for name in args.ops:
        t0 = time.time()
        ops[name] = OPS[name]()
        print(f"[op] built {name} in {time.time() - t0:.1f} s", flush=True)

    import torch
    df = None
    for phi in args.phis:
        for seed in args.seeds:
            config, forces, velocity = get_truth(args.N, phi, seed)
            rows = []
            for name, op in ops.items():
                t0 = time.time()
                with torch.no_grad():
                    pred = op.apply(config, forces, 1.0)
                wall = time.time() - t0
                stats = compute_error_stats(np.asarray(pred, dtype=np.float64), velocity)
                rows.append({"N": args.N, "phi": phi, "seed": seed, "op": name + args.tag, "wall_s": wall, **stats})
                print(f"N={args.N} phi={phi} seed={seed} {name + args.tag:<22} rel_rmse={stats['rel_rmse']:7.3f}%  "
                      f"lin={stats['prmse_lin']:7.3f}%  ang={stats['prmse_ang']:7.3f}%  "
                      f"max={stats['max_rel_rmse']:7.2f}%  ({wall:.1f} s)", flush=True)
            df = append_rows(args.csv, rows)
    if df is not None:
        print_summary(df, [o + args.tag for o in args.ops])


if __name__ == "__main__":
    main()
