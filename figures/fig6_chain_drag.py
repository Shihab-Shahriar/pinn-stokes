#!/usr/bin/env python3
"""Figure 6 refresh: horizontal line-of-spheres drag test (Fig. 1 of Durlofsky et al. 1987),
regenerated with the new NeMO stack: Mob_Op_Nbody_Moments with the pc8 moments pair model
(nbody_moments_v2_kinf_rc8_pc8.pt) + learned per-particle diagonal (nbody_diag_v2_pc8.pt),
pair_cutoff = switch_dist = 8.

15 unit spheres on a line at center-to-center spacing 4.0, unit force perpendicular to the
line on every sphere; the instantaneous drag coefficient is lambda_i = F/(6 pi mu a U_i).
Truth is the reference MFS solver (src/mfs.py, the original figure's generator) at both
"fine" and "Xfine" discretizations, tol 1e-9; errors are quoted against Xfine, with the
fine-vs-Xfine gap bounding the discretization error. The paper's previous operator
(Mob_Op_Nbody with nbody_pinn_b1.pt, switch 6 — exactly what drew the published figure)
and the 2-body-only stack are evaluated alongside for the old-vs-new comparison.

    TORCH_COMPILE_DISABLE=1 python figures/fig6_chain_drag.py     # eval + figure
    python figures/fig6_chain_drag.py --plot-only                 # re-render from the CSV

Outputs: figures/fig6_chain_drag.{pdf,png}, per-sphere coefficients in
figures/fig6_chain_drag.csv, and a drop-in copy at the paper's include name
figures/horizontal_chain_drag_3.{pdf,png} (plot style matches the published figure:
half-chain, indexed outermost 0 -> central 7).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path = [p for p in sys.path if p != str(ROOT)]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "src"))  # grpy_tensors is imported bare by mob_op_2b_combined
os.chdir(ROOT)
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")  # accuracy work

N_SPHERES = 15
SPACING = 4.0
SELF_PATH = "data/models/self_interaction_model.pt"
TWO_BODY_PATH = "data/models/two_body_combined_model.pt"
OPS = ["2b", "b1_paper", "moments", "diag"]
OP_LABELS = {"2b": "2-body only", "b1_paper": "n-body b1 (published figure)",
             "moments": "moments pc8", "diag": "moments pc8 + diag (NeMO)"}

# Digitized from Fig. 1 of Durlofsky et al. (1987), central sphere -> chain end
# (8 unique values of the mirror-symmetric 15-chain).
DURLOFSKY_CENTER_TO_END = [0.5018, 0.5029, 0.5054, 0.5102, 0.5183, 0.5321, 0.5559, 0.6170]
# Reference MFS values hard-coded in src/horizontal_chain.py (acc "fine", tol 1e-7),
# kept as a consistency check on the recomputed truth.
LEGACY_FINE = [0.61787316, 0.55555363, 0.53188422, 0.51888724, 0.51095464,
               0.50609488, 0.50343569, 0.502587, 0.50343569, 0.50609488,
               0.51095464, 0.51888724, 0.53188422, 0.55555363, 0.61787315]


def chain_positions() -> np.ndarray:
    pos = np.zeros((N_SPHERES, 3))
    pos[:, 0] = SPACING * np.arange(N_SPHERES)
    return pos


def drag_from_uz(u_z: np.ndarray) -> np.ndarray:
    # lambda = F/(6 pi mu a U), F = 1, mu = 1, a = 1, U = -u_z (force is -z)
    return -1.0 / (6.0 * np.pi * u_z)


def mfs_drags(pos: np.ndarray, acc: str, tol: float = 1e-9) -> np.ndarray:
    from src.mfs import imp_mfs_mobility_vec
    from src.mfs_utils import build_B

    boundary = np.loadtxt(f"data/points/b_sphere_{acc}.txt", dtype=np.float64)
    source = np.loadtxt(f"data/points/s_sphere_{acc}.txt", dtype=np.float64)
    B_inv = np.linalg.pinv(build_B(boundary, source, np.zeros(3)))
    b_list = [boundary + p for p in pos]
    s_list = [source + p for p in pos]
    F = [np.array([0.0, 0.0, -1.0])] * N_SPHERES
    T = [np.zeros(3)] * N_SPHERES
    V = imp_mfs_mobility_vec(b_list, s_list, F, T, [B_inv] * N_SPHERES,
                             max_iter=2000, tol=tol)
    M = source.shape[0]
    u_z = np.array([V[i][3 * M + 2] for i in range(N_SPHERES)])
    return drag_from_uz(u_z)


MOMENTS_PATH = "data/models/nbody_moments_v2_kinf_rc8_pc8.pt"
DIAG_PATH = "data/models/nbody_diag_v2_pc8.pt"


def build_ops(moments_path: str = MOMENTS_PATH, diag_path: str = DIAG_PATH):
    from src.mob_op_2b_combined import NNMob
    from src.mob_op_nbody import Mob_Op_Nbody
    from src.mob_op_nbody_moments import Mob_Op_Nbody_Moments

    common = dict(shape="sphere", self_nn_path=SELF_PATH, two_nn_path=TWO_BODY_PATH)
    mom = dict(common, nbody_nn_path=moments_path,
               switch_dist=8.0, pair_cutoff=8.0, neighbor_cutoff=8.0, max_neighbors=None)
    return {"2b": NNMob(**common),
            "b1_paper": Mob_Op_Nbody(**common, nbody_nn_path="data/models/nbody_pinn_b1.pt",
                                     switch_dist=6.0),
            "moments": Mob_Op_Nbody_Moments(**mom),
            "diag": Mob_Op_Nbody_Moments(**mom, diag_nn_path=diag_path, diag_cutoff=8.0)}


def op_drags(pos: np.ndarray, moments_path: str = MOMENTS_PATH,
             diag_path: str = DIAG_PATH) -> dict[str, np.ndarray]:
    import torch

    config = np.zeros((N_SPHERES, 7))
    config[:, :3] = pos
    config[:, 6] = 1.0  # identity quaternion — the operators are scalar-last (as in fig2_nbody_acc.py)
    forces = np.zeros((N_SPHERES, 6))
    forces[:, 2] = -1.0
    out = {}
    for name, op in build_ops(moments_path, diag_path).items():
        with torch.no_grad():
            v = op.apply(config, forces, 1.0)
        v = np.asarray(v.cpu() if torch.is_tensor(v) else v, dtype=np.float64)
        out[name] = drag_from_uz(v[:, 2])
    return out


def evaluate(moments_path: str = MOMENTS_PATH, diag_path: str = DIAG_PATH) -> pd.DataFrame:
    pos = chain_positions()
    print("MFS truth (fine, tol 1e-9) ...", flush=True)
    lam_fine = mfs_drags(pos, "fine")
    print("MFS truth (Xfine, tol 1e-9) ...", flush=True)
    lam_xfine = mfs_drags(pos, "Xfine")
    lam_ops = op_drags(pos, moments_path, diag_path)

    durlofsky = np.array(DURLOFSKY_CENTER_TO_END)
    lam_durlofsky = np.concatenate([durlofsky[:0:-1], durlofsky])  # mirrored to all 15

    df = pd.DataFrame({"sphere": np.arange(N_SPHERES),
                       "lambda_mfs_fine": lam_fine,
                       "lambda_mfs_xfine": lam_xfine,
                       "lambda_durlofsky": lam_durlofsky,
                       **{f"lambda_{k}": lam_ops[k] for k in OPS}})
    return df


def report(df: pd.DataFrame) -> None:
    truth = df["lambda_mfs_xfine"].values
    legacy_gap = np.abs(df["lambda_mfs_fine"].values - np.array(LEGACY_FINE)).max()
    disc_gap = np.abs(df["lambda_mfs_fine"] / truth - 1).max() * 100
    print(f"\nMFS fine vs legacy hard-coded values: max |dlambda| = {legacy_gap:.2e}")
    print(f"MFS discretization (fine vs Xfine):   max rel = {disc_gap:.4f} %")

    print(f"\n{'operator':34s} {'max rel err %':>13s} {'mean rel err %':>14s} {'asymmetry %':>12s}")
    rows = [("Durlofsky et al. (digitized)", df["lambda_durlofsky"].values)] + \
           [(OP_LABELS[k], df[f"lambda_{k}"].values) for k in OPS]
    for label, lam in rows:
        rel = np.abs(lam / truth - 1) * 100
        asym = np.abs(lam / lam[::-1] - 1).max() * 100
        print(f"{label:34s} {rel.max():13.3f} {rel.mean():14.3f} {asym:12.2g}")

    print("\nper-sphere relative error vs MFS Xfine (%), outermost (0) -> central (7):")
    half = pd.DataFrame({"sphere": np.arange(8)})
    for k in OPS:
        half[k] = np.abs(df[f"lambda_{k}"].values[:8] / truth[:8] - 1) * 100
    print(half.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def plot(df: pd.DataFrame, out_stem: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    half = slice(0, 8)  # outermost (sphere 0) -> central (sphere 7)
    fig, ax = plt.subplots(figsize=(9.5, 6.0), dpi=300)
    ax.plot(np.arange(8), df["lambda_durlofsky"].values[half], "x")
    ax.plot(np.arange(8), df["lambda_diag"].values[half])
    ax.legend(["Durlofsky et al.", "NeMO"], fontsize=13)
    ax.set_xlabel("Sphere number", fontsize=16)
    ax.set_ylabel("λ", rotation=0, fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=14, width=1.1, length=5)
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)
    fig.tight_layout()
    for stem in (out_stem, Path("figures/horizontal_chain_drag_3")):  # paper include name
        for ext in ("pdf", "png"):
            fig.savefig(stem.with_suffix(f".{ext}"), dpi=900)
    print(f"-> {out_stem}.{{pdf,png}} + figures/horizontal_chain_drag_3.{{pdf,png}}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("figures/fig6_chain_drag"))
    ap.add_argument("--plot-only", action="store_true", help="re-render from the existing CSV")
    ap.add_argument("--moments-model", default=MOMENTS_PATH, help="pair moments model (.pt)")
    ap.add_argument("--diag-model", default=DIAG_PATH, help="diagonal model (.pt)")
    args = ap.parse_args()
    csv = args.out.with_suffix(".csv")
    if args.plot_only:
        df = pd.read_csv(csv)
    else:
        df = evaluate(args.moments_model, args.diag_model)
        df.to_csv(csv, index=False, float_format="%.10g")
        print(f"-> {csv}")
    report(df)
    plot(df, args.out)


if __name__ == "__main__":
    main()
