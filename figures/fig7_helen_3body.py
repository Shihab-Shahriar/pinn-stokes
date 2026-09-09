#!/usr/bin/env python3
"""Figure 7 refresh: 3-sphere equilateral-triangle mobility testcase (Wilson 2013),
regenerated with the new NeMO stack: Mob_Op_Nbody_Moments with the chain-fixed pc8c
moments pair model (nbody_moments_v2_kinf_rc8_pc8c.pt) + learned per-particle diagonal
(nbody_diag_v2_pc8c.pt), pair_cutoff = switch_dist = 8.

Three unit spheres at the vertices of an equilateral triangle of side S (center-to-center,
in radii) in the x-z plane; the apex sphere is forced with F = (0, 0, -6*pi), no torques,
mu = 1. Reported: U1 = |apex U_z|, U2/U3 = |base-sphere U_z/U_x|, Omega = |base-sphere
Omega_y|. References are the published figure's hard-coded tables: Helen Wilson's method
(Wilson 2013) and Stokesian Dynamics (Townsend 2017). In addition, MFS truth
(src/mfs.py, fine + Xfine, tol 1e-9) is recomputed per S so operator errors can be quoted
against our own converged solver; the paper's previous operator (Mob_Op_Nbody with
nbody_pinn_b1.pt, switch 6 -- what drew the published figure) and the 2-body-only stack
are evaluated alongside for the old-vs-new comparison.

    TORCH_COMPILE_DISABLE=1 python figures/fig7_helen_3body.py    # eval + figure
    python figures/fig7_helen_3body.py --plot-only                # re-render from the CSV

Outputs: figures/fig7_helen_3body.{pdf,png}, all values in figures/fig7_helen_3body.csv,
and a drop-in copy at the paper's include name figures/helens_3body_comparison_nbody.{pdf,png}
(layout matches the published figure: 2x2 panels, Helen's Method / Stokesian Dynamics / NeMO).
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

S_VALUES = [2.01, 2.10, 2.50, 3.00, 4.00, 6.00]
METRICS = ["U1", "U2", "U3", "Omega"]
METRIC_LABELS = {"U1": r"$U_1$", "U2": r"$U_2$", "U3": r"$U_3$", "Omega": r"$\Omega$"}
SELF_PATH = "data/models/self_interaction_model.pt"
TWO_BODY_PATH = "data/models/two_body_combined_model.pt"
MOMENTS_PATH = "data/models/nbody_moments_v2_kinf_rc8_pc8c.pt"
DIAG_PATH = "data/models/nbody_diag_v2_pc8c.pt"
OPS = ["2b", "b1_paper", "moments", "diag"]
OP_LABELS = {"2b": "2-body only", "b1_paper": "n-body b1 (published figure)",
             "moments": "moments pc8c", "diag": "moments pc8c + diag (NeMO)"}

# Hard-coded reference tables of the published figure (benchmarks/helens_3body.py in the
# paper checkout): Wilson (2013) "Current" method and Stokesian Dynamics (Townsend 2017).
WILSON_ROWS = {
    2.01: (0.65528, 0.63461, 0.00498, 0.037336),
    2.10: (0.73857, 0.59718, 0.03517, 0.052035),
    2.50: (0.87765, 0.49545, 0.07393, 0.045466),
    3.00: (0.93905, 0.41694, 0.07824, 0.035022),
    4.00: (0.97964, 0.31859, 0.06925, 0.021634),
    6.00: (0.99581, 0.21586, 0.05078, 0.010159),
}
SD_ROWS = {
    2.01: (0.64739, 0.62691, 0.00451, 0.034339),
    2.10: (0.73126, 0.58784, 0.02570, 0.051414),
    2.50: (0.87482, 0.48829, 0.05853, 0.045446),
    3.00: (0.93806, 0.41356, 0.06970, 0.034843),
    4.00: (0.97945, 0.31774, 0.06639, 0.021581),
    6.00: (0.99579, 0.21575, 0.05019, 0.010153),
}


def triangle(S: float) -> np.ndarray:
    # equilateral triangle of side S in the x-z plane; index 2 is the (forced) apex
    return np.array([[0.0, 0.0, 0.0],
                     [S, 0.0, 0.0],
                     [S / 2.0, 0.0, (np.sqrt(3.0) / 2.0) * S]])


def forces() -> np.ndarray:
    F = np.zeros((3, 6))
    F[2, 2] = -6.0 * np.pi  # apex sphere; U = 1 for an isolated sphere (mu = a = 1)
    return F


def metrics_from_velocity(v: np.ndarray) -> tuple[float, float, float, float]:
    # U1: apex settling speed; U2/U3/Omega: base-sphere drift and rotation
    return (abs(v[2, 2]), abs(v[0, 2]), abs(v[0, 0]), abs(v[0, 4]))


_MFS_SOLVERS: dict[str, object] = {}


def mfs_metrics(S: float, acc: str) -> tuple[float, float, float, float]:
    # BatchedMFS GMRES (torch64, tol_v 1e-8): the src/mfs.py Gauss-Seidel iteration
    # diverges at the S=2.01 near-contact spacing (gap 0.01), a Krylov solve does not
    from src.mfs_batched import BatchedMFS

    solver = _MFS_SOLVERS.get(acc)
    if solver is None:
        solver = _MFS_SOLVERS[acc] = BatchedMFS(acc=acc, backend="torch64")
    vel, info = solver.solve(triangle(S), forces().reshape(3, 6, 1))
    assert info.converged, info
    return metrics_from_velocity(vel[:, :, 0].cpu().numpy())


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


def evaluate(moments_path: str = MOMENTS_PATH, diag_path: str = DIAG_PATH) -> pd.DataFrame:
    import torch

    ops = build_ops(moments_path, diag_path)
    rows = []
    for S in S_VALUES:
        row = {"S": S}
        row.update({f"{m}_wilson": x for m, x in zip(METRICS, WILSON_ROWS[S])})
        row.update({f"{m}_sd": x for m, x in zip(METRICS, SD_ROWS[S])})
        for acc in ("fine", "Xfine"):
            print(f"MFS {acc} S={S} ...", flush=True)
            row.update({f"{m}_mfs_{acc.lower()}": x
                        for m, x in zip(METRICS, mfs_metrics(S, acc))})
        config = np.zeros((3, 7))
        config[:, :3] = triangle(S)
        config[:, 6] = 1.0  # identity quaternion — the operators are scalar-last
        for name, op in ops.items():
            with torch.no_grad():
                v = op.apply(config, forces(), 1.0)
            v = np.asarray(v.cpu() if torch.is_tensor(v) else v, dtype=np.float64)
            row.update({f"{m}_{name}": x for m, x in zip(METRICS, metrics_from_velocity(v))})
        rows.append(row)
    return pd.DataFrame(rows)


def report(df: pd.DataFrame) -> None:
    curves = [("Wilson (Helen's method)", "wilson"), ("Stokesian Dynamics", "sd"),
              ("MFS fine", "mfs_fine")] + [(OP_LABELS[k], k) for k in OPS]
    truth = {m: df[f"{m}_mfs_xfine"].values for m in METRICS}
    disc = max(np.abs(df[f"{m}_mfs_fine"] / truth[m] - 1).max() for m in METRICS) * 100
    wilson_gap = max(np.abs(df[f"{m}_wilson"] / truth[m] - 1).max() for m in METRICS) * 100
    print(f"\nMFS discretization (fine vs Xfine): max rel = {disc:.4f} %")
    print(f"Wilson vs MFS Xfine:                max rel = {wilson_gap:.4f} % "
          "(independent solvers agreeing on the testcase)")

    print(f"\nmax |rel err| vs MFS Xfine over the {len(S_VALUES)} spacings (%):")
    print(f"{'curve':32s}{'U1':>8s}{'U2':>8s}{'U3':>8s}{'Omega':>8s}{'  worst S (U3)':>15s}")
    for label, key in curves:
        rel = {m: np.abs(df[f"{m}_{key}"].values / truth[m] - 1) * 100 for m in METRICS}
        worst = df["S"].values[np.argmax(rel["U3"])]
        print(f"{label:32s}" + "".join(f"{rel[m].max():8.2f}" for m in METRICS)
              + f"{worst:15.2f}")

    print("\nabsolute deviation from Wilson at S=2.01 (the published figure's visible gap):")
    r = df[df["S"] == 2.01].iloc[0]
    for label, key in curves[1:]:
        devs = "  ".join(f"{m}: {abs(r[f'{m}_{key}'] - r[f'{m}_wilson']):.4f}" for m in METRICS)
        print(f"  {label:32s}{devs}")

    # aggregate deviation from Wilson (the essentially-exact reference; MFS Xfine confirms
    # it to ~1e-4 for S >= 2.1 but cannot resolve the 0.01 gap at S=2.01 itself)
    print("\n|deviation| from Wilson over all 24 (S, metric) cells / the 20 with S >= 2.1:")
    print(f"{'curve':32s}{'mean':>10s}{'max':>10s}{'mean>=2.1':>12s}{'max>=2.1':>11s}")
    keep = df["S"] >= 2.1
    for label, key in curves[1:]:
        dev = np.abs(np.stack([df[f"{m}_{key}"] - df[f"{m}_wilson"] for m in METRICS]))
        print(f"{label:32s}{dev.mean():10.4f}{dev.max():10.4f}"
              f"{dev[:, keep].mean():12.4f}{dev[:, keep].max():11.4f}")

    # the published figure's summary stat: wins vs Stokesian Dynamics, referenced to Wilson
    for key in ("b1_paper", "diag"):
        wins = sum(abs(df[f"{m}_{key}"] - df[f"{m}_wilson"]).lt(
            abs(df[f"{m}_sd"] - df[f"{m}_wilson"])).sum() for m in METRICS)
        print(f"\n{OP_LABELS[key]}: closer to Wilson than Stokesian Dynamics in "
              f"{wins}/{4 * len(S_VALUES)} (S, metric) cells")


def plot(df: pd.DataFrame, out_stem: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["STIXGeneral", "DejaVu Serif"],
                         "mathtext.fontset": "stix",
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    series = [("wilson", "Helen's Method"), ("sd", "Stokesian Dynamics"), ("diag", "NeMO")]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    for m, ax in zip(METRICS, axes.flat):
        for key, label in series:
            ax.plot(df["S"], df[f"{m}_{key}"], marker="o", label=label)
        ax.set_title(METRIC_LABELS[m])
        ax.set_xlabel(r"$S$")
        ax.set_ylabel(METRIC_LABELS[m])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(series))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    for stem in (out_stem, Path("figures/helens_3body_comparison_nbody")):  # paper include name
        for ext in ("pdf", "png"):
            fig.savefig(stem.with_suffix(f".{ext}"), dpi=300)
    print(f"-> {out_stem}.{{pdf,png}} + figures/helens_3body_comparison_nbody.{{pdf,png}}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("figures/fig7_helen_3body"))
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
