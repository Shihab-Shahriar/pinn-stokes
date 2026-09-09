#!/usr/bin/env python3
"""Ring-like array of P spheres (Jordan & Lockerby, JCP 520 (2025) 113487, Section 4.2): NeMO vs RPY
against the paper's exact IMP-MFS coefficients.

Geometry (Fig. 10, R = mu = 1): a planar regular P-gon of unit spheres with surface separation S between
neighbours; side L = S + 2, ring radius R_c = L / (2 sin(pi/P)), sphere p at angle
theta_p = (p-1) 2pi/P measured clockwise from +y, centre (R_c sin theta_p, R_c cos theta_p, 0);
tangent t_p = (cos theta_p, -sin theta_p, 0).  Sedimentation mobility problem: the same force on every
sphere, F = (0,-1,0) "parallel" (in the ring plane, Fig. 11a) or F = (0,0,-1) "perpendicular"
(Fig. 11b).  By symmetry the per-sphere response is fixed by five coefficients (Eqs. 59-60):
    parallel:      V_x = M_o F sin 2theta,  V_y = -M_par F + M_o F cos 2theta,  Omega_z = N_zt F sin theta
    perpendicular: V_z = -M_perp F,          Omega = -N_tz F t_p
Tables B.9/B.10 (data/ring_array_jordan2025.csv, values x 6 pi mu R or x 6 pi mu R^2) are the reference;
touching rings (S/R = 0) are outside the learned operators' training range (surface gap >= 0.1) and skipped.

    TORCH_COMPILE_DISABLE=1 python figures/fig_ring_array.py [--ops ...] [--P ...] [--grid b10|b9|all]
    python figures/fig_ring_array.py --zero-crossing          # S* where M_o = 0 for P = 7 (exact 1.58 R)
    python figures/fig_ring_array.py --plot-only

Results accumulate in figures/fig_ring_array.csv (one row per op, P, S; --skip-done by default).
Figures: fig_ring_array_A (Fig.-20-style velocity/rotation pattern, exact / NeMO / RPY),
fig_ring_array_B (coefficients vs S/R), fig_ring_array_C (global M vs P), fig_ring_array_err (relative
error vs S/R).
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
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "src"))
os.chdir(ROOT)
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

REF = "data/ring_array_jordan2025.csv"
OUT = "figures/fig_ring_array.csv"
SELF_PATH = "data/models/self_interaction_model.pt"
TWO_BODY_PATH = "data/models/two_body_combined_model.pt"
MOMENTS_PATH = "data/models/nbody_moments_v2_kinf_rc8_pc8c.pt"
DIAG_PATH = "data/models/nbody_diag_v2_pc8c.pt"
COEFS = ["M_par", "M_perp", "M_o", "N_tz", "N_zt"]
COEF_TEX = {"M_par": r"$M_\parallel$", "M_perp": r"$M_\perp$", "M_o": r"$M_\circ$",
            "N_tz": r"$N_{tz}$", "N_zt": r"$N_{zt}$"}
OPS = ["rpy", "2b", "nemo", "b1_paper"]
OP_LABELS = {"rpy": "RPY", "2b": "2-body only", "nemo": "NeMO", "b1_paper": "NeMO b1 (published)"}
OP_STYLE = {"rpy": dict(color="#0173B2", ls="--", lw=1.7), "2b": dict(color="#CC78BC", ls="-.", lw=1.2),
            "nemo": dict(color="#D55E00", ls="-", lw=1.9), "b1_paper": dict(color="#949494", ls=":", lw=1.4)}
B10_P = [3, 4, 5, 6, 7, 8, 9, 10]
B10_S = [0.1, 0.2, 0.5, 1, 2, 5, 10, 1000]
B9_P = [20, 50, 100, 200]
B9_S = [0.5, 1, 2, 5, 10, 100, 1000]
SIXPI = 6 * np.pi


# ----------------------------------------------------------------------------- geometry
def ring(P: int, S: float, theta1: float = 0.0):
    L = S + 2.0
    Rc = L / (2.0 * np.sin(np.pi / P))
    theta = theta1 + np.arange(P) * 2 * np.pi / P
    pos = np.stack([Rc * np.sin(theta), Rc * np.cos(theta), np.zeros(P)], axis=1)
    return pos, theta, Rc


def tangents(theta):
    return np.stack([np.cos(theta), -np.sin(theta), np.zeros_like(theta)], axis=1)


def config_of(pos):
    cfg = np.zeros((len(pos), 7))
    cfg[:, :3] = pos
    cfg[:, 6] = 1.0  # identity quaternion, scalar-last
    return cfg


def forces(P: int, orientation: str, F: float = 1.0):
    f = np.zeros((P, 6))
    f[:, 1 if orientation == "par" else 2] = -F
    return f


# ----------------------------------------------------------------------------- coefficient fits
def fit_parallel(theta, v, F: float = 1.0):
    """(M_par, M_o, N_zt, residual) from per-sphere [U, Omega] under F = (0,-F,0); values x 6 pi."""
    s2, c2 = np.sin(2 * theta), np.cos(2 * theta)
    # joint least squares on V_x = M_o F sin2t, V_y = -M_par F + M_o F cos2t  (unknowns M_par, M_o)
    A = np.concatenate([np.stack([np.zeros_like(s2), F * s2], 1), np.stack([-F * np.ones_like(c2), F * c2], 1)])
    b = np.concatenate([v[:, 0], v[:, 1]])
    (M_par, M_o), *_ = np.linalg.lstsq(A, b, rcond=None)
    st = np.sin(theta)
    N_zt = (v[:, 5] @ st) / (F * (st @ st))
    fit = np.concatenate([A @ np.array([M_par, M_o]), N_zt * F * st])
    data = np.concatenate([v[:, 0], v[:, 1], v[:, 5]])
    # everything not in the symmetry form: misfit + the components that must vanish (V_z, Omega_x, Omega_y)
    res = np.sqrt(np.sum((fit - data) ** 2) + np.sum(v[:, 2] ** 2) + np.sum(v[:, 3:5] ** 2)) / np.linalg.norm(data)
    return SIXPI * M_par, SIXPI * M_o, SIXPI * N_zt, res


def fit_perpendicular(theta, v, F: float = 1.0):
    """(M_perp, N_tz, residual) from per-sphere [U, Omega] under F = (0,0,-F); values x 6 pi."""
    t = tangents(theta)
    M_perp = -v[:, 2].mean() / F
    om_t = np.sum(v[:, 3:] * t, axis=1)
    N_tz = -om_t.mean() / F
    fit_v = np.zeros_like(v)
    fit_v[:, 2] = -M_perp * F
    fit_v[:, 3:] = -N_tz * F * t
    res = np.linalg.norm(v - fit_v) / np.linalg.norm(v)
    return SIXPI * M_perp, SIXPI * N_tz, res


def exact_velocities(P, S, coefs, orientation, F: float = 1.0):
    """Per-sphere [U, Omega] reconstructed from table coefficients (x 6 pi units) via Eqs. 59-60."""
    pos, theta, _ = ring(P, S)
    v = np.zeros((P, 6))
    if orientation == "par":
        v[:, 0] = coefs["M_o"] * F * np.sin(2 * theta) / SIXPI
        v[:, 1] = (-coefs["M_par"] * F + coefs["M_o"] * F * np.cos(2 * theta)) / SIXPI
        v[:, 5] = coefs["N_zt"] * F * np.sin(theta) / SIXPI
    else:
        v[:, 2] = -coefs["M_perp"] * F / SIXPI
        v[:, 3:] = -coefs["N_tz"] * F * tangents(theta) / SIXPI
    return v


# ----------------------------------------------------------------------------- operators
def build_op(name: str):
    from src.mob_op_2b_combined import NNMob
    from src.mob_op_nbody import Mob_Op_Nbody
    from src.mob_op_nbody_moments import Mob_Op_Nbody_Moments

    common = dict(shape="sphere", self_nn_path=SELF_PATH, two_nn_path=TWO_BODY_PATH)
    if name == "rpy":
        return NNMob(**common, rpy_only=True)
    if name == "2b":
        return NNMob(**common)
    if name == "nemo":
        return Mob_Op_Nbody_Moments(**common, nbody_nn_path=MOMENTS_PATH, switch_dist=8.0, pair_cutoff=8.0,
                                    neighbor_cutoff=8.0, max_neighbors=None, diag_nn_path=DIAG_PATH,
                                    diag_cutoff=8.0)
    if name == "b1_paper":
        return Mob_Op_Nbody(**common, nbody_nn_path="data/models/nbody_pinn_b1.pt", switch_dist=6.0)
    raise ValueError(name)


def apply_op(op, cfg, f):
    import torch
    with torch.no_grad():
        v = op.apply(cfg, f, 1.0)
    return np.asarray(v.cpu() if torch.is_tensor(v) else v, dtype=np.float64)


def coefficients(op, P: int, S: float) -> dict:
    pos, theta, Rc = ring(P, S)
    cfg = config_of(pos)
    M_par, M_o, N_zt, res_par = fit_parallel(theta, apply_op(op, cfg, forces(P, "par")))
    M_perp, N_tz, res_perp = fit_perpendicular(theta, apply_op(op, cfg, forces(P, "perp")))
    return dict(M_par=M_par, M_perp=M_perp, M_o=M_o, N_tz=N_tz, N_zt=N_zt, res_par=res_par, res_perp=res_perp,
                Rc=Rc)


def load_ref() -> pd.DataFrame:
    d = pd.read_csv(REF)
    return d.pivot_table(index=["table", "P", "S_over_R"], columns="coef", values="value").reset_index()


def sweep(ops, grid, P_filter=None, skip_done=True):
    ref = load_ref()
    cells = []
    if grid in ("b10", "all"):
        cells += [("B10", P, S) for P in B10_P for S in B10_S]
    if grid in ("b9", "all"):
        cells += [("B9", P, S) for P in B9_P for S in B9_S]
    if P_filter:
        cells = [c for c in cells if c[1] in P_filter]
    if skip_done and Path(OUT).exists():
        done = pd.read_csv(OUT)
    else:
        done = pd.DataFrame({"op": pd.Series(dtype=str), "P": pd.Series(dtype=int), "S_over_R": pd.Series(dtype=float)})
    for name in ops:
        todo = [c for c in cells if not ((done.op == name) & (done.P == c[1]) & (np.isclose(done.S_over_R, c[2]))).any()]
        if not todo:
            continue
        op = build_op(name)
        for table, P, S in todo:
            t0 = time.time()
            c = coefficients(op, P, S)
            r = ref[(ref.table == table) & (ref.P == P) & np.isclose(ref.S_over_R, S)].iloc[0]
            row = dict(op=name, table=table, P=P, S_over_R=S, **c)
            for k in COEFS:
                if k in r and not pd.isna(r[k]):
                    row[f"{k}_ref"] = r[k]
                    row[f"{k}_relerr"] = abs(c[k] - r[k]) / abs(r[k]) if r[k] != 0 else np.nan
            pd.DataFrame([row]).to_csv(OUT, mode="a", header=not Path(OUT).exists(), index=False)
            print(f"{name:9s} P={P:4d} S={S:<6g} M_par {c['M_par']:.5f} ({r['M_par']:.5f})  M_perp {c['M_perp']:.5f}"
                  f" ({r['M_perp']:.5f})  M_o {c['M_o']:+.5f} ({r.get('M_o', np.nan):+.5f})  N_tz {c['N_tz']:+.5f}"
                  f" ({r.get('N_tz', np.nan):+.5f})  N_zt {c['N_zt']:+.5f} ({r.get('N_zt', np.nan):+.5f})"
                  f"  res {c['res_par']:.1e}/{c['res_perp']:.1e}  [{time.time() - t0:.1f}s]", flush=True)


def zero_crossing(ops, P: int = 7, lo: float = 0.5, hi: float = 5.0):
    """S* with M_o(P, S*) = 0 by bisection (the exact value is 1.58 R for P = 7, Section 4.2.3)."""
    out = {}
    for name in ops:
        op = build_op(name)
        f = lambda S: coefficients(op, P, S)["M_o"]
        a, b = lo, hi
        fa, fb = f(a), f(b)
        if np.sign(fa) == np.sign(fb):
            print(f"{name}: M_o has the same sign at S={a} ({fa:+.5f}) and S={b} ({fb:+.5f}); no crossing")
            out[name] = np.nan
            continue
        for _ in range(18):
            m = 0.5 * (a + b)
            fm = f(m)
            if np.sign(fm) == np.sign(fa):
                a, fa = m, fm
            else:
                b, fb = m, fm
        out[name] = 0.5 * (a + b)
        print(f"{name}: S*(P={P}) = {out[name]:.3f} R  (exact 1.58 R)", flush=True)
    pd.DataFrame([dict(P=P, **out)]).to_csv("figures/fig_ring_array_zero_crossing.csv", index=False)
    return out


# ----------------------------------------------------------------------------- figures
def _panel_ring(ax, P, S, v, title, scale, omega_scale, lim=None):
    from matplotlib.patches import Circle
    pos, theta, Rc = ring(P, S)
    vrel = v[:, :2] - v[:, :2].mean(axis=0)
    for p in range(P):
        col = plt_cmap((v[p, 5] / omega_scale + 1) / 2) if omega_scale > 0 else "#cfe8cf"
        ax.add_patch(Circle(pos[p, :2], 1.0, facecolor=col, edgecolor="k", lw=0.8, zorder=1))
    ax.quiver(pos[:, 0], pos[:, 1], vrel[:, 0], vrel[:, 1], angles="xy", scale_units="xy", scale=1.0 / scale,
              color="k", width=0.012, zorder=3)
    ax.add_patch(Circle((0, 0), Rc, fill=False, ls="--", color="0.6", lw=0.8, zorder=0))
    ax.set_aspect("equal")
    lim = Rc + 2.2 if lim is None else lim
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=9.5)


plt_cmap = None


def plot_all(res: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    global plt_cmap
    plt_cmap = plt.get_cmap("coolwarm")
    ref = load_ref()

    # ---- Figure A: Fig.-20-style pattern, exact / NeMO / RPY, for P = 6 and P = 10 at S = 0.5 R
    cells = [(6, 0.5), (10, 0.5)]
    cols = ["exact", "nemo", "rpy"]
    fig, axes = plt.subplots(len(cells), len(cols), figsize=(3.6 * len(cols), 3.7 * len(cells)))
    for i, (P, S) in enumerate(cells):
        r = ref[(ref.table == "B10") & (ref.P == P) & np.isclose(ref.S_over_R, S)].iloc[0]
        vex = exact_velocities(P, S, r, "par")
        scale = 0.55 * ring(P, S)[2] / np.abs(vex[:, :2] - vex[:, :2].mean(0)).max()
        om = np.abs(vex[:, 5]).max()
        # frame: fit the longest arrow of any column (RPY overshoots the exact pattern)
        reach = ring(P, S)[2] + 1.0
        for name in cols[1:]:
            row = res[(res.op == name) & (res.P == P) & np.isclose(res.S_over_R, S)]
            if not row.empty:
                vv = exact_velocities(P, S, row.iloc[0], "par")
                pos = ring(P, S)[0]
                tips = pos[:, :2] + scale * (vv[:, :2] - vv[:, :2].mean(0))
                reach = max(reach, np.abs(tips).max() + 0.6)
        lim = max(ring(P, S)[2] + 2.2, reach)
        for j, name in enumerate(cols):
            if name == "exact":
                v, title = vex, f"exact (Jordan & Lockerby)   P = {P}, S = {S} R"
            else:
                row = res[(res.op == name) & (res.P == P) & np.isclose(res.S_over_R, S)]
                if row.empty:
                    axes[i, j].axis("off"); continue
                row = row.iloc[0]
                v = exact_velocities(P, S, row, "par")
                e_v = np.linalg.norm((v[:, :2] - v[:, :2].mean(0)) - (vex[:, :2] - vex[:, :2].mean(0))) / np.linalg.norm(vex[:, :2] - vex[:, :2].mean(0))
                e_w = np.linalg.norm(v[:, 5] - vex[:, 5]) / np.linalg.norm(vex[:, 5])
                title = f"{OP_LABELS[name]}:  M$_\\circ$ err {abs(row['M_o'] - row['M_o_ref']) / abs(row['M_o_ref']) * 100:.0f} %, " \
                        f"$\\Omega_z$ err {e_w * 100:.0f} %, $M_\\parallel$ err {row['M_par_relerr'] * 100:.1f} %"
            _panel_ring(axes[i, j], P, S, v, title, scale, om, lim=lim)
    fig.suptitle("Ring sedimenting parallel to its plane (force −y): velocity relative to the centre of mass (arrows, common scale per row)"
                 " and $\\Omega_z$ (colour, blue −, red +)", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"figures/fig_ring_array_A.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure B: coefficients vs S/R for P = 4, 6, 8, 10
    Ps = [4, 6, 8, 10]
    fig, axes = plt.subplots(len(COEFS), len(Ps), figsize=(3.4 * len(Ps), 2.5 * len(COEFS)), sharex=True)
    for j, P in enumerate(Ps):
        rP = ref[(ref.table == "B10") & (ref.P == P) & (ref.S_over_R <= 10)].sort_values("S_over_R")
        for i, k in enumerate(COEFS):
            ax = axes[i, j]
            ax.axvspan(0.09, 2, color="0.93", zorder=0)
            ax.plot(rP.S_over_R, rP[k], "ko", ms=4.5, label="exact", zorder=4)
            for name in ("nemo", "rpy", "2b"):
                d = res[(res.op == name) & (res.P == P) & (res.S_over_R <= 10)].sort_values("S_over_R")
                if not d.empty:
                    ax.plot(d.S_over_R, d[k], label=OP_LABELS[name], **OP_STYLE[name])
            ax.set_xscale("log")
            if k == "M_o":
                ax.axhline(0, color="0.5", lw=0.6)
            if i == 0:
                ax.set_title(f"P = {P}", fontsize=10)
            if j == 0:
                ax.set_ylabel(COEF_TEX[k] + (r" $\cdot 6\pi\mu R$" if k.startswith("M") else r" $\cdot 6\pi\mu R^2$"), fontsize=9)
            if i == len(COEFS) - 1:
                ax.set_xlabel("S / R")
            ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Ring-array mobility coefficients vs sphere separation (shaded: near field, S ≤ 2R)", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"figures/fig_ring_array_B.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure C: global M_par, M_perp vs P (both tables) for several S
    Ss = [0.5, 1, 2, 5, 10]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, k in zip(axes, ["M_par", "M_perp"]):
        for S in Ss:
            rr = ref[np.isclose(ref.S_over_R, S)].sort_values("P")
            rr = rr.drop_duplicates("P")
            ax.plot(rr.P, rr[k], "k.", ms=5, zorder=4)
            for name in ("nemo", "rpy"):
                d = res[(res.op == name) & np.isclose(res.S_over_R, S)].sort_values("P")
                if not d.empty:
                    ax.plot(d.P, d[k], **OP_STYLE[name])
            ax.text(rr.P.max() * 1.15, rr[k].iloc[-1], f"S/R = {S:g}", fontsize=8, va="center")
        ax.set_xscale("log"); ax.set_xlabel("P"); ax.set_ylabel(COEF_TEX[k] + r" $\cdot 6\pi\mu R$"); ax.grid(alpha=0.25)
    from matplotlib.lines import Line2D
    axes[0].legend([Line2D([], [], color="k", marker=".", ls="none"), Line2D([], [], **OP_STYLE["nemo"]), Line2D([], [], **OP_STYLE["rpy"])],
                   ["exact", "NeMO", "RPY"], fontsize=9, loc="upper left")
    fig.suptitle("Global ring mobility vs number of spheres", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"figures/fig_ring_array_C.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure err: relative error vs S/R (mean over P = 3..10) per coefficient
    fig, axes = plt.subplots(1, len(COEFS), figsize=(3.2 * len(COEFS), 3.4), sharey=True)
    for ax, k in zip(axes, COEFS):
        for name in ("nemo", "2b", "rpy", "b1_paper"):
            d = res[(res.op == name) & (res.table == "B10") & (res.S_over_R <= 10)]
            if d.empty:
                continue
            g = d.groupby("S_over_R")[f"{k}_relerr"].mean()
            ax.plot(g.index, 100 * g.values, marker="o", ms=3.5, label=OP_LABELS[name], **OP_STYLE[name])
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(COEF_TEX[k], fontsize=10); ax.set_xlabel("S / R"); ax.grid(alpha=0.25)
    axes[0].set_ylabel("relative error, mean over P = 3…10  (%)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"figures/fig_ring_array_err.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def add_scaled_errors(res: pd.DataFrame) -> pd.DataFrame:
    """M_o crosses zero with P, so its plain relative error explodes at the crossing cells; report it
    relative to the coefficient's scale at that S, max_P |M_o,exact(P, S)| over the B.10 rows."""
    ref = load_ref()
    scale = ref[ref.table == "B10"].groupby("S_over_R")["M_o"].apply(lambda x: np.abs(x).max())
    res = res.copy()
    res["M_o_relerr"] = np.abs(res["M_o"] - res["M_o_ref"]) / res["S_over_R"].map(scale)
    return res


def summary(res: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, d in res[res.table == "B10"].groupby("op"):
        for label, sel in [("S<=2 (near field)", d.S_over_R <= 2), ("S>2", (d.S_over_R > 2) & (d.S_over_R <= 10)), ("all S<=10", d.S_over_R <= 10)]:
            dd = d[sel]
            row = dict(op=name, cells=label, n=len(dd))
            for k in COEFS:
                row[f"{k} mean %"] = 100 * dd[f"{k}_relerr"].mean()
                row[f"{k} max %"] = 100 * dd[f"{k}_relerr"].max()
            row["max fit residual"] = max(dd.res_par.max(), dd.res_perp.max())
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ops", nargs="*", default=["rpy", "2b", "nemo", "b1_paper"], choices=OPS)
    ap.add_argument("--grid", default="all", choices=["b10", "b9", "all"])
    ap.add_argument("--P", nargs="*", type=int, default=None)
    ap.add_argument("--no-skip-done", action="store_true")
    ap.add_argument("--zero-crossing", action="store_true")
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    import torch
    torch.set_num_threads(args.threads)
    if args.zero_crossing:
        zero_crossing(args.ops)
        return
    if not args.plot_only:
        sweep(args.ops, args.grid, args.P, skip_done=not args.no_skip_done)
    res = pd.read_csv(OUT).drop_duplicates(["op", "P", "S_over_R"], keep="last")
    res = add_scaled_errors(res)
    with pd.option_context("display.width", 250, "display.float_format", "{:.2f}".format):
        print(summary(res).to_string(index=False))
    plot_all(res)
    print("wrote figures/fig_ring_array_{A,B,C,err}.{png,pdf}")


if __name__ == "__main__":
    main()
