#!/usr/bin/env python3
"""Dynamic companion to figures/fig_ring_array.py: a ring of P unit spheres (Jordan & Lockerby 2025,
Fig. 10 geometry, surface separation S) sedimenting *parallel* to its plane (force (0,-1,0) on every
sphere, mu = 1), integrated with RK4 on the positions.  Truth is BatchedMFS (src/mfs_batched.py) applied
at every RK4 stage; operators are the CPU NeMO / RPY / 2-body stacks of fig_ring_array.build_op.

The ring deforms at the rate set by the local anisotropy M_o (Eq. 60): leading/trailing spheres run
ahead of the centre of mass for P < 7.  The run stops after the centre of mass has fallen --fall radii
or when any surface gap drops below --gap-floor (0.1: the learned near field's training range and the
MFS point clouds' resolution).  Each sphere's accumulated rotation about z is integrated alongside.

    python figures/fig_ring_sedimentation.py --P 6 --S 0.5 --ops nemo rpy 2b        # CPU, minutes
    python figures/fig_ring_sedimentation.py --P 6 --S 0.5 --truth                  # GPU, ~1 h
    python figures/fig_ring_sedimentation.py --P 6 --S 0.5 --plot-only

Outputs: figures/fig_ring_sed_P{P}_S{S}_{op}.npz (frames: t, pos, vel, phi) and
figures/fig_ring_sed_P{P}_S{S}.{png,pdf} (storyboard at equal fall distances + deformation metrics).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path = [p for p in sys.path if p != str(ROOT)]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "src"))
os.chdir(ROOT)
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("fig_ring_array", ROOT / "figures" / "fig_ring_array.py")
fra = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fra)

LABELS = {"truth": "MFS (truth)", "nemo": "NeMO", "rpy": "RPY", "2b": "2-body only"}
COLORS = {"truth": "black", "nemo": "#D55E00", "rpy": "#0173B2", "2b": "#CC78BC"}


def out_path(P, S, name):
    return ROOT / "figures" / f"fig_ring_sed_P{P}_S{S:g}_{name}.npz"


class TruthOp:
    def __init__(self, acc="fine", backend="triton32"):
        from src.mfs_batched import BatchedMFS
        self.solver = BatchedMFS(acc=acc, backend=backend)

    def apply(self, cfg, f, mu):
        vel, info = self.solver.solve(cfg[:, :3], f.reshape(len(f), 6, 1))
        assert info.converged, info
        return vel[:, :, 0].cpu().numpy()


def integrate(op, P, S, dt, fall, gap_floor, frame_every, log=print):
    pos0, theta, Rc = fra.ring(P, S)
    pos = pos0.copy()
    phi = np.zeros(P)  # accumulated rotation about z
    f = fra.forces(P, "par")

    def vel(p):
        cfg = fra.config_of(p)
        return fra.apply_op(op, cfg, f)

    frames = []
    t, t0, next_frame = 0.0, time.time(), 0.0
    v = vel(pos)
    # cross-check at t = 0: fit the five coefficients (parallel case only gives three)
    M_par, M_o, N_zt, res = fra.fit_parallel(theta, v)
    log(f"t=0 fit: M_par {M_par:.5f} M_o {M_o:+.5f} N_zt {N_zt:+.5f} (residual {res:.1e})")
    while True:
        if t >= next_frame - 1e-9:
            d = np.linalg.norm(pos[:, None] - pos[None], axis=-1) + np.eye(P) * 1e9
            gap = d.min() - 2.0
            y_cm = pos[:, 1].mean()
            frames.append((t, pos.copy(), v.copy(), phi.copy()))
            log(f"t={t:7.1f} fallen {-y_cm:7.2f} min gap {gap:.3f}  [{time.time() - t0:.0f}s]")
            next_frame += frame_every
            if -y_cm >= fall or gap < gap_floor:
                break
        k1 = v
        p2 = pos + 0.5 * dt * k1[:, :3]; k2 = vel(p2)
        p3 = pos + 0.5 * dt * k2[:, :3]; k3 = vel(p3)
        p4 = pos + dt * k3[:, :3]; k4 = vel(p4)
        pos = pos + dt / 6.0 * (k1[:, :3] + 2 * k2[:, :3] + 2 * k3[:, :3] + k4[:, :3])
        phi = phi + dt / 6.0 * (k1[:, 5] + 2 * k2[:, 5] + 2 * k3[:, 5] + k4[:, 5])
        t += dt
        v = vel(pos)
    return dict(t=np.array([fr[0] for fr in frames]), pos=np.array([fr[1] for fr in frames]),
                vel=np.array([fr[2] for fr in frames]), phi=np.array([fr[3] for fr in frames]))


def metrics(run):
    pos = run["pos"]
    cm = pos.mean(axis=1, keepdims=True)
    rel = pos - cm
    ext_y = rel[:, :, 1].max(1) - rel[:, :, 1].min(1)
    ext_x = rel[:, :, 0].max(1) - rel[:, :, 0].min(1)
    r = np.linalg.norm(rel[:, :, :2], axis=-1)
    d = np.linalg.norm(pos[:, :, None] - pos[:, None], axis=-1) + np.eye(pos.shape[1])[None] * 1e9
    return dict(fallen=-cm[:, 0, 1], elong=ext_y / ext_x, rms_dev=np.sqrt(((r - r[0:1].mean()) ** 2).mean(1)),
                min_gap=d.min(axis=(1, 2)) - 2.0)


def plot(P, S, runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    names = [n for n in ("truth", "nemo", "rpy", "2b") if n in runs]
    mets = {n: metrics(runs[n]) for n in names}
    fmax = mets["truth"]["fallen"].max() if "truth" in mets else min(mets[n]["fallen"].max() for n in names)
    falls = np.linspace(0.0, np.floor(fmax), 4)
    fig = plt.figure(figsize=(2.9 * len(falls) + 0.5, 2.9 * len(names) + 3.4))
    gs = fig.add_gridspec(len(names) + 1, len(falls), height_ratios=[1] * len(names) + [1.1])
    Rc = fra.ring(P, S)[2]
    lim = Rc + 2.6
    for i, name in enumerate(names):
        run, m = runs[name], mets[name]
        for j, fd in enumerate(falls):
            k = int(np.argmin(np.abs(m["fallen"] - fd)))
            pinched = m["fallen"][k] < fd - 0.5  # run ended (gap floor) before this fall distance
            ax = fig.add_subplot(gs[i, j])
            rel = run["pos"][k] - run["pos"][k].mean(0)
            if name != "truth" and "truth" in runs:
                kt = int(np.argmin(np.abs(mets["truth"]["fallen"] - fd)))
                relt = runs["truth"]["pos"][kt] - runs["truth"]["pos"][kt].mean(0)
                for p in range(P):
                    ax.add_patch(Circle(relt[p, :2], 1.0, fill=False, edgecolor="k", ls=":", lw=1.0, zorder=3))
            for p in range(P):
                ax.add_patch(Circle(rel[p, :2], 1.0, facecolor="#e6e6e6" if name == "truth" else "#f7e3d6" if name == "nemo" else "#dde9f5",
                                    edgecolor=COLORS[name], lw=1.4, zorder=2))
                ph = run["phi"][k][p]
                ax.plot([rel[p, 0], rel[p, 0] + np.cos(ph)], [rel[p, 1], rel[p, 1] + np.sin(ph)], color="k", lw=1.2, zorder=4)
            ax.set_aspect("equal"); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(f"fallen {abs(m['fallen'][k]):.0f} R  (t = {run['t'][k]:.0f})", fontsize=10)
            if j == 0:
                ax.set_ylabel(LABELS[name], fontsize=11, color=COLORS[name])
            if pinched:
                ax.set_facecolor("#f4f4f4")
                ax.text(0.5, 0.06, f"pinched to gap 0.1 after {m['fallen'][k]:.1f} R\n(last frame shown)", transform=ax.transAxes,
                        fontsize=8.5, color="#a00000", ha="center", va="bottom", fontweight="bold")
            else:
                ax.text(0.03, 0.03, f"min gap {m['min_gap'][k]:.2f}", transform=ax.transAxes, fontsize=8, color="0.3")
    sub = gs[len(names), :].subgridspec(1, 3, wspace=0.35)
    axs = [fig.add_subplot(sub[0, c]) for c in range(3)]
    for name in names:
        m, kw = mets[name], dict(color=COLORS[name], lw=1.8, ls="--" if name == "truth" else "-", label=LABELS[name])
        axs[0].plot(m["fallen"], m["min_gap"], **kw)
        axs[1].plot(m["fallen"], m["elong"], **kw)
        axs[2].plot(m["fallen"], np.degrees(runs[name]["phi"][:, 1]), **kw)
    axs[0].axhline(0.1, color="0.5", lw=0.8); axs[0].text(0.5, 0.11, "gap floor 0.1 R (training range / MFS resolution)", fontsize=7.5, color="0.4")
    axs[0].set_ylabel("smallest surface gap (R)"); axs[0].legend(fontsize=8)
    axs[1].set_ylabel("ring elongation  y-extent / x-extent")
    axs[2].set_ylabel("rotation of sphere 2 about z (deg)")
    for ax in axs:
        ax.set_xlabel("distance fallen (R)"); ax.grid(alpha=0.3)
    fig.suptitle(f"Ring of P = {P} spheres, S = {S:g} R, sedimenting parallel to its plane (force −y). "
                 f"Frames relative to the centre of mass; dotted circles = truth; ticks = accumulated rotation.", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "figures" / f"fig_ring_sed_P{P}_S{S:g}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)



def error_series(runs):
    """Per-frame errors vs the truth run at common frame times: mean over spheres of |r_i - r_i^truth| (radii),
    the same after removing each ring's centre of mass (shape error), and mean |phi_i - phi_i^truth| (deg)."""
    tr = runs["truth"]
    out = {}
    for name, r in runs.items():
        if name == "truth":
            continue
        T = min(len(r["t"]), len(tr["t"]))
        assert np.allclose(r["t"][:T], tr["t"][:T])
        dpos = r["pos"][:T] - tr["pos"][:T]
        rel = (r["pos"][:T] - r["pos"][:T].mean(1, keepdims=True)) - (tr["pos"][:T] - tr["pos"][:T].mean(1, keepdims=True))
        v_cm, v_cm_t = r["vel"][:T, :, 1].mean(1), tr["vel"][:T, :, 1].mean(1)
        out[name] = dict(t=r["t"][:T], fallen=-tr["pos"][:T, :, 1].mean(1),
                         settle_rel=np.abs(v_cm - v_cm_t) / np.abs(v_cm_t),
                         pos=np.linalg.norm(dpos, axis=-1).mean(1),
                         shape=np.linalg.norm(rel, axis=-1).mean(1),
                         rot=np.degrees(np.abs(r["phi"][:T] - tr["phi"][:T])).mean(1),
                         rot_rel=np.abs(r["phi"][:T] - tr["phi"][:T]).mean(1) / np.maximum(np.abs(tr["phi"][:T]).mean(1), 1e-12))
    return out


def plot_errors(P, S, runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    err = error_series(runs)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, e in err.items():
        axes[0].plot(e["t"], e["pos"], color=COLORS[name], lw=1.9, label=f"{LABELS[name]}: position")
        axes[0].plot(e["t"], e["shape"], color=COLORS[name], lw=1.3, ls=":", label=f"{LABELS[name]}: shape (centre of mass removed)")
        axes[1].plot(e["t"], e["rot"], color=COLORS[name], lw=1.9, label=LABELS[name])
    axes[0].set_ylabel("translational error  ⟨|r$_i$ − r$_i^{MFS}$|⟩  (radii)")
    axes[1].set_ylabel("rotational error  ⟨|φ$_i$ − φ$_i^{MFS}$|⟩  (deg)")
    for ax in axes:
        ax.set_xlabel("t   (a = μ = F = 1;  the ring falls ≈ 0.13 R per unit time)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"Ring of P = {P} spheres, S = {S:g} R, sedimenting parallel to its plane: error vs the MFS run,"
                 f" averaged over the {P} spheres (each curve ends where that operator reached the 0.1 R gap floor)", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "figures" / f"fig_ring_sed_P{P}_S{S:g}_errors.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    rows = []
    for name, e in err.items():
        for k in range(len(e["t"])):
            rows.append(dict(op=name, t=e["t"][k], fallen=e["fallen"][k], settle_rel_err=e["settle_rel"][k], pos_err=e["pos"][k], shape_err=e["shape"][k],
                             rot_err_deg=e["rot"][k], rot_err_rel=e["rot_rel"][k]))
    pd.DataFrame(rows).to_csv(ROOT / "figures" / f"fig_ring_sed_P{P}_S{S:g}_errors.csv", index=False)
    return err



def plot_metrics(P, S, runs):
    """Three-panel version: settling speed (instantaneous centre-of-mass velocity, % error), shape (mean sphere
    displacement after removing the centre of mass, radii) and rotation (mean |phi_i - phi_i^MFS|, deg)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    err = error_series(runs)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.1))
    for name, e in err.items():
        axes[0].plot(e["t"], 100 * e["settle_rel"], color=COLORS[name], lw=1.9,
                     label=f"{LABELS[name]}  (mean {100 * e['settle_rel'].mean():.2f} %)")
        axes[1].plot(e["t"], e["shape"], color=COLORS[name], lw=1.9, label=LABELS[name])
        axes[2].plot(e["t"], e["rot"], color=COLORS[name], lw=1.9,
                     label=f"{LABELS[name]}  ({100 * e['rot_rel'][-1]:.0f} % of the accumulated angle)")
    axes[0].set_ylabel("settling-speed error  |ΔV$_{cm}$| / V$_{cm}^{MFS}$  (%)")
    axes[0].set_yscale("log")
    axes[0].set_title("(a) settling speed", fontsize=10)
    axes[1].set_ylabel("shape error  ⟨|r$_i$ − r$_i^{MFS}$|⟩  (radii)")
    axes[1].set_title("(b) ring shape (centre of mass removed)", fontsize=10)
    axes[2].set_ylabel("rotation error  ⟨|φ$_i$ − φ$_i^{MFS}$|⟩  (deg)")
    axes[2].set_title("(c) sphere rotation", fontsize=10)
    for ax in axes:
        ax.set_xlabel("t   (a = μ = F = 1; the ring falls ≈ 0.13 R per unit time)")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.suptitle(f"Ring of P = {P} spheres, S = {S:g} R, sedimenting parallel to its plane: errors vs the MFS run, averaged over"
                 f" the {P} spheres (each curve ends where that operator reached the 0.1 R gap floor)", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "figures" / f"fig_ring_sed_P{P}_S{S:g}_metrics.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return err



def plot_relative_storyboard(P, S, runs, falls=(4, 8, 12, 16), magnify=10.0):
    """Storyboard of *relative* errors vs the truth. Each sphere is drawn at its true position (centre of mass
    removed), filled by its relative shape error |dr_i| / RMS_j |d_j^truth| (d = displacement of the true ring
    from its initial shape), with an arrow = position error x magnify, a black tick = true rotation and a coloured
    tick = predicted rotation. Bottom row: settling-speed, shape and rotation relative errors vs fall distance."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from matplotlib import colors as mcolors

    tr = runs["truth"]
    names = [n for n in ("nemo", "rpy", "2b") if n in runs]
    rel_t = tr["pos"] - tr["pos"].mean(1, keepdims=True)
    deform_t = np.sqrt((np.linalg.norm(rel_t - rel_t[0:1], axis=-1) ** 2).mean(1))  # RMS true deformation per frame
    fallen_t = -tr["pos"][:, :, 1].mean(1)
    err = error_series(runs)
    cmap, norm = plt.get_cmap("YlOrRd"), mcolors.Normalize(0, 100)
    ncol = len(falls)
    fig = plt.figure(figsize=(2.9 * ncol + 1.2, 2.9 * len(names) + 3.3))
    gs = fig.add_gridspec(len(names) + 1, ncol, height_ratios=[1] * len(names) + [1.05])
    Rc = fra.ring(P, S)[2]
    lim = Rc + 2.6
    for i, name in enumerate(names):
        r = runs[name]
        rel = r["pos"] - r["pos"].mean(1, keepdims=True)
        T = min(len(r["t"]), len(tr["t"]))
        for j, fd in enumerate(falls):
            k = int(np.argmin(np.abs(fallen_t[:T] - fd)))
            pinched = fallen_t[k] < fd - 0.5
            ax = fig.add_subplot(gs[i, j])
            d = rel[k] - rel_t[k]
            rel_shape = np.linalg.norm(d, axis=-1) / max(deform_t[k], 1e-9)
            phi_t, phi = tr["phi"][k], r["phi"][k]
            for q in range(P):
                ax.add_patch(Circle(rel_t[k, q, :2], 1.0, facecolor=cmap(norm(100 * rel_shape[q])), edgecolor="k", lw=0.8, zorder=2))
                ax.plot([rel_t[k, q, 0], rel_t[k, q, 0] + np.cos(phi_t[q])], [rel_t[k, q, 1], rel_t[k, q, 1] + np.sin(phi_t[q])],
                        color="k", lw=1.4, zorder=4)
                ax.plot([rel_t[k, q, 0], rel_t[k, q, 0] + np.cos(phi[q])], [rel_t[k, q, 1], rel_t[k, q, 1] + np.sin(phi[q])],
                        color=COLORS[name], lw=1.4, zorder=5)
                if np.linalg.norm(d[q, :2]) * magnify > 0.05:
                    ax.annotate("", xy=rel_t[k, q, :2] + magnify * d[q, :2], xytext=rel_t[k, q, :2],
                                arrowprops=dict(arrowstyle="-|>", color=COLORS[name], lw=1.3, mutation_scale=9), zorder=6)
            e = err[name]
            ax.text(0.02, 0.02, f"ΔV$_{{cm}}$ {100 * e['settle_rel'][k]:.1f} %   shape {100 * rel_shape.mean():.0f} %   "
                                f"rot {100 * e['rot_rel'][k]:.0f} %", transform=ax.transAxes, fontsize=8, color="0.2")
            if pinched:
                ax.set_facecolor("#f4f4f4")
                ax.text(0.5, 0.9, f"pinched after {fallen_t[k]:.1f} R (last frame)", transform=ax.transAxes, fontsize=8.5,
                        color="#a00000", ha="center", fontweight="bold")
            ax.set_aspect("equal"); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(f"fallen {fallen_t[k]:.0f} R  (t = {tr['t'][k]:.0f})", fontsize=10)
            if j == 0:
                ax.set_ylabel(LABELS[name], fontsize=11, color=COLORS[name])
    sub = gs[len(names), :].subgridspec(1, 3, wspace=0.35)
    axs = [fig.add_subplot(sub[0, c]) for c in range(3)]
    for name in names:
        e = err[name]
        T = len(e["t"])
        rel_shape_curve = np.array([np.linalg.norm((runs[name]["pos"][k] - runs[name]["pos"][k].mean(0)) - rel_t[k], axis=-1).mean()
                                    / max(deform_t[k], 1e-9) for k in range(T)])
        kw = dict(color=COLORS[name], lw=1.8, label=LABELS[name])
        axs[0].plot(e["fallen"], 100 * e["settle_rel"], **kw)
        axs[1].plot(e["fallen"][1:], 100 * rel_shape_curve[1:], **kw)
        axs[2].plot(e["fallen"][1:], 100 * e["rot_rel"][1:], **kw)
    axs[0].set_ylabel("settling-speed error  |ΔV$_{cm}$| / V$_{cm}^{MFS}$  (%)"); axs[0].set_yscale("log"); axs[0].legend(fontsize=8)
    axs[1].set_ylabel("shape error / true deformation  (%)"); axs[1].set_ylim(0, 100)
    axs[2].set_ylabel("rotation error / true rotation  (%)"); axs[2].set_ylim(0, 40)
    for ax in axs:
        ax.set_xlabel("distance fallen (R)"); ax.grid(alpha=0.3, which="both")
    cax = fig.add_axes([0.92, 0.42, 0.012, 0.35])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, label="per-sphere shape error / true ring deformation  (%)")
    fig.suptitle(f"Ring of P = {P} spheres, S = {S:g} R: relative errors vs the MFS run. Spheres at the true positions (centre of mass"
                 f" removed); arrows = position error ×{magnify:g}; black tick = true rotation, coloured tick = predicted.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 0.91, 1))
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "figures" / f"fig_ring_sed_P{P}_S{S:g}_relerr.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--P", type=int, default=6)
    ap.add_argument("--S", type=float, default=0.5)
    ap.add_argument("--ops", nargs="*", default=[], choices=["nemo", "rpy", "2b"])
    ap.add_argument("--truth", action="store_true")
    ap.add_argument("--truth-acc", default="fine")
    ap.add_argument("--truth-backend", default="triton32")
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--fall", type=float, default=150.0)
    ap.add_argument("--gap-floor", type=float, default=0.1)
    ap.add_argument("--frame-every", type=float, default=5.0)
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--errors", action="store_true", help="only write the error-vs-time figure/CSV (needs the truth run)")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    import torch
    torch.set_num_threads(args.threads)
    P, S = args.P, args.S
    if args.errors:
        runs = {n: dict(np.load(out_path(P, S, n))) for n in ("truth", "nemo", "rpy", "2b") if out_path(P, S, n).exists()}
        err = plot_errors(P, S, runs)
        plot_metrics(P, S, runs)
        plot_relative_storyboard(P, S, runs)
        for name, e in err.items():
            for tq in (50.0, 95.0, e["t"][-1]):
                k = int(np.argmin(np.abs(e["t"] - tq)))
                print(f"{name:5s} t={e['t'][k]:5.0f} fallen {e['fallen'][k]:5.1f} R | settling speed err {100 * e['settle_rel'][k]:.2f} % | position err {e['pos'][k]:.3f} R"
                      f"  shape err {e['shape'][k]:.3f} R | rotation err {e['rot'][k]:5.2f} deg ({100 * e['rot_rel'][k]:.1f} %)")
        return
    if not args.plot_only:
        todo = [(n, fra.build_op(n)) for n in args.ops]
        if args.truth:
            todo.append(("truth", TruthOp(args.truth_acc, args.truth_backend)))
        for name, op in todo:
            print(f"== {name} P={P} S={S} dt={args.dt}", flush=True)
            run = integrate(op, P, S, args.dt, args.fall, args.gap_floor, args.frame_every,
                            log=lambda s: print(f"   [{name}] {s}", flush=True))
            np.savez(out_path(P, S, name), **run)
    runs = {n: dict(np.load(out_path(P, S, n))) for n in ("truth", "nemo", "rpy", "2b") if out_path(P, S, n).exists()}
    if runs:
        plot(P, S, runs)
        for n, r in runs.items():
            m = metrics(r)
            print(f"{n:6s} frames {len(r['t'])} fallen {m['fallen'][-1]:.1f} elong {m['elong'][-1]:.3f} "
                  f"min gap {m['min_gap'][-1]:.3f} phi2 {np.degrees(r['phi'][-1, 1]):.1f} deg")


if __name__ == "__main__":
    main()
