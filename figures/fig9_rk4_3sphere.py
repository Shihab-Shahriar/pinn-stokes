#!/usr/bin/env python3
"""Figure 9 refresh: dynamic simulation of 3 falling spheres (Durlofsky et al. 1987, Fig 5),
regenerated with the new NeMO stack: Mob_Op_Nbody_Moments with the chain-fixed pc8c moments
pair model (nbody_moments_v2_kinf_rc8_pc8c.pt) + learned per-particle diagonal
(nbody_diag_v2_pc8c.pt), pair_cutoff = switch_dist = 8.

Protocol (benchmarks/rk4-3sphere.py, the published figure's integrator): three unit spheres
on the x-axis at x = -5, 0, 7, unit gravity F = (0, 0, -1) on each, mu = 1, classical RK4 on
the positions only with dt = 1 for 90 frames of 128 time units (11,520 steps; positions are
recorded every 128 steps -> 91 frames including t = 0). Plotted as x vs z ("y" in the paper).

References (data/fig9_rk4_3sphere_refs.npz, bundled from the paper checkout):
  mfs               the method-of-fundamental-solutions run (MobOpMFS coarse, same integrator)
  sd_townsend       Stokesian Dynamics, Townsend 2017 code (100fr-t128 run, first 91 frames)
  durlofsky         digitised Fig 5 of Durlofsky, Brady & Bossis 1987 (x, z only)
  published_*       the trajectories behind the published panels (nemo_b1 = Mob_Op_Nbody
                    with nbody_pinn_b1.pt at switch 6, 2b, rpy) for the old-vs-new comparison

    TORCH_COMPILE_DISABLE=1 python figures/fig9_rk4_3sphere.py --ops nemo      # ~20 min CPU
    TORCH_COMPILE_DISABLE=1 python figures/fig9_rk4_3sphere.py --ops 2b rpy    # ~3 min each
    python figures/fig9_rk4_3sphere.py --plot-only                             # re-render

Outputs: trajectories figures/fig9_rk4_3sphere_<op>.npy; the three paper panels as drop-in
copies at the paper's include names figures/rk4-dynamics.{pdf,png} (NeMO), rk4-2b.*, rk4-rpy.*;
a combined 3-panel figures/fig9_rk4_3sphere.{pdf,png}; the deviation-from-reference diagnostic
figures/fig9_rk4_3sphere_error.png and the per-operator metrics figures/fig9_rk4_3sphere.csv.
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
sys.path.insert(1, str(ROOT / "src"))  # grpy_tensors is imported bare by mob_op_2b_combined
os.chdir(ROOT)
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")  # accuracy work

REFS = "data/fig9_rk4_3sphere_refs.npz"
SELF_PATH = "data/models/self_interaction_model.pt"
TWO_BODY_PATH = "data/models/two_body_combined_model.pt"
MOMENTS_PATH = "data/models/nbody_moments_v2_kinf_rc8_pc8c.pt"
DIAG_PATH = "data/models/nbody_diag_v2_pc8c.pt"

INIT_POS = np.array([[-5.0, 0.0, 0.0], [0.0, 0.0, 0.0], [7.0, 0.0, 0.0]])
N_FRAMES, FRAME_STEPS, DT = 90, 128, 1.0

OPS = ["nemo", "2b", "rpy", "b1_paper"]
OP_LABELS = {"nemo": "NeMO", "2b": r"$M_{2b}$", "rpy": "RPY", "b1_paper": "NeMO (b1, published)"}
PANEL_FILES = {"nemo": "rk4-dynamics", "2b": "rk4-2b", "rpy": "rk4-rpy"}  # the paper's include names
PANEL_TITLES = {"nemo": "NeMO (moments pc8c + diag)", "2b": "learned two-body $M_{2b}$", "rpy": "RPY"}

# seaborn "colorblind" palette, the published figure's hues: MFS / Durlofsky / Townsend / ours
COLORS = {"MFS": "#0173B2", "SD (Durlofsky et al.)": "#DE8F05",
          "SD (Townsend et al.)": "#029E73", "ours": "#D55E00", "published": "#949494"}


def build_op(name: str):
    from src.mob_op_2b_combined import NNMob
    from src.mob_op_nbody import Mob_Op_Nbody
    from src.mob_op_nbody_moments import Mob_Op_Nbody_Moments

    common = dict(shape="sphere", self_nn_path=SELF_PATH, two_nn_path=TWO_BODY_PATH)
    if name == "nemo":
        return Mob_Op_Nbody_Moments(**common, nbody_nn_path=MOMENTS_PATH, switch_dist=8.0,
                                    pair_cutoff=8.0, neighbor_cutoff=8.0, max_neighbors=None,
                                    diag_nn_path=DIAG_PATH, diag_cutoff=8.0)
    if name == "2b":
        return NNMob(**common)
    if name == "rpy":
        return NNMob(**common, rpy_only=True)
    if name == "b1_paper":
        return Mob_Op_Nbody(**common, nbody_nn_path="data/models/nbody_pinn_b1.pt", switch_dist=6.0)
    raise ValueError(name)


def integrate(op, n_frames: int = N_FRAMES, frame_steps: int = FRAME_STEPS, dt: float = DT) -> np.ndarray:
    """RK4 on the sphere positions (orientations stay identity; only U is used), verbatim the
    published integrator. Returns (n_frames + 1, 3, 3) positions at t = 0, 128, ..., 128*n_frames."""
    import torch

    config = np.zeros((3, 7))
    config[:, :3] = INIT_POS
    config[:, 6] = 1.0  # identity quaternion, scalar-last
    forces = np.zeros((3, 6))
    forces[:, 2] = -1.0

    def vel(cfg: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            v = op.apply(cfg, forces, 1.0)
        v = np.asarray(v.cpu() if torch.is_tensor(v) else v, dtype=np.float64)
        return v[:, :3]

    frames = [config[:, :3].copy()]
    t0 = time.time()
    for it in range(n_frames * frame_steps):
        k1 = vel(config)
        c2 = config.copy(); c2[:, :3] += 0.5 * dt * k1
        k2 = vel(c2)
        c3 = config.copy(); c3[:, :3] += 0.5 * dt * k2
        k3 = vel(c3)
        c4 = config.copy(); c4[:, :3] += dt * k3
        k4 = vel(c4)
        config[:, :3] += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if (it + 1) % frame_steps == 0:
            frames.append(config[:, :3].copy())
            f = (it + 1) // frame_steps
            if f % 10 == 0 or f == n_frames:
                print(f"  frame {f}/{n_frames}  t={(it + 1) * dt:.0f}  z={config[:, 2].round(1).tolist()}"
                      f"  x={config[:, 0].round(2).tolist()}  [{time.time() - t0:.0f} s]", flush=True)
    return np.array(frames)


def traj_path(name: str) -> Path:
    return ROOT / "figures" / f"fig9_rk4_3sphere_{name}.npy"


def load_trajectories(names) -> dict[str, np.ndarray]:
    out = {}
    for name in names:
        p = traj_path(name)
        if p.exists():
            out[name] = np.load(p)
    return out


def deviation(traj: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Per-frame mean over the 3 spheres of |r_i(t) - r_i^ref(t)| (radii)."""
    n = min(len(traj), len(ref))
    return np.linalg.norm(traj[:n] - ref[:n], axis=-1).mean(axis=1)


def metrics(trajs: dict[str, np.ndarray], refs) -> "pd.DataFrame":
    import pandas as pd

    rows = []
    mfs, sd = refs["mfs"], refs["sd_townsend"]
    z_mfs = mfs[:, :, 2].mean(axis=1)
    early = z_mfs > -400.0  # the window where the references agree with one another
    ref_gap = deviation(sd, mfs)
    curves = dict(trajs)
    curves["b1_paper"] = refs["published_nemo_b1"]
    curves["2b_paper"] = refs["published_2b"]
    curves["rpy_paper"] = refs["published_rpy"]
    for name, tr in curves.items():
        d_mfs, d_sd = deviation(tr, mfs), deviation(tr, sd)
        n = len(d_mfs)
        rows.append({"op": name,
                     "mean_dev_mfs": d_mfs.mean(), "max_dev_mfs": d_mfs.max(), "final_dev_mfs": d_mfs[-1],
                     "mean_dev_mfs_y>-400": d_mfs[early[:n]].mean(), "max_dev_mfs_y>-400": d_mfs[early[:n]].max(),
                     "mean_dev_sd": d_sd.mean(), "max_dev_sd": d_sd.max(), "final_dev_sd": d_sd[-1],
                     "mean_dev_sd_y>-400": d_sd[early[:n]].mean(), "max_dev_sd_y>-400": d_sd[early[:n]].max(),
                     "frames": n})
    rows.append({"op": "SD (Townsend) vs MFS", "mean_dev_mfs": ref_gap.mean(), "max_dev_mfs": ref_gap.max(),
                 "final_dev_mfs": ref_gap[-1], "mean_dev_mfs_y>-400": ref_gap[early].mean(),
                 "max_dev_mfs_y>-400": ref_gap[early].max(), "frames": len(ref_gap)})
    return pd.DataFrame(rows)


def _plot_refs(ax, refs, n_frames: int | None = None):
    def plot_method(points, label, **kw):
        for i in range(3):
            ax.plot(points[:, i, 0], points[:, i, 2], color=COLORS[label], label=label if i == 0 else None, **kw)

    plot_method(refs["mfs"][: (n_frames + 1) if n_frames else None], "MFS", linestyle="--", lw=1.2)
    plot_method(refs["durlofsky"], "SD (Durlofsky et al.)", marker=".", linestyle="None", ms=4, zorder=0)
    plot_method(refs["sd_townsend"][: (n_frames + 1) if n_frames else None], "SD (Townsend et al.)", linestyle="--", lw=1.2)


def _plot_ours(ax, traj, label, color=COLORS["ours"], **kw):
    for i in range(3):
        ax.plot(traj[:, i, 0], traj[:, i, 2], color=color, label=label if i == 0 else None, **kw)


def _finish(ax, legend=True):
    ax.set_xlabel("x")
    ax.set_ylabel("y", rotation=0)
    ax.set_xlim([-6, 13])
    ax.set_ylim([-850, 10])
    if legend:
        ax.legend(loc="upper right", fontsize=9)


def plot(trajs: dict[str, np.ndarray], refs) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # single panels at the paper's include names
    for name, stem in PANEL_FILES.items():
        if name not in trajs:
            continue
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        _plot_refs(ax, refs)
        _plot_ours(ax, trajs[name], OP_LABELS[name], linestyle="-", lw=1.4)
        _finish(ax)
        for ext in ("pdf", "png"):
            fig.savefig(f"figures/{stem}.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # combined 3-panel (paper layout) + the published NeMO trajectory greyed into the NeMO panel
    panels = [n for n in ("nemo", "2b", "rpy") if n in trajs]
    if panels:
        fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.8), sharey=True, squeeze=False)
        for ax, name in zip(axes[0], panels):
            _plot_refs(ax, refs)
            if name == "nemo":
                _plot_ours(ax, refs["published_nemo_b1"], OP_LABELS["b1_paper"], color=COLORS["published"],
                           linestyle=":", lw=1.2)
            _plot_ours(ax, trajs[name], OP_LABELS[name], linestyle="-", lw=1.4)
            ax.set_title(PANEL_TITLES[name], fontsize=10)
            _finish(ax, legend=(name == panels[0]))
        for ax in axes[0][1:]:
            ax.set_ylabel("")
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(f"figures/fig9_rk4_3sphere.{ext}", dpi=200, bbox_inches="tight")
        plt.close(fig)

    # deviation diagnostic: mean |r - r_ref| per frame vs the MFS run and vs Townsend's SD
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
    t = np.arange(len(refs["mfs"])) * FRAME_STEPS * DT
    series = [("b1_paper", refs["published_nemo_b1"], COLORS["published"], ":"),
              ("2b", trajs.get("2b", refs["published_2b"]), "#CC78BC", "--"),
              ("rpy", trajs.get("rpy", refs["published_rpy"]), "#56B4E9", "--")]
    if "nemo" in trajs:
        series.insert(0, ("nemo", trajs["nemo"], COLORS["ours"], "-"))
    for ax, (ref_name, key) in zip(axes, [("MFS", "mfs"), ("SD (Townsend et al.)", "sd_townsend")]):
        for name, tr, color, ls in series:
            d = deviation(tr, refs[key])
            ax.plot(t[: len(d)], d, color=color, linestyle=ls, lw=1.6, label=OP_LABELS[name])
        other = "sd_townsend" if key == "mfs" else "mfs"
        d = deviation(refs[other], refs[key])
        ax.plot(t[: len(d)], d, color="k", linestyle="-.", lw=1.0, label="other reference")
        ax.set_title(f"mean sphere displacement from {ref_name} (radii)", fontsize=10)
        ax.set_xlabel("t")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("|r − r_ref| (radii)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("figures/fig9_rk4_3sphere_error.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ops", nargs="*", default=["nemo"], choices=OPS)
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--frames", type=int, default=N_FRAMES)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    refs = np.load(REFS)
    if not args.plot_only:
        import torch
        torch.set_num_threads(args.threads)
        for name in args.ops:
            print(f"== {name}: RK4 dt={DT} x {args.frames} frames of {FRAME_STEPS} steps", flush=True)
            traj = integrate(build_op(name), n_frames=args.frames)
            np.save(traj_path(name), traj)
            print(f"   saved {traj_path(name)}  final z={traj[-1, :, 2].round(2).tolist()}", flush=True)

    trajs = load_trajectories(OPS)
    if not trajs:
        sys.exit("no trajectories found; run without --plot-only first")
    import pandas as pd
    df = metrics(trajs, refs)
    df.to_csv("figures/fig9_rk4_3sphere.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(df.to_string(index=False))
    plot(trajs, refs)
    print("wrote figures/fig9_rk4_3sphere.{pdf,png,csv}, figures/fig9_rk4_3sphere_error.png,"
          " and the paper panels figures/rk4-{dynamics,2b,rpy}.{pdf,png}")


if __name__ == "__main__":
    main()
