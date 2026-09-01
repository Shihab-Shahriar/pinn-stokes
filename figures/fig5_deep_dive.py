#!/usr/bin/env python3
"""Figure 5 (spatial structure of the NeMO prediction in the opposing-force experiment),
regenerated with the new NeMO stack: Mob_Op_Nbody_Moments with the pc8 moments pair model
(nbody_moments_v2_kinf_rc8_pc8.pt) + learned per-particle diagonal (nbody_diag_v2_pc8.pt),
pair_cutoff = switch_dist = 8.

The protocol is experiments/accuracy_deep_dive.ipynb verbatim, only the operator is swapped:
  (b) representative configuration: N=300, seed 42, phi=0.10, vertical-split ("opposing") unit
      force; in-plane error vectors e_i = u_MFS - u_NeMO coloured by |e_i,xy|.
  (a,c,d) KDE fields over 10 configurations (N=250, seeds 42..51): predicted velocity magnitude
      (with force arrows), per-particle translational RMSE, and relative RMSE (%).
Truth is MobMFSTriton Xfine (tol 1e-8).  KDE surfaces: expected value = weighted KDE /
unweighted KDE x mean weight on a 220x220 grid, masked below the 8th density percentile.

    TORCH_COMPILE_DISABLE=1 python figures/fig5_deep_dive.py            # full run (GPU for MFS truth)
    python figures/fig5_deep_dive.py --plot-only                        # re-render from the NPZ

Outputs: figures/fig5_deep_dive.{pdf,png}, the collected data in figures/fig5_deep_dive_data.npz,
and a drop-in copy at figures/accuracy_deep_dive_fields_2x2.pdf (the paper's include name).
"""
from __future__ import annotations

import argparse
import os
import shutil
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

VOL_FRAC = 0.10
PAPER_FONT_SIZES = {"font.size": 24, "axes.labelsize": 30, "axes.titlesize": 28, "xtick.labelsize": 26,
                    "ytick.labelsize": 26, "legend.fontsize": 24, "figure.titlesize": 32}


def make_vertical_split(centers: np.ndarray) -> np.ndarray:
    """Unit +/-y force split at the box midplane (the paper's opposing-force field)."""
    x = centers[:, 0]
    x_mid = 0.5 * (x.min() + x.max())
    F = np.stack([np.zeros_like(x), np.where(x < x_mid, 1.0, -1.0), np.zeros_like(x)], axis=1)
    return F / np.linalg.norm(F, axis=1, keepdims=True)


def particle_rel_rmse(predicted: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    """Per-particle relative RMSE (%) of the linear velocity, per-particle RMS normalisation."""
    diff = velocity[:, :3] - predicted[:, :3]
    rmse = np.sqrt(np.mean(diff * diff, axis=1))
    ref = np.sqrt(np.mean(velocity[:, :3] ** 2, axis=1))
    return np.where(ref > 1e-8, rmse / ref * 100.0, 0.0)


def build_models():
    from src.mob_op_nbody_moments import Mob_Op_Nbody_Moments
    from src.triton_mfs import MobMFSTriton

    mob = Mob_Op_Nbody_Moments(shape="sphere", self_nn_path="data/models/self_interaction_model.pt",
                               two_nn_path="data/models/two_body_combined_model.pt",
                               nbody_nn_path="data/models/nbody_moments_v2_kinf_rc8_pc8.pt",
                               switch_dist=8.0, pair_cutoff=8.0, neighbor_cutoff=8.0, max_neighbors=None,
                               diag_nn_path="data/models/nbody_diag_v2_pc8.pt", diag_cutoff=8.0)
    truth = MobMFSTriton(shape="sphere", acc="Xfine", tol=1e-8)
    return mob, truth


def run_case(mob, truth, N: int, seed: int):
    import torch
    from benchmarks.cluster import uniform_sphere_cluster

    centers, _ = uniform_sphere_cluster(volume_fraction=VOL_FRAC, numParticles=N, radius=1.0, seed=seed)
    config = np.concatenate([centers, np.tile([0.0, 0.0, 0.0, 1.0], (N, 1))], axis=1)
    forces = np.concatenate([make_vertical_split(centers), np.zeros((N, 3))], axis=1)
    t0 = time.time()
    velocity = truth.apply(config, forces, viscosity=1.0)
    t1 = time.time()
    with torch.no_grad():
        predicted = mob.apply(config, forces, 1.0)
    predicted = np.asarray(predicted.cpu() if torch.is_tensor(predicted) else predicted, dtype=np.float64)
    return centers, forces, np.asarray(velocity, dtype=np.float64), predicted, t1 - t0, time.time() - t1


def collect(args) -> dict:
    mob, truth = build_models()
    # representative configuration for the error-vector panel (b)
    centers, forces, velocity, predicted, t_mfs, t_op = run_case(mob, truth, args.rep_particles, args.base_seed)
    err = velocity[:, :3] - predicted[:, :3]
    print(f"[rep] N={args.rep_particles} seed={args.base_seed}: mean |e_xy|={np.linalg.norm(err[:, :2], axis=1).mean():.4e} "
          f"max={np.linalg.norm(err[:, :2], axis=1).max():.4e}  [mfs {t_mfs:.0f} s, op {t_op:.0f} s]", flush=True)
    out = {"rep_xy": centers[:, :2], "rep_err_xy": err[:, :2], "rep_err_mag": np.linalg.norm(err, axis=1)}

    # KDE dataset for panels (a), (c), (d)
    xy, force_xy, rel_rmse, abs_rmse, velocity_mag = [], [], [], [], []
    for run_idx in range(args.num_configs):
        seed = args.base_seed + run_idx
        centers, forces, velocity, predicted, t_mfs, t_op = run_case(mob, truth, args.num_particles, seed)
        diff = velocity[:, :3] - predicted[:, :3]
        xy.append(centers[:, :2]); force_xy.append(forces[:, :2])
        rel_rmse.append(particle_rel_rmse(predicted, velocity))
        abs_rmse.append(np.sqrt(np.mean(diff * diff, axis=1)))
        velocity_mag.append(np.linalg.norm(predicted[:, :3], axis=1))  # caption: *predicted* velocity magnitude
        print(f"[KDE] run {run_idx + 1}/{args.num_configs} seed={seed}: mean rel-RMSE={rel_rmse[-1].mean():.3f}% "
              f"mean RMSE={abs_rmse[-1].mean():.4e} mean |u|={velocity_mag[-1].mean():.4e}  "
              f"[mfs {t_mfs:.0f} s, op {t_op:.0f} s]", flush=True)
    out.update({"xy": np.concatenate(xy), "force_xy": np.concatenate(force_xy),
                "rel_rmse": np.concatenate(rel_rmse), "abs_rmse": np.concatenate(abs_rmse),
                "velocity_mag": np.concatenate(velocity_mag),
                "runs": np.repeat(np.arange(args.num_configs), args.num_particles)})

    # per-run 5 %-trimmed robust mean of the relative RMSE (the notebook's summary number)
    robust = []
    for r in range(args.num_configs):
        v = np.sort(out["rel_rmse"][out["runs"] == r])
        k = int(np.floor(0.05 * v.size))
        robust.append(v[k:v.size - k].mean())
    print(f"[robust rel-RMSE] per-run 5%-trimmed means: " + " ".join(f"{v:.2f}" for v in robust)
          + f"  | avg {np.mean(robust):.3f}%", flush=True)
    return out


def plot(d: dict, out_stem: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter, FuncFormatter
    from scipy.stats import gaussian_kde

    plt.rcParams.update(PAPER_FONT_SIZES)

    # ---- KDE surfaces (accuracy_deep_dive.ipynb cell 7) ----
    x, y = d["xy"][:, 0], d["xy"][:, 1]
    points = np.vstack([x, y])
    density_kde = gaussian_kde(points)
    x_pad = 0.03 * (x.max() - x.min() + 1e-12); y_pad = 0.03 * (y.max() - y.min() + 1e-12)
    gx = np.linspace(x.min() - x_pad, x.max() + x_pad, 220)
    gy = np.linspace(y.min() - y_pad, y.max() + y_pad, 220)
    grid_x, grid_y = np.meshgrid(gx, gy)
    grid = np.vstack([grid_x.ravel(), grid_y.ravel()])
    density = density_kde(grid).reshape(grid_x.shape)

    def expected_surface(values):
        w = np.clip(values, 0.0, None)
        density_w = gaussian_kde(points, weights=w)(grid).reshape(grid_x.shape)
        return (density_w / np.maximum(density, 1e-15)) * w.mean()

    mask = density < np.quantile(density[density > 0.0], 0.08)
    surfaces = {"Velocity Magnitude": np.ma.array(expected_surface(d["velocity_mag"]), mask=mask),
                "RMSE": np.ma.array(expected_surface(d["abs_rmse"]), mask=mask),
                "Relative RMSE (%)": np.ma.array(expected_surface(d["rel_rmse"]), mask=mask)}

    # ---- 2x2 export (accuracy_deep_dive.ipynb cell 12) ----
    FIG_WIDTH_IN, FIG_HEIGHT_IN = 6.5, 4.7
    AXIS_LABEL_SIZE, TICK_LABEL_SIZE, CBAR_LABEL_SIZE, CBAR_TICK_SIZE, PANEL_LABEL_SIZE = 9, 7, 9, 7, 10
    NUM_ARROWS = 150
    panel_defs = [("Velocity Magnitude", "cividis", "Velocity magnitude", True),
                  ("RMSE", "plasma", "RMSE", False),
                  ("Relative RMSE (%)", "viridis", "PRMSE (%)", False)]

    fig, axes_grid = plt.subplots(2, 2, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), constrained_layout=True)
    axes = [axes_grid[0, 0], axes_grid[1, 0], axes_grid[1, 1], axes_grid[0, 1]]
    stride = max(1, d["xy"].shape[0] // NUM_ARROWS)

    for ax, (metric_name, cmap_name, cbar_label, show_arrows) in zip(axes[:3], panel_defs):
        heat = ax.contourf(grid_x, grid_y, surfaces[metric_name], levels=75, cmap=cmap_name)
        heat.set_edgecolor("face")  # close sub-pixel seams between filled bands in vector output
        if show_arrows:
            ax.quiver(d["xy"][::stride, 0], d["xy"][::stride, 1],
                      d["force_xy"][::stride, 0], d["force_xy"][::stride, 1],
                      angles="xy", scale_units="xy", scale=0.55, width=0.0075, color="white", alpha=0.95,
                      headwidth=3.2, headlength=4.0, headaxislength=3.6, linewidths=0.3, edgecolors="black")
        ax.set_aspect("equal", "box")
        ax.set_xlabel("x", fontsize=AXIS_LABEL_SIZE); ax.set_ylabel("y", fontsize=AXIS_LABEL_SIZE)
        ax.tick_params(axis="x", labelsize=TICK_LABEL_SIZE); ax.tick_params(axis="y", labelsize=TICK_LABEL_SIZE)
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f")); ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        cbar = fig.colorbar(heat, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label(cbar_label, fontsize=CBAR_LABEL_SIZE)
        cbar.ax.tick_params(labelsize=CBAR_TICK_SIZE)
        # The notebook rounded tick positions to 2-3 decimals; the new operator's RMSE is ~10x
        # smaller, which collapses those ticks, so format the default ticks instead.
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.4g}"))

    q = axes[3].quiver(d["rep_xy"][:, 0], d["rep_xy"][:, 1], d["rep_err_xy"][:, 0], d["rep_err_xy"][:, 1],
                       d["rep_err_mag"], angles="xy", scale=None, width=0.007, cmap="inferno")
    axes[3].scatter(d["rep_xy"][:, 0], d["rep_xy"][:, 1], s=2, c="k", alpha=0.35)
    axes[3].set_aspect("equal", "box")
    axes[3].set_xlabel("x", fontsize=AXIS_LABEL_SIZE); axes[3].set_ylabel("y", fontsize=AXIS_LABEL_SIZE)
    axes[3].tick_params(axis="x", labelsize=TICK_LABEL_SIZE); axes[3].tick_params(axis="y", labelsize=TICK_LABEL_SIZE)
    axes[3].xaxis.set_major_formatter(FormatStrFormatter("%.2f")); axes[3].yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    cbar = fig.colorbar(q, ax=axes[3], fraction=0.046, pad=0.02)
    cbar.set_label("|error| (in-plane)", fontsize=CBAR_LABEL_SIZE)
    cbar.ax.tick_params(labelsize=CBAR_TICK_SIZE)

    for ax, tag in [(axes_grid[0, 0], "(a)"), (axes_grid[0, 1], "(b)"), (axes_grid[1, 0], "(c)"), (axes_grid[1, 1], "(d)")]:
        ax.set_title("")
        ax.text(0.0, 1.02, tag, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=PANEL_LABEL_SIZE, fontweight="bold")

    fig.savefig(out_stem.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    shutil.copy(out_stem.with_suffix(".pdf"), "figures/accuracy_deep_dive_fields_2x2.pdf")
    print(f"-> {out_stem}.pdf, {out_stem}.png, figures/accuracy_deep_dive_fields_2x2.pdf "
          f"(6.5x4.7 in, include at width=\\columnwidth with no height override)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num-particles", type=int, default=250, help="particles per KDE configuration")
    ap.add_argument("--num-configs", type=int, default=10, help="KDE configurations")
    ap.add_argument("--rep-particles", type=int, default=300, help="particles in the representative (b) configuration")
    ap.add_argument("--base-seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("figures/fig5_deep_dive"))
    ap.add_argument("--plot-only", action="store_true", help="re-render the figure from the existing NPZ")
    args = ap.parse_args()
    npz = args.out.parent / (args.out.name + "_data.npz")
    if args.plot_only:
        d = dict(np.load(npz))
    else:
        d = collect(args)
        np.savez(npz, **d)
        print(f"-> {npz}")
    plot(d, args.out)


if __name__ == "__main__":
    main()
