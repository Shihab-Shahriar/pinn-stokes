#!/usr/bin/env python3
"""Deep-dive diagnostics for mobility-operator KDE error maps.

This script reproduces and extends the notebook analysis by measuring how
particle-wise error correlates with:
1) local force-direction variation,
2) force/speed magnitude,
3) geometric crowding and cutoff neighborhoods.

It also generates diagnostic plots in `artifacts/deep_dive`.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde, pearsonr, spearmanr

# Disable torch.compile graph capture for this diagnostic workflow to avoid
# extra startup overhead and improve run-to-run consistency.
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

from benchmarks.cluster import uniform_sphere_cluster
from src.gpu_mob_2b import NNMobTorch
from src.gpu_nbody_mob import Mob_Nbody_Torch
from src.triton_mfs import MobMFSTriton


@dataclass(frozen=True)
class SweepConfig:
    volume_fraction: float
    num_particles: int
    num_configs: int
    base_seed: int
    force_scale: float
    k_neighbors: int = 8
    switch_dist: float = 6.0


def ensure_repo_root() -> None:
    if not os.path.exists("data/points") and os.path.exists("../data/points"):
        os.chdir("..")


def make_periodic_force(centers: np.ndarray, period: float) -> np.ndarray:
    x = centers[:, 0]
    y = centers[:, 1]
    fx = -np.sin(x / period) * np.cos(y / period)
    fy = np.cos(x / period) * np.sin(y / period)
    return np.stack([fx, fy, np.zeros_like(fx)], axis=1).astype(np.float64)


def make_vertical_split_force(
    centers: np.ndarray,
    normalize: bool = True,
    mean_sub: bool = False,
) -> np.ndarray:
    x = centers[:, 0]
    x_mid = 0.5 * (x.min() + x.max())
    fx = np.zeros_like(x)
    fy = np.where(x < x_mid, 1.0, -1.0)
    force = np.stack([fx, fy, np.zeros_like(x)], axis=1).astype(np.float64)

    if mean_sub:
        force = force - force.mean(axis=0, keepdims=True)
    if normalize:
        norms = np.linalg.norm(force, axis=1, keepdims=True)
        np.divide(force, np.maximum(norms, 1e-12), out=force, where=norms > 1e-12)
    return force


def project_linear_error_along_force(
    predicted: np.ndarray,
    truth: np.ndarray,
    force: np.ndarray,
) -> np.ndarray:
    force_dir = np.zeros_like(force, dtype=np.float64)
    force_norm = np.linalg.norm(force, axis=1, keepdims=True)
    np.divide(force, np.maximum(force_norm, 1e-12), out=force_dir, where=force_norm > 1e-12)
    return np.sum((predicted[:, :3] - truth[:, :3]) * force_dir, axis=1)


def local_direction_metrics(
    centers_xy: np.ndarray,
    force_xy: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tree = cKDTree(centers_xy)
    dists, indices = tree.query(centers_xy, k=k + 1)
    dists = dists[:, 1:]
    indices = indices[:, 1:]

    force_mag = np.linalg.norm(force_xy, axis=1)
    force_unit = force_xy / (force_mag[:, None] + 1e-12)
    neigh_unit = force_unit[indices]
    cos_sim = np.clip(np.sum(force_unit[:, None, :] * neigh_unit, axis=2), -1.0, 1.0)
    angle = np.arccos(cos_sim)

    dir_change = angle.mean(axis=1)
    dir_grad = (angle / (dists + 1e-12)).mean(axis=1)
    min_dist = dists[:, 0]
    inv_knn_dist_sum = (1.0 / (dists + 1e-6)).sum(axis=1)
    return dir_change, dir_grad, min_dist, inv_knn_dist_sum


def count_neighbors(centers_xy: np.ndarray, radius: float) -> np.ndarray:
    tree = cKDTree(centers_xy)
    return np.array(
        [len(tree.query_ball_point(centers_xy[i], r=radius)) - 1 for i in range(centers_xy.shape[0])],
        dtype=np.int32,
    )


def count_neighbors_in_band(
    centers_xy: np.ndarray,
    low: float,
    high: float,
) -> np.ndarray:
    n = centers_xy.shape[0]
    out = np.zeros(n, dtype=np.int32)
    for i in range(n):
        d = np.linalg.norm(centers_xy - centers_xy[i], axis=1)
        out[i] = int(np.sum((d >= low) & (d <= high)))
    return out


def build_models(switch_dist: float) -> tuple[Mob_Nbody_Torch, MobMFSTriton]:
    mob = Mob_Nbody_Torch(
        shape="sphere",
        self_nn_path="data/models/self_interaction_model.pt",
        two_nn_path="data/models/combined_2body.wt",
        nbody_nn_path="data/models/nbody_cross_tmp.wt",
        near_field_2b="nn",
        far_field_2b="rpy",
        near_far_switch=switch_dist,
    )
    truth = MobMFSTriton(shape="sphere", acc="Xfine", tol=1e-8)
    return mob, truth


def evaluate_truth_base_full(
    mob: Mob_Nbody_Torch,
    truth: MobMFSTriton,
    centers: np.ndarray,
    orientations: np.ndarray,
    forces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    config = np.concatenate([centers, orientations], axis=1)
    vel_true = truth.apply(config, forces, viscosity=1.0)

    pos_t = torch.as_tensor(np.ascontiguousarray(centers, dtype=np.float32), device=mob.device)
    ori_t = torch.as_tensor(np.ascontiguousarray(orientations, dtype=np.float32), device=mob.device)
    force_t = torch.as_tensor(np.ascontiguousarray(forces, dtype=np.float32), device=mob.device)

    t_idx, s_idx = mob.get_neighbor_pairs(pos_t)
    vel_base = NNMobTorch.apply(mob, pos_t, ori_t, force_t, viscosity=1.0, t_idx=t_idx, s_idx=s_idx)
    vel_nbody = mob.get_nbody_velocity(pos_t, force_t, t_idx, s_idx)
    vel_full = vel_base + vel_nbody

    return (
        vel_true,
        vel_base.detach().cpu().numpy(),
        vel_full.detach().cpu().numpy(),
    )


def run_feature_sweep(cfg: SweepConfig, out_dir: Path) -> pd.DataFrame:
    mob, truth = build_models(cfg.switch_dist)
    rows: list[dict[str, float]] = []

    for run_idx in range(cfg.num_configs):
        seed = cfg.base_seed + run_idx
        centers, _ = uniform_sphere_cluster(
            volume_fraction=cfg.volume_fraction,
            numParticles=cfg.num_particles,
            radius=1.0,
            seed=seed,
        )
        period = float(np.max(centers.max(axis=0) - centers.min(axis=0)) + 2.0)
        orientations = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64), (cfg.num_particles, 1))
        config = np.concatenate([centers, orientations], axis=1)

        force = make_periodic_force(centers, period) * cfg.force_scale
        forces = np.concatenate([force, np.zeros((cfg.num_particles, 3), dtype=np.float64)], axis=1)

        vel_true = truth.apply(config, forces, viscosity=1.0)
        vel_pred = mob.apply_cpu(centers, orientations, forces, viscosity=1.0)

        diff = vel_true[:, :2] - vel_pred[:, :2]
        rmse_xy = np.sqrt(np.mean(diff * diff, axis=1))
        err_norm = np.linalg.norm(diff, axis=1)
        true_speed = np.linalg.norm(vel_true[:, :2], axis=1)
        pred_speed = np.linalg.norm(vel_pred[:, :2], axis=1)
        rel_err_xy = err_norm / (true_speed + 1e-12)

        angle_err = np.full(cfg.num_particles, np.nan, dtype=np.float64)
        valid = (true_speed > 1e-8) & (pred_speed > 1e-8)
        if np.any(valid):
            u_true = vel_true[valid, :2] / true_speed[valid, None]
            u_pred = vel_pred[valid, :2] / pred_speed[valid, None]
            cos_sim = np.clip(np.sum(u_true * u_pred, axis=1), -1.0, 1.0)
            angle_err[valid] = np.degrees(np.arccos(cos_sim))

        centers_xy = centers[:, :2]
        force_xy = force[:, :2]
        force_mag = np.linalg.norm(force_xy, axis=1)
        dir_change, dir_grad, min_dist, inv_knn_dist_sum = local_direction_metrics(
            centers_xy,
            force_xy,
            k=cfg.k_neighbors,
        )
        mean_knn_dist = np.full_like(min_dist, np.nan)
        tree = cKDTree(centers_xy)
        d_knn, _ = tree.query(centers_xy, k=cfg.k_neighbors + 1)
        mean_knn_dist = d_knn[:, 1:].mean(axis=1)

        cnt_within_switch = count_neighbors(centers_xy, radius=cfg.switch_dist)
        cnt_near_switch = count_neighbors_in_band(
            centers_xy,
            low=cfg.switch_dist - 0.5,
            high=cfg.switch_dist + 0.5,
        )

        x = centers[:, 0]
        y = centers[:, 1]
        x_l = x / period
        y_l = y / period
        dfx_dx = -(np.cos(x_l) * np.cos(y_l)) / period
        dfx_dy = (np.sin(x_l) * np.sin(y_l)) / period
        dfy_dx = -(np.sin(x_l) * np.sin(y_l)) / period
        dfy_dy = (np.cos(x_l) * np.cos(y_l)) / period
        force_grad_frob = np.sqrt(dfx_dx**2 + dfx_dy**2 + dfy_dx**2 + dfy_dy**2)

        for i in range(cfg.num_particles):
            rows.append(
                {
                    "run": run_idx,
                    "seed": seed,
                    "L": period,
                    "x": x[i],
                    "y": y[i],
                    "rmse_xy": rmse_xy[i],
                    "rel_err_xy": rel_err_xy[i],
                    "angle_err_deg": angle_err[i],
                    "true_speed": true_speed[i],
                    "pred_speed": pred_speed[i],
                    "force_mag": force_mag[i],
                    "force_dir_change": dir_change[i],
                    "force_dir_grad": dir_grad[i],
                    "force_grad_frob": force_grad_frob[i],
                    "min_dist": min_dist[i],
                    "mean_knn_dist": mean_knn_dist[i],
                    "inv_knn_dist_sum": inv_knn_dist_sum[i],
                    "cnt_within_switch": cnt_within_switch[i],
                    "cnt_near_switch_band": cnt_near_switch[i],
                }
            )

        print(
            f"[feature_sweep] run {run_idx + 1}/{cfg.num_configs}, seed={seed}, "
            f"mean_rmse={rmse_xy.mean():.4e}"
        )

    df = pd.DataFrame(rows)
    out_path = out_dir / "error_feature_table.csv"
    df.to_csv(out_path, index=False)
    print(f"[feature_sweep] saved: {out_path} ({len(df)} rows)")
    return df


def run_base_vs_nbody(cfg: SweepConfig, out_dir: Path, num_configs: int = 6) -> pd.DataFrame:
    mob, truth = build_models(cfg.switch_dist)
    rows: list[dict[str, float]] = []

    for run_idx in range(num_configs):
        seed = cfg.base_seed + run_idx
        centers, _ = uniform_sphere_cluster(
            volume_fraction=cfg.volume_fraction,
            numParticles=cfg.num_particles,
            radius=1.0,
            seed=seed,
        )
        period = float(np.max(centers.max(axis=0) - centers.min(axis=0)) + 2.0)
        orientations = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64), (cfg.num_particles, 1))
        force = make_periodic_force(centers, period) * cfg.force_scale
        forces = np.concatenate([force, np.zeros((cfg.num_particles, 3), dtype=np.float64)], axis=1)

        vel_true, vel_base_np, vel_full_np = evaluate_truth_base_full(
            mob,
            truth,
            centers,
            orientations,
            forces,
        )

        rmse_base = np.sqrt(np.mean((vel_true[:, :2] - vel_base_np[:, :2]) ** 2, axis=1))
        rmse_full = np.sqrt(np.mean((vel_true[:, :2] - vel_full_np[:, :2]) ** 2, axis=1))
        improve = rmse_base - rmse_full

        force_xy = force[:, :2]
        dir_change, _, _, _ = local_direction_metrics(centers[:, :2], force_xy, k=cfg.k_neighbors)

        for i in range(cfg.num_particles):
            rows.append(
                {
                    "run": run_idx,
                    "rmse_base": rmse_base[i],
                    "rmse_full": rmse_full[i],
                    "improve": improve[i],
                    "dir_change": dir_change[i],
                }
            )

        print(
            f"[base_vs_nbody] run {run_idx + 1}/{num_configs}, seed={seed}, "
            f"mean_base={rmse_base.mean():.4e}, mean_full={rmse_full.mean():.4e}"
        )

    out = pd.DataFrame(rows)
    out_path = out_dir / "base_vs_nbody.csv"
    out.to_csv(out_path, index=False)
    print(f"[base_vs_nbody] saved: {out_path} ({len(out)} rows)")
    return out


def run_vertical_split_overshoot_compare(
    cfg: SweepConfig,
    out_dir: Path,
    num_configs: int = 6,
) -> pd.DataFrame:
    mob, truth = build_models(cfg.switch_dist)
    rows: list[dict[str, float]] = []

    for run_idx in range(num_configs):
        seed = cfg.base_seed + run_idx
        centers, _ = uniform_sphere_cluster(
            volume_fraction=cfg.volume_fraction,
            numParticles=cfg.num_particles,
            radius=1.0,
            seed=seed,
        )
        orientations = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64), (cfg.num_particles, 1))
        force = make_vertical_split_force(centers, normalize=True, mean_sub=False) * cfg.force_scale
        forces = np.concatenate([force, np.zeros((cfg.num_particles, 3), dtype=np.float64)], axis=1)

        vel_true, vel_base_np, vel_full_np = evaluate_truth_base_full(
            mob,
            truth,
            centers,
            orientations,
            forces,
        )

        error_base = vel_base_np[:, :2] - vel_true[:, :2]
        error_full = vel_full_np[:, :2] - vel_true[:, :2]
        rmse_base = np.sqrt(np.mean(error_base * error_base, axis=1))
        rmse_full = np.sqrt(np.mean(error_full * error_full, axis=1))

        parallel_error_base = project_linear_error_along_force(vel_base_np, vel_true, force)
        parallel_error_full = project_linear_error_along_force(vel_full_np, vel_true, force)

        pos_frac_base = float(np.mean(parallel_error_base > 0.0))
        pos_frac_full = float(np.mean(parallel_error_full > 0.0))
        pos_mean_base = float(parallel_error_base[parallel_error_base > 0.0].mean()) if np.any(parallel_error_base > 0.0) else 0.0
        pos_mean_full = float(parallel_error_full[parallel_error_full > 0.0].mean()) if np.any(parallel_error_full > 0.0) else 0.0

        for i in range(cfg.num_particles):
            rows.append(
                {
                    "run": run_idx,
                    "seed": seed,
                    "x": centers[i, 0],
                    "y": centers[i, 1],
                    "force_x": force[i, 0],
                    "force_y": force[i, 1],
                    "error_base_x": error_base[i, 0],
                    "error_base_y": error_base[i, 1],
                    "error_full_x": error_full[i, 0],
                    "error_full_y": error_full[i, 1],
                    "parallel_error_base": parallel_error_base[i],
                    "parallel_error_full": parallel_error_full[i],
                    "parallel_error_gap": parallel_error_base[i] - parallel_error_full[i],
                    "rmse_base": rmse_base[i],
                    "rmse_full": rmse_full[i],
                }
            )

        print(
            f"[vertical_split_overshoot] run {run_idx + 1}/{num_configs}, seed={seed}, "
            f"mean_parallel_base={parallel_error_base.mean():+.4e}, "
            f"mean_parallel_full={parallel_error_full.mean():+.4e}, "
            f"positive_frac_base={100.0 * pos_frac_base:.1f}%, "
            f"positive_frac_full={100.0 * pos_frac_full:.1f}%, "
            f"mean_positive_base={pos_mean_base:.4e}, "
            f"mean_positive_full={pos_mean_full:.4e}, "
            f"mean_rmse_base={rmse_base.mean():.4e}, "
            f"mean_rmse_full={rmse_full.mean():.4e}"
        )

    out = pd.DataFrame(rows)
    out_path = out_dir / "vertical_split_overshoot_compare.csv"
    out.to_csv(out_path, index=False)
    print(f"[vertical_split_overshoot] saved: {out_path} ({len(out)} rows)")
    return out


def run_volume_fraction_sweep(
    cfg: SweepConfig,
    out_dir: Path,
    volume_fractions: list[float] | None = None,
    seeds_per_vf: int = 4,
) -> pd.DataFrame:
    if volume_fractions is None:
        volume_fractions = [0.05, 0.10, 0.20]

    mob, truth = build_models(cfg.switch_dist)
    rows: list[dict[str, float]] = []

    for vf in volume_fractions:
        for i in range(seeds_per_vf):
            seed = cfg.base_seed + i
            centers, _ = uniform_sphere_cluster(
                volume_fraction=vf,
                numParticles=cfg.num_particles,
                radius=1.0,
                seed=seed,
            )
            period = float(np.max(centers.max(axis=0) - centers.min(axis=0)) + 2.0)
            orientations = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64), (cfg.num_particles, 1))
            config = np.concatenate([centers, orientations], axis=1)
            force = make_periodic_force(centers, period) * cfg.force_scale
            forces = np.concatenate([force, np.zeros((cfg.num_particles, 3), dtype=np.float64)], axis=1)

            vel_true = truth.apply(config, forces, viscosity=1.0)
            vel_pred = mob.apply_cpu(centers, orientations, forces, viscosity=1.0)

            diff = vel_true[:, :2] - vel_pred[:, :2]
            rmse_xy = np.sqrt(np.mean(diff * diff, axis=1))
            speed = np.linalg.norm(vel_true[:, :2], axis=1)
            rel = np.linalg.norm(diff, axis=1) / (speed + 1e-12)

            dir_change, _, min_dist, _ = local_direction_metrics(
                centers[:, :2],
                force[:, :2],
                k=cfg.k_neighbors,
            )

            rows.append(
                {
                    "vf": vf,
                    "seed": seed,
                    "mean_rmse": rmse_xy.mean(),
                    "p90_rmse": np.quantile(rmse_xy, 0.9),
                    "mean_rel": rel.mean(),
                    "mean_min_dist": min_dist.mean(),
                    "mean_dir_change": dir_change.mean(),
                    "mean_speed": speed.mean(),
                    "L": period,
                }
            )
            print(
                f"[vf_sweep] vf={vf:.2f}, seed={seed}, "
                f"mean_rmse={rmse_xy.mean():.4e}, mean_min_dist={min_dist.mean():.3f}"
            )

    out = pd.DataFrame(rows)
    out_path = out_dir / "vf_sweep_summary.csv"
    out.to_csv(out_path, index=False)
    print(f"[vf_sweep] saved: {out_path} ({len(out)} rows)")
    return out


def report_correlations(df: pd.DataFrame) -> None:
    metrics = [
        "force_dir_change",
        "force_dir_grad",
        "force_mag",
        "force_grad_frob",
        "min_dist",
        "mean_knn_dist",
        "inv_knn_dist_sum",
        "cnt_within_switch",
        "cnt_near_switch_band",
        "true_speed",
    ]
    for target in ["rmse_xy", "rel_err_xy"]:
        print(f"\n=== correlations with {target} ===")
        rows = []
        y = df[target].to_numpy()
        for metric in metrics:
            x = df[metric].to_numpy()
            rows.append(
                (
                    metric,
                    spearmanr(x, y, nan_policy="omit").correlation,
                    pearsonr(x, y)[0],
                )
            )
        rows.sort(key=lambda t: abs(t[1]), reverse=True)
        for metric, s, p in rows:
            print(f"{metric:22s} spearman={s:+.3f} pearson={p:+.3f}")

    print("\n=== force-direction vs scale confound ===")
    for a, b in [
        ("force_dir_change", "force_mag"),
        ("force_dir_grad", "force_mag"),
        ("force_dir_change", "true_speed"),
        ("force_dir_grad", "true_speed"),
    ]:
        s = spearmanr(df[a], df[b], nan_policy="omit").correlation
        p = pearsonr(df[a], df[b])[0]
        print(f"{a:20s} vs {b:12s}: spearman={s:+.3f}, pearson={p:+.3f}")


def plot_scatter_and_bins(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), dpi=160)
    sc = axs[0].scatter(
        df["force_dir_change"],
        df["rmse_xy"],
        c=df["true_speed"],
        s=14,
        alpha=0.45,
        cmap="viridis",
    )
    axs[0].set_xlabel("Local force direction change (rad, kNN mean)")
    axs[0].set_ylabel("Absolute in-plane RMSE")
    axs[0].set_title("Absolute RMSE vs direction change")
    fig.colorbar(sc, ax=axs[0], label="True in-plane speed")

    sc2 = axs[1].scatter(
        df["force_dir_change"],
        df["rel_err_xy"],
        c=df["true_speed"],
        s=14,
        alpha=0.45,
        cmap="viridis",
    )
    axs[1].set_xlabel("Local force direction change (rad, kNN mean)")
    axs[1].set_ylabel("Relative in-plane error")
    axs[1].set_title("Relative error vs direction change")
    fig.colorbar(sc2, ax=axs[1], label="True in-plane speed")
    for ax in axs:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "01_scatter_direction_vs_error.png", bbox_inches="tight")
    plt.close(fig)

    bins = np.unique(np.quantile(df["force_dir_change"], np.linspace(0.0, 1.0, 11)))
    tmp = df.copy()
    tmp["dir_bin"] = pd.cut(tmp["force_dir_change"], bins=bins, include_lowest=True, duplicates="drop")
    agg = (
        tmp.groupby("dir_bin", observed=False)
        .agg(
            dir_mid=("force_dir_change", "mean"),
            rmse_mean=("rmse_xy", "mean"),
            rel_mean=("rel_err_xy", "mean"),
            speed_mean=("true_speed", "mean"),
        )
        .dropna()
    )

    fig, ax1 = plt.subplots(figsize=(7, 5), dpi=160)
    ax1.plot(agg["dir_mid"], agg["rmse_mean"], marker="o", color="tab:blue", label="Abs RMSE")
    ax1.plot(agg["dir_mid"], agg["rel_mean"], marker="s", color="tab:red", label="Rel error")
    ax1.set_xlabel("Local force direction change (rad)")
    ax1.set_ylabel("Error")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(
        agg["dir_mid"],
        agg["speed_mean"],
        marker="^",
        linestyle="--",
        color="tab:green",
        label="True speed",
    )
    ax2.set_ylabel("True speed")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper center")
    ax1.set_title("Binned trend: direction-change confounded by speed")
    fig.tight_layout()
    fig.savefig(out_dir / "02_binned_direction_trend.png", bbox_inches="tight")
    plt.close(fig)


def plot_phase_and_kde(df: pd.DataFrame, out_dir: Path) -> None:
    phase_x = df["x"] / df["L"]
    phase_y = df["y"] / df["L"]

    fig, axs = plt.subplots(1, 2, figsize=(12, 5), dpi=160)
    hb1 = axs[0].hexbin(
        phase_x,
        phase_y,
        C=df["rmse_xy"],
        gridsize=36,
        reduce_C_function=np.mean,
        cmap="magma",
        mincnt=1,
    )
    axs[0].set_xlabel("x / L")
    axs[0].set_ylabel("y / L")
    axs[0].set_title("Mean absolute RMSE in phase coordinates")
    fig.colorbar(hb1, ax=axs[0], label="mean abs RMSE")

    hb2 = axs[1].hexbin(
        phase_x,
        phase_y,
        C=df["rel_err_xy"],
        gridsize=36,
        reduce_C_function=np.mean,
        cmap="magma",
        mincnt=1,
    )
    axs[1].set_xlabel("x / L")
    axs[1].set_ylabel("y / L")
    axs[1].set_title("Mean relative error in phase coordinates")
    fig.colorbar(hb2, ax=axs[1], label="mean relative error")
    for ax in axs:
        ax.axvline(0.0, color="w", alpha=0.4, linewidth=1)
        ax.axhline(0.0, color="w", alpha=0.4, linewidth=1)
    fig.tight_layout()
    fig.savefig(out_dir / "03_phase_hexbin_abs_vs_rel.png", bbox_inches="tight")
    plt.close(fig)

    def kde_surface(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ma.MaskedArray]:
        pts = np.vstack([x, y])
        w_clip = np.clip(w, 1e-12, None)
        kde_w = gaussian_kde(pts, weights=w_clip)
        kde = gaussian_kde(pts)
        x_pad = 0.03 * (x.max() - x.min() + 1e-12)
        y_pad = 0.03 * (y.max() - y.min() + 1e-12)
        gx = np.linspace(x.min() - x_pad, x.max() + x_pad, 220)
        gy = np.linspace(y.min() - y_pad, y.max() + y_pad, 220)
        grid_x, grid_y = np.meshgrid(gx, gy)
        grid = np.vstack([grid_x.ravel(), grid_y.ravel()])
        density_w = kde_w(grid).reshape(grid_x.shape)
        density = kde(grid).reshape(grid_x.shape)
        surf = (density_w / np.maximum(density, 1e-15)) * w_clip.mean()
        valid = density[density > 0.0]
        cut = np.quantile(valid, 0.08) if valid.size else 0.0
        return grid_x, grid_y, np.ma.array(surf, mask=(density < cut))

    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    gx, gy, surf_abs = kde_surface(x, y, df["rmse_xy"].to_numpy())
    _, _, surf_rel = kde_surface(x, y, df["rel_err_xy"].to_numpy())
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), dpi=160)
    c1 = axs[0].contourf(gx, gy, surf_abs, levels=70, cmap="viridis")
    fig.colorbar(c1, ax=axs[0], label="Local abs RMSE")
    axs[0].set_title("KDE map: absolute RMSE")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    axs[0].set_aspect("equal", "box")
    c2 = axs[1].contourf(gx, gy, surf_rel, levels=70, cmap="viridis")
    fig.colorbar(c2, ax=axs[1], label="Local relative error")
    axs[1].set_title("KDE map: relative error")
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("y")
    axs[1].set_aspect("equal", "box")
    fig.tight_layout()
    fig.savefig(out_dir / "05_kde_abs_vs_rel_xy.png", bbox_inches="tight")
    plt.close(fig)


def plot_vf_sweep(df_vf: pd.DataFrame, out_dir: Path) -> None:
    g = df_vf.groupby("vf", as_index=False).agg(
        mean_rmse=("mean_rmse", "mean"),
        mean_rel=("mean_rel", "mean"),
        mean_dir_change=("mean_dir_change", "mean"),
        mean_min_dist=("mean_min_dist", "mean"),
    )
    fig, ax1 = plt.subplots(figsize=(7, 5), dpi=160)
    ax1.plot(g["vf"], g["mean_rmse"], marker="o", color="tab:blue", label="Abs RMSE")
    ax1.plot(g["vf"], g["mean_rel"], marker="s", color="tab:red", label="Rel error")
    ax1.set_xlabel("Volume fraction")
    ax1.set_ylabel("Error")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(
        g["vf"],
        g["mean_dir_change"],
        marker="^",
        linestyle="--",
        color="tab:green",
        label="Force dir-change",
    )
    ax2.plot(
        g["vf"],
        g["mean_min_dist"],
        marker="d",
        linestyle="--",
        color="tab:purple",
        label="Mean min distance",
    )
    ax2.set_ylabel("Geometry / force metrics")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")
    ax1.set_title("Volume-fraction sweep: error tracks crowding, not dir-change")
    fig.tight_layout()
    fig.savefig(out_dir / "04_vf_sweep.png", bbox_inches="tight")
    plt.close(fig)


def plot_vertical_split_overshoot(df: pd.DataFrame, out_dir: Path) -> None:
    rep = df[df["run"] == int(df["run"].min())].copy()
    centers = rep[["x", "y"]].to_numpy()
    force_xy = rep[["force_x", "force_y"]].to_numpy()
    error_full_xy = rep[["error_full_x", "error_full_y"]].to_numpy()
    error_base_xy = rep[["error_base_x", "error_base_y"]].to_numpy()
    parallel_error_full = rep["parallel_error_full"].to_numpy()
    parallel_error_base = rep["parallel_error_base"].to_numpy()

    vlim = float(
        np.max(
            np.abs(
                np.concatenate(
                    [
                        parallel_error_full,
                        parallel_error_base,
                    ]
                )
            )
        )
    )
    vlim = max(vlim, 1e-8)
    err_scale = float(
        max(
            np.linalg.norm(error_full_xy, axis=1).max(initial=0.0),
            np.linalg.norm(error_base_xy, axis=1).max(initial=0.0),
            1e-8,
        )
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5),
        dpi=160,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    axes[0].quiver(
        centers[:, 0],
        centers[:, 1],
        force_xy[:, 0],
        force_xy[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.003,
        color="black",
    )
    axes[0].scatter(centers[:, 0], centers[:, 1], s=10, c="k", alpha=0.30)
    axes[0].set_title("Vertical split force field")

    q_full = axes[1].quiver(
        centers[:, 0],
        centers[:, 1],
        error_full_xy[:, 0],
        error_full_xy[:, 1],
        parallel_error_full,
        angles="xy",
        scale_units="xy",
        scale=err_scale,
        width=0.004,
        cmap="coolwarm",
    )
    q_full.set_clim(-vlim, vlim)
    axes[1].scatter(centers[:, 0], centers[:, 1], s=10, c="k", alpha=0.25)
    axes[1].set_title(
        "With n-body correction\n"
        f"mean along-force error = {parallel_error_full.mean():+.3e}"
    )

    q_base = axes[2].quiver(
        centers[:, 0],
        centers[:, 1],
        error_base_xy[:, 0],
        error_base_xy[:, 1],
        parallel_error_base,
        angles="xy",
        scale_units="xy",
        scale=err_scale,
        width=0.004,
        cmap="coolwarm",
    )
    q_base.set_clim(-vlim, vlim)
    axes[2].scatter(centers[:, 0], centers[:, 1], s=10, c="k", alpha=0.25)
    axes[2].set_title(
        "Without n-body correction\n"
        f"mean along-force error = {parallel_error_base.mean():+.3e}"
    )

    for ax in axes:
        ax.set_aspect("equal", "box")
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")

    fig.colorbar(
        q_base,
        ax=axes[1:],
        label="Signed along-force error (predicted - truth)",
    )
    fig.savefig(out_dir / "06_vertical_split_overshoot_compare.png", bbox_inches="tight")
    plt.close(fig)

    fig_hist, ax_hist = plt.subplots(figsize=(7, 5), dpi=160)
    hist_vlim = float(
        np.quantile(
            np.abs(
                np.concatenate(
                    [
                        df["parallel_error_full"].to_numpy(),
                        df["parallel_error_base"].to_numpy(),
                    ]
                )
            ),
            0.99,
        )
    )
    hist_vlim = max(hist_vlim, 1e-8)
    bins = np.linspace(-hist_vlim, hist_vlim, 61)
    ax_hist.hist(
        df["parallel_error_full"],
        bins=bins,
        density=True,
        alpha=0.60,
        label="with n-body",
        color="tab:blue",
    )
    ax_hist.hist(
        df["parallel_error_base"],
        bins=bins,
        density=True,
        alpha=0.60,
        label="without n-body",
        color="tab:red",
    )
    ax_hist.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax_hist.set_xlabel("Signed along-force error (predicted - truth)")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title("Overshoot distribution across random vertical-split clusters")
    ax_hist.legend()
    ax_hist.grid(alpha=0.25)
    fig_hist.tight_layout()
    fig_hist.savefig(out_dir / "07_vertical_split_overshoot_hist.png", bbox_inches="tight")
    plt.close(fig_hist)


def report_base_vs_nbody(df: pd.DataFrame) -> None:
    print("\n=== base-vs-nbody summary ===")
    means = df[["rmse_base", "rmse_full", "improve"]].mean()
    q = df[["rmse_base", "rmse_full", "improve"]].quantile([0.1, 0.5, 0.9])
    print(means.to_string())
    print(q.to_string())
    for c in ["rmse_base", "rmse_full", "improve"]:
        s = spearmanr(df[c], df["dir_change"], nan_policy="omit").correlation
        print(f"{c:12s} vs dir_change spearman={s:+.3f}")


def report_vertical_split_overshoot(df: pd.DataFrame) -> None:
    print("\n=== vertical-split overshoot summary ===")
    pos_full = df.loc[df["parallel_error_full"] > 0.0, "parallel_error_full"]
    pos_base = df.loc[df["parallel_error_base"] > 0.0, "parallel_error_base"]
    summary = {
        "mean_parallel_error_with_nbody": float(df["parallel_error_full"].mean()),
        "mean_parallel_error_without_nbody": float(df["parallel_error_base"].mean()),
        "positive_frac_with_nbody": float(np.mean(df["parallel_error_full"] > 0.0)),
        "positive_frac_without_nbody": float(np.mean(df["parallel_error_base"] > 0.0)),
        "mean_positive_with_nbody": float(pos_full.mean()) if not pos_full.empty else 0.0,
        "mean_positive_without_nbody": float(pos_base.mean()) if not pos_base.empty else 0.0,
        "mean_rmse_with_nbody": float(df["rmse_full"].mean()),
        "mean_rmse_without_nbody": float(df["rmse_base"].mean()),
        "median_gap_base_minus_full": float(df["parallel_error_gap"].median()),
    }
    for key, value in summary.items():
        if "frac" in key:
            print(f"{key:35s} = {100.0 * value:6.2f}%")
        else:
            print(f"{key:35s} = {value:+.6e}")

    with_nbody = max(summary["mean_positive_with_nbody"], 1e-12)
    without_nbody = summary["mean_positive_without_nbody"]
    print(f"positive overshoot ratio (no nbody / with nbody) = {without_nbody / with_nbody:.2f}x")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deep-dive diagnostics for KDE mobility error plots.")
    parser.add_argument("--volume-fraction", type=float, default=0.10)
    parser.add_argument("--num-particles", type=int, default=100)
    parser.add_argument("--num-configs", type=int, default=12)
    parser.add_argument("--base-seed", type=int, default=44)
    parser.add_argument("--force-scale", type=float, default=2.0)
    parser.add_argument("--output-dir", type=str, default="artifacts/deep_dive")
    parser.add_argument("--skip-base-vs-nbody", action="store_true")
    parser.add_argument("--skip-vf-sweep", action="store_true")
    parser.add_argument("--skip-vertical-overshoot", action="store_true")
    parser.add_argument("--overshoot-num-configs", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_repo_root()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = SweepConfig(
        volume_fraction=args.volume_fraction,
        num_particles=args.num_particles,
        num_configs=args.num_configs,
        base_seed=args.base_seed,
        force_scale=args.force_scale,
    )

    df = run_feature_sweep(cfg, out_dir=out_dir)
    report_correlations(df)
    plot_scatter_and_bins(df, out_dir=out_dir)
    plot_phase_and_kde(df, out_dir=out_dir)

    if not args.skip_base_vs_nbody:
        df_base = run_base_vs_nbody(cfg, out_dir=out_dir, num_configs=min(6, cfg.num_configs))
        report_base_vs_nbody(df_base)

    if not args.skip_vertical_overshoot:
        df_overshoot = run_vertical_split_overshoot_compare(
            cfg,
            out_dir=out_dir,
            num_configs=min(args.overshoot_num_configs, cfg.num_configs),
        )
        report_vertical_split_overshoot(df_overshoot)
        plot_vertical_split_overshoot(df_overshoot, out_dir=out_dir)

    if not args.skip_vf_sweep:
        df_vf = run_volume_fraction_sweep(cfg, out_dir=out_dir)
        print("\n=== per-vf means ===")
        print(
            df_vf.groupby("vf")[
                ["mean_rmse", "p90_rmse", "mean_rel", "mean_min_dist", "mean_dir_change", "mean_speed"]
            ]
            .mean()
            .to_string()
        )
        plot_vf_sweep(df_vf, out_dir=out_dir)

    print(f"\nSaved diagnostics to: {out_dir}")


if __name__ == "__main__":
    main()
