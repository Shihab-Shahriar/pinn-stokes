#!/usr/bin/env python3
"""Figure 8 (dynamic sedimentation of a falling cloud, N0 ~ 3000) regenerated with the new
moments-based NeMO stack.

Protocol is experiments/single_drop_sedimentation_executed.ipynb verbatim -- same drop
(R = 40, phi = 0.048, cubic lattice + 5 % jitter, seed 42, N = 3071), same gravity
(Fz = -6*pi, so the single-particle Stokes speed is 1), same adaptive RKF45
(Dormand-Prince, dt in [5e-4, 0.03], rtol 5e-3 / atol 2e-3), T = 500, snapshots every 3,
and the same pre-run U_d validation (mean settling speed while the tail fraction < 1 %).
Only the operator changes: Mob_Nbody_Moments_Torch (pc8 moments pair model + learned
diagonal, switch_dist = pair_cutoff = 8) with the production widebvh far field
(bary, mac 0.8, near cutoff 8; --fp32-level 3 is the consumer-GPU operating point),
replacing the b1 GPU operator + WarpFMM theta 0.28 of the published run.

Needs warp -> run the simulation inside the local docker image:

    bash docker/run_local.sh python figures/fig8_single_drop.py               # simulate + figure
    python figures/fig8_single_drop.py --plot-only                            # re-render storyboard
    python figures/fig8_single_drop.py --compare                              # consistency vs the old run

--compare parses the published run's per-snapshot mean/max |V| series and U_d validation
out of the executed notebook's embedded outputs and prints the consistency metrics +
overlay figure (figures/fig8_consistency.png).

Outputs: figures/fig8_single_drop_storyboard.{png,pdf} (+ drop-in copies at the paper's
include name figures/single_drop_storyboard_late.{png,pdf}), snapshots + U_d validation
in figures/fig8_single_drop_data.npz.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import re
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

# ----------------------------------------------------------------------------- notebook parameters
R_DROP = 40.0
TARGET_PHI = 0.048
VISCOSITY = 1.0
GRAVITY = -6.0 * math.pi
PERTURBATION_FRAC = 0.05
DT_INITIAL, DT_MIN, DT_MAX = 0.01, 5e-4, 0.03
RKF_RTOL, RKF_ATOL, RKF_SAFETY = 5e-3, 2e-3, 0.9
RKF_MIN_FACTOR, RKF_MAX_FACTOR = 0.2, 2.5
SNAPSHOT_INTERVAL = 3.0
NEAR_FIELD_CUTOFF = 8.0            # the pc8 stack's switch distance (published run: 6.0)
UD_TAIL_THRESHOLD, UD_MAX_TIME = 0.01, 25.0
WARMUP_STEPS = 3
STORY_TIMES = np.array([400, 420, 440, 460, 480], dtype=float)
OLD_NOTEBOOK = ROOT / "experiments" / "single_drop_sedimentation_executed.ipynb"


@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as fnull, contextlib.redirect_stdout(fnull):
        yield


def generate_single_drop(R, target_phi, perturbation_frac=0.05, seed=42):
    """Spherical drop on a primitive cubic lattice, auto-tuned to the target volume fraction."""
    rng = np.random.default_rng(seed)
    a_lattice = (4.0 * np.pi / (3.0 * target_phi)) ** (1.0 / 3.0)
    n_pts = int(np.ceil(2 * R / a_lattice)) + 2
    grid_1d = np.arange(n_pts) * a_lattice
    grid_1d -= np.mean(grid_1d)
    x, y, z = np.meshgrid(grid_1d, grid_1d, grid_1d, indexing="ij")
    coords = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T
    particles = coords[np.linalg.norm(coords, axis=1) <= R]
    particles += rng.uniform(-perturbation_frac * a_lattice, perturbation_frac * a_lattice, size=particles.shape)
    print(f"Drop: lattice {a_lattice:.4f}, N={len(particles)}, phi={len(particles) / R ** 3:.4f}")
    return particles.astype(np.float32)


# ----------------------------------------------------------------------------- simulation
def simulate(args) -> dict:
    import torch
    import torch._inductor.config as inductor_config

    inductor_config.triton.cudagraph_skip_dynamic_graphs = True
    inductor_config.triton.cudagraphs = False
    inductor_config.freezing = True
    from src.gpu_nbody_moments import Mob_Nbody_Moments_Torch
    from src.treecode_widebvh import WidebvhFMM

    particles = generate_single_drop(R_DROP, TARGET_PHI, PERTURBATION_FRAC)
    N = len(particles)
    mob_near = Mob_Nbody_Moments_Torch(
        shape="sphere", self_nn_path="data/models/self_interaction_model.pt",
        two_nn_path="data/models/combined_2body.wt",
        moments_nn_path="experiments/nbody_moments_v2_kinf_rc8_pc8.wt",
        diag_nn_path="experiments/nbody_diag_v2_pc8.wt",
        near_field_2b="nn", far_field_2b=None, switch_dist=NEAR_FIELD_CUTOFF)
    warp_solver = WidebvhFMM(near_field_operator=mob_near, near_field_cutoff=NEAR_FIELD_CUTOFF,
                             device="cuda", fp32_level=args.fp32_level)
    print(f"Far field: widebvh, policy={warp_solver.policy}, mac={warp_solver.mac}, "
          f"pdeg={warp_solver.pdeg}, fp32_level={warp_solver.fp32_level}", flush=True)

    device = torch.device("cuda")
    positions = torch.from_numpy(particles).to(device)
    initial_positions = positions.clone()
    orientations = torch.zeros((N, 4), dtype=torch.float32, device=device); orientations[:, 3] = 1.0
    forces = torch.zeros((N, 6), dtype=torch.float32, device=device); forces[:, 2] = GRAVITY
    vis_arr = torch.full((N,), VISCOSITY, dtype=torch.float32, device=device)

    @torch.no_grad()
    def compute_velocity(pos):
        with suppress_stdout():
            return warp_solver.apply(pos, orientations, forces, vis_arr)

    @torch.no_grad()
    def rkf45_adaptive_step(pos, dt):
        """One Dormand-Prince 5(4) trial step -> (pos_5th, err_norm, dt_next, accept)."""
        f = lambda x: compute_velocity(x)[:, :3]
        k1 = f(pos)
        k2 = f(pos + dt * (1.0 / 5.0) * k1)
        k3 = f(pos + dt * ((3.0 / 40.0) * k1 + (9.0 / 40.0) * k2))
        k4 = f(pos + dt * ((44.0 / 45.0) * k1 + (-56.0 / 15.0) * k2 + (32.0 / 9.0) * k3))
        k5 = f(pos + dt * ((19372.0 / 6561.0) * k1 + (-25360.0 / 2187.0) * k2
                           + (64448.0 / 6561.0) * k3 + (-212.0 / 729.0) * k4))
        k6 = f(pos + dt * ((9017.0 / 3168.0) * k1 + (-355.0 / 33.0) * k2 + (46732.0 / 5247.0) * k3
                           + (49.0 / 176.0) * k4 + (-5103.0 / 18656.0) * k5))
        pos_5th = pos + dt * ((35.0 / 384.0) * k1 + (500.0 / 1113.0) * k3 + (125.0 / 192.0) * k4
                              + (-2187.0 / 6784.0) * k5 + (11.0 / 84.0) * k6)
        k7 = f(pos_5th)
        pos_4th = pos + dt * ((5179.0 / 57600.0) * k1 + (7571.0 / 16695.0) * k3 + (393.0 / 640.0) * k4
                              + (-92097.0 / 339200.0) * k5 + (187.0 / 2100.0) * k6 + (1.0 / 40.0) * k7)
        err = pos_5th - pos_4th
        scale = RKF_ATOL + RKF_RTOL * torch.maximum(torch.abs(pos), torch.abs(pos_5th))
        err_norm = torch.max(torch.abs(err) / torch.clamp(scale, min=1e-12)).item()
        factor = RKF_MAX_FACTOR if err_norm == 0.0 else float(
            np.clip(RKF_SAFETY * err_norm ** -0.2, RKF_MIN_FACTOR, RKF_MAX_FACTOR))
        return pos_5th, err_norm, float(np.clip(dt * factor, DT_MIN, DT_MAX)), err_norm <= 1.0

    print(f"Warmup ({WARMUP_STEPS} RKF45 steps)...", flush=True)
    warm_pos, warm_dt = positions.clone(), DT_INITIAL
    t0 = time.perf_counter()
    for i in range(WARMUP_STEPS):
        for _ in range(8):
            trial, _, warm_dt, ok = rkf45_adaptive_step(warm_pos, warm_dt)
            if ok:
                warm_pos = trial
                break
        print(f"  warmup {i + 1}/{WARMUP_STEPS}  [{time.perf_counter() - t0:.0f} s]", flush=True)
    del warm_pos
    torch.cuda.empty_cache()

    # -------- pre-run U_d validation (paper protocol: mean settling speed, tail < 1 %) --------
    pos, t, dt = initial_positions.clone(), 0.0, DT_INITIAL
    init_center = initial_positions.mean(dim=0, keepdim=True)
    drop_radius_ref = max(R_DROP, torch.quantile(
        torch.linalg.norm(initial_positions - init_center, dim=1), 0.995).item())
    ud_w, ud_wsum, tail_cross_time, n_ud = 0.0, 0.0, None, 0
    while t < UD_MAX_TIME - 1e-12:
        center = pos.mean(dim=0, keepdim=True)
        tail = (torch.linalg.norm(pos - center, dim=1) > drop_radius_ref)
        vel = compute_velocity(pos)[:, :3]
        in_drop = ~tail
        ud_inst = (-vel[in_drop, 2]).mean().item() if in_drop.any() else (-vel[:, 2]).mean().item()
        if tail.float().mean().item() >= UD_TAIL_THRESHOLD and n_ud > 0:
            tail_cross_time = t
            break
        local_dt = min(dt, UD_MAX_TIME - t)
        nxt, _, dt, ok = rkf45_adaptive_step(pos, local_dt)
        if ok:
            pos = nxt; t += local_dt
            ud_wsum += ud_inst * local_dt; ud_w += local_dt; n_ud += 1
        elif dt <= DT_MIN + 1e-12:
            break
    ud_mean = ud_wsum / max(ud_w, 1e-12)
    ud_hr = 1.0 + (6.0 / 5.0) * N / R_DROP  # Eq. (8), epsilon = a/R_drop
    print(f"[Ud] mean (tail<1%) = {ud_mean:.4f} | Eq.(8) HR = {ud_hr:.4f} | "
          f"tail crossed 1% at t={tail_cross_time}", flush=True)

    # -------- main loop --------
    positions = initial_positions.clone()
    current_time, dt = 0.0, DT_INITIAL
    accepted = rejected = 0
    snap_t, snap_pos, snap_vel = [], [], []

    def take_snapshot(t_now, pos_now):
        vel = compute_velocity(pos_now)
        vmag = torch.linalg.norm(vel[:, :3], dim=1).cpu().numpy()
        snap_t.append(t_now); snap_pos.append(pos_now.cpu().numpy().copy()); snap_vel.append(vmag)
        print(f"  Snapshot at t={t_now:.3f}: mean |v|={vmag.mean():.4f}, max |v|={vmag.max():.4f}", flush=True)

    take_snapshot(0.0, positions)
    next_snap = SNAPSHOT_INTERVAL
    sim_start = time.perf_counter()
    with torch.no_grad():
        while current_time < args.t_final - 1e-12:
            remaining = min(args.t_final, next_snap) - current_time
            if remaining <= 1e-12:
                take_snapshot(current_time, positions)
                next_snap += SNAPSHOT_INTERVAL
                continue
            local_dt = min(dt, remaining)
            nxt, err_norm, dt_next, ok = rkf45_adaptive_step(positions, local_dt)
            if ok:
                positions = nxt; current_time += local_dt; accepted += 1; dt = dt_next
                if current_time >= next_snap - 1e-9:
                    take_snapshot(current_time, positions)
                    next_snap += SNAPSHOT_INTERVAL
                if accepted % 200 == 0:
                    el = time.perf_counter() - sim_start
                    print(f"accepted {accepted:6d} (+{rejected} rej) t={current_time:8.3f} dt={local_dt:.5f} "
                          f"err={err_norm:.2e} [{el:.0f} s, {el / current_time:.1f} s/unit-t]", flush=True)
            else:
                rejected += 1; dt = dt_next
                if dt <= DT_MIN + 1e-12:
                    print("RKF45 reached DT_MIN with rejected step; stopping early.", flush=True)
                    break
    print(f"Done: accepted={accepted} rejected={rejected} wall={time.perf_counter() - sim_start:.0f} s "
          f"snapshots={len(snap_t)}", flush=True)
    return {"times": np.array(snap_t), "positions": np.array(snap_pos, dtype=np.float32),
            "vel_mags": np.array(snap_vel, dtype=np.float32),
            "ud_mean": ud_mean, "ud_hr": ud_hr, "ud_tail_cross_time": float(tail_cross_time or np.nan),
            "N": N, "accepted": accepted, "rejected": rejected, "fp32_level": args.fp32_level}


# ----------------------------------------------------------------------------- storyboard (notebook cell 8c)
def plot_storyboard(d: dict, out_stem: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize

    times, all_pos, all_vel = d["times"], d["positions"], d["vel_mags"]
    idx = [int(np.argmin(np.abs(times - t))) for t in STORY_TIMES]
    story_norm = Normalize(vmin=0.0, vmax=float(np.percentile(all_vel.ravel(), 99)))

    side_coords, top_coords = [], []
    for i in idx:
        c = np.median(all_pos[i], axis=0)
        side_coords.append((all_pos[i][:, 1] - c[1], all_pos[i][:, 2] - c[2]))
        top_coords.append((all_pos[i][:, 0] - c[0], all_pos[i][:, 1] - c[1]))

    def limits(coord_sets, pad=0.08, cap=None):
        xs = np.concatenate([s[0] for s in coord_sets]); ys = np.concatenate([s[1] for s in coord_sets])
        if cap is not None:
            ys = ys[ys <= cap]
        x0, x1 = np.percentile(xs, [0.2, 99.8]); y0, y1 = np.percentile(ys, [0.2, 99.8])
        dx, dy = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
        return (x0 - pad * dx, x1 + pad * dx), (y0 - pad * dy, y1 + pad * dy)

    # The side-view window excludes the far leakage trail (> 6 R above the cloud) before taking
    # percentiles: the trail runs off the top of the frame exactly as in the published figure, whose
    # raw-percentile rule only worked because that run had shed a handful of particles by t=400.
    side_xlim, side_ylim = limits(side_coords, cap=6.0 * R_DROP)
    top_xlim, top_ylim = limits(top_coords)

    fig = plt.figure(figsize=(14.2, 5.8))
    gs = fig.add_gridspec(2, 6, left=0.075, right=0.965, top=0.90, bottom=0.075,
                          wspace=0.08, hspace=0.18, width_ratios=[1, 1, 1, 1, 1, 0.05])
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(5)] for r in range(2)])
    cax = fig.add_subplot(gs[:, 5])

    for col, i in enumerate(idx):
        vmag = all_vel[i]
        for row, (xx, yy) in enumerate((side_coords[col], top_coords[col])):
            ax = axes[row, col]
            ax.scatter(xx, yy, c=vmag, cmap=cm.viridis, norm=story_norm, s=2.8, alpha=0.88,
                       rasterized=True, linewidths=0)
            ax.set_xlim(*(side_xlim if row == 0 else top_xlim))
            ax.set_ylim(*(side_ylim if row == 0 else top_ylim))
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
        axes[0, col].set_title(f"t={int(round(times[i]))}", fontsize=10, pad=6)

    fig.text(0.028, 0.66, "side view, y-z", rotation=90, ha="center", va="center", fontsize=11)
    fig.text(0.028, 0.28, "top view, x-y", rotation=90, ha="center", va="center", fontsize=11)
    sm = cm.ScalarMappable(norm=story_norm, cmap=cm.viridis); sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("|V|", rotation=90, labelpad=10)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0, labelsize=9)

    for ext in ("png", "pdf"):
        p = out_stem.parent / f"{out_stem.name}_storyboard.{ext}"
        fig.savefig(p, dpi=450, bbox_inches="tight", pad_inches=0.03)
        shutil.copy(p, ROOT / "figures" / f"single_drop_storyboard_late.{ext}")
    print(f"-> {out_stem}_storyboard.{{png,pdf}} (+ figures/single_drop_storyboard_late.*)")


# ----------------------------------------------------------------------------- consistency vs the old run
def old_run_series():
    """(t, mean|v|, max|v|) series + U_d numbers of the published run, parsed from the
    executed notebook's embedded stdout."""
    nb = json.load(open(OLD_NOTEBOOK))
    txt = "".join("".join(o.get("text", [])) for c in nb["cells"] for o in c.get("outputs", [])
                  if o.get("output_type") == "stream")
    snaps = re.findall(r"Snapshot at t=([\d.]+): mean \|v\|=([\d.]+), max \|v\|=([\d.]+)", txt)
    arr = np.array(snaps, dtype=np.float64)
    order = np.argsort(arr[:, 0])
    ud = float(re.search(r"Ud mean from simulation \(tail < 1%\) = ([\d.]+)", txt).group(1))
    tail_t = float(re.search(r"Tail threshold crossed at t = ([\d.]+)", txt).group(1))
    return arr[order], ud, tail_t


def compare(d: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    old, old_ud, old_tail_t = old_run_series()
    # the old loop snapshotted some times twice; keep the first of each
    _, uniq = np.unique(old[:, 0].round(3), return_index=True)
    old = old[np.sort(uniq)]
    new_t, new_mean = d["times"], d["vel_mags"].mean(axis=1)
    new_max = d["vel_mags"].max(axis=1)

    print("=== consistency: new moments NeMO vs published run ===")
    print(f"U_d (tail<1%):   new {d['ud_mean']:.3f} | old {old_ud:.3f} | Eq.(8) HR {d['ud_hr']:.2f} "
          f"(delta new-old {100 * (d['ud_mean'] - old_ud) / old_ud:+.2f}%)")
    print(f"tail crossed 1%: new t={d['ud_tail_cross_time']:.3f} | old t={old_tail_t:.3f}")
    print(f"initial mean |v|: new {new_mean[0]:.3f} | old {old[0, 1]:.3f} "
          f"({100 * (new_mean[0] - old[0, 1]) / old[0, 1]:+.2f}%)")
    on_new = np.interp(new_t, old[:, 0], old[:, 1])
    on_new_max = np.interp(new_t, old[:, 0], old[:, 2])
    for lo, hi in [(0, 100), (100, 250), (250, 400), (400, 500), (0, 500)]:
        m = (new_t >= lo) & (new_t <= hi)
        rel = np.abs(new_mean[m] - on_new[m]) / on_new[m]
        print(f"  mean|v| rel dev, t in [{lo:3d},{hi:3d}]: median {100 * np.median(rel):5.2f}%  "
              f"max {100 * rel.max():5.2f}%")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (ys_old, ys_new, lbl) in zip(axes, [(old[:, 1], new_mean, "mean |V|"),
                                                (old[:, 2], new_max, "max |V|")]):
        ax.plot(old[:, 0], ys_old, "-", color="#D55E00", lw=1.4, label="published (b1 + WarpFMM)")
        ax.plot(new_t, ys_new, "-", color="#0072B2", lw=1.4, label="new NeMO (moments pc8 + diag)")
        ax.set_xlabel("t"); ax.set_ylabel(lbl)
        ax.grid(alpha=0.3)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Falling-cloud velocity statistics: published vs new NeMO", fontsize=11)
    fig.tight_layout()
    fig.savefig("figures/fig8_consistency.png", dpi=200, bbox_inches="tight")
    print("-> figures/fig8_consistency.png")

    # breakup timing: when the pooled mean |V| first drops below a fraction of its initial value
    def cross_time(ts, ys, level):
        below = np.nonzero(ys < level)[0]
        return float(ts[below[0]]) if len(below) else float("nan")

    print("breakup timing (t at which mean|v| first drops below f x initial):")
    for f in (0.8, 0.6, 0.4, 0.25):
        print(f"  f={f:.2f}: old t={cross_time(old[:, 0], old[:, 1], f * old[0, 1]):6.1f} | "
              f"new t={cross_time(new_t, new_mean, f * new_mean[0]):6.1f}")

    # instability-sequence metrics of the new run (old counterparts are read off its diagnostics
    # panels; its COM-based leakage metric saturates by t~60 and is not reused here)
    pos = d["positions"]
    horiz = np.array([np.percentile(np.linalg.norm(p[:, :2] - np.median(p[:, :2], axis=0), axis=1), 90)
                      for p in pos])
    trail = np.array([100.0 * np.mean(p[:, 2] - np.median(p[:, 2]) > 2 * R_DROP) for p in pos])
    print("new-run horizontal radius (90th pct): "
          + "  ".join(f"t={tq:g}:{horiz[np.argmin(np.abs(new_t - tq))]:.1f}" for tq in (0, 200, 300, 400, 450, 500)))
    print("new-run radius crossings: "
          + "  ".join(f">{lvl:g} at t={cross_time(new_t, -horiz, -lvl):.0f}" for lvl in (50, 60, 100, 150)))
    print("new-run trail fraction (z > median + 2R): "
          + "  ".join(f"t={tq:g}:{trail[np.argmin(np.abs(new_t - tq))]:.2f}%" for tq in (100, 200, 300, 400, 500)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--t-final", type=float, default=500.0)
    ap.add_argument("--fp32-level", type=int, default=3, help="widebvh far-field fp32 level (3 = consumer GPU)")
    ap.add_argument("--out", type=Path, default=Path("figures/fig8_single_drop"))
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--compare", action="store_true", help="consistency metrics vs the published run")
    args = ap.parse_args()
    npz = args.out.parent / (args.out.name + "_data.npz")
    if args.plot_only or args.compare:
        d = dict(np.load(npz))
    else:
        d = simulate(args)
        np.savez_compressed(npz, **d)
        print(f"-> {npz}")
    if args.compare:
        compare(d)
        return
    plot_storyboard(d, args.out)


if __name__ == "__main__":
    main()
