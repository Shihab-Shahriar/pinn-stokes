"""
Consistency check between Figure 3 (operator comparison, N=300, CPU operators)
and Figure 4 (n-body accuracy vs system size, GPU operator).

Facet A -- OPERATOR identity: on IDENTICAL N=300 configs, how far apart are the
  Fig-3 CPU n-body operator (Mob_Op_Nbody, nbody_pinn_b1.pt) and the Fig-4 GPU
  n-body operator (Mob_Nbody_Torch, nbody_cross_tmp.wt)?  No MFS needed -- we
  compare the two operators' velocity outputs directly.

Facet B -- TREND continuity: overlay the new Fig-3 N=300 n-body points on Fig-4's
  committed n-body-vs-N curves and check the N=300 point continues each vf trend.

Outputs:
  figures/consistency_fig3_fig4_nbody.png   (diagnostic overlay)
  prints a per-vf operator-agreement table.

Run: TORCH_COMPILE_DISABLE=1 python consistency_fig3_fig4.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import benchmarks.accuracy_grand_M as A
assert "pinn-stokes" in A.__file__, A.__file__

from benchmarks.cluster import uniform_sphere_cluster
from src.mob_op_nbody import Mob_Op_Nbody
from src.gpu_nbody_mob import Mob_Nbody_Torch

SHAPE = "sphere"
N = 300
VFS = [0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20]
SEEDS = [123, 124, 125]  # a few configs per vf for a stable operator-agreement number

# ---- Fig-3 CPU n-body operator (the paper's operator-comparison n-body) ----
cpu_nbody = Mob_Op_Nbody(
    shape=SHAPE, self_nn_path=A.SELF_PATH, two_nn_path=A.TWO_BODY_PATH,
    nbody_nn_path=A.NBODY_PATH, nn_only=False, rpy_only=False, switch_dist=6.0,
)
# ---- Fig-4 GPU n-body operator ----
gpu_nbody = Mob_Nbody_Torch(
    shape=SHAPE, self_nn_path=A.SELF_PATH, two_nn_path="data/models/combined_2body.wt",
    nbody_nn_path="data/models/nbody_cross_tmp.wt", near_field_2b="nn",
    far_field_2b="rpy", near_far_switch=6.0,
)
dev = gpu_nbody.device


def make_config(vf, seed):
    """Reproduce Fig-3's config+forces WITHOUT the MFS ground-truth solve."""
    np.random.seed(seed)  # matches generate_uniform_testcase ordering
    centers, _ = uniform_sphere_cluster(vf, N, seed=seed)
    ori = np.tile([0.0, 0.0, 0.0, 1.0], (N, 1))
    F = np.array([np.random.uniform(-1, 1, 3) for _ in range(N)])
    T = np.array([np.random.uniform(-1, 1, 3) for _ in range(N)])
    F = F / np.linalg.norm(F, axis=1, keepdims=True)
    T = T / np.linalg.norm(T, axis=1, keepdims=True)
    forces = np.concatenate([F, T], axis=1)
    return centers.astype(np.float64), ori.astype(np.float64), forces


rows = []
for vf in VFS:
    reldiffs = []
    for seed in SEEDS:
        centers, ori, forces = make_config(vf, seed)
        # CPU n-body
        if hasattr(cpu_nbody, "apply_cpu"):
            cpu_v = cpu_nbody.apply_cpu(centers, ori, forces, viscosity=1.0)
        else:
            cfg = np.concatenate([centers, ori], axis=1)
            cpu_v = cpu_nbody.apply(cfg, forces, viscosity=1.0)
        cpu_v = np.asarray(cpu_v)
        # GPU n-body
        pos_t = torch.as_tensor(centers, device=dev, dtype=torch.float32)
        ori_t = torch.as_tensor(ori, device=dev, dtype=torch.float32)
        f_t = torch.as_tensor(forces, device=dev, dtype=torch.float32)
        gpu_v = gpu_nbody.apply(pos_t, ori_t, f_t, viscosity=1.0).detach().cpu().numpy()
        rel = np.linalg.norm(gpu_v - cpu_v) / (np.linalg.norm(cpu_v) + 1e-12) * 100.0
        reldiffs.append(rel)
    m = float(np.mean(reldiffs))
    rows.append({"vf": vf, "cpu_gpu_reldiff_pct": m})
    print(f"vf={vf:<6} CPU-vs-GPU n-body rel diff = {m:6.2f}%  (seeds {SEEDS})", flush=True)

agree = pd.DataFrame(rows)
print("\nMean operator disagreement across all vf: "
      f"{agree['cpu_gpu_reldiff_pct'].mean():.2f}%  "
      f"(max {agree['cpu_gpu_reldiff_pct'].max():.2f}% at vf={agree.loc[agree['cpu_gpu_reldiff_pct'].idxmax(),'vf']})",
      flush=True)

# ---------------- Facet B: overlay figure ----------------
fig4 = pd.read_csv("data/M_accuracy_nbody_diff_sizes.csv")
fig3 = pd.read_csv("data/grand_M_acc_uniform_fixed_N.csv")

import matplotlib.cm as cm
colors = cm.viridis(np.linspace(0, 0.95, len(VFS)))
plt.figure(figsize=(9, 6))
for c, vf in zip(colors, VFS):
    sub = fig4[np.isclose(fig4["volume_fraction"], vf)].sort_values("num_particles")
    if not sub.empty:
        plt.plot(sub["num_particles"], sub["avg_rel_rmse"], "-o", color=c, ms=4,
                 label=f"vf={vf} (Fig4 GPU)")
    row = fig3[np.isclose(fig3["volume_fraction"], vf)]
    if not row.empty:
        plt.plot([N], [row["M_nbody"].values[0]], marker="*", color=c, ms=18,
                 markeredgecolor="k", linestyle="none")
plt.xlabel("num particles (N)")
plt.ylabel("n-body operator  rel_rmse of velocity (%)")
plt.title("Consistency: Fig-4 n-body-vs-N (lines, GPU)  +  Fig-3 N=300 n-body (stars, CPU)")
plt.legend(fontsize=7, ncol=2)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figures/consistency_fig3_fig4_nbody.png", dpi=150)
plt.close()
print("\nSaved figures/consistency_fig3_fig4_nbody.png", flush=True)

# Numeric trend check: Fig-3 N=300 vs Fig-4 largest-N (N=200)
print("\nTrend continuity (n-body rel_rmse):")
print(f"{'vf':<7}{'Fig4 N=200':>12}{'Fig3 N=300':>12}{'delta':>9}")
for vf in VFS:
    f4 = fig4[np.isclose(fig4["volume_fraction"], vf) & (fig4["num_particles"] == 200)]
    f3 = fig3[np.isclose(fig3["volume_fraction"], vf)]
    v4 = f4["avg_rel_rmse"].values[0] if not f4.empty else float("nan")
    v3 = f3["M_nbody"].values[0] if not f3.empty else float("nan")
    print(f"{vf:<7}{v4:>12.3f}{v3:>12.3f}{v3-v4:>9.3f}")
print("\nDone.", flush=True)
