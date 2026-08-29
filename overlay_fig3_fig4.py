"""
Refresh the Fig-3 / Fig-4 consistency overlay now that Fig 3 exists at BOTH N=200
and N=300.

N=200 is Fig-4's maximum N, so the Fig-3 N=200 n-body points can be compared to
Fig-4's committed n-body curve at the SAME N -- a direct check rather than an
extrapolation argument.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

VFS = [0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20]

fig4 = pd.read_csv("data/M_accuracy_nbody_diff_sizes.csv")
fig3 = pd.read_csv("data/grand_M_acc_uniform_fixed_N.csv")

colors = cm.viridis(np.linspace(0, 0.95, len(VFS)))
plt.figure(figsize=(10, 6.5))
for c, vf in zip(colors, VFS):
    sub = fig4[np.isclose(fig4["volume_fraction"], vf)].sort_values("num_particles")
    if not sub.empty:
        plt.plot(sub["num_particles"], sub["avg_rel_rmse"], "-o", color=c, ms=4,
                 lw=1.4, label=f"vf={vf}")
    for N, marker, ms in [(200, "D", 9), (300, "*", 18)]:
        row = fig3[np.isclose(fig3["volume_fraction"], vf) & (fig3["num_particles"] == N)]
        if not row.empty:
            plt.plot([N], [row["M_nbody"].values[0]], marker=marker, color=c, ms=ms,
                     markeredgecolor="k", markeredgewidth=1.0, linestyle="none")

plt.plot([], [], "o-", color="0.35", label="Fig-4 n-body vs N (GPU)")
plt.plot([], [], "D", color="0.35", markeredgecolor="k", linestyle="none",
         label="Fig-3 n-body @ N=200 (CPU, direct overlap)")
plt.plot([], [], "*", color="0.35", ms=15, markeredgecolor="k", linestyle="none",
         label="Fig-3 n-body @ N=300 (CPU)")
plt.axvline(200, color="0.7", ls="--", lw=1, zorder=0)
plt.xlabel("num particles (N)")
plt.ylabel("n-body operator  rel_rmse of velocity (%)")
plt.title("Fig-3 / Fig-4 consistency: n-body error vs system size")
plt.legend(fontsize=8, ncol=2)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figures/consistency_fig3_fig4_nbody.png", dpi=150)
plt.close()
print("Saved figures/consistency_fig3_fig4_nbody.png\n")

# ---- Direct same-N comparison at N=200 (Fig-4's max N) ----
print("DIRECT same-N check: n-body rel_rmse at N=200")
print(f"{'vf':<8}{'Fig3 (CPU)':>12}{'Fig4 (GPU)':>12}{'delta':>9}{'rel %':>9}")
deltas = []
for vf in VFS:
    f3 = fig3[np.isclose(fig3["volume_fraction"], vf) & (fig3["num_particles"] == 200)]
    f4 = fig4[np.isclose(fig4["volume_fraction"], vf) & (fig4["num_particles"] == 200)]
    if f3.empty or f4.empty:
        continue
    v3 = f3["M_nbody"].values[0]
    v4 = f4["avg_rel_rmse"].values[0]
    rel = (v3 - v4) / v4 * 100
    deltas.append(rel)
    print(f"{vf:<8}{v3:>12.3f}{v4:>12.3f}{v3-v4:>9.3f}{rel:>+8.1f}%")
d = np.array(deltas)
print(f"\nmean signed deviation: {d.mean():+.1f}%   mean |deviation|: {np.abs(d).mean():.1f}%"
      f"   (max |{np.abs(d).max():.1f}|%)")

# ---- 3b vs nbody gap at both N ----
print("\n3-body vs n-body gap (negative => n-body WORSE):")
print(f"{'vf':<8}{'N=200 gap':>12}{'N=300 gap':>12}")
for vf in VFS:
    out = [f"{vf:<8}"]
    for N in (200, 300):
        r = fig3[np.isclose(fig3["volume_fraction"], vf) & (fig3["num_particles"] == N)]
        if r.empty:
            out.append(f"{'-':>12}")
        else:
            gap = r["M_3b"].values[0] - r["M_nbody"].values[0]
            out.append(f"{gap:>+12.3f}")
    print("".join(out))
print("\nDone.")
