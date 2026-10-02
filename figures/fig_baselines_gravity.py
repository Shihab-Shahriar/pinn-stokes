#!/usr/bin/env python3
"""NeMO against the three baselines (RPY, HIGNN, Stokesian Dynamics with and without lubrication) on one chart: translational PRMSE vs volume
fraction at N = 200, torque-free spheres settling under uniform gravity (exp fig4g: F = (0, 0, -9.81), T = 0).

Render-only from data/paper_accuracy_v2.csv. The gravity protocol is the one every baseline can be scored on: HIGNN
predicts translational velocities only (no torques in, no angular velocities out), so the metric is prmse_lin,
100 ||U_pred - U_true||_F / ||U_true||_F over the translational components (benchmarks/compare_nbody_moments.py).

Series (harness op -> label):
  M_mom_v3_nb8lin_tr2_pc8c_diag  NeMO n-body          moments v3 pair correction + learned diagonal (Fig 3's NeMO)
  HIGNN_full                     HIGNN                their 2-body + 3-body (cutoff 5) + self networks (src/hignn_ops.py)
  M_rpy                          RPY                  pairwise RPY, all pairs
  SD                             Stokesian Dynamics   Townsend's SD as shipped: FTS far field + pairwise lubrication
  SD_Minf                        ..., no lubrication  the same SD solve with the lubrication term (R2Bexact) dropped:
                                                       its FTS far-field mobility alone (src/sd_ops.py minfinity_only)
Rows (all 8 phi x 10 seeds per op; HIGNN is registered as a GPU op):
  TORCH_COMPILE_DISABLE=1 python benchmarks/sd_gravity_truth.py           # gravity truths for phi 0.075/0.125/0.175/0.2
  TORCH_COMPILE_DISABLE=1 python benchmarks/paper_accuracy_v2.py --exp fig4g --N 200 \
      --ops M_rpy SD SD_Minf M_mom_v3_nb8lin_tr2_pc8c_diag --skip-done
  TORCH_COMPILE_DISABLE=1 python benchmarks/paper_accuracy_v2.py --exp fig4g --N 200 --ops HIGNN_full --gpu-ops --skip-done

    python figures/fig_baselines_gravity.py        # -> figures/fig_baselines_gravity.pdf
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

MAIN_CSV = Path("data/paper_accuracy_v2.csv")
EXP, METRIC = "fig4g", "prmse_lin"
PHIS = [0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]
N_SEEDS = 10
# Drawn bottom to top in reverse list order (so the curve stack and z-order agree); the legend sits above the axes in
# columns, LEGEND_COLUMNS (the lubrication-off SD under SD), since the five curves span three decades and leave no
# empty corner.
# NeMO red and RPY blue are Figs 3/4's; aqua and amber pass the CVD checks against both (dataviz validator, all pairs).
# Marker and dash carry identity alongside colour; the two SD curves share SD's amber (one method, lubrication on/off).
SERIES = [("SD", "Stokesian Dynamics", "#c98500", dict(ls="-.", marker="^", ms=6.5)),
          ("M_rpy", "RPY", "#4C72B0", dict(ls="--", marker="o", ms=5.5, mfc="white")),
          ("HIGNN_full", "HIGNN", "#1baf7a", dict(ls="-", marker="s", ms=5.5)),
          ("M_mom_v3_nb8lin_tr2_pc8c_diag", "NeMO n-body", "#C44E52", dict(ls="-", marker="o", ms=6.5, lw=2.4)),
          ("SD_Minf", "Stokesian Dynamics, no lubrication", "#c98500", dict(ls=":", marker="^", ms=6.5, mfc="white"))]
LEGEND_COLUMNS = [["M_mom_v3_nb8lin_tr2_pc8c_diag"], ["HIGNN_full"], ["M_rpy"], ["SD", "SD_Minf"]]


def load(N: int) -> pd.DataFrame:
    df = pd.read_csv(MAIN_CSV)
    d = df[(df["exp"] == EXP) & (df["N"] == N) & df["op"].isin([s[0] for s in SERIES])]
    g = d.groupby(["op", "phi"])[METRIC]
    a = pd.DataFrame({"mean": g.mean(), "std": g.std(ddof=0), "n": g.count()}).reset_index()
    for op, *_ in SERIES:
        got = sorted(a.loc[a["op"] == op, "phi"].round(4))
        assert got == PHIS, f"{op}: {EXP} N={N} rows at phi {got}, need {PHIS} (run the commands in the docstring)"
    assert (a["n"] == N_SEEDS).all(), f"incomplete cells:\n{a[a['n'] != N_SEEDS].to_string(index=False)}"
    return a


def plot(a: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

    fig, ax = plt.subplots(figsize=(8, 5.4))
    for z, (op, label, color, style) in enumerate(SERIES):
        r = a[a["op"] == op].sort_values("phi")
        kw = dict(lw=2.0, mew=1.2, mec=color if style.get("mfc") == "white" else "white")
        kw.update(style)
        ax.fill_between(r["phi"], r["mean"] - r["std"], r["mean"] + r["std"], color=color, alpha=0.15, lw=0,
                        zorder=2 + z)
        ax.plot(r["phi"], r["mean"], color=color, label=label, zorder=10 + z, **kw)

    ax.set_xticks(PHIS)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{100 * x:g}%"))
    ax.set_xlim(0.015, 0.21)
    ax.set_yscale("log")
    ax.set_ylim(0.05, 40)
    ax.yaxis.set_major_locator(FixedLocator([0.1, 0.2, 0.5, 1, 2, 5, 10, 20]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="major", color="#e3e3e3", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel(r"Volume fraction $\phi$", fontsize=12)
    ax.set_ylabel("Translational PRMSE (%)", fontsize=12)
    ax.tick_params(labelsize=10)
    # fig.legend fills column by column; pad the short columns with invisible entries
    by_op = dict(zip([s[0] for s in SERIES], zip(*ax.get_legend_handles_labels())))
    blank = (Line2D([], [], ls="none"), "")
    nrow = max(len(col) for col in LEGEND_COLUMNS)
    entries = [by_op[col[r]] if r < len(col) else blank for col in LEGEND_COLUMNS for r in range(nrow)]
    fig.legend(*zip(*entries), loc="upper center", ncol=len(LEGEND_COLUMNS), frameon=False, fontsize=10.5,
               handlelength=3.2, columnspacing=1.6, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--N", type=int, default=200)
    ap.add_argument("--out", type=Path, default=Path("figures/fig_baselines_gravity"))
    args = ap.parse_args()

    a = load(args.N)
    labels = {op: label for op, label, *_ in SERIES}
    p = a.pivot(index="phi", columns="op", values="mean")[[s[0] for s in SERIES]].rename(columns=labels)
    nemo = p["NeMO n-body"]
    print(f"{METRIC} (%) vs phi at N={args.N}, {EXP}, mean over {N_SEEDS} seeds:")
    print(p.round(2).to_string())
    print("ratio to NeMO:")
    print(p.drop(columns="NeMO n-body").div(nemo, axis=0).round(2).to_string())

    fig = plot(a)
    fig.savefig(args.out.with_suffix(".pdf"))
    print(f"-> {args.out}.pdf")


if __name__ == "__main__":
    main()
