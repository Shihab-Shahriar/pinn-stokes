#!/usr/bin/env python3
"""Figure 4 candidate: grand mobility accuracy vs N under gravity forcing, NeMO n-body (v3 + diag) against RPY,
one panel per volume fraction. N = 20 .. 10^4 (phi = 0.1 to 3 x 10^4), truths = widebvh Broms MFS beyond N = 200
(`benchmarks/broms_truth.py --forcing gravity`, `artifacts/fig4_large_n_report.md` §4b).

Render-only: the rows come from the paper-accuracy harness (GPU ops, docker for warp):

    RUN_LOCAL_DOCKER_ARGS="-e TORCH_COMPILE_DISABLE=1" bash docker/run_local.sh \
        python benchmarks/paper_accuracy_v2.py --exp fig4g --phis 0.025 0.05 0.1 0.15 \
        --gpu-ops --ops M_rpy_gpu M_mom_gpu_v3_diag --skip-done

    python figures/fig4_n_acc.py
    python figures/fig4_n_acc.py --max-n 1000 --out figures/fig4_n_acc_n1000   # small-N zoom (10 seeds per cell, 5 at N = 1000)
    python figures/fig4_n_acc.py --max-n 1000 --metric max_rel_rmse --out figures/fig4_n_acc_max_n1000   # worst particle

Random forcing (exp fig4: unit forces and torques in random directions, Fig 3's protocol) with `--forcing random`;
rows from the same harness command with `--exp fig4` (truths exist to N = 10^4 at these four phi):

    python figures/fig4_n_acc.py --forcing random --max-n 1000 --out figures/fig4_n_acc_random_n1000
    python figures/fig4_n_acc.py --forcing random --max-n 1000 --metric max_rel_rmse --out figures/fig4_n_acc_random_max_n1000
    python figures/fig4_n_acc.py --forcing random --max-n 1000 --single --out figures/fig4_n_acc_random_n1000_single

`--single` merges the four panels into one axes: colour = volume fraction (4 hues, CVD-validated), line style = operator
(NeMO solid, RPY dashed); no seed bands. `--single --nemo-only` drops RPY: four NeMO curves with +-1 std seed bands
(paper Fig 4, `~/nemo/figs/fig4_n_acc.pdf`, with a linear N axis):

    python figures/fig4_n_acc.py --forcing random --max-n 1000 --single --nemo-only --linear-x \
        --out figures/fig4_n_acc_random_n1000_nemo_linx

Series (harness op -> label):
  M_rpy_gpu          RPY          analytic self + RPY over all pairs (== M_rpy to 1e-5 pts; GPU for large N)
  M_mom_gpu_v3_diag  NeMO n-body  self + 2-body NN within 8 + moments v3 pair correction + learned diagonal, RPY beyond
Plotted value: rel_rmse (%) averaged over the seeds of each (N, phi) cell (10 up to N = 500, tapering to 2 at N = 3e4),
band = +-1 std across seeds. `--metric max_rel_rmse` plots the worst particle instead: max_i |v_i - v_i^ref| / |v_i^ref|
(6-vectors), per configuration, then averaged over seeds. Outputs: figures/fig4_n_acc.{pdf,png}.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

MAIN_CSV = Path("data/paper_accuracy_v2.csv")
SERIES = [("M_rpy_gpu", "RPY", "#4C72B0"),  # Fig 3's operator colours
          ("M_mom_gpu_v3_diag", "NeMO n-body", "#C44E52")]
PHIS = [0.025, 0.05, 0.1, 0.15]
EXP = {"gravity": "fig4g", "random": "fig4"}
PHI_COLORS = ["#eb6834", "#1baf7a", "#2a78d6", "#4a3aa7"]  # orange, aqua, blue, violet: CVD-safe on all pairs (curves cross)
YLABEL = {"rel_rmse": "PRMSE (%)", "max_rel_rmse": "Max per-particle error (%)"}


def load(max_n: int | None = None, metric: str = "rel_rmse", forcing: str = "gravity") -> pd.DataFrame:
    df = pd.read_csv(MAIN_CSV)
    d = df[(df["exp"] == EXP[forcing]) & df["op"].isin([s[0] for s in SERIES])
           & np.isclose(df["phi"].values[:, None], PHIS).any(axis=1)]
    if max_n is not None:
        d = d[d["N"] <= max_n]
    keys = {op: set(map(tuple, g[["N", "phi", "seed"]].round(6).values)) for op, g in d.groupby("op")}
    assert set(keys) == {s[0] for s in SERIES}, f"missing ops: {sorted({s[0] for s in SERIES} - set(keys))}"
    a, b = (keys[s[0]] for s in SERIES)
    assert a == b, f"ops cover different cells: {len(a ^ b)} differ"
    g = d.groupby(["op", "phi", "N"])[metric]
    return pd.DataFrame({"mean": g.mean(), "std": g.std(ddof=0), "n": g.count()}).reset_index()


def _format_axes(ax, n_max, linear_x: bool = False):
    from matplotlib.ticker import FixedLocator, NullLocator
    if linear_x:
        ticks = list(range(0, int(n_max) + 1, 200 if n_max <= 1000 else 2000))
        ax.set_xlim(0, n_max * 1.03)
    else:
        ticks = [n for n in (20, 50, 100, 200, 500, 1000) if n <= n_max] if n_max <= 1000 else [20, 100, 1000, 10000]
        ax.set_xscale("log")
        ax.set_xlim(ticks[0] * 0.8, n_max * 1.3)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([f"{t:,}" for t in ticks])
    ax.grid(True, color="#e3e3e3", lw=0.8)
    ax.set_axisbelow(True)


def plot(a: pd.DataFrame, metric: str = "rel_rmse", linear_x: bool = False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(8, 5.6), sharex=True)
    for ax, phi in zip(axes.flat, PHIS):
        for op, label, color in SERIES:
            r = a[(a["op"] == op) & np.isclose(a["phi"], phi)].sort_values("N")
            ax.fill_between(r["N"], r["mean"] - r["std"], r["mean"] + r["std"], color=color, alpha=0.15, lw=0)
            ax.plot(r["N"], r["mean"], color=color, lw=2, marker="o", ms=4.5, mec="white", mew=0.8, label=label)
        _format_axes(ax, a["N"].max(), linear_x)
        ax.set_title(f"$\\phi$ = {100 * phi:g}%", fontsize=11)
        ax.tick_params(labelsize=9)
    for ax in axes[1]:
        ax.set_xlabel("Number of particles $N$", fontsize=11)
    for ax in axes[:, 0]:
        ax.set_ylabel(YLABEL[metric], fontsize=11)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_nemo(a: pd.DataFrame, metric: str = "rel_rmse", linear_x: bool = False):
    """NeMO n-body alone, one curve per volume fraction (the paper's original Fig 4 layout)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    op = SERIES[1][0]
    fig, ax = plt.subplots(figsize=(8, 5))
    for phi, color in zip(PHIS, PHI_COLORS):
        r = a[(a["op"] == op) & np.isclose(a["phi"], phi)].sort_values("N")
        ax.fill_between(r["N"], r["mean"] - r["std"], r["mean"] + r["std"], color=color, alpha=0.15, lw=0)
        ax.plot(r["N"], r["mean"], color=color, lw=2, marker="o", ms=5, mec="white", mew=1.0, label=f"{100 * phi:g}%")
    _format_axes(ax, a["N"].max(), linear_x)
    ax.set_xlabel("Number of particles $N$", fontsize=12)
    ax.set_ylabel(YLABEL[metric], fontsize=12)
    ax.tick_params(labelsize=10)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=c, lw=2, marker="o", ms=5) for c in PHI_COLORS]  # no white ring: reads as a gap
    ax.legend(handles, [f"{100 * p:g}%" for p in PHIS], title="Volume fraction", loc="upper left", fontsize=10,
              title_fontsize=10)
    fig.tight_layout()
    return fig


def plot_single(a: pd.DataFrame, metric: str = "rel_rmse", linear_x: bool = False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    style = {SERIES[0][0]: dict(ls="--", marker="o", mfc="white"),  # RPY: dashed, open markers
             SERIES[1][0]: dict(ls="-", marker="o", mec="white")}   # NeMO: solid, filled markers
    fig, ax = plt.subplots(figsize=(8.6, 5))
    for phi, color in zip(PHIS, PHI_COLORS):
        for op, _, _ in SERIES:
            r = a[(a["op"] == op) & np.isclose(a["phi"], phi)].sort_values("N")
            ax.plot(r["N"], r["mean"], color=color, lw=2, ms=5, mew=1.0, **style[op])
    _format_axes(ax, a["N"].max(), linear_x)
    ax.set_xlabel("Number of particles $N$", fontsize=12)
    ax.set_ylabel(YLABEL[metric], fontsize=12)
    ax.tick_params(labelsize=10)
    # both legends outside the axes: the RPY curves at high phi fill the upper-left corner
    kw = dict(loc="upper left", fontsize=10, title_fontsize=10, frameon=False, handlelength=3.2, alignment="left")
    phi_handles = [Line2D([], [], color=c, lw=2.5) for c in PHI_COLORS]
    leg = ax.legend(phi_handles, [f"{100 * p:g}%" for p in PHIS], title="Volume fraction",
                    bbox_to_anchor=(1.02, 1.0), **kw)
    ax.add_artist(leg)
    op_style = {op: {**st, "mec": "#333333"} for op, st in style.items()}  # no white ring: it reads as a line break
    op_handles = [Line2D([], [], color="#333333", lw=2, ms=5, mew=1.0, **op_style[op]) for op, _, _ in SERIES]
    ax.legend(op_handles, [label for _, label, _ in SERIES], title="Mobility operator",
              bbox_to_anchor=(1.02, 0.6), **kw)
    fig.subplots_adjust(left=0.08, right=0.78, bottom=0.12, top=0.97)  # tight_layout would count the outside legends
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("figures/fig4_n_acc"))
    ap.add_argument("--max-n", type=int, default=None, help="drop cells with N above this")
    ap.add_argument("--metric", choices=sorted(YLABEL), default="rel_rmse")
    ap.add_argument("--forcing", choices=sorted(EXP), default="gravity")
    ap.add_argument("--single", action="store_true", help="all four volume fractions in one axes")
    ap.add_argument("--nemo-only", action="store_true", help="with --single: NeMO curves only (no RPY)")
    ap.add_argument("--linear-x", action="store_true", help="linear N axis (default log)")
    args = ap.parse_args()

    a = load(args.max_n, args.metric, args.forcing)
    labels = {op: label for op, label, _ in SERIES}
    p = a.pivot_table(index=["phi", "N"], columns="op", values="mean")[[s[0] for s in SERIES]].rename(columns=labels)
    p["RPY / NeMO"] = p["RPY"] / p["NeMO n-body"]
    p["seeds"] = a[a["op"] == SERIES[1][0]].set_index(["phi", "N"])["n"]
    print(f"{args.forcing} forcing, {args.metric} (%) seed mean:")
    print(p.round(2).to_string())

    assert args.single or not args.nemo_only, "--nemo-only needs --single"
    fig = (plot_nemo if args.nemo_only else plot_single if args.single else plot)(a, args.metric, args.linear_x)
    for ext in ("pdf", "png"):
        fig.savefig(args.out.with_suffix(f".{ext}"), dpi=600)
    print(f"-> {args.out}.pdf, {args.out}.png")


if __name__ == "__main__":
    main()
