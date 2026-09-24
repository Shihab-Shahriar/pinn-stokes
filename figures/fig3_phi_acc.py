#!/usr/bin/env python3
"""Figure 3: grand mobility accuracy vs volume fraction at N = 200, for the v3 stack with every learned
term cut off at 8 radii (RPY beyond).

Render-only: the rows come from the paper-accuracy harness. If an op is missing from the CSV:

    TORCH_COMPILE_DISABLE=1 python benchmarks/paper_accuracy_v2.py --exp fig3 --N 200 \
        --ops M_rpy M_2b_sw8 M_3b_sw8 M_mom_v3_nb8lin_tr2_pc8c_diag --workers 8 --skip-done

    python figures/fig3_phi_acc.py

Series (harness op -> label):
  M_rpy                          RPY                     pairwise RPY, all pairs
  M_2b_sw8                       NeMO 2-body             self + 2-body NN within 8, RPY beyond
  M_3b_sw8                       NeMO 3-body summations  ... + summed triplet corrections (triplets within 6)
  M_mom_v3_nb8lin_tr2_pc8c_diag  NeMO n-body             ... + moments v3 pair correction + learned diagonal (8)
Plotted value: rel_rmse (%) averaged over the 10 seeds per volume fraction (the statistic the original
Figure 3 plotted). Outputs: figures/fig3_phi_acc.{pdf,png}.
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
SERIES = [("M_rpy", "RPY", "#4C72B0"),  # seaborn "deep", as in the original figure
          ("M_2b_sw8", "NeMO 2-body", "#DD8452"),
          ("M_3b_sw8", "NeMO 3-body summations", "#55A868"),
          ("M_mom_v3_nb8lin_tr2_pc8c_diag", "NeMO n-body", "#C44E52")]
N_SEEDS = 10


def load(N: int) -> pd.DataFrame:
    df = pd.read_csv(MAIN_CSV)
    d = df[(df["exp"] == "fig3") & (df["N"] == N) & df["op"].isin([s[0] for s in SERIES])]
    g = d.groupby(["op", "phi"])["rel_rmse"]
    a = pd.DataFrame({"mean": g.mean(), "std": g.std(ddof=0), "n": g.count()}).reset_index()
    for op, _, _ in SERIES:
        assert (a["op"] == op).any(), f"no fig3 N={N} rows for {op!r} (run the harness command in the docstring)"
    assert (a["n"] == N_SEEDS).all(), f"incomplete cells:\n{a[a['n'] != N_SEEDS].to_string(index=False)}"
    return a


def plot(a: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    fig, ax = plt.subplots(figsize=(8, 5))
    for op, label, color in SERIES:
        r = a[a["op"] == op].sort_values("phi")
        ax.plot(r["phi"], r["mean"], color=color, lw=2, marker="o", ms=6, mec="white", mew=1.0, label=label)
    phis = sorted(a["phi"].unique())
    ax.set_xticks(phis)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{100 * x:g}%"))
    ax.set_xlabel("Volume fraction", fontsize=12)
    ax.set_ylabel("PRMSE (%)", fontsize=12)
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=10)
    ax.legend(title="Mobility operator", fontsize=10, title_fontsize=10, loc="upper left")
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--N", type=int, default=200)
    ap.add_argument("--out", type=Path, default=Path("figures/fig3_phi_acc"))
    args = ap.parse_args()

    a = load(args.N)
    labels = {op: label for op, label, _ in SERIES}
    p = a.pivot(index="phi", columns="op", values="mean")[[s[0] for s in SERIES]].rename(columns=labels)
    print(f"rel_rmse (%) vs phi at N={args.N}, mean over {N_SEEDS} seeds:")
    print(p.round(2).to_string())

    fig = plot(a)
    for ext in ("pdf", "png"):
        fig.savefig(args.out.with_suffix(f".{ext}"), dpi=600)
    print(f"-> {args.out}.pdf, {args.out}.png")


if __name__ == "__main__":
    main()
