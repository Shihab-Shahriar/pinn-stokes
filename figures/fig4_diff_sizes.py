#!/usr/bin/env python3
"""Figure 4: grand mobility accuracy vs number of particles, for the latest published stack
(moments pc8c pair model + learned diagonal, switch_dist = pair_cutoff = 8).

Render-only: the rows come from the paper-accuracy harness. If the op is missing from the CSV:

    TORCH_COMPILE_DISABLE=1 python benchmarks/paper_accuracy_v2.py --exp fig4 \
        --ops M_mom_v2_kinf_rc8_pc8c M_mom_v2_kinf_rc8_pc8c_diag --workers 8 --skip-done

    python figures/fig4_diff_sizes.py

Outputs: figures/fig4_diff_sizes.{pdf,png} and drop-in copies at the paper's include name
figures/M_accuracy_nbody_diff_sizes_avg_rel_rmse.{pdf,png}.  Layout follows the paper's original
(figures/plot_M_accuracy.py): seed-mean rel. RMSE vs N, one curve per volume fraction, warm
colorblind palette; shaded band = ±1 std over the 10 seeds.
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
PHIS = [0.025, 0.05, 0.1, 0.15]
WARM = {0.025: "#CC79A7", 0.05: "#E69F00", 0.1: "#D55E00", 0.15: "#F0E442", 0.2: "#8C510A"}  # the paper's palette


def load(op: str, max_n: int, exp: str = "fig4") -> pd.DataFrame:
    df = pd.read_csv(MAIN_CSV)
    d = df[(df["exp"] == exp) & (df["op"] == op) & (df["N"] <= max_n)]
    assert not d.empty, f"no {exp} rows for op {op!r} in {MAIN_CSV} (run the harness command in the docstring)"
    g = d.groupby(["N", "phi"])["rel_rmse"]
    a = pd.DataFrame({"mean": g.mean(), "std": g.std(ddof=0), "n": g.count()}).reset_index()
    try:  # the large-N cells run tapered seed counts; compare against the harness' schedule
        import sys
        sys.path.insert(0, str(ROOT))
        from benchmarks.paper_accuracy_v2 import FIG4_REPEATS, NUM_REPEATS
        expected = a["N"].map(lambda n: FIG4_REPEATS.get(int(n), NUM_REPEATS))
    except ImportError:
        expected = a["n"].max()
    short = a[a["n"] < expected]
    if len(short):
        print(f"[warn] incomplete cells for {op}:\n{short.to_string(index=False)}")
    return a


def plot(a: pd.DataFrame, phis: list[float], band: bool, log_x: bool = False, title_note: str = ""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    for phi in phis:
        c = WARM[phi]
        r = a[np.isclose(a["phi"], phi)].sort_values("N")
        ax.plot(r["N"], r["mean"], marker="o", color=c, lw=2, label=f"{phi:g}")
        if band:
            ax.fill_between(r["N"], r["mean"] - r["std"], r["mean"] + r["std"], color=c, alpha=0.15, lw=0)
    if log_x:
        from matplotlib.ticker import NullFormatter, ScalarFormatter
        ax.set_xscale("log")
        big = (2000, 3000) if a["N"].max() <= 3000 else (2000, 5000, 10000)
        ticks = [n for n in (20, 50, 100, 200, 500, 1000, *big) if a["N"].min() <= n <= a["N"].max()]
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("N")
    ax.set_ylabel("Relative RMSE Error (%)")
    ax.set_title("Nemo Accuracy vs Particle Count" + title_note)
    ax.set_ylim(bottom=0)
    ax.legend(title="Volume fraction")
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--op", default="M_mom_v2_kinf_rc8_pc8c_diag", help="harness op to plot")
    ap.add_argument("--phis", type=float, nargs="+", default=PHIS)
    ap.add_argument("--no-band", action="store_true", help="drop the ±1 std seed band")
    ap.add_argument("--out", type=Path, default=Path("figures/fig4_diff_sizes"))
    ap.add_argument("--max-n", type=int, default=200,
                    help="plot cells with N <= this (default 200 = the paper's original range; "
                         "use 2000 with --out-suffix _large for the large-N figure)")
    ap.add_argument("--out-suffix", default="",
                    help="suffix on both output stems (e.g. _large) so the defaults stay untouched")
    ap.add_argument("--exp", default="fig4", choices=["fig4", "fig4g"],
                    help="fig4g = the uniform-gravity-forcing protocol rows")
    args = ap.parse_args()

    a = load(args.op, args.max_n, args.exp)
    print("rel. RMSE (%) vs N, mean over seeds:")
    p = a[a["phi"].isin(args.phis)].pivot(index="phi", columns="N", values="mean")
    print(p.round(2).to_string())

    fig = plot(a, args.phis, band=not args.no_band, log_x=args.max_n > 200,
               title_note=" (gravity forcing)" if args.exp == "fig4g" else "")
    for stem in (args.out, Path("figures/M_accuracy_nbody_diff_sizes_avg_rel_rmse")):
        stem = stem.with_name(stem.name + args.out_suffix)
        for ext in ("pdf", "png"):
            fig.savefig(stem.with_suffix(f".{ext}"), dpi=600)
        print(f"-> {stem}.pdf, {stem}.png")


if __name__ == "__main__":
    main()
