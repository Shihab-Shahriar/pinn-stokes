#!/usr/bin/env python3
"""HIGNN baseline vs NeMO on the gravity-forcing (torque-free) Fig 3 / Fig 4 protocols.

Render-only from data/paper_accuracy_v2.csv, exp `fig4g` (uniform F = (0, 0, -9.81), T = 0, the same configs
and seeds as Fig 4). HIGNN is translational-only, so every panel uses a translational metric:
  prmse_lin    100 ||U_pred - U_true||_F / ||U_true||_F                   (headline)
  prmse_fluct  same on the fluctuations U - mean(U): the collective settling speed is removed from both
               numerator and denominator, so the near-field / disorder part of the error is what remains
Rows come from the harness:
  TORCH_COMPILE_DISABLE=1 python benchmarks/paper_accuracy_v2.py --exp fig4g --phis 0.025 0.05 0.1 0.15 \
      --ops HIGNN_2b HIGNN_full --gpu-ops --part --skip-done          # + the NeMO/RPY ops, then --merge
Outputs (per metric; `_fluct` suffix for prmse_fluct):
  figures/fig_hignn_compare_phi[_fluct].{pdf,png}   Fig-3 style: metric vs phi at N = 200 and N = 300
  figures/fig_hignn_compare_N[_fluct].{pdf,png}     Fig-4 style: metric vs N, one panel per phi
  figures/fig_hignn_compare.csv                     seed means / stds behind the plots (all ops, both metrics)
  artifacts/hignn_comparison_tables.md              markdown tables (Fig-3 style per N, Fig-4 style per op, ratios)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
from benchmarks.paper_accuracy_v2 import FIG4_REPEATS, NUM_REPEATS  # noqa: E402

MAIN_CSV = Path("data/paper_accuracy_v2.csv")
EXP = "fig4g"
PHIS = [0.025, 0.05, 0.1, 0.15]
HEADLINE_GPU, HEADLINE_CPU = "M_mom_gpu_pc8c_diag", "M_mom_v2_kinf_rc8_pc8c_diag"
NEMO = "NeMO"   # the GPU op spans the whole N range; the CPU op (N <= 300) fills cells the GPU rows lack
# fixed categorical assignment (dataviz reference palette, light mode); HIGNN dashed, NeMO/RPY solid
STYLE = {  # op: (label, colour, linestyle, marker, family)
    NEMO: ("NeMO (moments + learned diagonal)", "#2a78d6", "-", "o", "nemo"),
    "HIGNN_full": ("HIGNN: 2-body + 3-body + self", "#eb6834", "--", "s", "hignn"),
    "M_3b": ("NeMO 3-body summations", "#1baf7a", "-", "^", "nemo"),
    "M_2b": ("NeMO 2-body", "#eda100", "-", "D", "nemo"),
    "HIGNN_2b": ("HIGNN: 2-body (their engine's kernel)", "#e87ba4", "--", "v", "hignn"),
    "M_nbody_b1": ("NeMO n-body (paper, b1)", "#008300", "-", "P", "nemo"),
    "M_rpy": ("RPY", "#4a3aa7", "-", "x", "rpy"),
}
PHI_ORDER = list(STYLE)
METRIC_LABEL = {"prmse_lin": "translational PRMSE (%)", "prmse_fluct": "fluctuation PRMSE (%)",
                "err_mean_pct": "mean-velocity error (%)", "max_rel_lin": "max per-particle rel. error (%)"}


def load(metrics: list[str]) -> pd.DataFrame:
    df = pd.read_csv(MAIN_CSV)
    d = df[df["exp"] == EXP].copy()
    assert not d.empty, f"no {EXP} rows in {MAIN_CSV}"
    # headline series: GPU op rows, with CPU-op rows filling any (N, phi, seed) cell where the GPU op lacks the metric
    gpu = d[d["op"] == HEADLINE_GPU].copy()
    cpu = d[d["op"] == HEADLINE_CPU].copy()
    key = ["N", "phi", "seed"]
    parts = [gpu]
    for m in metrics:
        have = gpu[gpu[m].notna()][key] if m in gpu else gpu.iloc[0:0][key]
        fill = cpu.merge(have, on=key, how="left", indicator=True)
        fill = fill[fill["_merge"] == "left_only"].drop(columns="_merge")
        if len(fill):
            parts.append(fill)
    head = pd.concat(parts, ignore_index=True).drop_duplicates(subset=key, keep="first")
    head["op"] = NEMO
    d = pd.concat([d[~d["op"].isin([HEADLINE_GPU, HEADLINE_CPU])], head], ignore_index=True)
    return d


def agg(d: pd.DataFrame, metric: str) -> pd.DataFrame:
    g = d.groupby(["op", "N", "phi"])[metric]
    a = pd.DataFrame({"mean": g.mean(), "std": g.std(ddof=0), "n": g.count()}).reset_index()
    a = a[a["n"] > 0]
    a["expected"] = a["N"].map(lambda n: FIG4_REPEATS.get(int(n), NUM_REPEATS))
    short = a[a["n"] < a["expected"]]
    if len(short):
        print(f"[warn] {metric}: incomplete cells\n{short.to_string(index=False)}")
    return a


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 7.5,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e6e6e3", "grid.linewidth": 0.6, "axes.edgecolor": "#8a8a86",
                         "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.labelcolor": "#0b0b0b",
                         "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 300})
    return plt


def plot_vs_phi(a: pd.DataFrame, metric: str, Ns=(200, 300), out=Path("figures/fig_hignn_compare_phi")):
    plt = _style()
    fig, axes = plt.subplots(1, len(Ns), figsize=(3.6 * len(Ns) + 0.8, 4.3), sharey=True)
    axes = np.atleast_1d(axes)
    top = a[a["N"].isin(Ns)]
    ymax = float((top["mean"] + top["std"]).max()) * 1.08
    for ax, N in zip(axes, Ns):
        sub = a[a["N"] == N]
        for op in PHI_ORDER:
            r = sub[sub["op"] == op].sort_values("phi")
            if r.empty:
                continue
            label, c, ls, mk, fam = STYLE[op]
            ax.errorbar(r["phi"], r["mean"], yerr=r["std"], color=c, ls=ls, marker=mk, ms=4.5, lw=1.6 if fam != "rpy" else 1.2,
                        capsize=2, elinewidth=0.8, label=label, alpha=1.0 if fam != "rpy" else 0.85,
                        markerfacecolor="white" if fam == "hignn" else c, markeredgewidth=1.2)
        ax.set_title(f"N = {N}")
        ax.set_xlabel("volume fraction φ")
        ax.set_xticks(PHIS)
        ax.set_ylim(0, ymax)
    axes[0].set_ylabel(METRIC_LABEL.get(metric, metric))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Uniform gravity, torque-free spheres (10 configurations per point, ±1 std)", fontsize=9, color="#52514e")
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"))
    plt.close(fig)
    print(f"-> {out}.pdf/.png")


def plot_vs_N(a: pd.DataFrame, metric: str, phis=PHIS, out=Path("figures/fig_hignn_compare_N"), band=True):
    plt = _style()
    from matplotlib.ticker import NullFormatter
    from matplotlib.ticker import FuncFormatter
    fmt = FuncFormatter(lambda v, _: f"{v / 1000:g}k" if v >= 1000 else f"{v:g}")
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 7.0), sharex=False, sharey=False)
    for ax, phi in zip(axes.ravel(), phis):
        sub = a[np.isclose(a["phi"], phi)]
        for op in PHI_ORDER:
            r = sub[sub["op"] == op].sort_values("N")
            if r.empty:
                continue
            label, c, ls, mk, fam = STYLE[op]
            main = op in (NEMO, "HIGNN_full", "HIGNN_2b")
            ax.plot(r["N"], r["mean"], color=c, ls=ls, marker=mk, ms=4 if main else 3, lw=1.8 if main else 1.0,
                    alpha=1.0 if main else 0.75, label=label, markerfacecolor="white" if fam == "hignn" else c,
                    markeredgewidth=1.1)
            if band and main:
                ax.fill_between(r["N"], r["mean"] - r["std"], r["mean"] + r["std"], color=c, alpha=0.12, lw=0)
        ax.set_xscale("log")
        ticks = [n for n in (20, 50, 100, 200, 500, 1000, 3000, 10000, 30000) if sub["N"].min() <= n <= sub["N"].max()]
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(fmt)
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(f"φ = {phi:g}")
        ax.set_ylim(bottom=0)
    for ax in axes[1]:
        ax.set_xlabel("number of particles N")
    for ax in axes[:, 0]:
        ax.set_ylabel(METRIC_LABEL.get(metric, metric))
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Uniform gravity, torque-free spheres: error vs N (seed mean, ±1 std band on the main curves)",
                 fontsize=9, color="#52514e")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"))
    plt.close(fig)
    print(f"-> {out}.pdf/.png")


def tables(aggs: dict[str, pd.DataFrame], Ns=(200, 300)) -> str:
    lines = ["# HIGNN vs NeMO -- gravity-forcing protocol tables (generated by figures/fig_hignn_compare.py)", ""]
    for metric, a in aggs.items():
        lines.append(f"## {METRIC_LABEL.get(metric, metric)} -- mean over seeds (± std)")
        for N in Ns:
            sub = a[a["N"] == N]
            if sub.empty:
                continue
            lines += ["", f"**N = {N}** (Fig-3 style: vs volume fraction)", "",
                      "| operator | " + " | ".join(f"φ={p:g}" for p in PHIS) + " |", "|---|" + "---|" * len(PHIS)]
            for op in PHI_ORDER:
                r = sub[sub["op"] == op]
                if r.empty:
                    continue
                cells = []
                for p in PHIS:
                    q = r[np.isclose(r["phi"], p)]
                    cells.append("-" if q.empty else f"{q['mean'].iloc[0]:.2f} ± {q['std'].iloc[0]:.2f}")
                lines.append(f"| {STYLE[op][0]} | " + " | ".join(cells) + " |")
        lines += ["", "**vs N** (Fig-4 style; rows φ, columns N; seed means)"]
        for op in PHI_ORDER:
            r = a[a["op"] == op]
            if r.empty:
                continue
            Ns_op = sorted(r["N"].unique())
            lines += ["", f"*{STYLE[op][0]}*", "", "| φ | " + " | ".join(str(n) for n in Ns_op) + " |", "|---|" + "---|" * len(Ns_op)]
            for p in PHIS:
                cells = []
                for n in Ns_op:
                    q = r[(r["N"] == n) & np.isclose(r["phi"], p)]
                    cells.append("-" if q.empty else f"{q['mean'].iloc[0]:.2f}")
                lines.append(f"| {p:g} | " + " | ".join(cells) + " |")
        # ratios HIGNN / NeMO per phi: at N=200, N=300 and averaged over the N grid both share
        lines += ["", "**HIGNN error / NeMO error** (ratio of seed means; >1 = NeMO more accurate)", "",
                  "| φ | HIGNN full / NeMO @N=200 | HIGNN 2b / NeMO @N=200 | HIGNN full / NeMO @N=300 | HIGNN full / NeMO, mean over shared N |",
                  "|---|---|---|---|---|"]
        for p in PHIS:
            def cell(op, N):
                q1 = a[(a["op"] == op) & (a["N"] == N) & np.isclose(a["phi"], p)]
                q2 = a[(a["op"] == NEMO) & (a["N"] == N) & np.isclose(a["phi"], p)]
                return "-" if q1.empty or q2.empty else f"{q1['mean'].iloc[0] / q2['mean'].iloc[0]:.2f}"
            m = a[(a["op"].isin(["HIGNN_full", NEMO])) & np.isclose(a["phi"], p)].pivot(index="N", columns="op", values="mean")
            m = m.dropna() if {"HIGNN_full", NEMO} <= set(m.columns) else m.iloc[0:0]
            shared = "-" if m.empty else f"{(m['HIGNN_full'] / m[NEMO]).mean():.2f} (N = {int(m.index.min())}..{int(m.index.max())})"
            lines.append(f"| {p:g} | {cell('HIGNN_full', 200)} | {cell('HIGNN_2b', 200)} | {cell('HIGNN_full', 300)} | {shared} |")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics", nargs="+", default=["prmse_lin", "prmse_fluct"])
    ap.add_argument("--no-band", action="store_true")
    ap.add_argument("--tables", type=Path, default=Path("artifacts/hignn_comparison_tables.md"))
    args = ap.parse_args()
    d = load(args.metrics)
    aggs, frames = {}, []
    for m in args.metrics:
        if m not in d.columns or d[m].isna().all():
            print(f"[skip] no {m} rows yet")
            continue
        a = agg(d, m)
        aggs[m] = a
        frames.append(a.assign(metric=m))
        suffix = "" if m == "prmse_lin" else f"_{m.replace('prmse_', '')}"
        plot_vs_phi(a, m, out=Path(f"figures/fig_hignn_compare_phi{suffix}"))
        plot_vs_N(a, m, out=Path(f"figures/fig_hignn_compare_N{suffix}"), band=not args.no_band)
    pd.concat(frames, ignore_index=True).to_csv("figures/fig_hignn_compare.csv", index=False, float_format="%.5g")
    text = tables(aggs)
    args.tables.parent.mkdir(parents=True, exist_ok=True)
    args.tables.write_text(text + "\n")
    print(text)
    print(f"-> figures/fig_hignn_compare.csv, {args.tables}")


if __name__ == "__main__":
    main()
