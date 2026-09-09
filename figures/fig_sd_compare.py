#!/usr/bin/env python3
"""Stokesian Dynamics vs NeMO on the Fig 3 protocol (N = 200, 10 configurations per volume fraction, MFS truth).

Render-only from data/paper_accuracy_v2.csv. Two forcings, the same Fig 4 configurations at N = 200:
  exp fig4g  uniform gravity F = (0, 0, -9.81), T = 0        -> figures/fig_sd_compare_phi.*  (main, Fig-3 layout)
                                                               figures/fig_sd_compare_metrics.*  (fluctuation / rotational / max)
  exp fig4   random unit force + unit torque (the paper's Fig 3 wrench) -> figures/fig_sd_compare_random.*
Operators (harness names): SD = Townsend's Stokesian Dynamics (FTS far field + pairwise lubrication, src/sd_ops.py),
SD_Minf = its far-field part alone (many-body FTS multipole inversion, no lubrication correction),
NeMO = M_mom_v2_kinf_rc8_pc8c_diag (moments + learned diagonal), M_rpy = the shared pairwise far field (--with-rpy).
Metrics (benchmarks/compare_nbody_moments.compute_error_stats), all in %:
  prmse_lin    100 ||U_pred - U_true||_F / ||U_true||_F         translational PRMSE (gravity headline)
  prmse_fluct  the same on U - mean(U): the collective settling speed removed from both sides
  prmse_ang    rotational rel-L2; max_rel_lin  worst particle's translational relative error
  rel_rmse     the paper's PRMSE over all six components (random-wrench headline)
Rows come from the harness (all CPU, in-process):
  TORCH_COMPILE_DISABLE=1 python benchmarks/sd_gravity_truth.py           # gravity truths for the 4 missing phi
  TORCH_COMPILE_DISABLE=1 python benchmarks/paper_accuracy_v2.py --exp fig4g --N 200 --ops SD SD_Minf M_rpy M_mom_v2_kinf_rc8_pc8c_diag --skip-done
  TORCH_COMPILE_DISABLE=1 python benchmarks/paper_accuracy_v2.py --exp fig4  --N 200 --ops SD SD_Minf --skip-done
Also writes figures/fig_sd_compare.csv (seed means / stds) and artifacts/sd_comparison_tables.md.
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
from benchmarks.paper_accuracy_v2 import NUM_REPEATS, PHIS  # noqa: E402

MAIN_CSV = Path("data/paper_accuracy_v2.csv")
N_PARTICLES = 200
NEMO = "M_mom_v2_kinf_rc8_pc8c_diag"
NEMO_FTS = "M_mom_v2_kinf_rc8_pc8c_fts_diag"   # + analytic stresslet single reflection, models retrained on that base
# fixed categorical assignment (dataviz reference palette, light mode; validated): NeMO solid, SD dashed with
# hollow markers (third-party family), RPY thin. Line style + marker carry identity alongside colour.
STYLE = {  # op: (label, colour, linestyle, marker, family)
    NEMO: ("NeMO (moments + learned diagonal)", "#2a78d6", "-", "o", "nemo"),
    NEMO_FTS: ("NeMO + FTS reflection", "#0b3d91", "-", "D", "nemo"),
    "SD": ("Stokesian Dynamics (far field + lubrication)", "#eb6834", "--", "s", "sd"),
    "SD_Minf": ("Stokesian Dynamics, far field only (FTS)", "#1baf7a", "--", "^", "sd"),
    "M_rpy": ("RPY", "#4a3aa7", ":", "x", "rpy"),
}
ORDER = list(STYLE)
METRIC_LABEL = {"prmse_lin": "translational PRMSE (%)", "prmse_fluct": "fluctuation PRMSE (%)",
                "prmse_ang": "rotational PRMSE (%)", "max_rel_lin": "max per-particle rel. error (%)",
                "rel_rmse": "PRMSE (%)", "max_rel_rmse": "max per-particle rel. error (%)",
                "err_mean_pct": "mean-velocity error (%)"}
FORCING_TITLE = {"fig4g": "uniform gravity, torque-free spheres", "fig4": "random unit force and torque per sphere"}


def load() -> pd.DataFrame:
    df = pd.read_csv(MAIN_CSV)
    d = df[(df["N"] == N_PARTICLES) & df["exp"].isin(["fig4g", "fig4"]) & df["op"].isin(ORDER)].copy()
    assert not d.empty, f"no N={N_PARTICLES} fig4g/fig4 rows for {ORDER} in {MAIN_CSV}"
    return d


def agg(d: pd.DataFrame, metric: str) -> pd.DataFrame:
    g = d.groupby(["exp", "op", "phi"])[metric]
    a = pd.DataFrame({"mean": g.mean(), "std": g.std(ddof=0), "n": g.count()}).reset_index()
    a = a[a["n"] > 0]
    short = a[a["n"] < NUM_REPEATS]
    if len(short):
        print(f"[warn] {metric}: incomplete cells\n{short.to_string(index=False)}")
    return a


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9.5, "legend.fontsize": 8,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e6e6e3", "grid.linewidth": 0.6, "axes.edgecolor": "#8a8a86",
                         "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.labelcolor": "#0b0b0b",
                         "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 300})
    return plt


def _draw(ax, sub: pd.DataFrame, ops, direct_labels=True, ticks=PHIS):
    """One errorbar curve per op (mean ± 1 std over seeds) with a direct label at the right end."""
    ends = []
    for op in ops:
        r = sub[sub["op"] == op].sort_values("phi")
        if r.empty:
            continue
        label, c, ls, mk, fam = STYLE[op]
        main = fam != "rpy"
        ax.errorbar(r["phi"], r["mean"], yerr=r["std"], color=c, ls=ls, marker=mk, ms=5.5 if main else 5,
                    lw=1.6 if main else 1.1, capsize=2, elinewidth=0.8, label=label, alpha=1.0 if main else 0.85,
                    markerfacecolor="white" if fam == "sd" else c, markeredgewidth=1.2, zorder=3 if main else 2)
        ends.append((float(r["mean"].iloc[-1]), float(r["phi"].iloc[-1]), op, c))
    if direct_labels and ends:
        short = {NEMO: "NeMO", NEMO_FTS: "NeMO + FTS", "SD": "SD", "SD_Minf": "SD far field", "M_rpy": "RPY"}
        ymax = max(e[0] for e in ends)
        placed = []
        for y, x, op, c in sorted(ends):  # nudge apart when two ends collide
            yy = y
            while any(abs(yy - p) < 0.045 * ymax for p in placed):
                yy += 0.045 * ymax
            placed.append(yy)
            ax.annotate(short[op], (x, y), xytext=(6, 0), textcoords="offset points", ha="left", va="center",
                        fontsize=8, color="#52514e", annotation_clip=False)
    ax.set_xticks(ticks)
    ax.set_xlim(PHIS[0] - 0.01, PHIS[-1] + 0.025)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("volume fraction φ")


def plot_main(a: pd.DataFrame, metric: str, exp: str, out: Path, ops, title: str):
    plt = _style()
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    _draw(ax, a[a["exp"] == exp], ops)
    ax.set_ylabel(METRIC_LABEL.get(metric, metric))
    ax.set_title(f"N = {N_PARTICLES}, {FORCING_TITLE[exp]} ({NUM_REPEATS} configurations per point, ±1 std)",
                 fontsize=9, color="#52514e", loc="left")
    ax.legend(title="Mobility operator", loc="upper left", frameon=False, title_fontsize=8.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"))
    plt.close(fig)
    print(f"-> {out}.pdf/.png  ({title})")


def plot_metrics(aggs: dict[str, pd.DataFrame], exp: str, metrics, out: Path, ops):
    plt = _style()
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.4 * len(metrics) + 0.6, 3.9))
    for ax, m in zip(np.atleast_1d(axes), metrics):
        _draw(ax, aggs[m][aggs[m]["exp"] == exp], ops, direct_labels=False, ticks=[0.05, 0.1, 0.15, 0.2])
        ax.set_ylabel(METRIC_LABEL.get(m, m))
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"N = {N_PARTICLES}, {FORCING_TITLE[exp]} ({NUM_REPEATS} configurations per point, ±1 std)",
                 fontsize=9, color="#52514e")
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"))
    plt.close(fig)
    print(f"-> {out}.pdf/.png")


def tables(aggs: dict[str, pd.DataFrame], ops) -> str:
    lines = ["# Stokesian Dynamics vs NeMO -- Fig 3 protocol tables, N = 200 (generated by figures/fig_sd_compare.py)", ""]
    for exp, mets in [("fig4g", ["prmse_lin", "prmse_fluct", "prmse_ang", "max_rel_lin", "err_mean_pct"]),
                      ("fig4", ["rel_rmse", "prmse_lin", "prmse_ang", "max_rel_rmse"])]:
        lines += [f"## {FORCING_TITLE[exp]} (exp `{exp}`)", ""]
        for m in mets:
            if m not in aggs:
                continue
            a = aggs[m][aggs[m]["exp"] == exp]
            if a.empty:
                continue
            lines += [f"**{METRIC_LABEL.get(m, m)}** -- mean over seeds (± std; n in brackets if < {NUM_REPEATS})", "",
                      "| operator | " + " | ".join(f"φ={p:g}" for p in PHIS) + " |", "|---|" + "---|" * len(PHIS)]
            for op in ops:
                r = a[a["op"] == op]
                if r.empty:
                    continue
                cells = []
                for p in PHIS:
                    q = r[np.isclose(r["phi"], p)]
                    if q.empty:
                        cells.append("-")
                    else:
                        n = int(q["n"].iloc[0])
                        cells.append(f"{q['mean'].iloc[0]:.2f} ± {q['std'].iloc[0]:.2f}" + (f" ({n})" if n != NUM_REPEATS else ""))
                lines.append(f"| {STYLE[op][0]} | " + " | ".join(cells) + " |")
            lines.append("")
        # ratios vs NeMO on the headline metric
        head = "prmse_lin" if exp == "fig4g" else "rel_rmse"
        if head in aggs:
            a = aggs[head][aggs[head]["exp"] == exp]
            lines += [f"**error ratio vs NeMO** ({METRIC_LABEL[head]}; > 1 = NeMO more accurate)", "",
                      "| φ | " + " | ".join(f"{STYLE[op][0].split(' (')[0]} / NeMO" for op in ops if op != NEMO) + " |",
                      "|---|" + "---|" * (len(ops) - 1)]
            for p in PHIS:
                cells = []
                q0 = a[(a["op"] == NEMO) & np.isclose(a["phi"], p)]
                for op in ops:
                    if op == NEMO:
                        continue
                    q = a[(a["op"] == op) & np.isclose(a["phi"], p)]
                    cells.append("-" if q.empty or q0.empty else f"{q['mean'].iloc[0] / q0['mean'].iloc[0]:.2f}")
                lines.append(f"| {p:g} | " + " | ".join(cells) + " |")
            lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metric", default="prmse_lin", help="y metric of the main gravity figure")
    ap.add_argument("--with-rpy", action="store_true", help="add the RPY curve to the figures")
    ap.add_argument("--tables", type=Path, default=Path("artifacts/sd_comparison_tables.md"))
    args = ap.parse_args()
    ops = [o for o in ORDER if args.with_rpy or o != "M_rpy"]
    d = load()
    metrics = ["prmse_lin", "prmse_fluct", "prmse_ang", "max_rel_lin", "err_mean_pct", "rel_rmse", "max_rel_rmse"]
    aggs = {m: agg(d, m) for m in metrics if m in d.columns}
    frames = [a.assign(metric=m) for m, a in aggs.items()]
    pd.concat(frames, ignore_index=True).to_csv("figures/fig_sd_compare.csv", index=False, float_format="%.5g")

    if (d["exp"] == "fig4g").any():
        plot_main(aggs[args.metric], args.metric, "fig4g", Path("figures/fig_sd_compare_phi"), ops, "gravity, main")
        plot_metrics(aggs, "fig4g", ["prmse_fluct", "prmse_ang", "max_rel_lin"], Path("figures/fig_sd_compare_metrics"), ops)
    if (d["exp"] == "fig4").any():
        plot_main(aggs["rel_rmse"], "rel_rmse", "fig4", Path("figures/fig_sd_compare_random"), ops, "random wrench")
    text = tables(aggs, [o for o in ORDER])  # tables always carry RPY
    args.tables.parent.mkdir(parents=True, exist_ok=True)
    args.tables.write_text(text + "\n")
    print(text)
    print(f"-> figures/fig_sd_compare.csv, {args.tables}")


if __name__ == "__main__":
    main()
