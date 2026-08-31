#!/usr/bin/env python3
"""Pair-cutoff ablation report: moments v2 (all, r_c=8) at pair_cutoff 6 vs 8.

Reads the merged paper-protocol results (data/paper_accuracy_v2.csv), the exact-residual ceiling
decompositions (artifacts/nbody_v2_ceiling{,_pc8}.csv) and the two training runs' metrics, and writes
  artifacts/pair_cutoff_ablation.md                      tables (Fig 3, Fig 4, cluster, ceiling, validation)
  figures/pair_cutoff_ablation_fig3.{pdf,png}            PRMSE vs phi at N=200/300, pc6 vs pc8 + context + floors
  figures/pair_cutoff_ablation_fig4.{pdf,png}            PRMSE vs N per phi (pc6 dashed, pc8 solid) + improvement

    python benchmarks/pair_cutoff_ablation.py
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PC6, PC8 = "M_mom_v2_kinf_rc8", "M_mom_v2_kinf_rc8_pc8"
CTX = ["M_2b", "M_nbody_b1"]
LABELS = {"M_2b": "NeMO 2-body", "M_nbody_b1": "NeMO n-body (paper)",
          PC6: "moments v2 (pairs<=6)", PC8: "moments v2 (pairs<=8)"}
RUNS = {PC6: ROOT / "experiments/runs_v2/mom_kinf_rc8/metrics.json",
        PC8: ROOT / "experiments/runs_v2/mom_kinf_rc8_pc8/metrics.json"}
SIDES = {PC6: ROOT / "data/models/nbody_moments_v2_kinf_rc8.json",
         PC8: ROOT / "data/models/nbody_moments_v2_kinf_rc8_pc8.json"}


def seed_mean(df: pd.DataFrame, exp: str, metric: str) -> pd.DataFrame:
    """(N, phi, op) -> mean metric over seeds (asserting the full 10-seed protocol for fig3/fig4)."""
    d = df[df["exp"] == exp]
    g = d.groupby(["N", "phi", "op"])[metric].agg(["mean", "count"]).reset_index()
    if exp in ("fig3", "fig4"):
        bad = g[g["count"] != 10]
        assert bad.empty, f"{exp}: incomplete seed coverage\n{bad}"
    return g.pivot_table(index=["N", "phi"], columns="op", values="mean")


def md_fig3_table(piv: pd.DataFrame, N: int) -> list[str]:
    md = [f"| phi | {' | '.join(LABELS[o] for o in CTX)} | pc6 | pc8 | delta | improvement |",
          "|---|" + "---:|" * (len(CTX) + 4)]
    for (n, phi), row in piv.iterrows():
        if n != N:
            continue
        d = row[PC6] - row[PC8]
        md.append(f"| {phi:g} | " + " | ".join(f"{row[o]:.2f}" for o in CTX)
                  + f" | {row[PC6]:.2f} | {row[PC8]:.2f} | {d:+.2f} | {100 * d / row[PC6]:.0f} % |")
    return md


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=ROOT / "data/paper_accuracy_v2.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts/pair_cutoff_ablation.md")
    ap.add_argument("--fig-dir", type=Path, default=ROOT / "figures")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    have = set(df["op"])
    assert PC6 in have and PC8 in have, f"need both {PC6} and {PC8} in {args.csv} (have {sorted(have)})"
    f3 = seed_mean(df, "fig3", "rel_rmse")
    f3max = seed_mean(df, "fig3", "max_rel_rmse")
    f4 = seed_mean(df, "fig4", "rel_rmse")
    phis4 = sorted({p for _, p in f4.index})

    ceil = {}
    for tag, p in [("pc6", ROOT / "artifacts/nbody_v2_ceiling.csv"), ("pc8", ROOT / "artifacts/nbody_v2_ceiling_pc8.csv")]:
        c = pd.read_csv(p)
        ceil[tag] = c[c["family"] == "uniform"].groupby("param")[["e_2b", "e_near"]].mean()
    floor = pd.DataFrame({"e_2b": ceil["pc6"]["e_2b"], "floor_pc6": ceil["pc6"]["e_near"],
                          "floor_pc8": ceil["pc8"]["e_near"]})
    metrics = {k: json.load(open(v)) for k, v in RUNS.items()}
    sides = {k: json.load(open(v)) for k, v in SIDES.items()}

    md = ["# Pair-cutoff ablation: moments v2 (all, r_c = 8), pair_cutoff 6 -> 8", ""]
    md += [f"Generated {time.strftime('%Y-%m-%d %H:%M')} from `{args.csv.name}` "
           f"({len(df[df['op'] == PC8])} pc8 rows) by `benchmarks/pair_cutoff_ablation.py`.",
           "",
           f"Both operators share everything except the corrected-pair range: `{PC6}` corrects pairs with "
           "d <= 6 (2b NN base to 6, RPY beyond), `" + PC8 + "` corrects d <= 8 and runs with `switch_dist=8` "
           "(the 2b NN, trained to d = 8, is the base wherever pairs are corrected; NN-vs-RPY base difference "
           "in the 6-8 shell is 2e-4 median, i.e. negligible). Both trained with the identical recipe on "
           "dataset v2; labels R = 0.5(M_ts + M_st^T) - M2b. Metric: PRMSE % (`rel_rmse`), mean over 10 seeds; "
           "MFS Xfine truths (cached, identical for every operator).", ""]

    for N in (200, 300):
        md += [f"## Figure 3 protocol, N = {N} (PRMSE %, mean over 10 seeds)", ""]
        md += md_fig3_table(f3, N) + [""]
    md += ["## Figure 3, N = 200: max per-particle error (max_rel_rmse %, mean over seeds)", "",
           "| phi | pc6 | pc8 | improvement |", "|---|---:|---:|---:|"]
    for (n, phi), row in f3max.iterrows():
        if n == 200:
            md.append(f"| {phi:g} | {row[PC6]:.1f} | {row[PC8]:.1f} | {100 * (row[PC6] - row[PC8]) / row[PC6]:.0f} % |")
    md += [""]

    md += ["## Figure 4 protocol (PRMSE %, mean over 10 seeds)", "",
           "Mean over the 14 sizes N = 20..200 per phi:", "",
           "| phi | pc6 | pc8 | improvement |", "|---|---:|---:|---:|"]
    for phi in phis4:
        s6 = np.mean([f4.loc[(n, phi), PC6] for n in sorted({n for n, p in f4.index if p == phi})])
        s8 = np.mean([f4.loc[(n, phi), PC8] for n in sorted({n for n, p in f4.index if p == phi})])
        md.append(f"| {phi:g} | {s6:.2f} | {s8:.2f} | {100 * (s6 - s8) / s6:.0f} % |")
    for phi in (0.1, 0.2):
        md += ["", f"Per size at phi = {phi:g}:", "", "| N | pc6 | pc8 | improvement |", "|---|---:|---:|---:|"]
        for n in sorted({n for n, p in f4.index if p == phi}):
            r6, r8 = f4.loc[(n, phi), PC6], f4.loc[(n, phi), PC8]
            md.append(f"| {n} | {r6:.2f} | {r8:.2f} | {100 * (r6 - r8) / r6:.0f} % |")
    md += [""]

    dc = df[df["exp"] == "cluster"]
    if PC8 in set(dc["op"]):
        md += ["## Clustered near-contact protocol (N = 10, PRMSE %)", "",
               "| delta | " + " | ".join(LABELS[o] for o in ["M_nbody_b1", PC6, PC8]) + " |", "|---|---:|---:|---:|"]
        pc = dc.pivot_table(index="phi", columns="op", values="rel_rmse")
        for delta, row in pc.iterrows():
            md.append(f"| {delta:g} | {row['M_nbody_b1']:.2f} | {row[PC6]:.2f} | {row[PC8]:.2f} |")
        md += [""]

    md += ["## Exact-residual floor (uniform v2 boxes, P <= 64): what a *perfect* pairwise correction leaves", "",
           "e_near from `experiments/nbody_v2_ceiling.py` on the two caches -- the model-independent floor of the",
           "design. Raising the cutoff moves pairs from the never-corrected far field into the corrected set:", "",
           "| phi (uniform) | e_2b | floor, pairs<=6 | floor, pairs<=8 |", "|---|---:|---:|---:|"]
    for phi, row in floor.iterrows():
        md.append(f"| {phi:g} | {row['e_2b']:.2f} | {row['floor_pc6']:.2f} | {row['floor_pc8']:.2f} |")
    md += ["", "Caveat: v2 boxes are small (half-box ~5.5-9), so the d > 8 far field is a smaller share there than",
           "at N = 200/300; the Fig-4 N-sweep above is the operative measurement of the remaining far-field term.", ""]

    m6, m8 = metrics[PC6], metrics[PC8]
    md += ["## Dataset-v2 validation (PRMSE % lin / ang)", "",
           "| model | prmse_lin | prmse_ang | residual capture % |", "|---|---:|---:|---:|",
           f"| 2-body only (pc6 rows) | {m6['twobody_only']['prmse_lin']:.2f} | {m6['twobody_only']['prmse_ang']:.2f} | -- |",
           f"| pc6 (`{sides[PC6]['name']}`) | {m6['prmse_lin']:.2f} | {m6['prmse_ang']:.2f} | {m6['capture']:.1f} |",
           f"| 2-body only (pc8 rows) | {m8['twobody_only']['prmse_lin']:.2f} | {m8['twobody_only']['prmse_ang']:.2f} | -- |",
           f"| pc8 (`{sides[PC8]['name']}`) | {m8['prmse_lin']:.2f} | {m8['prmse_ang']:.2f} | {m8['capture']:.1f} |", "",
           "These validation sets differ (the pc8 rows include the easier 6-8 shell), so pc6-vs-pc8 is NOT",
           "apples-to-apples here -- the paper protocols above, on identical truths, are the comparison.", ""]

    md += ["## Models", ""]
    for k in (PC6, PC8):
        s = sides[k]
        md.append(f"- `{s['name']}.pt`: max_neighbors={s['max_neighbors']}, neighbor_cutoff={s['neighbor_cutoff']}, "
                  f"pair_cutoff={s.get('pair_cutoff', 6.0)} (run with switch_dist=max(6, pair_cutoff)); "
                  f"trained {s['created']}.")
    args.out.write_text("\n".join(md) + "\n")
    print(f"-> {args.out}")

    # ---------------------------------------------------------------- figures
    plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, N in zip(axes, (200, 300)):
        sub = f3.xs(N, level="N")
        for o, c, ls, mk in [("M_2b", "0.55", "--", ""), ("M_nbody_b1", "0.35", "-", "^")]:
            ax.plot(sub.index, sub[o], ls, color=c, marker=mk, ms=4, label=LABELS[o])
        ax.plot(sub.index, sub[PC6], "o--", color="C0", label=LABELS[PC6])
        ax.plot(sub.index, sub[PC8], "s-", color="C3", label=LABELS[PC8])
        ax.plot(floor.index, floor["floor_pc6"], ":", color="C0", alpha=0.7, lw=1.5)
        ax.plot(floor.index, floor["floor_pc8"], ":", color="C3", alpha=0.7, lw=1.5,
                label="exact-residual floors (v2 boxes)")
        ax.set_xlabel("volume fraction $\\phi$"); ax.set_title(f"N = {N}")
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("PRMSE (%)")
    axes[0].legend(fontsize=9, loc="upper left")
    fig.suptitle("Figure 3 protocol: pair_cutoff 6 vs 8 (moments v2, all neighbours, $r_c$ = 8)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(args.fig_dir / f"pair_cutoff_ablation_fig3.{ext}", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    cmap = plt.get_cmap("viridis")
    for i, phi in enumerate(phis4):
        col = cmap(i / max(1, len(phis4) - 1))
        ns = sorted({n for n, p in f4.index if p == phi})
        r6 = [f4.loc[(n, phi), PC6] for n in ns]; r8 = [f4.loc[(n, phi), PC8] for n in ns]
        axes[0].plot(ns, r6, "--", color=col, alpha=0.8)
        axes[0].plot(ns, r8, "-", color=col, label=f"$\\phi$={phi:g}")
        axes[1].plot(ns, [100 * (a - b) / a for a, b in zip(r6, r8)], "-o", ms=3, color=col)
    axes[0].set_xlabel("N"); axes[0].set_ylabel("PRMSE (%)"); axes[0].set_ylim(bottom=0)
    axes[0].set_title("Figure 4 protocol: pairs<=6 (dashed) vs pairs<=8 (solid)")
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].set_xlabel("N"); axes[1].set_ylabel("improvement of pc8 over pc6 (%)")
    axes[1].set_title("relative improvement: steady 15-20 % beyond N \u2248 30")
    axes[1].axhline(0, color="k", lw=0.8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(args.fig_dir / f"pair_cutoff_ablation_fig4.{ext}", dpi=180)
    plt.close(fig)
    print(f"-> {args.fig_dir}/pair_cutoff_ablation_fig3/.fig4 (pdf+png)")

    # headline for the terminal
    for N in (200, 300):
        r = f3.loc[(N, 0.2)]
        print(f"fig3 N={N} phi=0.2: pc6 {r[PC6]:.2f} -> pc8 {r[PC8]:.2f} PRMSE% "
              f"({100 * (r[PC6] - r[PC8]) / r[PC6]:.0f} % better)")


if __name__ == "__main__":
    main()
