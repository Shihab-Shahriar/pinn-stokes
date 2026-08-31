#!/usr/bin/env python3
"""Render the comparison tables (markdown) from experiments/runs/*/metrics.json and
data/nbody_moments_compare.csv.  Used to build artifacts/nbody_moments_report.md.

    python experiments/nbody_moments_summary.py [--runs experiments/runs] [--csv data/nbody_moments_compare.csv]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

COMPS = ["Ux", "Uy", "Uz", "Ox", "Oy", "Oz"]


def validation_table(runs_dir: Path) -> str:
    rows = []
    for d in sorted(runs_dir.iterdir()):
        mj = d / "metrics.json"
        if not mj.exists() or d.name.startswith(("smoke", "pilot")):
            continue
        m = json.load(open(mj))
        rows.append((d.name, m))
    if not rows:
        return "(no runs)"
    m0 = rows[0][1]["twobody_only"]
    out = ["| model | run | RMSE | PRMSE lin % | PRMSE ang % | " + " | ".join(f"{c} %" for c in COMPS) + " | note |",
           "|---|---|---:|---:|---:|" + "---:|" * 6 + "---|"]
    out.append(f"| 2-body only | – | {rows[0][1]['twobody_only']['rmse']:.5f} | {m0['prmse_lin']:.2f} | {m0['prmse_ang']:.2f} | "
               + " | ".join(f"{v:.2f}" for v in m0["prmse_comp"]) + " | no n-body term |")
    for name, m in rows:
        label = m.get("model", "eval") + (" (inv_norm)" if m.get("inv_norm") else "") + (" (zero-init)" if m.get("zero_init_head") and m.get("model") == "baseline" else "")
        note = m.get("note", "") or (f"seed {m['seed']}" if "seed" in m else "")
        if "model_path" in m:
            label = f"shipped {Path(m['model_path']).name}"
        out.append(f"| {label} | {name} | {m['rmse']:.5f} | {m['prmse_lin']:.2f} | {m['prmse_ang']:.2f} | "
                   + " | ".join(f"{v:.2f}" for v in m["prmse_comp"]) + f" | {note} |")
    return "\n".join(out)


def by_k_table(runs_dir: Path, names) -> str:
    ms = {}
    for n in names:
        mj = runs_dir / n / "metrics.json"
        if mj.exists():
            ms[n] = json.load(open(mj))["prmse_by_K"]
    if not ms:
        return ""
    ks = sorted({int(k) for m in ms.values() for k in m}, key=int)
    out = ["| K | n | 2-body only % | " + " | ".join(ms) + " |", "|---:|---:|---:|" + "---:|" * len(ms)]
    first = next(iter(ms.values()))
    for k in ks:
        out.append(f"| {k} | {first[str(k)]['n']} | {first[str(k)]['prmse_2b']:.2f} | "
                   + " | ".join(f"{m[str(k)]['prmse']:.2f}" for m in ms.values()) + " |")
    return "\n".join(out)


def operator_table(csv: Path) -> str:
    if not csv.exists():
        return "(no benchmark rows yet)"
    df = pd.read_csv(csv)
    out = []
    for N in sorted(df.N.unique()):
        sub = df[df.N == N]
        phis = sorted(sub.phi.unique())
        ops = list(dict.fromkeys(sub.op))
        for col, label in [("rel_rmse", "rel-RMSE % (mean ± std over seeds)"), ("prmse_lin", "translational rel-L2 %"),
                           ("prmse_ang", "rotational rel-L2 %"), ("max_rel_rmse", "max per-particle rel. error %"),
                           ("wall_s", "wall time per apply [s]")]:
            out.append(f"\n**N={N}: {label}**\n")
            out.append("| operator | " + " | ".join(f"φ={p:g}" for p in phis) + " |")
            out.append("|---|" + "---:|" * len(phis))
            for op in ops:
                cells = []
                for p in phis:
                    r = sub[(sub.op == op) & (sub.phi == p)][col]
                    if r.empty:
                        cells.append("–")
                    elif col == "rel_rmse":
                        cells.append(f"{r.mean():.2f} ± {r.std(ddof=0):.2f} (n={len(r)})")
                    else:
                        cells.append(f"{r.mean():.2f}")
                out.append(f"| {op} | " + " | ".join(cells) + " |")
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=Path("experiments/runs"))
    ap.add_argument("--csv", type=Path, default=Path("data/nbody_moments_compare.csv"))
    args = ap.parse_args()
    print("## Validation split (9,880 rows, identical for all models)\n")
    print(validation_table(args.runs))
    print("\n## PRMSE by neighbour count K (validation rows)\n")
    print(by_k_table(args.runs, ["b1_shipped", "baseline_s411", "moments_s411"]))
    print("\n## Grand mobility vs MFS truth (uniform suspensions)\n")
    print(operator_table(args.csv))
