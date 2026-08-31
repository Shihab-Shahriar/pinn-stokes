#!/usr/bin/env python3
"""Render the dataset-v2 validation tables (experiments/runs_v2/*/metrics.json) as markdown.

    python experiments/nbody_v2_summary.py [--runs experiments/runs_v2] [--out artifacts/nbody_v2_validation_tables.md]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

FAMILIES = ["uniform", "grown", "lattice"]


def load_runs(root: Path) -> dict:
    runs = {}
    for d in sorted(root.iterdir()):
        f = d / "metrics.json"
        if d.is_dir() and f.exists() and not d.name.startswith(("pilot", "smoke")):
            runs[d.name] = json.load(open(f))
    return runs


def label(name: str, m: dict) -> str:
    if "model_path" in m:
        return f"{Path(m['model_path']).name} (old data) @ {m['variant']}"
    return f"{m['model']} v2 @ {m['variant']}" + (f" [{m['loss']}/{m['loss_form']}, bw={m['block_weights']}]" if m.get("block_weights", "none") != "none" or m.get("loss_form", "block") != "block" else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=Path("experiments/runs_v2"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/nbody_v2_validation_tables.md"))
    args = ap.parse_args()
    runs = load_runs(args.runs)
    lines = ["# Dataset-v2 validation (configuration-level split, seed % 10 == 0; unordered near pairs with >= 1 neighbour)", ""]
    any_m = next(iter(runs.values()))
    tb = any_m["twobody_only"]
    lines.append(f"Validation rows: {any_m['n']} pairs.  2-body only: PRMSE lin {tb['prmse_lin']:.2f} % / ang {tb['prmse_ang']:.2f} %, "
                 f"block rel-Frobenius {any_m['rel_2b']:.2f} %.")
    lines.append("")
    lines.append("| run | model @ selection | PRMSE lin % | PRMSE ang % | block rel % | TT | TR | RR | residual capture % | uniform lin/ang | grown lin/ang | lattice lin/ang |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, m in runs.items():
        fam = " | ".join(f"{m['by_family'][f]['prmse_lin']:.2f}/{m['by_family'][f]['prmse_ang']:.2f}" if f in m.get("by_family", {}) else "-" for f in FAMILIES)
        b = m["blocks"]
        lines.append(f"| {name} | {label(name, m)} | {m['prmse_lin']:.2f} | {m['prmse_ang']:.2f} | {m['rel_total']:.2f} | {b['TT']['rel_total']:.2f} | "
                     f"{b['TR']['rel_total']:.2f} | {b['RR']['rel_total']:.2f} | {m['capture']:.1f} | {fam} |")
    lines.append("")
    lines.append("PRMSE lin/ang by neighbour count (K under the run's selection):")
    lines.append("")
    keys = sorted({k for m in runs.values() for k in m.get("by_nnbr", {})}, key=lambda s: int(s.split("-")[0]))
    lines.append("| run | " + " | ".join(keys) + " |")
    lines.append("|---|" + "---:|" * len(keys))
    for name, m in runs.items():
        cells = [f"{m['by_nnbr'][k]['prmse_lin']:.2f}/{m['by_nnbr'][k]['prmse_ang']:.2f} ({m['by_nnbr'][k]['n']})" if k in m.get("by_nnbr", {}) else "-" for k in keys]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("PRMSE lin/ang by pair distance:")
    lines.append("")
    dkeys = list(next(iter(runs.values()))["by_dist"].keys())
    lines.append("| run | " + " | ".join(dkeys) + " |")
    lines.append("|---|" + "---:|" * len(dkeys))
    for name, m in runs.items():
        lines.append(f"| {name} | " + " | ".join(f"{m['by_dist'][k]['prmse_lin']:.2f}/{m['by_dist'][k]['prmse_ang']:.2f}" if k in m["by_dist"] else "-" for k in dkeys) + " |")
    text = "\n".join(lines) + "\n"
    print(text)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
