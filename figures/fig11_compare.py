"""Render Figure 11 (both panels) from the H200 rerun CSV and diff it against
the current numbers in data/fig11_breakdown_h200.csv (backend=widebvh).

Usage: python fig11_compare.py <repo_root> <new_csv>
Outputs figures/runtime_summary_two_panel_rerun.{png,pdf} (new run) and
figures/runtime_summary_two_panel_ac78eb4.{png,pdf} (current CSV), plus a
component-level comparison table on stdout.
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

ROOT = Path(sys.argv[1]).resolve()
NEW_CSV = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(ROOT))

import figures.grand_M_perf as perf  # noqa: E402

OPS = ("FMM_2body_RPY", "FMM_2body_NN", "FMM_Nbody_NN")
SIZES = (10_000, 50_000, 100_000, 200_000, 1_000_000)
COMPONENTS = ("far_ms", "self2b_ms", "nbody_ms", "nsearch_ms", "total_gpu_ms")


def load(path):
    with open(path) as fh:
        rows = [r for r in csv.DictReader(fh) if r["backend"] == "widebvh"]
    return {(int(r["n"]), r["operator"]): r for r in rows}


def render(csv_path, suffix):
    import os
    perf.FIG11_CSV = Path(csv_path)
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        perf.runtime_breakdown("widebvh")
        for ext in ("png", "pdf"):
            src = ROOT / f"figures/runtime_summary_two_panel.{ext}"
            dst = ROOT / f"figures/runtime_summary_two_panel_{suffix}.{ext}"
            src.rename(dst)
            print(f"  -> {dst.relative_to(ROOT)}")
    finally:
        os.chdir(cwd)


old = load(ROOT / "data" / "fig11_breakdown_h200.csv")
new = load(NEW_CSV)

print("=" * 100)
print("panel (a): total wall time per operator, old (ac78eb4) vs new")
print(f"{'N':>10} {'operator':16} {'old ms':>9} {'new ms':>9} {'delta':>8}")
for n in SIZES:
    for op in OPS:
        if (n, op) not in new:
            continue
        o, w = float(old[(n, op)]["total_gpu_ms"]), float(new[(n, op)]["total_gpu_ms"])
        print(f"{n:>10,} {op:16} {o:9.2f} {w:9.2f} {(w - o) / o:+8.1%}")

print()
print("panel (b): full NeMO (FMM_Nbody_NN) component breakdown, old vs new")
hdr = f"{'N':>10}"
for c in COMPONENTS:
    hdr += f" | {c.replace('_ms',''):>21}"
print(hdr + "\n" + " " * 10 + " |     old     new    d% " * len(COMPONENTS))
for n in SIZES:
    if (n, "FMM_Nbody_NN") not in new:
        continue
    line = f"{n:>10,}"
    for c in COMPONENTS:
        o = float(old[(n, "FMM_Nbody_NN")][c])
        w = float(new[(n, "FMM_Nbody_NN")][c])
        pct = (w - o) / o * 100 if o else float("nan")
        line += f" | {o:7.2f} {w:7.2f} {pct:+5.0f}"
    print(line)

print()
print("peak VRAM (PyTorch allocator), full NeMO:")
for n in SIZES:
    if (n, "FMM_Nbody_NN") not in new:
        continue
    o = old[(n, "FMM_Nbody_NN")].get("peak_vram_gb", "")
    w = new[(n, "FMM_Nbody_NN")].get("peak_vram_gb", "")
    print(f"{n:>10,}  old {float(o):6.2f} GiB   new {float(w):6.2f} GiB")

print("\nrendering new-run figure:")
render(NEW_CSV, "rerun")
print("rendering current-CSV figure:")
render(ROOT / "data" / "fig11_breakdown_h200.csv", "ac78eb4")
