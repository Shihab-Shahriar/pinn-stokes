"""Render Figure 12 from the H200 rerun CSV and diff it against the current
numbers in data/fig12_scaling_h200.csv (backend=widebvh).

Usage: python fig12_compare.py <repo_root> <new_csv>
Outputs figures/grand_M_scaling_test_h200_rerun.{png,pdf} (new run) and
figures/grand_M_scaling_test_h200_ac78eb4.{png,pdf} (current CSV).
"""
import csv
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

ROOT = Path(sys.argv[1]).resolve()
NEW_CSV = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(ROOT))

import figures.grand_M_perf as perf  # noqa: E402


def load(path):
    with open(path) as fh:
        rows = [r for r in csv.DictReader(fh) if r["backend"] == "widebvh"]
    return {int(r["n"]): r for r in rows}


def render(csv_path, suffix):
    perf.FIG12_CSV = Path(csv_path)
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        perf.scaling_test("widebvh")
        for ext in ("png", "pdf"):
            src = ROOT / f"figures/grand_M_scaling_test_h200.{ext}"
            dst = ROOT / f"figures/grand_M_scaling_test_h200_{suffix}.{ext}"
            src.rename(dst)
            print(f"  -> {dst.relative_to(ROOT)}")
    finally:
        os.chdir(cwd)


old = load(ROOT / "data" / "fig12_scaling_h200.csv")
new = load(NEW_CSV)

print("figure 12 (widebvh, full NeMO): old (ac78eb4) vs new")
print(f"{'N':>10} | {'total ms':>19} | {'far ms':>17} | {'near ms':>17} | "
      f"{'M upd/s':>15} | {'VRAM GiB':>13}")
for n in sorted(new):
    o, w = old[n], new[n]
    def f(key, scale=1.0):
        return float(o[key]) * scale, float(w[key]) * scale
    to, tw = f("total_ms"); fo, fw = f("far_ms"); no, nw = f("near_ms")
    uo, uw = f("updates_per_sec", 1e-6); vo, vw = f("peak_vram_gb")
    print(f"{n:>10,} | {to:7.2f} {tw:7.2f} {(tw-to)/to:+3.0%} | "
          f"{fo:6.2f} {fw:6.2f} {(fw-fo)/fo:+3.0%} | "
          f"{no:6.2f} {nw:6.2f} {(nw-no)/no:+3.0%} | "
          f"{uo:5.2f} {uw:5.2f} {(uw-uo)/uo:+3.0%} | "
          f"{vo:5.2f} {vw:5.2f}")

print("\nrendering new-run figure:")
render(NEW_CSV, "rerun")
print("rendering current-CSV figure:")
render(ROOT / "data" / "fig12_scaling_h200.csv", "ac78eb4")
