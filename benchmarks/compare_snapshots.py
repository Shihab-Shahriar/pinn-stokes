"""Trajectory A/B between two two_suspensions_1M.py --snapshots runs.

Reads positions_<t>.npy from two output directories (same seed, so the same
initial cloud) and reports, per snapshot, the RMS and max particle displacement
between the runs in units of the particle radius (1.0), plus bulk statistics --
the protocol of artifacts/fig13_mac_report.md sec 4, where mac 0.9/leaf 512
against production measured 0.136 radii at t = 1.0.

    python benchmarks/compare_snapshots.py figures/drop_1M_4060_fp64 figures/drop_1M_4060
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

R_DROP = 175.0


def snapshots(d: Path) -> dict[float, Path]:
    out = {}
    for p in d.glob("positions_*.npy"):
        m = re.match(r"positions_([\d.]+)\.npy$", p.name)
        if m:
            out[float(m.group(1))] = p
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("a", type=Path, help="reference run directory")
    ap.add_argument("b", type=Path, help="candidate run directory")
    args = ap.parse_args()

    sa, sb = snapshots(args.a), snapshots(args.b)
    ts = sorted(set(sa) & set(sb))
    assert ts, f"no common positions_<t>.npy between {args.a} and {args.b}"
    z0 = None
    print(f"{'t':>5} {'rms':>10} {'max':>10} {'max/R_drop':>11} "
          f"{'max/settled':>12}  {'z_mean A':>11} {'z_mean B':>11} "
          f"{'z_std A':>9} {'z_std B':>9}")
    for t in ts:
        a = np.load(sa[t]).astype(np.float64)
        b = np.load(sb[t]).astype(np.float64)
        assert a.shape == b.shape, f"t={t}: {a.shape} vs {b.shape}"
        d = np.linalg.norm(a - b, axis=1)
        if z0 is None:
            z0 = a[:, 2].mean()
        settled = abs(a[:, 2].mean() - z0)
        print(f"{t:5.1f} {np.sqrt((d**2).mean()):10.3e} {d.max():10.3e} "
              f"{d.max()/R_DROP:11.2e} "
              f"{(d.max()/settled if settled > 0 else float('nan')):12.2e}  "
              f"{a[:,2].mean():11.3f} {b[:,2].mean():11.3f} "
              f"{a[:,2].std():9.3f} {b[:,2].std():9.3f}")
    print("displacements are in particle radii (radius 1.0); "
          "settled = bulk z displacement of the reference since t=0")


if __name__ == "__main__":
    main()
