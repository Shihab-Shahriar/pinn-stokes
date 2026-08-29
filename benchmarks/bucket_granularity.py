"""Is the Cartesian policy's P2P blowup a bucket-granularity artifact?

At the calibrated operating points the two policies sit in very different
near-pair regimes (83M vs 7.3M pairs at 1M), and the bucketizer's automatic
cell edge -- q = cbrt(max(1024, n/maxLeaf)), so ~1024 cells no matter how large
n is -- was tuned for the sparse one. Sweep q for both policies so each is
compared at its own best granularity rather than at the other's.

Writes data/cartesian_bucket_granularity.csv.
"""
import contextlib
import csv
import io
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.mac_calibration import (  # noqa: E402
    CUTOFF, far_ref_tt, get_case, make_force, time_far)
from src.treecode_widebvh import WidebvhFMM  # noqa: E402

OUT = ROOT / "data" / "cartesian_bucket_granularity.csv"
FIELDS = ("case", "n", "loading", "policy", "order", "pdeg", "mac", "hilbert_q",
          "buckets", "near_pairs", "far_ms", "traverse_ms", "p2p_ms",
          "upward_ms", "build_ms", "rel_far")

CASES = sys.argv[1:] or ["uniform100k", "uniform750k", "twoball0"]
QS = [None, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 26.0, 28.0, 36.0]

CONFIGS = [
    ("bary", dict(policy="bary", mac=0.8)),
    ("cart", dict(policy="cart", order=4, mac=0.33)),
]

rows = []
for case in CASES:
    pos, descr = get_case(case)
    n = pos.shape[0]
    dev = torch.device("cuda")
    pos32 = torch.as_tensor(pos, dtype=torch.float32, device=dev).contiguous()
    pos64 = torch.as_tensor(pos, dtype=torch.float64, device=dev).contiguous()
    sample = torch.linspace(0, n - 1, min(2048, n), device=dev).long()

    f_np = make_force(n, "random")
    f3 = torch.as_tensor(f_np, dtype=torch.float32, device=dev).contiguous()
    f64 = torch.as_tensor(f_np, dtype=torch.float64, device=dev)
    ref = far_ref_tt(pos64, f64, sample)
    ref_norm = float(torch.linalg.norm(ref))
    print(f"\n=== {case}: N={n:,} ({descr}), random loading ===", flush=True)

    for label, kw in CONFIGS:
        for q in QS:
            s = WidebvhFMM(near_field_operator=None, near_field_cutoff=CUTOFF,
                           hilbert_q=q, **kw)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    u = s.get_far_field_vel(pos32, f3)
                err = float(torch.linalg.norm(u[sample] - ref)) / ref_norm
                ms = time_far(s, pos32, f3)
                st = s.last_stats
            finally:
                s.close()
            rows.append(dict(
                case=case, n=n, loading="random", policy=label,
                order=kw.get("order", ""), pdeg=7 if label == "bary" else "",
                mac=kw["mac"], hilbert_q="" if q is None else q,
                buckets=int(st["num_source_buckets"]),
                near_pairs=int(st["near_pairs"]), far_ms=ms,
                traverse_ms=st["traverse_ms"], p2p_ms=st["p2p_ms"],
                upward_ms=st["upward_ms"], build_ms=st["build_bvh_ms"],
                rel_far=err))
            print(f"{label:5s} q={str(q):>5s} buckets={st['num_source_buckets']:8.0f} "
                  f"pairs={st['near_pairs']:11.0f} far={ms:8.2f} ms "
                  f"(trav={st['traverse_ms']:6.2f} p2p={st['p2p_ms']:7.2f}) "
                  f"rel_far={err:.3e}", flush=True)

OUT.parent.mkdir(parents=True, exist_ok=True)
# Replace only the cases just measured and keep the rest, so running this on one
# case does not silently discard the others (figure12_grand_M.py does the same
# for its backends). Plain "w" here costs you the whole file.
kept = []
if OUT.exists():
    with OUT.open() as fh:
        kept = [r for r in csv.DictReader(fh) if r["case"] not in set(CASES)]
with OUT.open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    w.writeheader()
    w.writerows(kept)
    w.writerows(rows)
print(f"\nwrote {len(rows)} rows ({len(kept)} kept) -> {OUT}")
