"""Figure 11: per-component runtime breakdown of one mobility application.

Two panels, both over N = 10k ... 1M on uniform suspensions at 10% volume
fraction:

  (a) total wall time of three operators -- RPY, NeMO excluding the n-body
      correction, and the full NeMO operator;
  (b) the full operator's time split into far field / self + two-body /
      n-body correction / neighbour search.

The published figure was produced with the Warp far field and was never
reproducible from the repo: its numbers live in a 1319-line hand-transcribed
`RAW_DATA` dict in `figures/plot_runtime_breakdown.py`, itself transcribed from
the five stdout captures in `figures/runtime_breakdown/*_h200.txt`. This script
replaces the transcription step. `--from-logs` re-parses those same captures, so
the published Warp column lands in the CSV beside the new measurement without
spending GPU time, and -- more usefully -- it is a self-test: if the parser does
not reproduce the published figure from the archived logs, no new number it
produces can be trusted either.

Where the four components come from: they exist only as unconditional print()
calls (`src/treecode.py:307,400`, `src/gpu_nbody_mob.py:402,425`), so each
apply() is run under a stdout capture and the keys are parsed back out.
`WidebvhFMM` overrides only `get_far_field_vel`, so every near-field key is
identical across backends and one parser serves both.

Timing protocol is deliberately identical to Figure 12
(benchmarks/figure12_grand_M.py): 6 warmup applications so torch.compile can
specialize and 6 timed, reduced by a median rather than the published figure's
three-sample mean (see `_warm_median`). torch.compile stays ENABLED; this is a
performance run, but every size runs in its own process -- see main().

One deliberate deviation from the published figure: its "neighbour search"
segment is `Warp::HashGrid took` PLUS `[MobFMM] near-field construction`
(`plot_runtime_breakdown.py:_add_near_field_setup`), which are the same interval
timed two ways -- 25.07 vs 25.195 ms at N=1M. This script uses the CUDA-event
value alone, for both backends, so the A/B stays apples-to-apples.

Usage:
    source ~/warp_env.sh
    python benchmarks/figure11_breakdown.py --from-logs      # backfill warp, no GPU
    python benchmarks/figure11_breakdown.py --backend widebvh
    python figures/grand_M_perf.py                           # renders from the CSV
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import re
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.figure10_components import git_sha, write_rows  # noqa: E402
from benchmarks.performance_grand_M import (  # noqa: E402
    BenchmarkConfig, build_far_field, build_near_field, build_forces,
    load_configuration, TMP_DIR,
)

DEFAULT_SIZES = (10_000, 50_000, 100_000, 200_000, 1_000_000)
CSV_PATH = ROOT / "data" / "fig11_breakdown_h200.csv"
LOG_DIR = ROOT / "figures" / "runtime_breakdown"

# Bar/curve order in both panels. The labels are the published ones.
OPERATORS = ("2b_rpy", "2b_nn", "nbody")
OP_LABELS = {"2b_rpy": "FMM_2body_RPY", "2b_nn": "FMM_2body_NN",
             "nbody": "FMM_Nbody_NN"}

FIELDS = ("backend", "n", "operator", "far_ms", "self2b_ms", "nbody_ms",
          "nsearch_ms", "total_sum_ms", "total_gpu_ms", "unaccounted_ms",
          "overall_near_ms", "near_pairs", "peak_vram_gb", "peak_total_gb",
          "mac", "theta", "pdeg", "max_leaf", "gpu", "git_sha", "source")


# ---------------------------------------------------------------------------
# stdout parsing
# ---------------------------------------------------------------------------

# One entry per timed quantity. Every value a key emits during a capture is
# collected in order, so `_warm_median` can reduce over the warm tail.
PATTERNS = {
    "far_ms":          re.compile(r"\[MobFMM\] far-field FMM GPU time: ([\d.]+)"),
    "nsearch_ms":      re.compile(r"\[MobFMM\] near-field construction: ([\d.]+)"),
    "overall_near_ms": re.compile(r"\[MobFMM\] overall Nearfield time: ([\d,.]+)"),
    "total_gpu_ms":    re.compile(r"\[MobFMM\] total GPU time: ([\d.]+)"),
    "self2b_ms":       re.compile(r"\[Mob_Nbody\] Base velocity compute time: ([\d.]+)"),
    "nbody_ms":        re.compile(r"\[Mob_Nbody\] Post-base path time: ([\d.]+)"),
    "peak_alloc_mb":   re.compile(r"\[MobFMM\] peak GPU memory: allocated ([\d.]+) MB"),
    # Present only under NEMO_DEVICE_MEM=1. peak_alloc_mb is PyTorch's allocator
    # alone and misses the treecode's raw cudaMalloc traffic (~0.7-1.8 GiB at
    # N=1M); this is the figure that decides whether a card holds the run.
    "peak_total_mb":   re.compile(r"process total ([\d.]+) MB"),
    "near_pairs":      re.compile(r"Total near-field pairs found: (\d+)"),
    # Not a segment -- the same interval as nsearch_ms, timed from inside the
    # Warp scope instead of by CUDA events around it. Parsed only so the self
    # test can quantify the double count the published figure carried.
    "hashgrid_ms":     re.compile(r"Warp::HashGrid took ([\d.]+)"),
}


def parse_components(text: str) -> dict:
    """Every occurrence of each timed key, in emission order."""
    return {k: [float(m.replace(",", "")) for m in rx.findall(text)]
            for k, rx in PATTERNS.items()}


WARM_TAIL = 6


def _warm_median(values):
    """Median of the warm tail.

    The published figure averaged the last three samples
    (plot_runtime_breakdown.py:_avg_last3). That has no outlier protection, and
    the far-field timer is the noisiest thing here -- src/treecode.py:321 says
    so in a FIXME. One 10k run had a single apply blow up enough to drag the
    three-sample mean of a 7.6 ms far field to 85.6 ms, which looks like a
    result rather than a hiccup. A median over the same warm window is robust
    to that and agrees with the mean everywhere the mean is trustworthy (it
    still reproduces the published run; see check_against_published).

    The tail is the last WARM_TAIL samples: exactly the timed applies here,
    and the timed half of the archived logs, which interleave warmup with
    timed in a single stdout stream.
    """
    if not values:
        return None
    tail = sorted(values[-WARM_TAIL:])
    mid = len(tail) // 2
    return tail[mid] if len(tail) % 2 else 0.5 * (tail[mid - 1] + tail[mid])


def reduce_components(raw: dict) -> dict:
    """Collapse the per-call lists into the four figure segments.

    NNMobTorch (the 2b_rpy / 2b_nn operators) emits no [Mob_Nbody] lines, so its
    self+2-body cost is recovered as the near-field pass minus the neighbour
    search, and its n-body segment is zero. This mirrors
    plot_runtime_breakdown.py:_fill_base_velocity_from_overall, which is what
    makes panel (a)'s three curves comparable.
    """
    out = {k: _warm_median(v) for k, v in raw.items()}

    if out["self2b_ms"] is None:
        assert out["overall_near_ms"] is not None, "no near-field timing captured"
        out["self2b_ms"] = out["overall_near_ms"] - out["nsearch_ms"]
        out["nbody_ms"] = 0.0

    for key in ("far_ms", "self2b_ms", "nbody_ms", "nsearch_ms"):
        assert out[key] is not None, f"missing component {key}"

    out["total_sum_ms"] = (out["far_ms"] + out["self2b_ms"]
                           + out["nbody_ms"] + out["nsearch_ms"])

    # The four segments do not tile the step: `total GPU time` also covers
    # tensor staging inside the near-field pass and the final far+near add, all
    # of which sit outside any named timer. It is ~0.6 ms and roughly flat in N,
    # so it is 5% of the step at 10k and 0.3% at 1M. Recorded rather than
    # hidden -- panel (a) plots the measured total, panel (b) the segments.
    out["unaccounted_ms"] = out["total_gpu_ms"] - out["total_sum_ms"]

    # What the published figure plotted for the same run: it added the
    # Warp-scope timing of the neighbour search on top of the CUDA-event one.
    # That double count is ~0.55 ms and happens to cancel the unaccounted time
    # almost exactly, which is why the old panel (a) tracked the wall clock
    # despite summing four segments that do not add up to it.
    out["published_convention_ms"] = (out["total_sum_ms"]
                                      + (out["hashgrid_ms"] or 0.0))
    return out


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------

def measure(n: int, backend: str, bench_cfg: BenchmarkConfig, only=None,
            **kwargs) -> list:
    """One row per operator at this N, or just `only` if given."""
    device = torch.device("cuda")
    config = load_configuration(TMP_DIR / f"uniform_large_0.1_{n}.csv")
    assert config.shape[0] == n, f"{n} requested, CSV has {config.shape[0]}"
    forces = build_forces(n, seed=2024)

    positions = torch.as_tensor(config[:, :3], dtype=torch.float32, device=device)
    orientations = torch.as_tensor(config[:, 3:], dtype=torch.float32, device=device)
    forces_t = torch.as_tensor(forces, dtype=torch.float32, device=device)
    vis_arr = torch.full((n,), bench_cfg.viscosity, dtype=torch.float32,
                         device=device)

    if backend == "widebvh-cart":
        from src.treecode_widebvh import cart_hilbert_q
        kwargs.setdefault("hilbert_q", cart_hilbert_q(n))

    rows = []
    for kind in (OPERATORS if only is None else (only,)):
        op = build_far_field(build_near_field(kind, "sphere", 6.0), backend, 6.0,
                             **kwargs)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            for _ in range(bench_cfg.warmup_runs):
                op.apply(positions, orientations, forces_t, vis_arr)
                torch.cuda.synchronize(device)
            # Only the timed applies are parsed, so warmup chatter -- and the
            # cold hash-grid build, which is 10x the warm one -- cannot leak in.
            buf.truncate(0)
            buf.seek(0)
            for _ in range(bench_cfg.timed_runs):
                op.apply(positions, orientations, forces_t, vis_arr)
                torch.cuda.synchronize(device)
                time.sleep(0.1)          # let the clocks settle between samples

        raw = parse_components(buf.getvalue())
        comp = reduce_components(raw)

        # Peak VRAM comes from the operator's own print, not from
        # max_memory_allocated(): src/treecode.py:249 resets the peak stats
        # inside every apply, after the far field, so an outer reading would
        # only see the near field onward.
        peak_alloc = _warm_median(raw["peak_alloc_mb"])
        peak_total = _warm_median(raw["peak_total_mb"])
        near_pairs = raw["near_pairs"]

        rows.append(dict(
            backend=backend, n=n, operator=OP_LABELS[kind],
            far_ms=comp["far_ms"], self2b_ms=comp["self2b_ms"],
            nbody_ms=comp["nbody_ms"], nsearch_ms=comp["nsearch_ms"],
            total_sum_ms=comp["total_sum_ms"],
            total_gpu_ms=comp["total_gpu_ms"],
            unaccounted_ms=comp["unaccounted_ms"],
            overall_near_ms=comp["overall_near_ms"],
            near_pairs=int(near_pairs[-1]) if near_pairs else "",
            peak_vram_gb=peak_alloc / 1024.0 if peak_alloc else "",
            peak_total_gb=peak_total / 1024.0 if peak_total else "",
            mac=getattr(op, "mac", ""), theta=getattr(op, "theta", ""),
            pdeg=getattr(op, "pdeg", ""), max_leaf=getattr(op, "max_leaf", ""),
            gpu=torch.cuda.get_device_name(device), git_sha=git_sha(),
            source="measured",
        ))

        gap = comp["unaccounted_ms"]
        print(f"  {OP_LABELS[kind]:16s} far={comp['far_ms']:8.2f} "
              f"self+2b={comp['self2b_ms']:7.2f} nbody={comp['nbody_ms']:8.2f} "
              f"nsearch={comp['nsearch_ms']:5.2f} | sum={comp['total_sum_ms']:8.2f} "
              f"vs total={comp['total_gpu_ms']:8.2f} "
              f"(unaccounted {gap:+.2f} ms, {gap / comp['total_gpu_ms']:+.1%})",
              flush=True)
        # Staging is ~0.6 ms and flat in N, so bound it absolutely as well as
        # relatively -- a genuinely unparsed key would leave a segment-sized
        # hole (the far field alone is >35% of the step), not a millisecond.
        assert abs(gap) < max(2.0, 0.03 * comp["total_gpu_ms"]), (
            f"component sum and total disagree by {gap:.2f} ms for "
            f"{OP_LABELS[kind]} at N={n}: a stdout key is probably missing")

        if hasattr(op, "close"):
            op.close()
        del op
        torch.cuda.empty_cache()

    return rows


# ---------------------------------------------------------------------------
# archived-log backfill
# ---------------------------------------------------------------------------

LOG_SIZES = {"10k": 10_000, "50k": 50_000, "100k": 100_000, "200k": 200_000,
             "1000k": 1_000_000}


def rows_from_logs(log_dir: Path, backend: str = "warp-published") -> list:
    """Re-parse the published run's stdout captures into CSV rows.

    Labelled `warp-published`, not `warp`, because it is not interchangeable
    with a Warp run measured today: the near field has moved since (n-body
    pair chunking, ac78eb4), and its segment is 11-15% cheaper in these logs
    than at HEAD. Measure `--backend warp` for a control that isolates the far
    field; use these rows for "what the paper reported".

    Each capture holds all three operators back to back, separated by the
    benchmark's own 'Completed benchmark for operator:' marker.

    Returns the CSV rows and, alongside them, what each row would have totalled
    under the published double-counted convention -- kept out of the rows
    themselves so the CSV carries one definition of a segment.
    """
    rows, conv = [], {}
    for tag, n in sorted(LOG_SIZES.items(), key=lambda kv: kv[1]):
        path = log_dir / f"runtime_breakdown_{tag}_h200.txt"
        assert path.exists(), f"missing archived log: {path}"
        chunks = path.read_text().split("Completed benchmark for operator:")
        assert len(chunks) >= len(OPERATORS), f"{path} has too few operator sections"

        for kind, chunk in zip(OPERATORS, chunks):
            comp = reduce_components(parse_components(chunk))
            near_pairs = parse_components(chunk)["near_pairs"]
            rows.append(dict(
                backend=backend, n=n, operator=OP_LABELS[kind],
                far_ms=comp["far_ms"], self2b_ms=comp["self2b_ms"],
                nbody_ms=comp["nbody_ms"], nsearch_ms=comp["nsearch_ms"],
                total_sum_ms=comp["total_sum_ms"],
                total_gpu_ms=comp["total_gpu_ms"],
                unaccounted_ms=comp["unaccounted_ms"],
                overall_near_ms=comp["overall_near_ms"],
                near_pairs=int(near_pairs[-1]) if near_pairs else "",
                peak_vram_gb="", peak_total_gb="",
                mac="", theta=0.3, pdeg="", max_leaf="",
                gpu="NVIDIA H200", git_sha="archived", source="archived_log",
            ))
            conv[(n, OP_LABELS[kind])] = comp["published_convention_ms"]
    return rows, conv


# The trimmed-mean wall clock each archived log ends with, i.e. what
# benchmark_apply() measured for the published run. This is the only exact
# ground truth on disk -- the figure's own y-values can be read off the plot
# only approximately -- and it is independent of every regex here, since it
# comes from time.perf_counter() around apply() rather than from any [MobFMM]
# print. Reproducing it is what makes the parser trustworthy on new data.
PUBLISHED_WALL_MS = {
    (10_000, "FMM_2body_RPY"): 9.38, (10_000, "FMM_2body_NN"): 9.55,
    (10_000, "FMM_Nbody_NN"): 11.57,
    (50_000, "FMM_2body_RPY"): 13.63, (50_000, "FMM_2body_NN"): 14.10,
    (50_000, "FMM_Nbody_NN"): 21.60,
    (100_000, "FMM_2body_RPY"): 22.54, (100_000, "FMM_2body_NN"): 22.95,
    (100_000, "FMM_Nbody_NN"): 36.92,
    (200_000, "FMM_2body_RPY"): 53.69, (200_000, "FMM_2body_NN"): 55.16,
    (200_000, "FMM_Nbody_NN"): 81.87,
    (1_000_000, "FMM_2body_RPY"): 251.19, (1_000_000, "FMM_2body_NN"): 256.45,
    (1_000_000, "FMM_Nbody_NN"): 388.94,
}


def check_against_published(rows, conv) -> None:
    """Self-test: the reduced components must rebuild the published run.

    Two columns, because this script deliberately changed one convention. The
    'published' column re-adds the double-counted neighbour search and is what
    the old figure plotted -- it is the one that has to match. The 'fixed'
    column is what goes into the CSV; the gap between them is the double count,
    which is large in relative terms only where the whole step is a few ms.
    """
    print("\nparser self-test -- rebuild the published run from its own logs")
    print(f"  {'N':>9}  {'operator':16} {'published':>10} {'wall clock':>11} "
          f"{'err':>7} | {'fixed':>9} {'dbl count':>10}")
    worst = 0.0
    for r in rows:
        key = (r["n"], r["operator"])
        if key not in PUBLISHED_WALL_MS:
            continue
        wall, pub = PUBLISHED_WALL_MS[key], conv[key]
        rel = abs(pub - wall) / wall
        worst = max(worst, rel)
        print(f"  {r['n']:>9,}  {r['operator']:16} {pub:10.2f} {wall:11.2f} "
              f"{rel:+7.2%} | {r['total_sum_ms']:9.2f} "
              f"{pub - r['total_sum_ms']:10.2f}")
    assert worst < 0.03, (
        f"parser is {worst:.1%} off the published wall clock; a stdout key is "
        f"probably missing or a reduction rule is wrong")
    print(f"  worst deviation {worst:.2%} -- the parser reproduces the "
          f"published figure")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="widebvh",
                    choices=("widebvh", "widebvh-cart", "warp"))
    ap.add_argument("--sizes", default=",".join(str(s) for s in DEFAULT_SIZES))
    ap.add_argument("--operator", default=None, choices=OPERATORS,
                    help="measure one operator in this process; omit to fan "
                         "out over all three, one subprocess each")
    ap.add_argument("--warmup", type=int, default=6)
    ap.add_argument("--runs", type=int, default=6)
    ap.add_argument("--csv", default=str(CSV_PATH))
    ap.add_argument("--from-logs", nargs="?", const=str(LOG_DIR), default=None,
                    help="parse the archived stdout captures instead of "
                         "measuring; no GPU is touched")
    args = ap.parse_args()

    out = Path(args.csv)

    if args.from_logs is not None:
        rows, conv = rows_from_logs(Path(args.from_logs))
        check_against_published(rows, conv)
    else:
        sizes = [int(s) for s in args.sizes.split(",") if s]

        # One (size, operator) per process, always. Sharing a process across
        # measurements corrupts them in two independent ways, both silent:
        #
        #  * across sizes -- by the fifth size torch.compile has seen 12
        #    operator instances at 5 shapes, trips config.recompile_limit and
        #    falls back to eager permanently. That inflated the 1M two-body
        #    segment from 28.2 to 116.8 ms, only at the last size, so it read
        #    as a scaling result rather than an artifact. Same failure mode as
        #    artifacts/widebvh_far_field_report.md section 5.
        #  * across operators -- the third operator built in a process
        #    intermittently reports a ~10x far field (108 ms against 11 at 50k;
        #    85 ms against 7.6 at 10k), because building two solvers first
        #    leaves the caching allocator fragmented and the resulting
        #    cudaMalloc/cudaFree syncs land inside the far field's own CUDA
        #    event bracket. op.close() and empty_cache() do not clear it.
        #    src/treecode.py:321 already flags this timer as noisy.
        #
        # The published run happened to avoid the first (one invocation per
        # size) and never hit the second (WarpFMM allocates differently).
        if len(sizes) > 1 or args.operator is None:
            for n in sizes:
                for kind in (OPERATORS if args.operator is None
                             else (args.operator,)):
                    cmd = [sys.executable, __file__, "--backend", args.backend,
                           "--sizes", str(n), "--operator", kind,
                           "--warmup", str(args.warmup),
                           "--runs", str(args.runs), "--csv", str(out)]
                    print(f"\n===== subprocess: {args.backend} N={n:,} "
                          f"{OP_LABELS[kind]} =====", flush=True)
                    subprocess.run(cmd, check=True, cwd=ROOT)
            with out.open() as fh:
                rows = [r for r in csv.DictReader(fh)
                        if r["backend"] == args.backend and int(r["n"]) in sizes]
            rows.sort(key=lambda r: (int(r["n"]),
                                     list(OP_LABELS.values()).index(r["operator"])))
            for r in rows:
                for k in ("far_ms", "self2b_ms", "nbody_ms", "nsearch_ms",
                          "total_sum_ms", "total_gpu_ms"):
                    r[k] = float(r[k])
            _summarize(rows)
            return

        bench_cfg = BenchmarkConfig(warmup_runs=args.warmup, timed_runs=args.runs)
        rows = []
        for n in sizes:
            print(f"\n########## {args.backend}  N={n:,} "
                  f"{OP_LABELS[args.operator]} ##########", flush=True)
            rows.extend(measure(n, args.backend, bench_cfg, only=args.operator))

    write_rows(out, FIELDS, rows,
               key_fn=lambda r: (str(r["backend"]), int(r["n"]), r["operator"]))
    print(f"\nwrote {len(rows)} rows -> {out}")

    _summarize(rows)


def _summarize(rows) -> None:
    print("\n         N  operator            far    self+2b     nbody  nsearch"
          "       sum     total   far share")
    for r in rows:
        print(f" {int(r['n']):>9,}  {r['operator']:16s} {r['far_ms']:8.2f} "
              f"{r['self2b_ms']:9.2f} {r['nbody_ms']:9.2f} {r['nsearch_ms']:7.2f} "
              f"{r['total_sum_ms']:9.2f} {r['total_gpu_ms']:9.2f} "
              f"{r['far_ms'] / r['total_gpu_ms']:10.1%}")


if __name__ == "__main__":
    main()
