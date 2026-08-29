"""
Raw data:

Benchmark: N=10000, warmup=6, runs=6
------------------------------------
Operator                        Device     Mean (ms)  Std (ms)
--------------------------------------------------------------
FMM_2body_RPY                   cuda           12.81      0.34
FMM_2body_NN                    cuda           13.54      0.29
FMM_Nbody_NN                    cuda           16.76      0.09
Total Peak VRAM Usage: 1.31 GB


Benchmark: N=100000, warmup=6, runs=6
-------------------------------------
Operator                        Device     Mean (ms)  Std (ms)
--------------------------------------------------------------
FMM_2body_RPY                   cuda           44.83      0.21
FMM_2body_NN                    cuda           47.90      0.34
FMM_Nbody_NN                    cuda           72.02      0.02

Total Peak VRAM Usage: 3.04 GB

Benchmark: N=1000000, warmup=6, runs=6
--------------------------------------
Operator                        Device     Mean (ms)  Std (ms)
--------------------------------------------------------------
FMM_2body_RPY                   cuda          625.08     21.41
FMM_2body_NN                    cuda          667.98     11.94
FMM_Nbody_NN                    cuda          997.51     82.60

Total Peak VRAM Usage: 20.50 GB
"""

import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

def comp_operators():
    """
    n_values = [10_000, 100_000, 1_000_000]

    mean_ms_rpy = [12.81, 44.83, 625.08]
    mean_ms_nn = [13.54, 47.90, 667.98]
    mean_ms_m = [16.76, 72.02, 997.51]
    """
    n_values = [10000, 100_000]

    mean_ms_rpy = [12.81, 44.83,]
    mean_ms_nn = [13.54, 47.90]

    plt.figure(figsize=(6, 4))

    x = np.arange(len(n_values), dtype=float)
    bar_width = 0.36

    colors = ["#009E73", "#E69F00"]

    plt.bar(x - bar_width / 2.0, mean_ms_rpy, width=bar_width, color=colors[0], label=r"NeMO$_{2b}$ (RPY)")
    plt.bar(x + bar_width / 2.0, mean_ms_nn, width=bar_width, color=colors[1], label=r"NeMO$_{2b}$ (NN)")

    ax = plt.gca()
    ax.set_ylim(bottom=0)
    ax.set_xlim(-0.55, len(n_values) - 0.45)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n:,}" for n in n_values])

    plt.xlabel("N")
    plt.ylabel("Mean time (ms)")
    plt.legend()

    plt.tight_layout()
    plt.savefig("figures/Nemo_2b_rpy_vs_nn.pdf", dpi=600)
    plt.savefig("figures/Nemo_2b_rpy_vs_nn.png", dpi=600)

FIG11_CSV = Path(__file__).resolve().parents[1] / "data" / "fig11_breakdown_h200.csv"

# Figure 11 sizes and the operators drawn in panel (a), in curve order.
FIG11_SIZES = (10_000, 50_000, 100_000, 200_000, 1_000_000)
FIG11_OPERATORS = (
    ("FMM_2body_RPY", "RPY", "#4C78A8", "o"),
    ("FMM_2body_NN", "NeMO (excl. nbody)", "#F58518", "s"),
    ("FMM_Nbody_NN", "NeMO", "#54A24B", "^"),
)
# Panel (b) stacks these, bottom first.
FIG11_COMPONENTS = (
    ("far_ms", "far-field", "#4C78A8"),
    ("self2b_ms", "self + 2-body", "#F58518"),
    ("nbody_ms", "nbody correction", "#54A24B"),
    ("nsearch_ms", "neighbor search", "#B279A2"),
)


def _load_fig11(backend):
    """Rows for one far-field backend, keyed by (N, operator).

    Measured by benchmarks/figure11_breakdown.py. The published figure's
    numbers lived in a hand-transcribed dict in figures/plot_runtime_breakdown.py
    (kept for provenance); reading a CSV here makes the figure reproducible.
    """
    with FIG11_CSV.open() as fh:
        rows = [r for r in csv.DictReader(fh) if r["backend"] == backend]
    if not rows:
        raise SystemExit(
            f"no '{backend}' rows in {FIG11_CSV}; run\n"
            f"  python benchmarks/figure11_breakdown.py --backend {backend}")
    return {(int(r["n"]), r["operator"]): r for r in rows}


def _size_label(n):
    return f"{n // 1_000_000}M" if n >= 1_000_000 else f"{n // 1000}k"


def runtime_breakdown(backend="widebvh"):
    """Figure 11: where one mobility application spends its time (H200).

    (a) total wall time of the three operators against N; (b) the full NeMO
    operator's time split into its four components.

    Panel (a) plots the operator's own end-to-end timer rather than the sum of
    panel (b)'s segments: the segments leave ~0.6 ms of tensor staging and the
    final far+near add untimed, which is 5% of the step at 10k. The published
    figure summed the segments and happened to land on the right answer because
    it also double-counted the neighbour search by about the same amount.
    """
    rows = _load_fig11(backend)
    sizes = [n for n in FIG11_SIZES if (n, "FMM_Nbody_NN") in rows]
    assert sizes, f"no usable rows for backend {backend!r}"
    x = np.arange(len(sizes))

    fig, (ax_total, ax_break) = plt.subplots(
        1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.08, 1.0]})
    fig.patch.set_facecolor("white")

    for op, label, color, marker in FIG11_OPERATORS:
        totals = [float(rows[(n, op)]["total_gpu_ms"]) for n in sizes]
        ax_total.plot(x, totals, color=color, marker=marker, linewidth=2.0,
                      markersize=5.5, label=label)

    ax_total.set_title("(a) Total wall time", loc="left")
    ax_total.set_ylabel("Runtime (ms)")
    ax_total.legend(frameon=False, loc="upper left")

    bottom = np.zeros(len(sizes))
    for key, label, color in FIG11_COMPONENTS:
        values = np.array([float(rows[(n, "FMM_Nbody_NN")][key]) for n in sizes])
        ax_break.bar(x, values, width=0.68, bottom=bottom, color=color,
                     edgecolor="none", linewidth=0, label=label)
        bottom += values

    ax_break.set_title("(b) Full NeMO breakdown", loc="left")
    ax_break.set_ylabel("Runtime (ms)")
    ax_break.legend(frameon=False, loc="upper left")

    for ax in (ax_total, ax_break):
        ax.set_facecolor("white")
        ax.set_xlabel("Problem size")
        ax.set_xticks(x)
        ax.set_xticklabels([_size_label(n) for n in sizes])
        ax.set_ylim(bottom=0)
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"figures/runtime_summary_two_panel.{ext}", dpi=600,
                    bbox_inches="tight")
    print("wrote figures/runtime_summary_two_panel.{png,pdf}")

    print("\n        N     total    far    self+2b   nbody  nsearch  far share")
    for n in sizes:
        r = rows[(n, "FMM_Nbody_NN")]
        tot = float(r["total_gpu_ms"])
        print(f" {n:>8,}  {tot:8.2f} {float(r['far_ms']):7.2f} "
              f"{float(r['self2b_ms']):9.2f} {float(r['nbody_ms']):8.2f} "
              f"{float(r['nsearch_ms']):7.2f} {float(r['far_ms']) / tot:9.1%}")


FIG12_CSV = Path(__file__).resolve().parents[1] / "data" / "fig12_scaling_h200.csv"


def _load_fig12(backend):
    """Rows for one far-field backend, ordered by N.

    Measured by benchmarks/figure12_grand_M.py; reading them here instead of
    hardcoding keeps the figure and the numbers that produced it in sync.
    """
    with FIG12_CSV.open() as fh:
        rows = [r for r in csv.DictReader(fh) if r["backend"] == backend]
    if not rows:
        raise SystemExit(
            f"no '{backend}' rows in {FIG12_CSV}; run\n"
            f"  python benchmarks/figure12_grand_M.py --backend {backend}")
    rows.sort(key=lambda r: int(r["n"]))
    return rows


def scaling_test(backend="widebvh"):
    """Figure 12: end-to-end scaling of the grand mobility operator (H200).

    Bars are the total runtime of one mobility application; the line is the
    corresponding particle-update rate.
    """
    rows = _load_fig12(backend)
    n_values = [int(r["n"]) for r in rows]
    total_times_ms = [float(r["total_ms"]) for r in rows]
    updates_per_sec = [float(r["updates_per_sec"]) / 1e6 for r in rows]

    fig, ax = plt.subplots(figsize=(6.5, 4))

    x = np.arange(len(n_values))
    ax.bar(x, total_times_ms, width=0.6, color="#0072B2", alpha=0.85,
           label="H200 runtime")

    ax.set_xlabel("N (particles)")
    ax.set_ylabel("Total time (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n // 1000}k" for n in n_values])
    ax.set_ylim(bottom=0)

    ax2 = ax.twinx()
    ax2.plot(x, updates_per_sec, marker="o", color="#E69F00",
             label="H200 particles/sec")
    ax2.set_ylabel("Particle updates per sec (millions)")
    ax2.set_ylim(bottom=0.0, top=max(updates_per_sec) * 1.25)

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper left")

    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(f"figures/grand_M_scaling_test_h200.{ext}", dpi=600)
    print("wrote figures/grand_M_scaling_test_h200.{png,pdf}")
    for n, t, u in zip(n_values, total_times_ms, updates_per_sec):
        print(f"  N={n:>8,}  {t:8.2f} ms  {u:6.3f} M updates/s")


FIG12_5090_CSV = FIG12_CSV.with_name("fig12_scaling_5090.csv")
# H200 column re-measured 2026-08-22 at the current tree, 50k..2M, widebvh fp32
# level 2 (benchmarks/figure12_grand_M.py, one process per size above 1.5M;
# artifacts/fig12_h200_f32l2_report.md). The fp64 column stays in FIG12_CSV.
FIG12_H200_F32L2_CSV = FIG12_CSV.with_name("fig12_scaling_h200_f32l2.csv")


def _load_fig12_csv(path, backend):
    with path.open() as fh:
        rows = [r for r in csv.DictReader(fh) if r["backend"] == backend]
    assert rows, f"no '{backend}' rows in {path}"
    rows.sort(key=lambda r: int(r["n"]))
    return rows


def _n_label(n):
    return f"{n // 1000}k" if n < 1_000_000 else f"{n / 1e6:g}M"


# Sizes drawn in the published H200-vs-5090 figure. Both CSVs carry the full
# 50k..2M grid (10 sizes); the figure shows this subset so the bars stay
# readable. Set sizes=None to draw everything measured.
FIG12_VS_SIZES = (50_000, 200_000, 500_000, 1_000_000, 1_500_000)


def scaling_test_h200_vs_5090(backend="widebvh", h200_csv=FIG12_H200_F32L2_CSV,
                              sizes=FIG12_VS_SIZES):
    """Figure 12 as published: H200 against RTX 5090, same operator.

    H200 rows come from data/fig12_scaling_h200_f32l2.csv (50k..2M, widebvh
    fp32 level 2, current tree); RTX 5090 rows from data/fig12_scaling_5090.csv,
    measured 2026-08-23 on a RunPod RTX 5090 (driver 590.48.01, same image and
    tree, one process per size; artifacts/fig12_5090_runpod_report.md) with the
    same widebvh far field, pdeg 7, mac 0.8, fp32 level 3 -- the consumer-GPU
    operating point, 50k..2M. The size grid is the union of the two CSVs,
    restricted to `sizes` (a card with no row at a size simply has no
    bar/marker there). Styling is that of
    figs/gpu_scaling_h200_vs_5090_05_19.pdf in the paper repo.
    """
    h200 = {int(r["n"]): r for r in _load_fig12_csv(h200_csv, backend)}
    r5090 = {int(r["n"]): r for r in _load_fig12_csv(FIG12_5090_CSV, backend)}
    n_values = sorted(set(h200) | set(r5090))
    if sizes is not None:
        n_values = [n for n in n_values if n in sizes]
    assert n_values, "size filter left nothing to plot"

    def col(rows, key, scale=1.0):
        return [float(rows[n][key]) * scale if n in rows else np.nan
                for n in n_values]

    t_h200, t_5090 = col(h200, "total_ms"), col(r5090, "total_ms")
    u_h200 = col(h200, "updates_per_sec", 1e-6)
    u_5090 = col(r5090, "updates_per_sec", 1e-6)

    c_h200, c_5090 = "#1f77b4", "#ff7f0e"
    fig, ax = plt.subplots(figsize=(9.1, 4.7))
    x = np.arange(len(n_values))
    w = 0.36
    # Paired bars where both cards were measured; a lone bar sits on its tick.
    x_h200 = np.array([xi - w / 2 if n in r5090 else xi
                       for xi, n in zip(x, n_values)])
    ax.bar(x_h200, t_h200, width=w, color=c_h200, alpha=0.6,
           edgecolor=c_h200, linewidth=0.7, label="H200 runtime")
    ax.bar(x + w / 2, t_5090, width=w, color=c_5090, alpha=0.6,
           edgecolor=c_5090, linewidth=0.7, hatch="//",
           label="RTX 5090 runtime")

    ax.set_xlabel("N (particles)")
    ax.set_ylabel("Total time (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels([_n_label(n) for n in n_values])
    ax.set_ylim(bottom=0)

    ax2 = ax.twinx()
    ax2.plot(x, u_h200, marker="o", color=c_h200, linewidth=2,
             label="H200 particles/sec")
    ax2.plot(x, u_5090, marker="s", color=c_5090, linewidth=2,
             linestyle="--", label="RTX 5090 particles/sec")
    ax2.set_ylabel("Particle updates per sec (millions)")
    ax2.set_ylim(bottom=0.0, top=np.nanmax(u_h200 + u_5090) * 1.3)

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper left",
               framealpha=0.95)

    ax.grid(axis="y", alpha=0.55, linestyle=":", linewidth=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(f"figures/gpu_scaling_h200_vs_5090.{ext}", dpi=600)
    print("wrote figures/gpu_scaling_h200_vs_5090.{png,pdf}")
    print("\n        N   H200 ms  5090 ms   H200 Mu/s  5090 Mu/s  5090/H200")
    for n, a, b, ua, ub in zip(n_values, t_h200, t_5090, u_h200, u_5090):
        print(f" {n:>9,}  {a:8.2f} {b:8.2f}   {ua:8.3f}  {ub:8.3f}   {b / a:7.2f}x")


# Backends the A/B figure draws, in bar order: (csv key, label, total colour,
# far-field colour). A backend with no rows in the CSV is skipped rather than
# raising, so the figure renders from whatever has been measured.
AB_BACKENDS = (
    ("warp", "Warp treecode", "#7F7F7F", "#3B3B3B"),
    ("widebvh", "widebvh bary p7", "#7FB3D5", "#0072B2"),
    ("widebvh-cart", "widebvh Cartesian p4", "#F0C27A", "#D55E00"),
)


def scaling_test_backends():
    """Companion to Figure 12: the same measurement for each far field.

    Not in the paper -- it isolates what swapping the far-field solver buys,
    with the near field and the timing protocol held fixed. Each bar's dark
    segment is the far field, so the light remainder is the shared near field.
    """
    series = []
    for key, label, c_total, c_far in AB_BACKENDS:
        try:
            series.append((_load_fig12(key), label, c_total, c_far))
        except SystemExit:
            print(f"  (no '{key}' rows in the CSV; skipping)")
    assert series, "no backends to plot"
    n_values = [int(r["n"]) for r in series[0][0]]

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    x = np.arange(len(n_values))
    w = 0.8 / len(series)

    for i, (rows, label, c_total, c_far) in enumerate(series):
        off = (i - (len(series) - 1) / 2) * w
        ax.bar(x + off, [float(r["total_ms"]) for r in rows], width=w,
               color=c_total, alpha=0.95, label=f"{label} (total)")
        ax.bar(x + off, [float(r["far_ms"]) for r in rows], width=w,
               color=c_far, alpha=0.95, label=f"{label} (far field)")

    ax.set_xlabel("N (particles)")
    ax.set_ylabel("Time per mobility application (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n // 1000}k" for n in n_values])
    ax.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.9)
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(f"figures/grand_M_far_field_ab.{ext}", dpi=600)
    print("wrote figures/grand_M_far_field_ab.{png,pdf}")


if __name__ == "__main__":
    runtime_breakdown()
    scaling_test()
    scaling_test_h200_vs_5090()
    scaling_test_backends()
