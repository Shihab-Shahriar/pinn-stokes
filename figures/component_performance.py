"""Figure 10: performance of NeMO's individual components.

Two SEPARATE figures, meant to be placed side by side in LaTeX (subfigure /
minipage), which is why neither carries an "(a)"/"(b)" prefix and neither states
the treecode parameters in its title -- both belong to the caption:

  figures/component_pair_kernel.{png,pdf}
      Two-body pair-kernel throughput, analytic RPY vs the learned m_t^(2),
      against batch size -- the published Figure 10, now driven from measured
      data.
  figures/component_far_field.{png,pdf}
      The far-field treecode against particle count N: throughput on the left
      axis, and on a log right axis the far-field relative L2 error against a
      sampled fp64 direct sum together with the grand operator's relative
      asymmetry (Hutchinson estimate, with its standard error). Both of the
      latter are flat in N.

Data is produced by benchmarks/figure10_components.py; this script only renders.

    source ~/warp_env.sh
    python benchmarks/figure10_components.py --panel both
    python figures/component_performance.py
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

# Must precede the matplotlib import: the cluster home has no writable config
# dir, so redirect the caches into the repo (same preamble as figures/drop_1m.py).
REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".matplotlib_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / ".cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.titlesize": 8,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

FIGDIR = REPO_ROOT / "figures"
PAIR_CSV = REPO_ROOT / "data" / "fig10_pair_kernels.csv"
FAR_CSV = REPO_ROOT / "data" / "fig10_far_field.csv"

# Each figure occupies roughly half a text width once the two are set side by
# side, so they are sized for that rather than for a 1x2 subplot grid.
FIGSIZE = (3.7, 3.0)

# Okabe-Ito, as used across figures/grand_M_perf.py.
C_RPY = "#0072B2"
C_FT = "#E69F00"
C_THRU = "#0072B2"
C_RELL2 = "#D55E00"
C_ASYM = "#CC79A7"


def _load(path: Path, keep, producer: str):
    if not path.exists():
        raise SystemExit(f"missing {path}; run\n  {producer}")
    with path.open() as fh:
        rows = [r for r in csv.DictReader(fh) if keep(r)]
    if not rows:
        raise SystemExit(f"no matching rows in {path}; run\n  {producer}")
    return rows


def _load_pair_kernels(kernel, mode="compile", protocol="matched",
                       lo=2**15, hi=2**22):
    rows = _load(
        PAIR_CSV,
        lambda r: (r["kernel"] == kernel and r["mode"] == mode
                   and r["protocol"] == protocol
                   and lo <= int(r["batch"]) <= hi
                   # powers of two only: the CSV carries the producer's full
                   # list, the paper's axis is 2^15..2^22.
                   and int(r["batch"]) & (int(r["batch"]) - 1) == 0),
        "python benchmarks/figure10_components.py --panel left")
    rows.sort(key=lambda r: int(r["batch"]))
    return (np.array([int(r["batch"]) for r in rows]),
            np.array([float(r["throughput_per_sec"]) for r in rows]))


def _load_far_field(backend="widebvh", min_n=10_000):
    """Rows with N strictly greater than `min_n`.

    N <= 10k is dropped from the figure (the rows stay in the CSV): with
    max_leaf=1024 a 5-10k cloud is only a handful of buckets, so the far field
    is pinned by its ~7 ms fixed build cost and few nodes pass the MAC. Those
    points measure tree construction, not the asymptotic regime the figure is
    about, and they compress the throughput curve against the x-axis.
    """
    rows = _load(FAR_CSV,
                 lambda r: r["backend"] == backend and int(r["n"]) > min_n,
                 "python benchmarks/figure10_components.py --panel right")
    rows.sort(key=lambda r: int(r["n"]))
    col = lambda k: np.array([float(r[k]) for r in rows])
    return (np.array([int(r["n"]) for r in rows]), col("far_updates_per_sec"),
            col("rel_far"), col("rel_asym"), col("rel_asym_stderr"))


def _save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"{stem}.{ext}", dpi=600, bbox_inches="tight")
    print(f"wrote figures/{stem}.{{png,pdf}}")


def pair_kernel_figure(save=True, protocol="matched", mode="compile"):
    """Left figure: analytic RPY vs the learned two-body cross kernel."""
    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)

    b_rpy, t_rpy = _load_pair_kernels("rpy", mode, protocol)
    b_ft, t_ft = _load_pair_kernels("ft", mode, protocol)

    ax.plot(b_rpy, t_rpy / 1e9, marker="s", ms=4, lw=1.4, color=C_RPY,
            label="RPY (analytic)")
    ax.plot(b_ft, t_ft / 1e9, marker="o", ms=4, lw=1.4, color=C_FT,
            label=r"$m_t^{(2)}$ (learned)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(b_rpy)
    ax.set_xticklabels([rf"$2^{{{int(np.log2(b))}}}$" for b in b_rpy])
    ax.set_xlabel("Batch size")
    # Explicit label rather than matplotlib's offset text, which is easy to
    # crop away with bbox_inches="tight".
    ax.set_ylabel(r"Throughput ($10^9$ pair evals/s)")
    ax.set_ylim(bottom=0)
    ax.set_title("Two-body pair kernel", loc="left")
    ax.grid(axis="y", alpha=0.25, ls=":")
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=7, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    if save:
        _save(fig, "component_pair_kernel")
    return fig


def far_field_figure(save=True, prmse_ref=0.075):
    """Right figure: far-field treecode throughput and accuracy vs N."""
    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)

    n, thru, rel_far, rel_asym, asym_err = _load_far_field()

    ax.plot(n, thru / 1e6, marker="o", ms=4, lw=1.4, color=C_THRU,
            label="throughput (left)")
    ax.set_xscale("log")
    ax.set_xlabel("Number of particles, $N$")
    # Both axes stay black: the left/right binding is carried by the "(left)" /
    # "(right)" suffixes in the legend, which survive greyscale printing.
    ax.set_ylabel("Far-field throughput (M updates/s)")
    # Headroom, deliberately: the two series live on different axes, so any
    # apparent crossing is an artifact. Keeping throughput in the lower half
    # stops it from running through the right axis' reference line.
    ax.set_ylim(0, 1.3 * float(thru.max()) / 1e6)
    ax.set_title("Far-field treecode", loc="left")
    # Grid on the base axes only; gridding both halves of a twin pair is
    # unreadable.
    ax.grid(axis="y", alpha=0.25, ls=":")
    ax.set_axisbelow(True)
    ax.set_xticks([5e4, 1e5, 1e6, 4e6])
    ax.set_xticklabels(["50k", "100k", "1M", "4M"])
    ax.set_xticks(n, minor=True)
    ax.set_xticklabels([], minor=True)

    ax2 = ax.twinx()
    ax2.set_yscale("log")
    ax2.plot(n, rel_far, marker="s", ms=4, lw=1.4, ls="-", color=C_RELL2,
             mfc="none", label=r"far-field rel. $L_2$ error (right)")
    # The error bars are substantive, not decoration: the Hutchinson estimator's
    # own stderr is ~13% of the value while the spread across N is ~+-10%, so the
    # bars are what turn "looks flat" into "flat to within the uncertainty".
    # Open markers and a dashed line keep this distinguishable from rel_far,
    # which it nearly coincides with -- that near-coincidence is the point.
    ax2.errorbar(n, rel_asym, yerr=asym_err, marker="^", ms=4, lw=1.4, ls="--",
                 color=C_ASYM, capsize=2, mfc="none",
                 label="grand-operator asymmetry (right)")
    ax2.set_ylabel("Relative error / asymmetry")

    if prmse_ref:
        ax2.axhline(prmse_ref, color="0.45", lw=1.0, ls=(0, (4, 3)))
        ax2.text(n[-1] * 1.35, prmse_ref * 1.3, "learned near-field PRMSE",
                 fontsize=6, color="0.35", va="bottom", ha="right")
        # Pinned, not autoscaled: a 2.3e-4..5.0e-4 spread left to fill the axis
        # reads as a rising trend, which is the opposite of the message.
        ax2.set_ylim(1e-5, 3e-1)
    else:
        ax2.set_ylim(1e-5, 1e-2)

    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    # Last, and on the shared axis: the twin's own autoscale fires when it is
    # plotted on and would otherwise stretch x out to the largest major tick.
    # Margins so the end markers are not clipped by the spines.
    ax.set_xlim(n[0] * 0.7, n[-1] * 1.45)

    # A second legend() call replaces the first, so merge the two axes' handles
    # and draw once -- on the twin, so it sits above its lines.
    # Lower left is the one region all three series stay out of: the two error
    # curves sit at ~4e-4 (about a third of the way up the log axis) and the
    # throughput curve only climbs from there.
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=6, frameon=False,
               labelspacing=0.35, handlelength=2.0, borderaxespad=0.5)

    if save:
        _save(fig, "component_far_field")
    return fig


def component_performance(save=True, show=False, protocol="matched",
                          mode="compile", prmse_ref=0.075):
    figs = (pair_kernel_figure(save=save, protocol=protocol, mode=mode),
            far_field_figure(save=save, prmse_ref=prmse_ref))
    if show:
        plt.show()
    return figs


if __name__ == "__main__":
    component_performance()
