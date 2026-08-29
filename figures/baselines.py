"""Baseline plotting utilities."""

from __future__ import annotations

from typing import Optional, Tuple

import matplotlib.pyplot as plt


def plot_two_bar_runtime(
	ours_seconds: float = 1.1,
	h_hignn_seconds: float = 28.0,
	save_path: Optional[str] = None,
	show: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
	"""Plot a two-bar runtime comparison.

	Bars are labeled "ours" and "H-Hignn" with the provided runtimes.
	"""
	assert ours_seconds > 0
	assert h_hignn_seconds > 0

	labels = ["ours", "H-Hignn"]
	values = [ours_seconds, h_hignn_seconds]

	fig, ax = plt.subplots(figsize=(4.5, 3.0), dpi=150)
	bars = ax.bar(labels, values, color=["#2E86AB", "#C06014"])

	ax.set_ylabel("Runtime (s)")
	ax.set_ylim(0, max(values) * 1.15)
	ax.bar_label(bars, fmt="%.1f s", padding=3)
	ax.grid(axis="y", linestyle="--", alpha=0.35)

	fig.tight_layout()

	fig.savefig("figures/baseline_runtime_comparison.png", bbox_inches="tight")

	return fig, ax


if __name__ == "__main__":
    plot_two_bar_runtime()