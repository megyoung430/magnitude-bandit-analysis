"""Figures of the transient check: histograms of green against red, and example traces."""

import matplotlib.pyplot as plt
import numpy as np

from src.funcimg_analysis.transients import most_skewed
from src.funcimg_visualization.plot_funcimg_responses import middle_frames

GREEN, RED = "#009e73", "#d55e00"  # distinguishable under common colour-vision deficiencies
N_BINS = 30


def transients_png_name(
    kind: str, file_tag: str, variant: str, z_threshold: float, min_gap_s: float
) -> str:
    """File name of a transient-check figure; the thresholds that shape it are in it."""
    return f"transients_{kind}_{file_tag}_{variant}_z{z_threshold:g}_gap{min_gap_s:g}s.png"


def _pooled(runs: list[dict], channel: str, key: str) -> np.ndarray:
    values = np.concatenate([run[channel][key] for run in runs])
    return values[np.isfinite(values)]


def _overlaid(ax, green: np.ndarray, red: np.ndarray, xlabel: str) -> None:
    """Green and red histograms on shared bins, counts of cells."""
    both = np.concatenate([green, red])
    bins = np.linspace(both.min(), both.max(), N_BINS + 1) if both.size else N_BINS
    ax.hist(green, bins=bins, color=GREEN, alpha=0.6, label=f"green (n={len(green)})")
    ax.hist(red, bins=bins, color=RED, alpha=0.6, label=f"red, no indicator (n={len(red)})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Cells")
    ax.spines[["top", "right"]].set_visible(False)


def plot_transient_histograms(runs: list[dict], axes=None, title: str = ""):
    """Histograms over cells of skewness, and of positive minus negative events per minute.

    Cells of all *runs* are pooled. The skewness panel marks the 95th percentile of the red
    cells, the level that noise and artefacts alone reach.

    Args:
        runs (list[dict]): ``run_transients`` outputs.
        axes: Two axes, or None for a new figure.
        title (str): Figure title.

    Returns:
        matplotlib.figure.Figure: The figure.

    Raises:
        ValueError: If *axes* does not have two axes.
    """
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    if len(axes) != 2:
        raise ValueError(f"Need 2 axes, got {len(axes)}.")
    skew_ax, rate_ax = axes
    green_skew, red_skew = (_pooled(runs, c, "skew") for c in ("green", "red"))
    _overlaid(skew_ax, green_skew, red_skew, "Skewness of detrended robust-z ΔF/F0")
    if red_skew.size:
        skew_ax.axvline(
            np.percentile(red_skew, 95), color=RED, linestyle="--", linewidth=1, label="red 95th"
        )
    skew_ax.legend(fontsize=8, frameon=False)
    green_rate, red_rate = (_pooled(runs, c, "rate_diff") for c in ("green", "red"))
    _overlaid(rate_ax, green_rate, red_rate, "Positive minus negative events per min")
    rate_ax.axvline(0, color="0.4", linewidth=0.8)
    axes[0].figure.suptitle(title or "Transients: green cells against the red reference")
    return axes[0].figure


def plot_transient_examples(
    run: dict, stats: dict, n: int = 6, n_frames_shown: int | None = 2000, title: str = ""
):
    """Corrected ΔF/F0 of the *n* most positively skewed green cells, each with its red below.

    Fast rise and slow decay in green, and nothing like it in red, is what a transient looks like;
    motion artefacts show in both channels. Cells without a finite red skewness are not shown.

    Args:
        run (dict): One ``collect_residual_dff`` result.
        stats (dict): ``run_transients`` of the run.
        n (int): Number of cells.
        n_frames_shown (int | None): Frames drawn, centred in the run; None draws every frame.
        title (str): Figure title.

    Returns:
        matplotlib.figure.Figure: Two axes per cell, green over red, sharing the x axis.
    """
    paired = np.where(np.isfinite(stats["red"]["skew"]), stats["green"]["skew"], np.nan)
    cells = most_skewed(paired, n)
    time_s = np.asarray(run["time_s"])
    frames = middle_frames(len(time_s), n_frames_shown)
    fig, axes = plt.subplots(
        2 * len(cells), 1, figsize=(12, 1.3 * 2 * len(cells) + 1), sharex=True,
        constrained_layout=True, squeeze=False,
    )  # fmt: skip
    axes = axes[:, 0]
    for pair, cell in enumerate(cells):
        for ax, channel, color in (
            (axes[2 * pair], "green", GREEN),
            (axes[2 * pair + 1], "red", RED),
        ):
            ax.plot(
                time_s[frames],
                run["channels"][channel]["dff"][cell][frames],
                color=color,
                linewidth=0.7,
            )
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=7)
        skew = stats["green"]["skew"][cell]
        axes[2 * pair].set_ylabel(f"ROI {cell}\ngreen (skew {skew:.1f})", fontsize=7)
        axes[2 * pair + 1].set_ylabel("red", fontsize=7)
    axes[-1].set_xlabel("pyControl time (s)")
    fig.suptitle(title or "Most positively skewed green cells, with the same cells' red channel")
    return fig
