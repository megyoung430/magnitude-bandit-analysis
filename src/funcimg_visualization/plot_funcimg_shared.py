"""Figures for the shared-signal analysis: do cells share signal beyond the background?"""

import matplotlib.pyplot as plt
import numpy as np

from src.funcimg_analysis.shared_signal import NULL_PERCENTILE, paired_comparison
from src.funcimg_visualization.plot_funcimg_responses import (
    CHANNEL_TITLES,
    plot_pulse_responses,
    window_tag,
)

CHANNELS = ("green", "red")
# Okabe-Ito colours, which stay distinguishable under the common colour-vision deficiencies;
# runs beyond the eighth reuse them in order.
RUN_COLORS = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#F0E442",
    "#000000",
)
RAW_MEASURE_TITLES = {
    "F": "raw F (not neuropil-corrected)",
    "dF/F0": "ΔF/F0 (neuropil-corrected)",
}


def settings_tag(
    detrend_window_s: float, n_shifts: int, min_shift_s: float, cells_only: bool
) -> str:
    """The settings that shape the shared-signal figures, for file names."""
    rois = "cellsonly" if cells_only else "allrois"
    return f"detrend{detrend_window_s:g}s_shifts{n_shifts}_min{min_shift_s:g}s_{rois}"


def shared_png_name(kind: str, file_tag: str, variant: str, tag: str) -> str:
    """File name of a shared-signal figure (``kind``: ``scatter``, ``partial_r`` or ``r2``)."""
    return f"shared_{kind}_{file_tag}_{variant}_{tag}.png"


def residual_png_name(
    file_tag: str, variant: str, x_units: str, window_s: tuple[float, float], tag: str
) -> str:
    """File name of the original vs background-removed response figure."""
    return (
        f"shared_residual_responses_{file_tag}_{variant}_{x_units}_{window_tag(window_s)}_{tag}.png"
    )


def _run_label(run: dict) -> str:
    return f"{run['session'].split('_')[0]} {run['run']}"


def _color(index: int) -> str:
    return RUN_COLORS[index % len(RUN_COLORS)]


def _panel_grid(axes, measures: tuple[str, ...]):
    """The ``(2, len(measures))`` axes (rows green, red): the given ones, or a new figure."""
    shape = (len(CHANNELS), len(measures))
    if axes is None:
        _, axes = plt.subplots(
            *shape, figsize=(5.2 * shape[1], 4.6 * shape[0]), constrained_layout=True, squeeze=False
        )
    axes = np.asarray(axes)
    if axes.shape != shape:
        raise ValueError(f"Need a {shape[0]} x {shape[1]} array of axes, got {axes.shape}.")
    return axes


def _panel_title(channel: str, measure: str) -> str:
    return f"{CHANNEL_TITLES[channel]}: {RAW_MEASURE_TITLES[measure]}"


def plot_scatter_cell_vs_background(runs: list[dict], measures: tuple[str, ...], axes=None):
    """Each cell's r with the leave-one-out cell average (y) against its r with the background (x).

    Points above the identity line are cells that resemble the other cells more than the
    background. Colours are runs. The text gives, per run, the median difference (y - x) and a
    paired Wilcoxon p across cells, which is descriptive: cells within a run are not independent.

    Args:
        runs (list[dict]): Results with ``stats`` (``{(channel, measure): shared_signal_stats}``),
            ``session`` and ``run``.
        measures (tuple[str, ...]): Raw measures (``"F"``, ``"dF/F0"``), one column each.
        axes: A ``(2, len(measures))`` array of axes (rows green, red), or None for a new figure.

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    axes = _panel_grid(axes, measures)
    for row, channel in enumerate(CHANNELS):
        for col, measure in enumerate(measures):
            ax, lines = axes[row, col], []
            for index, run in enumerate(runs):
                stats = run["stats"][(channel, measure)]
                ax.scatter(
                    stats["r_bg"],
                    stats["r_loo"],
                    s=12,
                    alpha=0.7,
                    color=_color(index),
                    linewidths=0,
                    label=_run_label(run),
                )
                test = paired_comparison(stats["r_loo"], stats["r_bg"])
                lines.append(
                    f"{run['run']}: median diff {test['median_difference']:+.2f}, "
                    f"Wilcoxon p={test['p']:.1e}, n={test['n']}"
                )
            ax.plot([-1, 1], [-1, 1], color="0.5", linewidth=0.8, linestyle="--")
            ax.axhline(0, color="0.85", linewidth=0.6)
            ax.axvline(0, color="0.85", linewidth=0.6)
            ax.set_xlim(-0.4, 1)
            ax.set_ylim(-0.4, 1)
            ax.set_aspect("equal")
            ax.set_xlabel("r(cell, background)")
            ax.set_ylabel("r(cell, average of the other cells)")
            ax.set_title(_panel_title(channel, measure), fontsize=10)
            ax.text(
                0.98,
                0.02,
                "\n".join(lines),
                transform=ax.transAxes,
                va="bottom",
                ha="right",
                fontsize=7,
                color="0.25",
            )
    axes[0, -1].legend(fontsize=8, frameon=False, loc="upper left")
    axes[0, 0].figure.suptitle(
        "Cell vs other cells, against cell vs background\n"
        "(descriptive; cells within a run are not independent)"
    )
    return axes[0, 0].figure


def _histogram_panels(runs, measures, axes, values, xlabel, null_key=None, bins=None, suptitle=""):
    """Per-run step histograms of ``values(stats)`` in the green/red x measure grid."""
    axes = _panel_grid(axes, measures)
    for row, channel in enumerate(CHANNELS):
        for col, measure in enumerate(measures):
            ax = axes[row, col]
            for index, run in enumerate(runs):
                stats = run["stats"][(channel, measure)]
                data = values(stats)
                data = data[np.isfinite(data)]
                ax.hist(
                    data,
                    bins=bins,
                    histtype="step",
                    linewidth=1.4,
                    color=_color(index),
                    label=f"{_run_label(run)} (median {np.median(data):.2f})"
                    if len(data)
                    else None,
                )
                if null_key is not None:
                    chance = np.nanpercentile(stats[null_key], NULL_PERCENTILE)
                    ax.axvline(chance, color=_color(index), linewidth=1.0, linestyle="--")
            ax.axvline(0, color="0.5", linewidth=0.8)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Cells")
            ax.set_title(_panel_title(channel, measure), fontsize=10)
    axes[0, -1].legend(fontsize=8, frameon=False)
    axes[0, 0].figure.suptitle(suptitle)
    return axes[0, 0].figure


def plot_partial_r_histograms(runs: list[dict], measures: tuple[str, ...], axes=None):
    """Partial r(cell, other cells | background) per run, with the chance level of each run.

    The dashed line of a run is the 95th percentile of the null: the same statistic with the
    other cells' average circularly shifted in time, pooled over cells and shifts. Mass to its
    right means cells share signal beyond the background.

    Args:
        runs (list[dict]): See :func:`plot_scatter_cell_vs_background`.
        measures (tuple[str, ...]): Raw measures, one column each.
        axes: A ``(2, len(measures))`` array of axes, or None for a new figure.

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    return _histogram_panels(
        runs,
        measures,
        axes,
        lambda stats: stats["partial_r"],
        "Partial r(cell, other cells | background)",
        null_key="null_partial_average",
        bins=np.linspace(-0.5, 1, 31),
        suptitle="Cells' shared signal after removing the background\n"
        "(dashed: 95th percentile of the shifted null)",
    )


def plot_r2_histograms(runs: list[dict], measures: tuple[str, ...], axes=None):
    """R^2 of each cell regressed on the background (detrended z traces), per run.

    Args:
        runs (list[dict]): See :func:`plot_scatter_cell_vs_background`.
        measures (tuple[str, ...]): Raw measures, one column each.
        axes: A ``(2, len(measures))`` array of axes, or None for a new figure.

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    return _histogram_panels(
        runs,
        measures,
        axes,
        lambda stats: stats["r2"],
        "R² of cell ~ a + b · background",
        bins=np.linspace(0, 1, 26),
        suptitle="Variance of each cell explained by the background",
    )


def plot_residual_responses(
    original: dict,
    residual: dict,
    offsets: np.ndarray,
    period_s: float,
    x_units: str = "s",
    measures: tuple[str, ...] = ("F z", "dF/F0 z"),
    axes=None,
    title: str = "",
):
    """Pooled reward responses of the detrended cells next to the same with the background removed.

    One row per measure; columns are green original, green residual, red original, red residual,
    all on the same y limits so the sizes can be compared directly.

    Args:
        original (dict): Pooled responses of the detrended cells (``pool_runs`` output).
        residual (dict): Pooled responses after removing the background's fit.
        offsets (np.ndarray): Window offsets in frames.
        period_s (float): Frame period (s).
        x_units (str): ``"s"`` or ``"frames"``.
        measures (tuple[str, ...]): Rows: ``"F z"`` and/or ``"dF/F0 z"``.
        axes: A ``(len(measures), 4)`` array of axes, or None for a new figure.
        title (str): Figure title.

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    shape = (len(measures), 4)
    if axes is None:
        _, axes = plt.subplots(
            *shape, figsize=(18, 4 * shape[0]), sharex=True, constrained_layout=True, squeeze=False
        )
    axes = np.asarray(axes)
    if axes.shape != shape:
        raise ValueError(f"Need a {shape[0]} x {shape[1]} array of axes, got {axes.shape}.")
    for pooled, columns, name in ((original, [0, 2], "Original"), (residual, [1, 3], "Residual")):
        plot_pulse_responses(
            pooled, offsets, period_s, x_units, axes[:, columns], measures=measures
        )
        for ax in axes[:, columns].ravel():
            channel, measure = ax.get_title().split(": ", 1)
            ax.set_title(f"{name}, {channel}\n{measure}", fontsize=8)
    for row in axes:
        low = min(ax.get_ylim()[0] for ax in row)
        high = max(ax.get_ylim()[1] for ax in row)
        for ax in row:
            ax.set_ylim(low, high)
    axes[0, 0].figure.suptitle(
        title or "Reward responses of detrended cells, original and with the background removed"
    )
    return axes[0, 0].figure
