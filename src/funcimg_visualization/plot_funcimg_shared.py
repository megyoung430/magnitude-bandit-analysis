"""Figures for the shared-signal analysis: do cells share signal beyond the background?"""

from collections.abc import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from src.funcimg_analysis.event_frames import quiet_empty_means
from src.funcimg_analysis.shared_signal import paired_comparison
from src.funcimg_analysis.shared_signal_null import NULL_PERCENTILE
from src.funcimg_visualization.plot_funcimg_responses import (
    CHANNEL_TITLES,
    MEASURE_LABELS,
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
SCATTER_LIMITS = (-0.4, 1.0)
PARTIAL_BINS = np.linspace(-0.5, 1.0, 31)
R2_BINS = np.linspace(0.0, 1.0, 26)
TICK_FRACTION = 0.04  # length of the chance-level ticks on the scatter axes


def settings_tag(
    detrend_window_s: float,
    n_shifts: int,
    min_shift_s: float,
    cells_only: bool,
    fit_exclude_outcomes: bool = True,
    fit_exclude_window_s: tuple[float, float] | None = None,
) -> str:
    """The settings that shape the shared-signal figures, for file names."""
    rois = "cellsonly" if cells_only else "allrois"
    if not fit_exclude_outcomes:
        fit = "fitall"
    elif fit_exclude_window_s is None:
        fit = "fitoutside"
    else:
        fit = f"fitoutside{fit_exclude_window_s[0]:g}to{fit_exclude_window_s[1]:g}s"
    return f"detrend{detrend_window_s:g}s_shifts{n_shifts}_min{min_shift_s:g}s_{rois}_{fit}"


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


def _with_title(title: str, description: str) -> str:
    """*description* under the caller's *title*, if any."""
    return f"{title}\n{description}" if title else description


def _n_outside(values: Sequence[np.ndarray], low: float, high: float) -> int:
    """Number of cells with any finite value outside ``[low, high]`` (not drawn in a panel)."""
    stacked = np.stack([np.asarray(v, dtype=float) for v in values])
    with np.errstate(invalid="ignore"):
        outside = np.isfinite(stacked) & ((stacked < low) | (stacked > high))
    return int(outside.any(axis=0).sum())


def _panel_grid(axes, measures: tuple[str, ...]) -> np.ndarray:
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


def _pooled_null_level(null: np.ndarray, divisor: np.ndarray | None = None) -> float:
    """The pooled 95th percentile of a null ``(n_shifts, n_cells)``, optionally divided per cell."""
    values = null if divisor is None else null / divisor[None, :]
    with quiet_empty_means():
        return float(np.nanpercentile(np.clip(values, -1, 1), NULL_PERCENTILE))


def _scatter_one_run(ax, stats: dict, color: str, label: str) -> str:
    """Draw one run's cells and chance ticks into *ax*; return its line of text."""
    ax.scatter(
        stats["r_bg"],
        stats["r_loo_corrected"],
        s=12,
        alpha=0.7,
        color=color,
        linewidths=0,
        label=label,
    )
    ax.scatter(
        stats["r_bg"],
        stats["r_loo"],
        s=14,
        facecolors="none",
        edgecolors=color,
        linewidths=0.6,
        alpha=0.7,
    )
    if "null_r_bg" in stats:
        ax.axvline(_pooled_null_level(stats["null_r_bg"]), ymax=TICK_FRACTION, color=color, lw=1.4)
    if "null_r_loo" in stats:
        level = _pooled_null_level(stats["null_r_loo"], np.sqrt(stats["loo_reliability"]))
        ax.axhline(level, xmax=TICK_FRACTION, color=color, lw=1.4)
    corrected = paired_comparison(stats["r_loo_corrected"], stats["r_bg"])
    raw = paired_comparison(stats["r_loo"], stats["r_bg"])
    low, high = SCATTER_LIMITS
    outside = _n_outside([stats["r_bg"], stats["r_loo_corrected"], stats["r_loo"]], low, high)
    return (
        f"{label}: median diff {corrected['median_difference']:+.2f} corrected, "
        f"{raw['median_difference']:+.2f} raw; Wilcoxon p={corrected['p']:.1e} (corrected), "
        f"n={corrected['n']}" + (f"; {outside} cell(s) outside the axes" if outside else "")
    )


def plot_scatter_cell_vs_background(
    runs: list[dict], measures: tuple[str, ...], axes=None, title: str = ""
):
    """Each cell's r with the other cells' average (y) against its r with the background (x).

    The leave-one-out average of a few cells carries their private noise, which lowers every
    r(cell, average) below what the shared signal alone would give. Filled markers are therefore
    divided by the square root of the average's split-half (Spearman-Brown) reliability; hollow
    markers are the uncorrected values. Points above the identity line are cells that resemble
    the other cells more than the background. Colours are runs, and the short ticks on the
    axes are the 95th percentile of the circular-shift chance level of each correlation (x:
    background shifted; y: the other cells' average shifted, corrected like the markers).
    The text gives, per run, the median difference (y - x) and a paired Wilcoxon p across
    cells, which is descriptive: cells within a run are not independent. The background itself
    is noisy (see its split-half reliability in the partial-r figure), which lowers x too.

    Args:
        runs (list[dict]): Results with ``stats`` (``{(channel, measure): shared_signal_stats}``),
            ``session`` and ``run``.
        measures (tuple[str, ...]): Raw measures (``"F"``, ``"dF/F0"``), one column each.
        axes: A ``(2, len(measures))`` array of axes (rows green, red), or None for a new figure.
        title (str): Shown above the figure's own description.

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    axes = _panel_grid(axes, measures)
    for row, channel in enumerate(CHANNELS):
        for col, measure in enumerate(measures):
            ax = axes[row, col]
            lines = [
                _scatter_one_run(ax, run["stats"][(channel, measure)], _color(i), _run_label(run))
                for i, run in enumerate(runs)
            ]
            ax.plot([-1, 1], [-1, 1], color="0.5", linewidth=0.8, linestyle="--")
            ax.axhline(0, color="0.85", linewidth=0.6)
            ax.axvline(0, color="0.85", linewidth=0.6)
            ax.set_xlim(*SCATTER_LIMITS)
            ax.set_ylim(*SCATTER_LIMITS)
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
                fontsize=6.5,
                color="0.25",
            )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(
        Line2D([], [], marker="o", linestyle="", markerfacecolor="none", color="0.4", ms=4)
    )
    labels.append("not corrected for the average's noise")
    axes[0, -1].legend(handles, labels, fontsize=8, frameon=False, loc="upper left")
    axes[0, 0].figure.suptitle(
        _with_title(
            title,
            "Cell vs other cells (corrected for their noise), against cell vs background\n"
            "(descriptive; cells within a run are not independent)",
        )
    )
    return axes[0, 0].figure


def _histogram_panels(
    runs: list[dict],
    measures: tuple[str, ...],
    axes,
    series: Sequence[tuple[str, str, str | None]],
    xlabel: str,
    bins: np.ndarray,
    suptitle: str,
    title: str,
    note: Callable[[dict], str],
    style_legend: Sequence[Line2D] = (),
) -> Figure:
    """Per-run step histograms in the green/red x measure grid.

    *series* is ``(stats key, linestyle, null key or None)``: the null's pooled 95th percentile
    is drawn as a vertical line of the same colour in a matching dashed style when present.
    """
    axes = _panel_grid(axes, measures)
    null_styles = {"-": "--", ":": "-."}
    for row, channel in enumerate(CHANNELS):
        for col, measure in enumerate(measures):
            ax, lines = axes[row, col], []
            for index, run in enumerate(runs):
                stats = run["stats"][(channel, measure)]
                color = _color(index)
                for key, linestyle, null_key in series:
                    data = stats[key][np.isfinite(stats[key])]
                    ax.hist(
                        data,
                        bins=bins,
                        histtype="step",
                        linewidth=1.4,
                        color=color,
                        linestyle=linestyle,
                    )
                    if null_key is not None and null_key in stats:
                        ax.axvline(
                            float(np.nanpercentile(stats[null_key], NULL_PERCENTILE)),
                            color=color,
                            linewidth=1.0,
                            linestyle=null_styles[linestyle],
                        )
                lines.append(f"{_run_label(run)}: {note(stats)}")
                outside = _n_outside([stats[key] for key, *_ in series], bins[0], bins[-1])
                if outside:
                    lines[-1] += f"; {outside} cell(s) outside the bins"
            ax.axvline(0, color="0.5", linewidth=0.8)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Cells")
            ax.set_title(_panel_title(channel, measure), fontsize=10)
            ax.text(
                0.02,
                0.98,
                "\n".join(lines),
                transform=ax.transAxes,
                va="top",
                fontsize=6.5,
                color="0.25",
            )
    run_handles = [
        Line2D([], [], color=_color(i), label=_run_label(run)) for i, run in enumerate(runs)
    ]
    figure = axes[0, 0].figure
    # outside the panels, where it cannot cover the per-run text or the histograms
    figure.legend(
        handles=[*run_handles, *style_legend],
        fontsize=7,
        frameon=False,
        loc="outside lower center",
        ncol=len(run_handles) + len(style_legend),
    )
    figure.suptitle(_with_title(title, suptitle))
    return figure


def _median(values: np.ndarray) -> float:
    return float(np.nanmedian(values)) if np.isfinite(values).any() else float("nan")


def plot_partial_r_histograms(
    runs: list[dict], measures: tuple[str, ...], axes=None, title: str = ""
):
    """Partial r(cell, other cells | controls) per run, with the chance level of each run.

    Solid: given the background. Dotted: given the background and the cell's own and the other
    cells' neuropil. The dashed (dash-dotted) vertical line is the 95th percentile of the null
    of the solid (dotted) one: the same statistic with the other cells' average circularly
    shifted in time, pooled over cells and shifts. The null only tests partial r = 0. A noisy
    or spatially different background leaves some of the field signal in the cells, so partial
    r > 0 is not by itself evidence of shared neural signal; read it with the neuropil-controlled
    value, the background's split-half reliability (in the text) and the residual reward
    response.

    Args:
        runs (list[dict]): See :func:`plot_scatter_cell_vs_background`.
        measures (tuple[str, ...]): Raw measures, one column each.
        axes: A ``(2, len(measures))`` array of axes, or None for a new figure.
        title (str): Shown above the figure's own description.

    Returns:
        matplotlib.figure.Figure: The figure.
    """

    def note(stats: dict) -> str:
        return (
            f"median {_median(stats['partial_r']):.2f} (bg), "
            f"{_median(stats['partial_r_neuropil']):.2f} (bg + neuropil); "
            f"bg split-half r {stats['bg_half_r']:.2f}, reliability {stats['bg_reliability']:.2f}"
        )

    style = [
        Line2D([], [], color="0.3", linestyle="-", label="given the background"),
        Line2D([], [], color="0.3", linestyle=":", label="given background + neuropil"),
        Line2D([], [], color="0.3", linestyle="--", label="null 95th percentile (bg)"),
        Line2D([], [], color="0.3", linestyle="-.", label="null 95th percentile (bg + neuropil)"),
    ]
    return _histogram_panels(
        runs,
        measures,
        axes,
        series=(
            ("partial_r", "-", "null_partial_average"),
            ("partial_r_neuropil", ":", "null_partial_neuropil_average"),
        ),
        xlabel="Partial r(cell, other cells | controls)",
        bins=PARTIAL_BINS,
        suptitle="Cells' shared signal after removing the background (and neuropil)\n"
        "(the shifted null tests partial r = 0 only; an imperfect control leaves field signal)",
        title=title,
        note=note,
        style_legend=style,
    )


def plot_r2_histograms(runs: list[dict], measures: tuple[str, ...], axes=None, title: str = ""):
    """R^2 of each cell regressed on the background (detrended z traces), per run.

    Args:
        runs (list[dict]): See :func:`plot_scatter_cell_vs_background`.
        measures (tuple[str, ...]): Raw measures, one column each.
        axes: A ``(2, len(measures))`` array of axes, or None for a new figure.
        title (str): Shown above the figure's own description.

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    return _histogram_panels(
        runs,
        measures,
        axes,
        series=(("r2", "-", None),),
        xlabel="R² of cell ~ a + b · background",
        bins=R2_BINS,
        suptitle="Variance of each cell explained by the background",
        title=title,
        note=lambda stats: f"median {_median(stats['r2']):.2f}",
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
    # columns of each kind, green then red, as plot_pulse_responses lays out its two columns
    for pooled, columns, name in ((original, [0, 2], "Original"), (residual, [1, 3], "Residual")):
        plot_pulse_responses(
            pooled, offsets, period_s, x_units, axes[:, columns], measures=measures
        )
        for row, measure in enumerate(measures):
            for column, channel in zip(columns, CHANNELS, strict=True):
                axes[row, column].set_title(
                    f"{name}, {CHANNEL_TITLES[channel]}\n{MEASURE_LABELS[measure][0]}", fontsize=8
                )
    for row in axes:
        low = min(ax.get_ylim()[0] for ax in row)
        high = max(ax.get_ylim()[1] for ax in row)
        for ax in row:
            ax.set_ylim(low, high)
    axes[0, 0].figure.suptitle(
        title or "Reward responses of detrended cells, original and with the background removed"
    )
    return axes[0, 0].figure
