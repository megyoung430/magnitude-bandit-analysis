"""Figures for the background (non-ROI) fluorescence: its reward response and the pixels used."""

import matplotlib.pyplot as plt
import numpy as np

from src.funcimg_visualization.plot_funcimg_responses import (
    CHANNEL_TITLES,
    PULSE_COLORS,
    outline,
    plot_pulse_responses,
    window_x,
)

MASK_COLOR = (1.0, 0.55, 0.0)  # orange over the grey mean image
MASK_ALPHA = 0.35


def _share_ylim(first, second) -> None:
    low = min(first.get_ylim()[0], second.get_ylim()[0])
    high = max(first.get_ylim()[1], second.get_ylim()[1])
    first.set_ylim(low, high)
    second.set_ylim(low, high)


def _draw_run_lines(axes_row, per_run: list[dict], measure: str, x: np.ndarray) -> None:
    """One thin line per run and pulse count, under the pooled mean."""
    for ax, channel in zip(axes_row, ("green", "red"), strict=True):
        for means in per_run:
            for pulses, trace in means[(channel, measure)].items():
                ax.plot(x, trace[0], color=PULSE_COLORS[pulses], linewidth=0.5, alpha=0.5)


def plot_background_responses(
    background: dict,
    cells: dict | None,
    offsets: np.ndarray,
    period_s: float,
    x_units: str = "s",
    per_run: list[dict] | None = None,
    title: str = "",
    measures: tuple[str, ...] = ("F", "F z"),
):
    """The background's reward response (raw F, then robust z), and the cells' z beneath it.

    The background rows are drawn by ``plot_pulse_responses`` like the cell figure: one line per
    number of reward pulses, mean ± SEM across runs (one background trace per run). The cells'
    robust-z row shares its y limits with the background's z row in each channel, so the two
    can be compared directly.

    Args:
        background (dict): Pooled background views, from ``collect_background_responses``.
        cells (dict | None): Pooled cell views holding ``(channel, "F z")``, or None for no
            cell row.
        offsets (np.ndarray): Window offsets in frames.
        period_s (float): Frame period (s).
        x_units (str): ``"s"`` or ``"frames"``.
        per_run (list[dict] | None): Each run's ``means`` for a thin line per run, or None.
        title (str): Figure title.
        measures (tuple[str, ...]): Background rows, ``"F"`` and/or ``"F z"``.

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    n_rows = len(measures) + (cells is not None)
    fig, axes = plt.subplots(
        n_rows, 2, figsize=(11, 4 * n_rows), sharex=True, constrained_layout=True
    )
    axes = np.asarray(axes).reshape(n_rows, 2)
    bg_axes = axes[: len(measures)]
    plot_pulse_responses(
        background, offsets, period_s, x_units, bg_axes, measures=measures, unit=None
    )
    x = window_x(np.asarray(offsets), period_s, x_units)[0]
    for row, measure in enumerate(measures):
        for ax in bg_axes[row]:
            ax.set_title("Background: " + ax.get_title().split(": ", 1)[1])
        if per_run:
            _draw_run_lines(bg_axes[row], per_run, measure, x)
    if cells is not None:
        cell_axes = axes[len(measures) :]
        plot_pulse_responses(cells, offsets, period_s, x_units, cell_axes, measures=("F z",))
        for ax in cell_axes[0]:
            ax.set_title("Cells: " + ax.get_title().split(": ", 1)[1])
        if "F z" in measures:
            z_row = measures.index("F z")
            for col in range(2):
                _share_ylim(bg_axes[z_row, col], cell_axes[0, col])
    fig.suptitle(title or "Outcome-aligned background F; mean ± SEM across runs")
    return fig


def plot_background_mask(
    mean_images: dict,
    mask: np.ndarray,
    labels: np.ndarray,
    axes=None,
    title: str = "",
):
    """Mean images (green and red) with the background pixels shaded and ROIs outlined.

    Args:
        mean_images (dict): ``{"green": (Ly, Lx), "red": (Ly, Lx)}``.
        mask (np.ndarray): Boolean ``(Ly, Lx)`` of the pixels in the background.
        labels (np.ndarray): ROI label image (0 = none) whose boundaries are outlined.
        axes: Two axes (green, red), or None for a new figure.
        title (str): Figure title, e.g. the session and run.

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    axes = np.asarray(axes).ravel()
    fig = axes[0].figure
    shade = np.zeros((*mask.shape, 4))
    shade[mask] = (*MASK_COLOR, MASK_ALPHA)
    roi_edges = np.zeros((*mask.shape, 4))
    roi_edges[outline(labels)] = (0.0, 0.9, 1.0, 1.0)
    n = int(mask.sum())
    summary = f"{n} background pixels, {100 * n / mask.size:.1f}% of the FOV"
    for ax, channel in zip(axes, ("green", "red"), strict=True):
        image = mean_images[channel]
        low, high = np.nanpercentile(image, [1, 99.5])
        ax.imshow(image, cmap="gray", vmin=low, vmax=high)
        ax.imshow(shade)
        ax.imshow(roi_edges)
        ax.set_axis_off()
        ax.set_title(f"{CHANNEL_TITLES[channel]} mean image\n{summary}")
    fig.suptitle(title)
    return fig
