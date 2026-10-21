"""Figure of a run's most reward-responsive cells: small multiples, heatmap and field of view."""

import math

import matplotlib.pyplot as plt
import numpy as np

from src.funcimg_visualization.plot_funcimg_responses import (
    PULSE_COLORS,
    draw_fov,
    roi_colors,
    window_tag,
)

N_COLUMNS = 5
NOT_CROSS_VALIDATED = (
    "SELECTION BIAS: cells ranked and plotted on the same trials, not cross-validated"
)
HEATMAP_PERCENTILE = 99  # colour limits, so one extreme cell does not wash out the rest


def measure_tag(measure: str) -> str:
    """A measure name safe for a file name, e.g. ``bgcorrecteddFF0``."""
    return "".join(c for c in measure if c.isalnum())


def reward_cells_pdf_name(
    n: int,
    measure: str,
    contrast: str,
    rank_on: str,
    file_tag: str,
    variant: str,
    plot_window_s: tuple[float, float],
) -> str:
    """File name of the PDF of one measure's top cells; the settings that shape it are in it."""
    return (
        f"reward_cells_top{n}_{measure_tag(measure)}_{contrast}_{rank_on}_{file_tag}_{variant}_"
        f"{window_tag(plot_window_s)}.pdf"
    )


def _panel_title(run: dict, measure: dict, cell: int) -> str:
    p_cell = f"p={run['p_cell'][cell]:.2f}"
    if run["cross_validated"]:
        return (
            f"ROI {cell} | A {measure['index_a'][cell]:.2f} | B {measure['index_b'][cell]:.2f}"
            f" | {p_cell}"
        )
    return f"ROI {cell} | idx {measure['index_all'][cell]:.2f} | {p_cell}"


def draw_cell_panel(ax, run: dict, measure: dict, cell: int, color: str, title: str) -> None:
    """Mean ± SEM over the plotted trials of one cell, a line per pulse count, in seconds."""
    x = run["offsets"] * run["period_s"]
    for pulses, mean in sorted(measure["means"].items()):
        sem = measure["sems"][pulses][cell]
        ax.plot(x, mean[cell], color=PULSE_COLORS[pulses], linewidth=1.0)
        ax.fill_between(
            x,
            mean[cell] - sem,
            mean[cell] + sem,
            color=PULSE_COLORS[pulses],
            alpha=0.2,
            linewidth=0,
        )
    ax.axvline(0, color="0.5", linewidth=0.6, linestyle="--")
    ax.set_title(title, fontsize=7, color=color)
    ax.tick_params(labelsize=6)
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(color)


def draw_heatmap(ax, run: dict, measure: dict, n_marked: int) -> None:
    """Every ranked cell's trial-averaged rewarded response, best-ranked cell on top."""
    order = measure["order"]
    matrix = measure["rewarded_mean"][order]
    x = run["offsets"] * run["period_s"]
    limit = np.nanpercentile(np.abs(matrix), HEATMAP_PERCENTILE) if matrix.size else 1.0
    image = ax.imshow(
        matrix, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit,
        extent=(x[0], x[-1], len(order), 0), interpolation="nearest",
    )  # fmt: skip
    ax.axvline(0, color="0.3", linewidth=0.6, linestyle="--")
    ax.axhline(min(n_marked, len(order)), color="k", linewidth=0.6)
    ax.set_xlabel("Time from outcome (s)", fontsize=7)
    ax.set_ylabel("Cells, ranked", fontsize=7)
    ax.tick_params(labelsize=6)
    half = "half A" if run["cross_validated"] else "all trials"
    shown = "half B" if run["cross_validated"] else "all trials"
    ax.set_title(
        f"Rewarded response ({shown}), cells sorted by index on {half}; line = top {n_marked}",
        fontsize=7,
    )
    ax.figure.colorbar(image, ax=ax, shrink=0.8).set_label(
        "robust z, baseline-subtracted", fontsize=7
    )


def _summary(run: dict, measure: dict) -> str:
    null = measure["null"]
    if not run["cross_validated"]:
        return (
            f"{null['n_exceed_all']} of {null['n_cells']} cells exceed their shifted-outcome 95th "
            f"percentile on all trials (chance: {null['expected_all']:.1f}); "
            "no held-out trials to confirm them."
        )
    return (
        f"Half A: {null['n_exceed_a']} of {null['n_cells']} cells exceed their shifted-outcome "
        f"95th percentile (chance: {null['expected_a']:.1f}); of those, "
        f"{null['n_exceed_b_of_a']} also on half B (chance: {null['expected_b_of_a']:.1f}). "
        f"Spearman rank correlation of the indices between halves: "
        f"{measure['rank_correlation']:.2f}. Contrast: {run['contrast']}."
    )


def plot_reward_cells(run: dict, measure_name: str, n_cells: int, title: str = ""):
    """The top cells of one measure of a run: a panel each, a heatmap of all cells, and the field.

    Cells are ranked by the response index on half A and drawn from half B (cross-validated), or
    ranked and drawn on all trials, in which case the figure says so in red.

    Args:
        run (dict): One entry of ``collect_reward_cells``.
        measure_name (str): Which measure of the run to draw.
        n_cells (int): Number of top cells to give a panel.
        title (str): Figure title, e.g. the session and run.

    Returns:
        matplotlib.figure.Figure: The figure.

    Raises:
        KeyError: If the run has no such measure.
    """
    measure = run["measures"][measure_name]
    top = measure["order"][:n_cells]
    colors = roi_colors(len(top))
    n_rows = max(1, math.ceil(len(top) / N_COLUMNS))
    fig = plt.figure(figsize=(24, 13), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[3, 1.3], height_ratios=[30, 1])
    cells = grid[0, 0].subgridspec(n_rows, N_COLUMNS)
    side = grid[0, 1].subgridspec(2, 1)
    first = None
    for i, (cell, color) in enumerate(zip(top, colors, strict=True)):
        ax = fig.add_subplot(
            cells[i // N_COLUMNS, i % N_COLUMNS], sharex=first, sharey=first, label="cell"
        )
        first = first or ax
        draw_cell_panel(ax, run, measure, int(cell), color, _panel_title(run, measure, int(cell)))
        if i % N_COLUMNS == 0:
            ax.set_ylabel("robust z", fontsize=7)
        if i // N_COLUMNS == n_rows - 1:
            ax.set_xlabel("Time from outcome (s)", fontsize=7)
    draw_heatmap(fig.add_subplot(side[0], label="heatmap"), run, measure, n_cells)
    fov = fig.add_subplot(side[1], label="fov")
    draw_fov(
        fov,
        {
            "cells": np.asarray(top, dtype=int),
            "mean_image": run["mean_image"],
            "labels": run["labels"],
        },
        colors,
    )
    for pulses in sorted(PULSE_COLORS):
        (first or fov).plot([], [], color=PULSE_COLORS[pulses], label=f"{pulses} pulses")
    (first or fov).legend(fontsize=6, frameon=False, ncols=5, loc="upper right")
    heading = f"{title} {measure_name}: top {len(top)} cells".strip()
    if run["cross_validated"]:
        fig.suptitle(f"{heading} (ranked on half A, shown on half B)", fontsize=11)
    else:
        fig.suptitle(f"{heading}\n{NOT_CROSS_VALIDATED}", fontsize=11, color="#B00020")
    note = fig.add_subplot(grid[1, :], label="summary")
    note.set_axis_off()
    note.text(0, 0.5, _summary(run, measure), fontsize=8, ha="left", va="center")
    return fig
