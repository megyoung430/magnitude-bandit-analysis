"""Heatmaps and psychometric curves of P(best) / P(second) against option values.

Drawing only: the numbers come from :mod:`src.behavior_analysis.value_psychometrics`.
Each panel function takes ``ax=None`` and draws into it (making its own figure only if none is
given); the ``plot_*`` wrappers build a figure, then save it as PNG and PDF or show it.

Heatmap cells print the probability and, on a second line, the number of trials behind it
(``n=``). The count sits in the cell rather than in a companion heatmap so that a reader sees
each probability next to its evidence, and the per-mouse figure stays two panels wide.
"""

from contextlib import contextmanager
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.behavior_analysis.get_choice_probs_by_value import BEST_VALS, SECOND_VALS
from src.behavior_analysis.value_psychometrics import Curve, HeatmapTables
from src.behavior_visualization.plot_helpers import axes_or_new, despine, save_or_show
from src.behavior_visualization.plot_style import CHOICE_PROB_COLOR_MAP

COLOR_BEST = CHOICE_PROB_COLOR_MAP["prev_best"]
COLOR_SECOND = CHOICE_PROB_COLOR_MAP["next_best"]

HEATMAP_CMAP = "viridis"
EMPTY_CELL_COLOR = "#e6e6e6"

# Text height in data units (a 10 pt label on a ~4 inch tall panel) used to keep labels inside 0-1.
_LABEL_HEIGHT = 0.05
_LABEL_PAD = 0.05

PANELS = (
    ("best_val", "Value of Best"),
    ("second_val", "Value of Second"),
    ("diff", "Value of Best - Second"),
)


@contextmanager
def _new_figure(**subplots_kwargs):
    """Yield ``(fig, axes)`` of a new figure; the figure is closed if the body raises."""
    fig, axes = plt.subplots(**subplots_kwargs)
    try:
        yield fig, axes
    except BaseException:
        plt.close(fig)
        raise


def annotation_position(mean: float, *, above: bool) -> tuple[float, str]:
    """Where to put a mean's label so the whole label stays inside 0-1.

    The label goes above the point if *above*, else below, unless there is no room, when it
    flips to the other side.

    Args:
        mean (float): The annotated value, in [0, 1].
        above (bool): Preferred side.

    Returns:
        tuple[float, str]: The text's y position and its vertical alignment
        (``"bottom"`` when drawn above the point, ``"top"`` when below).
    """
    room = {
        True: mean + _LABEL_PAD + _LABEL_HEIGHT <= 1.0,
        False: mean - _LABEL_PAD - _LABEL_HEIGHT >= 0.0,
    }
    place_above = above if room[above] else not above
    if place_above:
        return mean + _LABEL_PAD, "bottom"
    return mean - _LABEL_PAD, "top"


# ---------------------------------------------------------------------------
# Heatmaps
# ---------------------------------------------------------------------------


def _cell_text_color(value: float, cmap) -> str:
    """White on dark cells, black on light ones (by the colour's luminance)."""
    r, g, b, _ = cmap(value)
    return "white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.5 else "black"


def _cell_label(value: float, n_trials: float) -> str:
    """``"0.62\\nn=134"``, or ``"NA\\nn=0"`` for a cell without trials."""
    prob = "NA" if np.isnan(value) else f"{value:.2f}"
    return f"{prob}\nn={n_trials:g}"


def plot_prob_heatmap(
    prob: pd.DataFrame,
    n_trials: pd.DataFrame,
    title: str,
    ax: plt.Axes | None = None,
    annotate: bool = True,
) -> plt.Axes:
    """Heatmap of a best value x second value probability table on a fixed 0-1 colour range.

    Best value 4 is the top row. Cells without trials are grey.

    Args:
        prob (pd.DataFrame): Probabilities, NaN where a cell has no trials.
        n_trials (pd.DataFrame): Trials behind each cell, same layout as *prob*.
        title (str): Axes title.
        ax (plt.Axes | None): Axes to draw into; a new figure if ``None``.
        annotate (bool): Print probability and trial count in each cell.

    Returns:
        plt.Axes: The axes drawn into.
    """
    prob = prob.reindex(index=BEST_VALS, columns=SECOND_VALS)
    n_trials = n_trials.reindex(index=BEST_VALS, columns=SECOND_VALS)
    values = np.flipud(prob.to_numpy(dtype=float))
    counts = np.flipud(n_trials.to_numpy(dtype=float))
    cmap = matplotlib.colormaps[HEATMAP_CMAP].with_extremes(bad=EMPTY_CELL_COLOR)
    with axes_or_new(ax, figsize=(4.6, 4.2)) as ax:
        image = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=0.0, vmax=1.0)
        ax.set_title(title)
        ax.set_xlabel("Second Value")
        ax.set_ylabel("Best Value")
        ax.set_xticks(range(len(SECOND_VALS)), SECOND_VALS)
        ax.set_yticks(range(len(BEST_VALS)), list(reversed(BEST_VALS)))
        if annotate:
            for i, j in np.ndindex(values.shape):
                color = "black" if np.isnan(values[i, j]) else _cell_text_color(values[i, j], cmap)
                ax.text(
                    j, i, _cell_label(values[i, j], counts[i, j]),
                    ha="center", va="center", fontsize=9, color=color,
                )  # fmt: skip
        ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    return ax


def plot_overall_heatmaps(
    tables: HeatmapTables, value_label: str, save_path: str | Path | None = None
) -> None:
    """Across-mice P(best) and P(second) heatmaps side by side; save or show.

    Args:
        tables (HeatmapTables): From ``heatmap_tables``.
        value_label (str): ``"True"`` or ``"Believed"``, for the titles.
        save_path (str | Path | None): Base path without extension for ``.png`` and ``.pdf``;
            ``None`` or ``""`` shows the figure.
    """
    n_mice = len(tables.per_mouse_p_best)
    with _new_figure(ncols=2, figsize=(10, 4.4)) as (fig, axs):
        for ax, prob, name in zip(
            axs, (tables.p_best, tables.p_second), ("Best", "Second"), strict=True
        ):
            plot_prob_heatmap(prob, tables.n_trials, f"P({name}) - {value_label} Value", ax=ax)
        fig.suptitle(f"Mean across {n_mice} mice (n = trials summed over mice)", fontsize=11)
        fig.tight_layout()
    save_or_show(fig, save_path or None)


def plot_per_mouse_heatmaps(
    tables: HeatmapTables,
    value_label: str,
    save_path: str | Path | None = None,
    mice: list[str] | None = None,
) -> None:
    """One row per mouse: P(best) and P(second) heatmaps; save or show.

    Args:
        tables (HeatmapTables): From ``heatmap_tables``.
        value_label (str): ``"True"`` or ``"Believed"``, for the titles.
        save_path (str | Path | None): Base path without extension; ``None`` or ``""`` shows.
        mice (list[str] | None): Mice to draw, in order; default all, sorted.

    Raises:
        ValueError: If there is no mouse to draw.
    """
    mice = sorted(tables.per_mouse_p_best) if mice is None else list(mice)
    if not mice:
        raise ValueError("no mice to draw per-mouse heatmaps for")
    with _new_figure(nrows=len(mice), ncols=2, figsize=(10, 4.4 * len(mice)), squeeze=False) as (
        fig,
        axs,
    ):
        for row, subject in enumerate(mice):
            n = tables.per_mouse_n[subject]
            for col, (name, probs) in enumerate(
                (("Best", tables.per_mouse_p_best), ("Second", tables.per_mouse_p_second))
            ):
                plot_prob_heatmap(probs[subject], n, f"P({name}) - {subject}", ax=axs[row, col])
        fig.suptitle(f"{value_label} Value", fontsize=12)
        fig.tight_layout()
    save_or_show(fig, save_path or None)


# ---------------------------------------------------------------------------
# Psychometric curves
# ---------------------------------------------------------------------------


def _draw_mouse_lines(ax: plt.Axes, curve: Curve, color: str) -> None:
    """Thin line per mouse; NaN breaks the line and lone points stay visible as markers."""
    for values in curve.per_mouse:
        ax.plot(curve.x, values, "-o", color=color, lw=1, ms=2.5, alpha=0.3)


def _draw_mean(ax: plt.Axes, curve: Curve, color: str, label: str) -> None:
    """Thick mean line with markers, and the SE band where the SE is defined."""
    has_se = ~np.isnan(curve.se) & ~np.isnan(curve.mean)
    if has_se.any():
        ax.fill_between(
            curve.x, curve.mean - curve.se, curve.mean + curve.se,
            where=has_se, color=color, alpha=0.15, zorder=3, linewidth=0,
        )  # fmt: skip
    ax.plot(curve.x, curve.mean, "-o", color=color, lw=2.2, ms=5, label=label, zorder=5)


def _annotate_means(ax: plt.Axes, curve: Curve, color: str, *, above: bool) -> None:
    """Write each mean next to its point, inside the 0-1 range."""
    for x, mean in zip(curve.x, curve.mean, strict=True):
        if not np.isnan(mean):
            y, va = annotation_position(mean, above=above)
            ax.text(x, y, f"{mean:.2f}", ha="center", va=va, fontsize=10, color=color)


def plot_psychometric_panel(
    curve_best: Curve,
    curve_second: Curve,
    xlabel: str,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """P(best) and P(second) against one x axis: mice thin, mean thick, SE band, mean values.

    Args:
        curve_best (Curve): P(best) against x.
        curve_second (Curve): P(second) against the same x.
        xlabel (str): X axis label (also the title).
        ax (plt.Axes | None): Axes to draw into; a new figure if ``None``.

    Returns:
        plt.Axes: The axes drawn into.
    """
    series = (
        (curve_best, COLOR_BEST, "P(best)", True),
        (curve_second, COLOR_SECOND, "P(second)", False),
    )
    with axes_or_new(ax, figsize=(4.6, 4.2)) as ax:
        for curve, color, label, above in series:
            _draw_mouse_lines(ax, curve, color)
            _draw_mean(ax, curve, color, label)
            _annotate_means(ax, curve, color, above=above)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("Choice Probability", fontsize=12)
        ax.set_title(xlabel, fontsize=14)
        ax.set_ylim(0, 1)
        ax.set_xticks(curve_best.x)
        ax.legend(fontsize=10)
        despine(ax)
    return ax


def plot_psychometric_curves(
    curves: dict[str, dict[str, Curve]],
    value_label: str,
    pool_trials: bool,
    save_path: str | Path | None = None,
) -> None:
    """Three panels (vs best, second and best - second value) of P(best) and P(second).

    Args:
        curves (dict[str, dict[str, Curve]]): From ``psychometric_curves``.
        value_label (str): ``"True"`` or ``"Believed"``, for the title.
        pool_trials (bool): The weighting the curves were computed with, for the title.
        save_path (str | Path | None): Base path without extension; ``None`` or ``""`` shows.
    """
    n_mice = len(curves["best"]["best_val"].mice)
    weighting = "trials pooled within mouse" if pool_trials else "cell probabilities averaged"
    with _new_figure(ncols=3, figsize=(13, 4.4)) as (fig, axs):
        for ax, (axis, xlabel) in zip(axs, PANELS, strict=True):
            plot_psychometric_panel(curves["best"][axis], curves["second"][axis], xlabel, ax=ax)
        fig.suptitle(
            f"Psychometric Curves ({value_label} Value): {weighting}; mean and SE across "
            f"{n_mice} mice",
            fontsize=13,
        )
        fig.tight_layout()
    save_or_show(fig, save_path or None)
