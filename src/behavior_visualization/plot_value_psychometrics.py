"""Heatmaps and psychometric curves of P(best) / P(second) against option values.

Drawing only: the numbers come from :mod:`src.behavior_analysis.value_psychometrics`.
Each panel function takes ``ax=None`` and draws into it (making its own figure only if none is
given); the ``plot_*`` wrappers build a figure, then save it (pdf, png, svg and a caption) or
show it.

Heatmap cells print the probability and, on a second line, the number of trials behind it
(``n=``). The count sits in the cell rather than in a companion heatmap so that a reader sees
each probability next to its evidence, and the per-mouse figure stays two panels wide.
"""

from contextlib import contextmanager
from pathlib import Path
from typing import NamedTuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.behavior_analysis.get_choice_probs_by_value import (
    BEST_VALS,
    SECOND_VALS,
    TIE_EXCLUDE,
    TIE_MODES,
)
from src.behavior_analysis.value_psychometrics import Curve, HeatmapTables
from src.behavior_visualization.plot_helpers import (
    Title,
    axes_or_new,
    count_text,
    despine,
    error_drawn,
    save_or_show,
    set_suptitle,
    set_title,
    stats_line,
)
from src.behavior_visualization.plot_style import CHOICE_PROB_COLOR_MAP

COLOR_BEST = CHOICE_PROB_COLOR_MAP["prev_best"]
COLOR_SECOND = CHOICE_PROB_COLOR_MAP["next_best"]

HEATMAP_CMAP = "viridis"
EMPTY_CELL_COLOR = "#e6e6e6"

# Text height in data units (a 10 pt label on a ~4 inch tall panel) used to keep labels inside 0-1.
_LABEL_HEIGHT = 0.05
_LABEL_PAD = 0.05

# Figure margins in inches: room for the title above the panels and a little below them.
_TITLE_MARGIN = 0.55
_FOOTER_MARGIN = 0.1

# Empty space left and right of the first and last tick, as a fraction of the tick range, so
# every panel is framed alike whatever its ticks are.
_X_PADDING = 0.1

PANELS = (
    ("best_val", "Value of Best"),
    ("second_val", "Value of Second"),
    ("diff", "Value of Best - Second"),
)


@contextmanager
def new_figure(**subplots_kwargs):
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


class Placement(NamedTuple):
    """Where one mean label goes: y, vertical and horizontal alignment, and x offset sign.

    Attributes:
        y (float): Y position of the text, in data units.
        va (str): Vertical alignment (``"bottom"``, ``"top"`` or ``"center"``).
        ha (str): Horizontal alignment (``"center"``, ``"left"`` or ``"right"``). Default
            ``"center"``.
        dx_sign (int): Sign of the horizontal offset from the point: 1 to the right, -1 to the
            left, 0 for none. Default 0.
    """

    y: float
    va: str
    ha: str = "center"
    dx_sign: int = 0


def _label_span(placement: Placement) -> tuple[float, float]:
    """The y range a label occupies, in data units."""
    y = placement.y
    if placement.va == "bottom":
        return y, y + _LABEL_HEIGHT
    if placement.va == "top":
        return y - _LABEL_HEIGHT, y
    return y - _LABEL_HEIGHT / 2, y + _LABEL_HEIGHT / 2


def _labels_collide(first: Placement, mean_first: float, second: Placement, mean_second: float):
    """Whether two stacked labels overlap each other or sit on the other curve's point."""
    box_a, box_b = _label_span(first), _label_span(second)
    overlap = box_a[0] < box_b[1] and box_b[0] < box_a[1]
    margin = 0.01
    on_point = lambda box, mean: box[0] - margin <= mean <= box[1] + margin  # noqa: E731
    return overlap or on_point(box_a, mean_second) or on_point(box_b, mean_first)


def label_placements(mean_best: float, mean_second: float) -> tuple[Placement, Placement]:
    """Label positions for the P(best) and P(second) means at one x, clear of each other.

    The higher mean's label goes above its point and the lower one's below, so the labels stay
    outside the gap between the curves, also when they are close or have crossed. If a label has
    to flip at the edge of the axes and then collides, the higher label goes to the right of its
    point and the lower one to the left.

    Args:
        mean_best (float): Mean P(best) at this x.
        mean_second (float): Mean P(second) at this x.

    Returns:
        tuple[Placement, Placement]: The placements of the best and of the second label.
    """
    best_higher = mean_best >= mean_second
    best = Placement(*annotation_position(mean_best, above=best_higher))
    second = Placement(*annotation_position(mean_second, above=not best_higher))
    if not _labels_collide(best, mean_best, second, mean_second):
        return best, second
    edge = _LABEL_HEIGHT / 2

    def side(mean: float, right: bool) -> Placement:
        y = min(max(mean, edge), 1.0 - edge)
        return Placement(y, "center", "left" if right else "right", 1 if right else -1)

    return side(mean_best, best_higher), side(mean_second, not best_higher)


def tie_label(tie_mode: str) -> str:
    """``"Excluded"`` or ``"Split"``, for titles and file names.

    Args:
        tie_mode (str): One of ``TIE_MODES``: exclude tied trials or split them between the
            tied arms.

    Returns:
        str: ``"Excluded"`` for the exclude mode, ``"Split"`` otherwise.

    Raises:
        ValueError: If *tie_mode* is not one of the tie modes.
    """
    if tie_mode not in TIE_MODES:
        raise ValueError(f"tie_mode must be one of {TIE_MODES}, got {tie_mode!r}")
    return "Excluded" if tie_mode == TIE_EXCLUDE else "Split"


def tie_summary(ties: dict[str, int], tie_mode: str) -> str:
    """One line with the tied-trial totals and what this figure did with them.

    The three categories are mutually exclusive and count trials eligible for binning.

    Args:
        ties (dict[str, int]): Tied-trial counts with keys ``"best_second"``,
            ``"second_third"`` and ``"all_three"``.
        tie_mode (str): The tie mode of the figure; the exclude mode reports the ties as left
            out, any other as split evenly between the tied arms.

    Returns:
        str: The one-line summary.

    Raises:
        KeyError: If *ties* lacks one of the keys above.
    """
    fate = (
        "left out of this figure"
        if tie_mode == TIE_EXCLUDE
        else "split evenly between the tied arms"
    )
    return (
        "Trials with ties among those eligible for binning - "
        f"best/second only: {ties['best_second']}, "
        f"second/third only: {ties['second_third']}, all three: {ties['all_three']} ({fate})"
    )


def _tie_caption(ties: dict[str, int], tie_mode: str) -> tuple[str, str]:
    """The caption lines of the tie mode and the tie totals."""
    return f"Ties {tie_label(tie_mode).lower()}.", f"{tie_summary(ties, tie_mode)}."


def finish_figure(fig: plt.Figure, title: Title) -> None:
    """Add the title above the panels, then lay the panels out below it.

    The top margin is fixed in inches so the title never overlaps the panels, however many
    rows the figure has; the title is centred over the panels (colour bars included).
    """
    height = fig.get_figheight()
    fig.tight_layout(rect=(0, _FOOTER_MARGIN / height, 1, 1 - _TITLE_MARGIN / height))
    fig.canvas.draw()  # equal-aspect panels and colour bars only settle their positions on draw
    positions = [ax.get_position() for ax in fig.axes]
    centre = (min(p.x0 for p in positions) + max(p.x1 for p in positions)) / 2
    set_suptitle(fig, title, x=centre, y=1 - 0.08 / height, va="top", fontsize=12)


# ---------------------------------------------------------------------------
# Heatmaps
# ---------------------------------------------------------------------------


def cell_text_color(value: float, cmap) -> str:
    """White on dark cells, black on light ones (by the colour's luminance)."""
    r, g, b, _ = cmap(value)
    return "white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.5 else "black"


def cell_label(value: float, n_trials: float) -> str:
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
                color = "black" if np.isnan(values[i, j]) else cell_text_color(values[i, j], cmap)
                ax.text(
                    j, i, cell_label(values[i, j], counts[i, j]),
                    ha="center", va="center", fontsize=9, color=color,
                )  # fmt: skip
        ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    return ax


def plot_overall_heatmaps(
    tables: HeatmapTables,
    value_label: str,
    tie_mode: str,
    save_path: str | Path | None = None,
) -> None:
    """Across-mice P(best) and P(second) heatmaps side by side; save or show.

    Args:
        tables (HeatmapTables): From ``heatmap_tables``.
        value_label (str): ``"True"`` or ``"Believed"``, for the titles.
        tie_mode (str): The tie mode the tables were made with, for the title and tie line.
        save_path (str | Path | None): Base path without extension for the figure and caption;
            ``None`` or ``""`` shows the figure.
    """
    n_mice = len(tables.per_mouse_p_best)
    with new_figure(ncols=2, figsize=(10, 4.4)) as (fig, axs):
        for ax, prob, name in zip(
            axs, (tables.p_best, tables.p_second), ("Best", "Second"), strict=True
        ):
            plot_prob_heatmap(prob, tables.n_trials, f"P({name}) - {value_label} Value", ax=ax)
        title = Title(
            f"Heatmaps ({value_label} Value)",
            stats_line("mean", "mice", [(n_mice, "mouse", "mice")]),
            ("Each cell: the mean across mice; n=trials summed over mice.",)
            + _tie_caption(tables.ties, tie_mode),
        )
        finish_figure(fig, title)
    save_or_show(fig, save_path or None)


def plot_per_mouse_heatmaps(
    tables: HeatmapTables,
    value_label: str,
    tie_mode: str,
    save_path: str | Path | None = None,
    mice: list[str] | None = None,
) -> None:
    """One row per mouse: P(best) and P(second) heatmaps; save or show.

    Args:
        tables (HeatmapTables): From ``heatmap_tables``.
        value_label (str): ``"True"`` or ``"Believed"``, for the titles.
        tie_mode (str): The tie mode the tables were made with, for the title and tie line.
        save_path (str | Path | None): Base path without extension; ``None`` or ``""`` shows.
        mice (list[str] | None): Mice to draw, in order; default all, sorted.

    Raises:
        ValueError: If there is no mouse to draw.
    """
    mice = sorted(tables.per_mouse_p_best) if mice is None else list(mice)
    if not mice:
        raise ValueError("no mice to draw per-mouse heatmaps for")
    with new_figure(nrows=len(mice), ncols=2, figsize=(10, 4.4 * len(mice)), squeeze=False) as (
        fig,
        axs,
    ):
        for row, subject in enumerate(mice):
            n = tables.per_mouse_n[subject]
            for col, (name, probs) in enumerate(
                (("Best", tables.per_mouse_p_best), ("Second", tables.per_mouse_p_second))
            ):
                plot_prob_heatmap(probs[subject], n, f"P({name}) - {subject}", ax=axs[row, col])
        caption = (f"{count_text(len(mice), 'mouse', 'mice')}.",)
        title = Title(
            f"Heatmaps per Mouse ({value_label} Value)",
            None,
            caption + _tie_caption(tables.ties, tie_mode),
        )
        finish_figure(fig, title)
    save_or_show(fig, save_path or None)


# ---------------------------------------------------------------------------
# Psychometric curves
# ---------------------------------------------------------------------------


def draw_mouse_lines(ax: plt.Axes, curve: Curve, color: str) -> None:
    """Thin line per mouse; NaN breaks the line and lone points stay visible as markers."""
    for values in curve.per_mouse:
        ax.plot(curve.x, values, "-o", color=color, lw=1, ms=2.5, alpha=0.3)


def draw_mean(ax: plt.Axes, curve: Curve, color: str, label: str) -> None:
    """Thick mean line with markers, and the SE band where the SE is defined."""
    has_se = ~np.isnan(curve.se) & ~np.isnan(curve.mean)
    if has_se.any():
        ax.fill_between(
            curve.x, curve.mean - curve.se, curve.mean + curve.se,
            where=has_se, color=color, alpha=0.15, zorder=3, linewidth=0,
        )  # fmt: skip
    ax.plot(curve.x, curve.mean, "-o", color=color, lw=2.2, ms=5, label=label, zorder=5)


def _placements_at(mean_best: float, mean_second: float) -> tuple[Placement | None, ...]:
    """Placements of the two labels at one x; ``(None, None)`` where there is no mean.

    Both means come from the same trials, so they are NaN together.

    Raises:
        ValueError: If exactly one of the means is NaN.
    """
    if np.isnan(mean_best) != np.isnan(mean_second):
        raise ValueError(f"P(best)={mean_best} and P(second)={mean_second} must be NaN together")
    if np.isnan(mean_best):
        return None, None
    return label_placements(mean_best, mean_second)


def _annotate_means(ax: plt.Axes, curve_best: Curve, curve_second: Curve) -> None:
    """Write each mean next to its point, inside 0-1 and clear of the other curve's label."""
    dx = 0.06 * float(np.ptp(curve_best.x))
    for x, mean_best, mean_second in zip(
        curve_best.x, curve_best.mean, curve_second.mean, strict=True
    ):
        placements = _placements_at(mean_best, mean_second)
        for placement, mean, color in zip(
            placements, (mean_best, mean_second), (COLOR_BEST, COLOR_SECOND), strict=True
        ):
            if placement is not None:
                ax.text(
                    x + placement.dx_sign * dx, placement.y, f"{mean:.2f}",
                    ha=placement.ha, va=placement.va, fontsize=10, color=color,
                )  # fmt: skip


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
    series = ((curve_best, COLOR_BEST, "P(best)"), (curve_second, COLOR_SECOND, "P(second)"))
    with axes_or_new(ax, figsize=(4.6, 4.2)) as ax:
        for curve, color, label in series:
            draw_mouse_lines(ax, curve, color)
            draw_mean(ax, curve, color, label)
        _annotate_means(ax, curve_best, curve_second)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("Choice Probability", fontsize=12)
        n_mice = len(curve_best.mice)
        average = "mean ± se" if error_drawn(curve_best.se, curve_second.se) else "mean"
        stats = stats_line(average, "mice", [(n_mice, "mouse", "mice")])
        set_title(ax, Title(xlabel, stats), fontsize=14)
        ax.set_ylim(0, 1)
        ax.set_xticks(curve_best.x)
        # Set explicitly: autoscaling ignores NaN points, so the limits would otherwise move
        # with whichever end points have data.
        pad = _X_PADDING * float(np.ptp(curve_best.x))
        ax.set_xlim(curve_best.x[0] - pad, curve_best.x[-1] + pad)
        ax.legend(fontsize=10)
        despine(ax)
    return ax


def plot_psychometric_curves(
    curves: dict[str, dict[str, Curve]],
    value_label: str,
    pool_trials: bool,
    ties: dict[str, int],
    tie_mode: str,
    save_path: str | Path | None = None,
) -> None:
    """Three panels (vs best, second and best - second value) of P(best) and P(second).

    Args:
        curves (dict[str, dict[str, Curve]]): From ``psychometric_curves``.
        value_label (str): ``"True"`` or ``"Believed"``, for the title.
        pool_trials (bool): The weighting the curves were computed with, for the title.
        ties (dict[str, int]): Tied-trial totals by category (``HeatmapTables.ties``).
        tie_mode (str): The tie mode the curves were made with, for the title and tie line.
        save_path (str | Path | None): Base path without extension; ``None`` or ``""`` shows.
    """
    weighting = "trials pooled within mouse" if pool_trials else "cell probabilities averaged"
    with new_figure(ncols=3, figsize=(13, 4.4)) as (fig, axs):
        for ax, (axis, xlabel) in zip(axs, PANELS, strict=True):
            plot_psychometric_panel(curves["best"][axis], curves["second"][axis], xlabel, ax=ax)
        caption = (
            f"{weighting.capitalize()}; thin lines: each mouse.",
            *_tie_caption(ties, tie_mode),
        )
        finish_figure(fig, Title(f"Psychometric Curves ({value_label} Value)", None, caption))
    save_or_show(fig, save_path or None)
