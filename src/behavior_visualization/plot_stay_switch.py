"""Stay/switch psychometric curves and the P(stay) heatmap.

Drawing only: the numbers come from :mod:`src.behavior_analysis.stay_switch_psychometrics`.
The panel functions take ``ax=None`` and draw into it (making their own figure only if none is
given); the ``plot_*`` wrappers build a figure, then save it (pdf, png, svg and a caption) or
show it.
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from src.behavior_analysis.get_choice_probs_by_value import TIE_EXCLUDE
from src.behavior_analysis.stay_switch_psychometrics import (
    CHOSEN_NONZERO,
    CHOSEN_ROWS,
    CHOSEN_ZERO,
    TIE_ALL_THREE,
    TIE_ALTERNATIVES,
    TIE_CHOSEN_ALTERNATIVE,
    StayHeatmap,
)
from src.behavior_analysis.value_psychometrics import Curve
from src.behavior_visualization.plot_helpers import (
    Title,
    axes_or_new,
    despine,
    error_drawn,
    save_or_show,
    set_title,
    stats_line,
)
from src.behavior_visualization.plot_value_psychometrics import (
    EMPTY_CELL_COLOR,
    HEATMAP_CMAP,
    cell_label,
    cell_text_color,
    draw_mean,
    draw_mouse_lines,
    finish_figure,
    new_figure,
    tie_label,
)

# Okabe-Ito colours, one per outcome.
OUTCOME_STYLE = {
    "stay": ("#0072B2", "Stay"),
    "switch": ("#D55E00", "Switch"),
    "switch_best_alt": ("#E69F00", "Switch to Best Alternative"),
    "switch_other": ("#CC79A7", "Switch to Other Arm"),
}
STAY_SWITCH_OUTCOMES = ("stay", "switch")
SWITCH_TARGET_OUTCOMES = ("switch_best_alt", "switch_other")
# Rows of the curve figures, top first, by the value of the option chosen before.
ROW_HEADINGS = {
    CHOSEN_ZERO: "Chose 0-Value Option",
    CHOSEN_NONZERO: "Chose Non-0 Option",
}
PANELS = (
    ("diff", "Value of Chosen - Best Alternative"),
    ("chosen", "Value of Chosen"),
    ("alt", "Value of Best Alternative"),
)
_X_PADDING = 0.1
# Half-width of the x range of a panel with a single x value, so its point sits mid-panel.
_SINGLE_X_PAD = 1.0


def definition_caption(value_label: str) -> tuple[str, ...]:
    """Caption lines that define the trial pairs, values and outcomes.

    Args:
        value_label (str): ``"True"`` or ``"Believed"``.

    Returns:
        tuple[str, ...]: One line per sentence.
    """
    lines = [
        "Pairs of consecutive trials (t-1, t) of one session; the chosen option is the arm "
        "chosen on t-1.",
        f"Values are the {value_label.lower()} values on trial t: V(chosen) is the chosen arm's "
        "value, V(alt) the higher value of the two other arms (the best alternative).",
        "Stay: trial t picks the chosen arm again. Switch = 1 - stay, split into a switch to "
        "the best alternative and to the other arm.",
        "Pairs across a block or value change are kept; pairs across sessions or a missed "
        "trial are not.",
    ]
    if value_label == "Believed":
        lines.append(
            "Believed value: what the arm paid the last time it was chosen before trial t; "
            "pairs count once all three arms have been chosen in the problem."
        )
    return tuple(lines)


def tie_caption(ties: dict[str, int], tie_mode: str) -> tuple[str, str]:
    """Caption lines of the tie mode and of the tied-pair totals.

    Args:
        ties (dict[str, int]): Tied pairs by ``TIE_*`` category.
        tie_mode (str): ``"exclude"`` or ``"split"``.

    Returns:
        tuple[str, str]: What the figure did with ties, and how many there were.
    """
    if tie_mode == TIE_EXCLUDE:
        mode = "Ties excluded: every pair with two equal values is left out."
        fate = "left out of this figure"
    else:
        mode = (
            "Ties split: pairs with V(chosen) = V(alt) are at a difference of 0, and a switch "
            "between two tied alternatives counts half to each."
        )
        fate = "kept"
    counts = (
        f"Tied pairs - V(chosen) = V(alt): {ties[TIE_CHOSEN_ALTERNATIVE]}, "
        f"the two alternatives: {ties[TIE_ALTERNATIVES]}, all three: {ties[TIE_ALL_THREE]} "
        f"({fate})."
    )
    return mode, counts


def row_caption(value_label: str) -> tuple[str, ...]:
    """Caption lines that say how the pairs are split into the two rows.

    Args:
        value_label (str): ``"True"`` or ``"Believed"``.

    Returns:
        tuple[str, ...]: One line per sentence.
    """
    lines = [
        f"Rows by the {value_label.lower()} value of the chosen option: top, V(chosen) = 0 "
        "(chose the 0-value option); bottom, V(chosen) > 0. A mouse without pairs in a row is "
        "left out of that row's mean and n.",
    ]
    if value_label == "Believed":
        lines.append(
            "Believed 0: the arm last paid 0 when the mouse chose it, which need not be the arm "
            "whose true magnitude is 0 on trial t."
        )
    return tuple(lines)


def _x_limits(ax: plt.Axes, x: np.ndarray) -> None:
    """Ticks at the x values and equal padding at both ends (NaN ends do not move them)."""
    ax.set_xticks(x)
    pad = _X_PADDING * float(np.ptp(x)) if len(x) > 1 else _SINGLE_X_PAD
    ax.set_xlim(x[0] - pad, x[-1] + pad)


def _draw_single_point_error(ax: plt.Axes, curve: Curve, color: str) -> None:
    """Error bar of a curve with one x value, whose se band would have no width."""
    if np.isfinite(curve.se).all():
        ax.errorbar(
            curve.x, curve.mean, yerr=curve.se, fmt="none", ecolor=color, capsize=5, lw=1.8,
            zorder=4,
        )  # fmt: skip


def plot_stay_switch_panel(
    curves: dict[str, Curve], heading: str, xlabel: str, ax: plt.Axes | None = None
) -> plt.Axes:
    """Curves of the given outcomes against one x axis: mice thin, mean thick with se band.

    A curve with a single x value gets an error bar instead of the band.

    Args:
        curves (dict[str, Curve]): ``{outcome: Curve}``, the outcomes of ``OUTCOME_STYLE`` to
            draw, all against the same x and from the same mice.
        heading (str): First line of the panel title (the row).
        xlabel (str): X axis label.
        ax (plt.Axes | None): Axes to draw into; a new figure if ``None``.

    Returns:
        plt.Axes: The axes drawn into.
    """
    first = next(iter(curves.values()))
    with axes_or_new(ax, figsize=(4.6, 4.2)) as ax:
        for outcome, curve in curves.items():
            color, label = OUTCOME_STYLE[outcome]
            draw_mouse_lines(ax, curve, color)
            draw_mean(ax, curve, color, label)
            if len(curve.x) == 1:
                _draw_single_point_error(ax, curve, color)
        average = "mean ± se" if error_drawn(*(c.se for c in curves.values())) else "mean"
        stats = stats_line(average, "mice", [(len(first.mice), "mouse", "mice")])
        set_title(ax, Title(heading, stats), fontsize=12)
        _x_limits(ax, first.x)
        ax.legend(fontsize=10)
        _finish_panel(ax, xlabel)
    return ax


def _finish_panel(ax: plt.Axes, xlabel: str) -> None:
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Probability", fontsize=12)
    ax.set_ylim(0, 1)
    despine(ax)


def _empty_panel(ax: plt.Axes, heading: str, xlabel: str) -> None:
    """A panel of a row without pairs: its heading and a note, axes kept so columns line up."""
    ax.set_title(heading, fontsize=12)
    ax.text(0.5, 0.5, "No pairs", ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([])
    _finish_panel(ax, xlabel)


def _curves_caption(value_label, pool_trials, ties, tie_mode, extra=()) -> tuple[str, ...]:
    weighting = "pairs pooled within mouse" if pool_trials else "cell probabilities averaged"
    return (
        *definition_caption(value_label),
        *row_caption(value_label),
        *extra,
        f"{weighting.capitalize()}; thin lines: each mouse.",
        *tie_caption(ties, tie_mode),
    )


def plot_outcome_curves(
    curves_by_row: dict[str, dict[str, dict[str, Curve]] | None],
    outcomes: tuple[str, ...],
    title: Title,
    save_path: str | Path | None = None,
) -> None:
    """*outcomes* against three x axes, one row per chosen-value row; save or show.

    Args:
        curves_by_row (dict): From ``stay_switch_curves_by_row``; a row of ``None`` draws empty
            panels.
        outcomes (tuple[str, ...]): The outcomes to draw in every panel.
        title (Title): The figure title and caption.
        save_path (str | Path | None): Base path without extension; ``None`` or ``""`` shows.
    """
    with new_figure(nrows=len(CHOSEN_ROWS), ncols=len(PANELS), figsize=(15, 9)) as (fig, axs):
        for axes_row, row in zip(axs, CHOSEN_ROWS, strict=True):
            curves = curves_by_row[row]
            for ax, (axis, xlabel) in zip(axes_row, PANELS, strict=True):
                if curves is None:
                    _empty_panel(ax, ROW_HEADINGS[row], xlabel)
                    continue
                panel = {outcome: curves[outcome][axis] for outcome in outcomes}
                plot_stay_switch_panel(panel, ROW_HEADINGS[row], xlabel, ax=ax)
        finish_figure(fig, title)
    save_or_show(fig, save_path or None)


def plot_stay_switch_curves(
    curves_by_row: dict,
    value_label: str,
    pool_trials: bool,
    ties: dict[str, int],
    tie_mode: str,
    save_path: str | Path | None = None,
) -> None:
    """P(stay) and P(switch) against three x axes, rows by the value of the chosen option.

    Args:
        curves_by_row (dict): From ``stay_switch_curves_by_row``.
        value_label (str): ``"True"`` or ``"Believed"``.
        pool_trials (bool): The weighting the curves were computed with, for the caption.
        ties (dict[str, int]): Tied pairs by category (``StaySwitchPairs.ties``).
        tie_mode (str): The tie mode of the curves.
        save_path (str | Path | None): Base path without extension; ``None`` or ``""`` shows.
    """
    caption = _curves_caption(value_label, pool_trials, ties, tie_mode)
    title = Title(f"Stay-Switch Curves ({value_label} Value)", None, caption)
    plot_outcome_curves(curves_by_row, STAY_SWITCH_OUTCOMES, title, save_path)


def plot_switch_target_curves(
    curves_by_row: dict,
    value_label: str,
    pool_trials: bool,
    ties: dict[str, int],
    tie_mode: str,
    save_path: str | Path | None = None,
) -> None:
    """P(switch to the best alternative) and P(switch to the other arm), rows as above.

    Both are fractions of all pairs, so they add up to P(switch).

    Args:
        curves_by_row (dict): From ``stay_switch_curves_by_row``.
        value_label (str): ``"True"`` or ``"Believed"``.
        pool_trials (bool): The weighting the curves were computed with, for the caption.
        ties (dict[str, int]): Tied pairs by category (``StaySwitchPairs.ties``).
        tie_mode (str): The tie mode of the curves.
        save_path (str | Path | None): Base path without extension; ``None`` or ``""`` shows.
    """
    fractions = (
        "P(switch to best alternative) and P(switch to other arm) are fractions of all pairs, "
        "not of switches: together they make P(switch).",
    )
    caption = _curves_caption(value_label, pool_trials, ties, tie_mode, fractions)
    title = Title(f"Switch Target Curves ({value_label} Value)", None, caption)
    plot_outcome_curves(curves_by_row, SWITCH_TARGET_OUTCOMES, title, save_path)


def plot_stay_heatmap_panel(
    tables: StayHeatmap, title: str | None = None, ax: plt.Axes | None = None
) -> plt.Axes:
    """P(stay) over V(chosen) (rows, highest on top) x V(alt) on a fixed 0-1 colour range.

    Each cell prints the probability and the pairs behind it; cells without pairs are grey.

    Args:
        tables (StayHeatmap): From ``stay_heatmap``.
        title (str | None): Short axes title, for a figure of several panels; ``None`` (default)
            leaves it to the figure title and the colour-bar label.
        ax (plt.Axes | None): Axes to draw into; a new figure if ``None``.

    Returns:
        plt.Axes: The axes drawn into.
    """
    values = np.flipud(tables.p_stay.to_numpy(dtype=float))
    counts = np.flipud(tables.n_pairs.to_numpy(dtype=float))
    rows = list(reversed(tables.p_stay.index))
    columns = list(tables.p_stay.columns)
    cmap = matplotlib.colormaps[HEATMAP_CMAP].with_extremes(bad=EMPTY_CELL_COLOR)
    size = (1.2 * len(columns) + 2.2, 1.0 * len(rows) + 1.6)
    with axes_or_new(ax, figsize=size) as ax:
        image = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=0.0, vmax=1.0)
        if title:
            ax.set_title(title)
        ax.set_xlabel("Value of Best Alternative")
        ax.set_ylabel("Value of Chosen")
        ax.set_xticks(range(len(columns)), [f"{c:g}" for c in columns])
        ax.set_yticks(range(len(rows)), [f"{r:g}" for r in rows])
        for i, j in np.ndindex(values.shape):
            color = "black" if np.isnan(values[i, j]) else cell_text_color(values[i, j], cmap)
            ax.text(
                j, i, cell_label(values[i, j], counts[i, j]),
                ha="center", va="center", fontsize=9, color=color,
            )  # fmt: skip
        bar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        bar.set_label("P(stay)")
    return ax


def plot_stay_heatmap(
    tables: StayHeatmap,
    value_label: str,
    ties: dict[str, int],
    tie_mode: str,
    save_path: str | Path | None = None,
) -> None:
    """The across-mice P(stay) heatmap; save or show.

    Args:
        tables (StayHeatmap): From ``stay_heatmap``.
        value_label (str): ``"True"`` or ``"Believed"``.
        ties (dict[str, int]): Tied pairs by category.
        tie_mode (str): The tie mode of the tables.
        save_path (str | Path | None): Base path without extension; ``None`` or ``""`` shows.
    """
    n_columns, n_rows = tables.p_stay.shape[1], tables.p_stay.shape[0]
    size = (max(6.0, 1.2 * n_columns + 2.6), 1.0 * n_rows + 2.4)
    with new_figure(figsize=size) as (fig, ax):
        plot_stay_heatmap_panel(tables, ax=ax)
        title = Title(
            f"Stay Heatmap ({value_label} Value)",
            stats_line("mean", "mice", [(tables.n_mice, "mouse", "mice")]),
            (
                *definition_caption(value_label),
                "Each cell: the mean across mice of each mouse's P(stay); n=pairs summed over "
                "mice.",
                *tie_caption(ties, tie_mode),
            ),
        )
        finish_figure(fig, title)
    save_or_show(fig, save_path or None)


def ties_in_name(tie_mode: str) -> str:
    """``"Ties Excluded"`` or ``"Ties Split"``, for figure names."""
    return f"Ties {tie_label(tie_mode)}"
