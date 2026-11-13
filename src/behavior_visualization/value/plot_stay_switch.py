"""Stay/switch psychometric curves and the P(stay) heatmap.

Drawing only: the numbers come from :mod:`src.behavior_analysis.value.stay_switch_psychometrics`.
The panel functions take ``ax=None`` and draw into it (making their own figure only if none is
given); the ``plot_*`` wrappers build a figure, then save it (pdf, png, svg and a caption) or
show it.
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from src.behavior_analysis.value.get_choice_probs_by_value import TIE_EXCLUDE
from src.behavior_analysis.value.stay_switch_psychometrics import (
    CHOSEN_NONZERO,
    CHOSEN_ROWS,
    CHOSEN_ZERO,
    TIE_ALL_THREE,
    TIE_ALTERNATIVES,
    TIE_CHOSEN_ALTERNATIVE,
    StayHeatmap,
)
from src.behavior_analysis.value.value_psychometrics import Curve
from src.behavior_visualization.common.plot_helpers import (
    Title,
    axes_or_new,
    despine,
    error_drawn,
    save_or_show,
    set_title,
    stats_line,
)
from src.behavior_visualization.value.plot_value_psychometrics import (
    EMPTY_CELL_COLOR,
    HEATMAP_CMAP,
    cell_label,
    cell_text_color,
    draw_baseline,
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
}
STAY_SWITCH_OUTCOMES = ("stay", "switch")
SWITCH_TARGET_XLABEL = "Value of Best Alternative - Other Arm"
SWITCH_TARGET_YLABEL = "P(Best Alternative | Switch)"
# P(best alternative | switch) when the mouse picks between the two unchosen arms at random.
SWITCH_TARGET_CHANCE = 0.5
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
        "Stay: trial t picks the chosen arm again. Switch = 1 - stay: trial t picks the best "
        "alternative or the other arm.",
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
    """Ticks at the x values and equal padding at both ends (NaN ends do not move them).

    Only exact (integer) values get a tick each; bin means of a continuous value can lie close
    together, so matplotlib chooses those ticks.
    """
    if np.allclose(x, np.round(x)):
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
    curves: dict[str, Curve],
    heading: str,
    xlabel: str,
    ax: plt.Axes | None = None,
    *,
    ylabel: str = "Probability",
    counted: tuple[str, str] | None = None,
    legend: bool = True,
    baselines: dict[str, Curve] | None = None,
) -> plt.Axes:
    """Curves of the given outcomes against one x axis: mice thin, mean thick with se band.

    A curve with a single x value gets an error bar instead of the band.

    Args:
        curves (dict[str, Curve]): ``{outcome: Curve}``, the outcomes of ``OUTCOME_STYLE`` to
            draw, all against the same x and from the same mice.
        heading (str): First line of the panel title (the row).
        xlabel (str): X axis label.
        ax (plt.Axes | None): Axes to draw into; a new figure if ``None``.
        ylabel (str): Y axis label. Default ``"Probability"``. Keyword-only.
        counted (tuple[str, str] | None): ``(singular, plural)`` of what ``Curve.n_trials``
            counts, to add its total to the statistics line (e.g. ``("switch",
            "switches")``); ``None`` (default) adds nothing. Keyword-only.
        legend (bool): Draw a legend of the outcomes. Default ``True``. Keyword-only.
        baselines (dict[str, Curve] | None): ``{outcome: Curve}`` reference curves on the same x
            (e.g. a model without the value), drawn above the observed ones as hollow squares on
            a dashed line, so a baseline that coincides with the observed curve stays visible.
            Keyword-only.

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
            if baselines is not None:
                draw_baseline(ax, baselines[outcome], color, f"{label} Baseline")
        average = "mean ± se" if error_drawn(*(c.se for c in curves.values())) else "mean"
        counts = [(len(first.mice), "mouse", "mice")]
        if counted is not None:
            counts.append((int(np.sum(first.n_trials)), *counted))
        set_title(ax, Title(heading, stats_line(average, "mice", counts)), fontsize=12)
        _x_limits(ax, first.x)
        if legend:
            ax.legend(fontsize=10)
        _finish_panel(ax, xlabel, ylabel)
    return ax


def _finish_panel(ax: plt.Axes, xlabel: str, ylabel: str = "Probability") -> None:
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_ylim(0, 1)
    despine(ax)


def _empty_panel(ax: plt.Axes, heading: str, xlabel: str, note: str, ylabel: str) -> None:
    """A panel of a row with nothing to draw: its heading and *note*, axes kept so columns
    line up."""
    ax.set_title(heading, fontsize=12)
    ax.text(0.5, 0.5, note, ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([])
    _finish_panel(ax, xlabel, ylabel)


def _weighting_caption(pool_trials: bool) -> str:
    if pool_trials:
        return "Pairs pooled within mouse; thin lines: each mouse."
    return (
        "Within a mouse, each (V(chosen), V(alt)) cell's probability counts once, whatever its "
        "number of pairs; thin lines: each mouse."
    )


def _curves_caption(value_label, pool_trials, ties_lines, extra=()) -> tuple[str, ...]:
    return (
        *definition_caption(value_label),
        *row_caption(value_label),
        *extra,
        _weighting_caption(pool_trials),
        *ties_lines,
    )


def plot_stay_switch_grid(
    curves_by_row: dict,
    title: Title,
    save_path: str | Path | None = None,
    *,
    row_headings: dict[str, str] = ROW_HEADINGS,
    panels: tuple[tuple[str, str], ...] = PANELS,
    baselines_by_row: dict | None = None,
) -> None:
    """P(stay) and P(switch) against the x axes of *panels*, one row per row heading; save or show.

    Args:
        curves_by_row (dict): ``{row: {outcome: {axis: Curve}} | None}`` for each row of
            *row_headings*; a row of ``None`` draws empty panels.
        title (Title): The figure title and caption.
        save_path (str | Path | None): Base path without extension; ``None`` or ``""`` shows.
        row_headings (dict[str, str]): ``{row key: panel heading}``, top row first. Default: the
            rows by the value of the chosen option. Keyword-only.
        panels (tuple[tuple[str, str], ...]): ``(axis, x label)`` of each column. Default
            ``PANELS``. Keyword-only.
        baselines_by_row (dict | None): Reference curves in the layout of *curves_by_row*, drawn
            in every panel. Default ``None``: none. Keyword-only.
    """
    size = (5 * len(panels), 4.5 * len(row_headings))
    # squeeze=False keeps the axes 2-D for a figure of a single row
    with new_figure(nrows=len(row_headings), ncols=len(panels), figsize=size, squeeze=False) as (
        fig,
        axs,
    ):
        for axes_row, (row, heading) in zip(axs, row_headings.items(), strict=True):
            curves = curves_by_row[row]
            for ax, (axis, xlabel) in zip(axes_row, panels, strict=True):
                if curves is None:
                    _empty_panel(ax, heading, xlabel, "No pairs", "Probability")
                    continue
                panel = {outcome: curves[outcome][axis] for outcome in STAY_SWITCH_OUTCOMES}
                reference = None
                if baselines_by_row is not None:
                    reference = {o: baselines_by_row[row][o][axis] for o in STAY_SWITCH_OUTCOMES}
                plot_stay_switch_panel(panel, heading, xlabel, ax=ax, baselines=reference)
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
        curves_by_row (dict): From ``stay_switch_curves_by_row``; a row of ``None`` draws empty
            panels.
        value_label (str): ``"True"`` or ``"Believed"``.
        pool_trials (bool): The weighting the curves were computed with, for the caption.
        ties (dict[str, int]): Tied pairs by category (``StaySwitchPairs.ties``).
        tie_mode (str): The tie mode of the curves.
        save_path (str | Path | None): Base path without extension; ``None`` or ``""`` shows.
    """
    caption = _curves_caption(value_label, pool_trials, tie_caption(ties, tie_mode))
    title = Title(f"Stay-Switch Curves ({value_label} Value)", None, caption)
    plot_stay_switch_grid(curves_by_row, title, save_path)


def switch_target_tie_caption(
    ties: dict[str, int], tie_mode: str, tied_switches: dict[str, int]
) -> tuple[str, ...]:
    """Caption lines of what the switch-target figure did with ties.

    Args:
        ties (dict[str, int]): Tied pairs by ``TIE_*`` category.
        tie_mode (str): ``"exclude"`` or ``"split"``.
        tied_switches (dict[str, int]): Switches between tied alternatives per row
            (``tied_switches_by_row``), left out of the curves.

    Returns:
        tuple[str, ...]: One line per sentence.
    """
    if tie_mode == TIE_EXCLUDE:
        return tie_caption(ties, tie_mode)
    return (
        "Pairs with V(chosen) = V(alt) are kept: their two alternatives differ, so a best "
        f"alternative exists ({ties[TIE_CHOSEN_ALTERNATIVE]} such pairs).",
        "Switches between tied alternatives (gap 0, all three tied included) left out: top "
        f"{tied_switches[CHOSEN_ZERO]}, bottom {tied_switches[CHOSEN_NONZERO]}.",
    )


def plot_switch_target_panel(
    curve: Curve | None,
    heading: str,
    ax: plt.Axes | None = None,
    *,
    baseline: Curve | None = None,
    xlabel: str = SWITCH_TARGET_XLABEL,
) -> plt.Axes:
    """P(best alternative | switch) against the gap between the unchosen arms, one row.

    Args:
        curve (Curve | None): From ``switch_target_curves``; ``None`` draws an empty panel.
        heading (str): First line of the panel title (the row).
        ax (plt.Axes | None): Axes to draw into; a new figure if ``None``.
        baseline (Curve | None): A reference curve on the same x (e.g. a model without the
            value), drawn above the observed one. Keyword-only.
        xlabel (str): X axis label. Default ``SWITCH_TARGET_XLABEL``. Keyword-only.

    Returns:
        plt.Axes: The axes drawn into.
    """
    with axes_or_new(ax, figsize=(6, 4.2)) as ax:
        if curve is None:
            _empty_panel(ax, heading, xlabel, "No untied switches", SWITCH_TARGET_YLABEL)
            return ax
        ax.axhline(SWITCH_TARGET_CHANCE, color="0.6", linestyle=":", linewidth=1, zorder=0)
        plot_stay_switch_panel(
            {"switch_best_alt": curve},
            heading,
            xlabel,
            ax=ax,
            ylabel=SWITCH_TARGET_YLABEL,
            counted=("switch", "switches"),
            legend=baseline is not None,
            baselines=None if baseline is None else {"switch_best_alt": baseline},
        )
    return ax


def plot_switch_target_curves(
    curves_by_row: dict[str, Curve | None],
    value_label: str,
    pool_trials: bool,
    ties: dict[str, int],
    tie_mode: str,
    tied_switches: dict[str, int],
    save_path: str | Path | None = None,
) -> None:
    """P(best alternative | switch) against V(best alternative) - V(other arm), one panel per row.

    Rows as in the stay/switch figure: the value of the option chosen on the earlier trial of
    each pair is 0 (top) or above 0 (bottom). Only pairs whose next choice is a switch count.

    Args:
        curves_by_row (dict[str, Curve | None]): From ``switch_target_curves_by_row``.
        value_label (str): ``"True"`` or ``"Believed"``.
        pool_trials (bool): The weighting the curves were computed with, for the caption.
        ties (dict[str, int]): Tied pairs by category (``StaySwitchPairs.ties``).
        tie_mode (str): The tie mode of the curves.
        tied_switches (dict[str, int]): Switches between tied alternatives per row
            (``tied_switches_by_row``), for the caption.
        save_path (str | Path | None): Base path without extension; ``None`` or ``""`` shows.
    """
    definition = (
        "Only the pairs whose next choice is a switch: y is the fraction of those switches that "
        "go to the best alternative (the higher-valued of the two arms not chosen before); "
        "P(other arm | switch) is one minus it.",
        "x: V(best alternative) - V(other arm), the gap between the two unchosen arms on the "
        "switch trial. Dotted line: 0.5, a random pick between the two.",
        "n=switches: the switches of all mice in the row. A mouse without switches in a row is "
        "not one of that row's mice; a point can rest on few switches.",
    )
    ties_lines = switch_target_tie_caption(ties, tie_mode, tied_switches)
    caption = _curves_caption(value_label, pool_trials, ties_lines, definition)
    title = Title(f"Switch Target Curves ({value_label} Value)", None, caption)
    with new_figure(nrows=len(CHOSEN_ROWS), ncols=1, figsize=(6, 9)) as (fig, axs):
        for ax, row in zip(axs, CHOSEN_ROWS, strict=True):
            plot_switch_target_panel(curves_by_row[row], ROW_HEADINGS[row], ax)
        finish_figure(fig, title)
    save_or_show(fig, save_path or None)


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
