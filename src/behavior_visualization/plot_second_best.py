"""Figures of the second-best test: where a switch goes, and how it grades with the value gap.

Drawing only: the numbers come from :mod:`src.pipelines.second_best` (one ``SourceSecondBest``
per source, and optionally a ``SecondBestCalibration`` per source). Each ``plot_*`` function
builds a figure, then saves it (pdf, png, svg and a caption file) or shows it. Every figure puts
the mice next to the controls (simulated Q-learner, null agents), one dot or line per mouse; a
source with fewer than ``choice_logit_extensions.MIN_TEST_MICE`` mice shows its mice only, with
no mean, error bar or test. A calibrated mouse has a gray band behind its dot: the middle 95% of
its own null. The 0.5 line is for orientation only, it is not the null.
"""

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from src.behavior_analysis import choice_logit_extensions as ext
from src.behavior_analysis import second_best as sb
from src.behavior_analysis import switch_regression as sr
from src.behavior_analysis.stats import nan_mean, nan_standard_error
from src.behavior_visualization.plot_helpers import (
    Title,
    despine,
    jitter_rng,
    save_or_show,
    set_title,
)
from src.behavior_visualization.plot_regression_extensions import (
    colors_of,
    counts_caption,
    draw_column,
    source_stats_line,
    style_source_axis,
)
from src.behavior_visualization.plot_switch_regression import (
    OTHER_MICE_COLOR,
    _add_headroom,
    _column_mean,
    _label_means,
)
from src.behavior_visualization.plot_value_psychometrics import finish_figure, new_figure
from src.pipelines.null_agents import AGENT_LABELS

SECOND_BEST_CHOICES = "Second Best Choices"
BETTER_ALTERNATIVE_CURVES = "Better Alternative Curves"
FIGURE_NAMES = (SECOND_BEST_CHOICES, BETTER_ALTERNATIVE_CURVES)
MEASURE_LABELS = {
    sb.SECOND_BEST: "Second Best After Leaving the Best",
    sb.BELIEVED_BETTER: "Believed-Better Alternative",
    sb.TRUE_BETTER: "Truly Better Alternative",
}
CHOSE_LABELS = {
    sb.SECOND_BEST: "P(Chose Second Best)",
    sb.BELIEVED_BETTER: "P(Chose Believed-Better)",
    sb.TRUE_BETTER: "P(Chose Truly Better)",
}
CURVE_LABELS = {sb.TRUE_BETTER: "True Value", sb.BELIEVED_BETTER: "Believed Value"}
CONDITION_LABELS = {
    sr.AFTER_ZERO: "After Zero",
    sr.AFTER_NONZERO: "After Non-Zero",
    sb.POOLED: "Pooled",
}
# A point of a graded curve from fewer used switches than this is not drawn.
MIN_BIN_USED = 10
# Line styles of the sources other than the first in the graded curves.
SOURCE_STYLES = ("--", ":", "-.")

DEFINITIONS = (
    "Switch trials: consecutive trials of one segment (a segment starts at a problem's start, a "
    "session's start and after a dropped trial) on which the mouse leaves the arm it chose on "
    "the previous trial. The choice set is the two other arms. After zero: the previous choice "
    "paid 0 pulses; after non-zero: 1 or more.",
    "Second best after leaving the best: switches whose previous choice was the arm with the "
    "highest true magnitude on the previous trial (a strict maximum). The outcome is whether the "
    "mouse chose the better of the two alternatives, ranked on that same trial, so the second "
    "best rather than the third. Switches from a tied top and switches with tied alternatives "
    "are excluded and counted. Leaving the best never follows a zero payout in this task (exactly "
    "one arm pays 0 and it is never the best), so the after-zero panel is empty.",
    "Believed-better alternative: switches on which both alternatives have been chosen before in "
    "the problem and the magnitudes they last paid (their believed values) differ; the outcome "
    "is whether the mouse chose the one that last paid more. Switches with a never-chosen "
    "alternative and ties are excluded and counted.",
    "Truly better alternative: switches on which the two alternatives' magnitudes on the switch "
    "trial differ; the outcome is whether the mouse chose the higher one. Ties are excluded and "
    "counted.",
    "The 0.5 line is for orientation only: it is not the null. The alternative chosen most "
    "recently tends to be the better one, so a mouse that ignores value still lies above 0.5. "
    "Read each mouse against its own null (gray band, calibrated z and p), the same proportion "
    "for its fitted win-stay/lose-shift agents played on its own task.",
    "Dots: mice, colored per mouse; a null agent fitted to a mouse has that mouse's color, and "
    "simulated mice not fitted to a mouse (the Q-learner next to the real mice) are gray. Black "
    "bars: mean ± se across mice, with the mean written at the top of the panel.",
)


# ---------------------------------------------------------------------------
# Shared pieces
# ---------------------------------------------------------------------------


def _check_sources(sources: Mapping) -> None:
    if not sources:
        raise ValueError("no sources to plot")
    empty = [name for name, s in sources.items() if s.proportions.empty]
    if empty:
        raise ValueError(f"no mice in {empty}")


def _mice_tables(sources: Mapping) -> dict:
    """``{source: table with one row per analyzed mouse}``, for the shared caption helpers."""
    return {name: s.graded[["mouse"]].drop_duplicates() for name, s in sources.items()}


def _mouse_colors(sources: Mapping) -> tuple[dict[str, str], frozenset[str]]:
    """Colors of every mouse, and the mice left out of the legend.

    The mice of the first source get the per-mouse palette (a null agent fitted to a mouse shares
    its name, so its color); a mouse in another source, such as a simulated Q-learner not fitted
    to any mouse, is gray and unnamed.
    """
    tables = list(_mice_tables(sources).values())
    colors = colors_of(tables[:1])
    others = frozenset(m for t in tables[1:] for m in t["mouse"]) - set(colors)
    return {**colors, **dict.fromkeys(others, OTHER_MICE_COLOR)}, others


def _ticks(sources: Mapping) -> list[str]:
    """``"<source>\nn=<analyzed>"``, with the skipped mice counted when there are any."""
    tables = _mice_tables(sources)
    return [
        f"{name}\nn={len(tables[name])}" + (f" ({len(s.skipped)} skipped)" if s.skipped else "")
        for name, s in sources.items()
    ]


def _skipped_lines(sources: Mapping) -> list[str]:
    return [
        f"Skipped, {name}: " + "; ".join(f"{m} ({reason})" for m, reason in s.skipped.items()) + "."
        for name, s in sources.items()
        if s.skipped
    ]


def _minimum_lines(sources: Mapping) -> list[str]:
    """The minimum, and the cells below it that are left out, per source."""
    lines = []
    for name, s in sources.items():
        lines.append(
            f"A cell needs at least {s.min_used} used switches; a cell with fewer is left out of "
            f"the figure ({name})."
        )
        for (measure, condition), cells in s.dropped.groupby(["measure", "condition"], sort=False):
            mice = ", ".join(f"{r.mouse} ({r.n_used})" for r in cells.itertuples())
            lines.append(
                f"Dropped cells, {name}, {MEASURE_LABELS[measure]}, "
                f"{CONDITION_LABELS[condition]}: {mice} used switches."
            )
    return lines


def _count_lines(sources: Mapping) -> list[str]:
    """Per mouse and cell: successes of the used switches, and the exclusions."""
    lines = []
    for name, s in sources.items():
        for r in s.proportions.itertuples():
            lines.append(
                f"{name}, {r.mouse}, {MEASURE_LABELS[r.measure]}, {CONDITION_LABELS[r.condition]}: "
                f"{r.n_success}/{r.n_used} = {r.p:.3f} of {r.n_switches} switches; excluded "
                f"{r.n_tied} tied, {r.n_unvisited} with a never-chosen alternative."
            )
    return lines


def _primary(calibration) -> pd.DataFrame:
    """Rows of the first null agent of *calibration* (the realistic WSLS (Magnitude) by default)."""
    table = calibration.calibration
    return table[table["agent"] == table["agent"].iloc[0]]


def _bands(calibration, measure: str, condition: str) -> dict:
    """``{mouse: (null_low, null_high)}`` of one panel, or ``{}`` without a calibration."""
    if calibration is None:
        return {}
    rows = _primary(calibration).query("measure == @measure and condition == @condition")
    return {r.mouse: (r.null_low, r.null_high) for r in rows.itertuples()}


def _calibration_lines(calibrations: Mapping) -> list[str]:
    """Per mouse: calibrated z and empirical p against each null agent; and the across-mice test."""
    lines = []
    for name, calibration in calibrations.items():
        table = calibration.calibration
        agents = list(dict.fromkeys(table["agent"]))
        first = AGENT_LABELS.get(agents[0], agents[0])
        lines.append(
            f"Null calibration, {name}: each mouse's fitted null agents ("
            + ", ".join(AGENT_LABELS.get(a, a) for a in agents)
            + f") played on its own task, {calibration.replicates} replicates each, through the "
            "identical analysis; gray band: 2.5th-97.5th percentile of the "
            f"{first} null. z = (observed - null mean) / null sd; p: two-sided empirical."
        )
        for (mouse, measure, condition), group in table.groupby(
            ["mouse", "measure", "condition"], sort=False
        ):
            parts = [
                f"z={r.z:.2f}, p={r.p_empirical:.3g} vs {AGENT_LABELS.get(r.agent, r.agent)}"
                for r in group.itertuples()
            ]
            lines.append(
                f"{name}, {mouse}, {MEASURE_LABELS[measure]}, {CONDITION_LABELS[condition]}: "
                f"observed {group['observed'].iloc[0]:.3f}; " + "; ".join(parts) + "."
            )
        for row in calibration.summary.itertuples():
            if np.isfinite(row.mean):
                lines.append(
                    f"{name}, calibrated z vs {AGENT_LABELS.get(row.agent, row.agent)}, "
                    f"{MEASURE_LABELS[row.measure]}, {CONDITION_LABELS[row.condition]}: mean "
                    f"{row.mean:.2f} ± se {row.se:.2f}; Wilcoxon p={row.p_wilcoxon:.3g} "
                    f"(n={row.n} mice)."
                )
    return lines


def _test_lines(sources: Mapping) -> list[str]:
    """Mean ± se across mice of each proportion, for each source with enough mice."""
    lines = []
    for name, s in sources.items():
        for row in sb.group_summary(s.proportions).itertuples():
            if np.isfinite(row.mean):
                lines.append(
                    f"{name}, {MEASURE_LABELS[row.measure]}, {CONDITION_LABELS[row.condition]}: "
                    f"mean {row.mean:.3f} ± se {row.se:.3f} across mice (n={row.n} mice)."
                )
    return lines


# ---------------------------------------------------------------------------
# Second Best Choices
# ---------------------------------------------------------------------------


def plot_second_best_choices(
    sources: Mapping, save_path: str | Path | None = None, calibrations: Mapping | None = None
) -> None:
    """P(second best | leaving the best), P(believed-better) and P(truly better) by source.

    One panel per measure (rows) and condition (columns, after zero and after non-zero); the
    pooled proportions are in the caption.

    Args:
        sources (Mapping): ``{source label: SourceSecondBest}``, the mice first.
        save_path (str | Path | None): Base path without extension; ``None`` shows the figure.
        calibrations (Mapping | None): ``{source label: SecondBestCalibration}`` of the sources
            whose mice are calibrated (gray null band per mouse, calibrated z and p in the
            caption). Default ``None``.

    Raises:
        ValueError: If there are no sources or a source has no mice.
    """
    _check_sources(sources)
    calibrations = calibrations or {}
    conditions = sr.CONDITIONS
    colors, unnamed = _mouse_colors(sources)
    ticks = _ticks(sources)
    caption = (
        *DEFINITIONS,
        counts_caption(_mice_tables(sources)),
        *_minimum_lines(sources),
        *_skipped_lines(sources),
        *_test_lines(sources),
        *_count_lines(sources),
        *_calibration_lines(calibrations),
    )
    width = max(9.0, 2 * (1.2 * len(sources) + 1.5))
    with new_figure(
        nrows=len(sb.MEASURES), ncols=len(conditions), figsize=(width, 3.9 * len(sb.MEASURES)),
        sharey=True, squeeze=False,
    ) as (fig, axs):  # fmt: skip
        rng = jitter_rng()
        for row, measure in enumerate(sb.MEASURES):
            for col, condition in enumerate(conditions):
                ax, labeled, counts, errors = axs[row, col], set(unnamed), [], []
                means = {}
                for x, (name, s) in enumerate(sources.items()):
                    mine = s.proportions.query("measure == @measure and condition == @condition")
                    if mine.empty:
                        ax.text(x, 0.5, "none", ha="center", va="center", color="0.5", fontsize=9)
                        continue
                    counts.append(len(mine))
                    bands = _bands(calibrations.get(name), measure, condition)
                    per_mouse = mine.set_index("mouse")["p"]
                    errors.append(draw_column(ax, x, per_mouse, colors, rng, labeled, bands=bands))
                    means[x] = _column_mean(per_mouse)
                _label_means(ax, means)
                style_source_axis(ax, ticks, CHOSE_LABELS[measure] if col == 0 else "")
                ax.axhline(0.5, color="0.6", lw=0.8, ls=":", zorder=1)
                ax.set(ylim=(0, 1), yticks=np.linspace(0, 1, 5))
                title = f"{MEASURE_LABELS[measure]}, {CONDITION_LABELS[condition]}"
                stats = source_stats_line(counts, errors) if counts else None
                set_title(ax, Title(title, stats), fontsize=12)
                if ax.get_legend_handles_labels()[1]:  # a panel of gray mice only has none
                    ax.legend(fontsize=9)
        _add_headroom(axs[0, 0])  # once: every panel shares the y axis
        finish_figure(fig, Title(SECOND_BEST_CHOICES, None, caption))
    save_or_show(fig, save_path or None)


# ---------------------------------------------------------------------------
# Better Alternative Curves
# ---------------------------------------------------------------------------


def _curve_matrix(graded: pd.DataFrame, measure: str, condition: str) -> pd.DataFrame:
    """``mouse x difference`` proportions of one panel; bins with too few used switches are NaN."""
    rows = graded.query("measure == @measure and condition == @condition")
    rows = rows.assign(p=rows["p"].where(rows["n_used"] >= MIN_BIN_USED))
    return rows.pivot(index="mouse", columns="difference", values="p").reindex(
        columns=list(sb.DIFFERENCES)
    )


def _mean_se(matrix: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Mean and se over mice per difference; NaN where fewer than ``MIN_TEST_MICE`` mice drew."""
    stack = matrix.to_numpy(dtype=float)
    enough = (~np.isnan(stack)).sum(axis=0) >= ext.MIN_TEST_MICE
    mean, se = nan_mean(stack), nan_standard_error(stack)
    return np.where(enough, mean, np.nan), np.where(enough, se, np.nan)


def _curve_stats_line(n_mice: int, se: np.ndarray) -> str:
    drawn = np.isfinite(se).any()
    return f"{'mean ± se across mice' if drawn else 'per mouse'}, n={n_mice} mice"


def plot_better_alternative_curves(sources: Mapping, save_path: str | Path | None = None) -> None:
    """P(chose the better alternative) against the magnitude difference of the alternatives.

    One panel per value (true, believed; rows) and condition (columns). The first source is drawn
    per mouse (colored lines) with its mean ± se across mice; every further source as a mean
    line. A point from fewer than ``MIN_BIN_USED`` used switches is not drawn.

    Args:
        sources (Mapping): ``{source label: SourceSecondBest}``, the mice first.
        save_path (str | Path | None): Base path without extension; ``None`` shows the figure.

    Raises:
        ValueError: If there are no sources or a source has no mice.
    """
    _check_sources(sources)
    mice_label, *others = list(sources)
    colors = colors_of(list(_mice_tables({mice_label: sources[mice_label]}).values()))
    conditions, measures = sr.CONDITIONS, tuple(CURVE_LABELS)
    caption = (
        "Each mouse's proportion of used switches on which it chose the better alternative, in "
        "bins of the absolute magnitude difference of the two alternatives (1 to 4 pulses). True "
        "value: the magnitudes on the switch trial. Believed value: the magnitudes the "
        "alternatives last paid (both chosen before in the problem). Ties are excluded.",
        f"A point from fewer than {MIN_BIN_USED} used switches is not drawn. A source with fewer "
        f"than {ext.MIN_TEST_MICE} mice shows its mice only: no mean, error bar or test.",
        "Thin colored lines: the mice of the first source; thick black line: its mean ± se across "
        "mice; other sources: their mean across mice (dashed, dotted, dash-dotted).",
        "0.5 is for orientation only: it is not the null (see Second Best Choices).",
        counts_caption(_mice_tables(sources)),
        *_skipped_lines(sources),
    )
    with new_figure(
        nrows=len(measures), ncols=len(conditions), figsize=(9.0, 3.9 * len(measures)),
        sharey=True, sharex=True, squeeze=False,
    ) as (fig, axs):  # fmt: skip
        x = np.array(sb.DIFFERENCES, dtype=float)
        for row, measure in enumerate(measures):
            for col, condition in enumerate(conditions):
                ax = axs[row, col]
                matrix = _curve_matrix(sources[mice_label].graded, measure, condition)
                for mouse, values in matrix.iterrows():
                    ax.plot(x, values.to_numpy(dtype=float), "-o", color=colors[mouse], ms=3,
                            lw=1.0, alpha=0.7, label=mouse)  # fmt: skip
                mean, se = _mean_se(matrix)
                ax.errorbar(x, mean, yerr=se, color="black", lw=2.2, capsize=3, label=mice_label)
                for style, name in zip(SOURCE_STYLES, others, strict=False):
                    other, _ = _mean_se(_curve_matrix(sources[name].graded, measure, condition))
                    ax.plot(x, other, style, color="black", lw=1.6, label=name)
                ax.axhline(0.5, color="0.6", lw=0.8, ls=":", zorder=1)
                ax.set(xlim=(0.6, 4.4), ylim=(0, 1), xticks=sb.DIFFERENCES)
                ax.set_xlabel("Magnitude Difference of the Alternatives (Pulses)", fontsize=12)
                ax.set_ylabel("P(Chose Better Alternative)" if col == 0 else "", fontsize=12)
                despine(ax)
                title = f"{CURVE_LABELS[measure]}, {CONDITION_LABELS[condition]}"
                set_title(ax, Title(title, _curve_stats_line(len(matrix), se)), fontsize=12)
                ax.legend(fontsize=9)
        finish_figure(fig, Title(BETTER_ALTERNATIVE_CURVES, None, caption))
    save_or_show(fig, save_path or None)


def plot_second_best(
    sources: Mapping, out_dir: str | Path, calibrations: Mapping | None = None
) -> None:
    """Save both second-best figures into *out_dir* (created if missing).

    Args:
        sources (Mapping): ``{source label: SourceSecondBest}``, the mice first.
        out_dir (str | Path): Folder of the figures.
        calibrations (Mapping | None): ``{source label: SecondBestCalibration}`` of the
            calibrated sources. Default ``None``.
    """
    out_dir = Path(out_dir)
    plot_second_best_choices(sources, out_dir / SECOND_BEST_CHOICES, calibrations)
    plot_better_alternative_curves(sources, out_dir / BETTER_ALTERNATIVE_CURVES)


__all__ = [
    "BETTER_ALTERNATIVE_CURVES",
    "FIGURE_NAMES",
    "SECOND_BEST_CHOICES",
    "plot_better_alternative_curves",
    "plot_second_best",
    "plot_second_best_choices",
]
