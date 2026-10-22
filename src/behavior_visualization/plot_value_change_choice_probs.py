"""Plots of choice probabilities aligned to changes in the true best / second-best magnitude.

:func:`plot_value_change_panel` draws one configuration's curves into an axis;
:func:`plot_value_change_choice_probs` is the wrapper that makes a figure and saves or shows
it. The curves come from :func:`src.behavior_analysis.value_change_windows.summarize_windows`.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from src.behavior_analysis.value_change_windows import (
    ValueChangeCurves,
    describe_value_change_kind,
)
from src.behavior_visualization.plot_helpers import (
    Title,
    axes_or_new,
    count_text,
    despine,
    save_or_show,
    set_title,
    stats_line,
)
from src.behavior_visualization.plot_style import CHOICE_PROB_COLOR_MAP

# prev_best / next_best in the shared map are the blue / green of the best / second curves.
COLORS = {
    "best": CHOICE_PROB_COLOR_MAP["prev_best"],
    "second": CHOICE_PROB_COLOR_MAP["next_best"],
    "third": CHOICE_PROB_COLOR_MAP["third"],
    "prev_best": CHOICE_PROB_COLOR_MAP["prev_best"],
    "prev_second": CHOICE_PROB_COLOR_MAP["next_best"],
}
LEGEND_LABELS = {
    "best": "Best",
    "second": "Second",
    "third": "Third",
    "prev_best": "Prev Best",
    "prev_second": "Prev Second Best",
}
X_LABEL = "Trials from Value Change"
Y_LABEL = "Choice Probability"


def _window_caption(curves: ValueChangeCurves) -> str:
    """The caption line of the window mode, with the events it affects."""
    if curves.symmetric:
        return f"Window: symmetric ±{curves.pre} trials, complete windows only."
    if curves.truncate_at_rank_swap:
        truncated = count_text(curves.num_truncated, "event", "events")
        return f"Window: truncated at the first rank swap ({truncated} truncated)."
    return "Window: full window."


def _title(curves: ValueChangeCurves, show_individual_mice: bool = False) -> Title:
    """Heading and statistics on the figure; the window mode, weighting and caveats in the caption."""
    events = (curves.num_events, "event", "events")
    mice = (curves.num_subjects, "mouse", "mice")
    if curves.weighting == "event":
        unit, counts = "events", [events, mice]
    else:
        unit, counts = "mice", [mice, events]
    average = "mean ± SE" if counts[0][0] >= 2 else "mean"
    caption = [_window_caption(curves)]
    if curves.weighting == "event":
        caption.append("Each event is weighted equally.")
        if curves.num_events >= 2:
            caption.append(
                "SE treats events as independent, so it is narrower than the true uncertainty."
            )
    if show_individual_mice:
        caption.append("Grey lines: each mouse's mean.")
    heading = f"Choice Probabilities Around {describe_value_change_kind(curves.config)}"
    return Title(heading, stats_line(average, unit, counts), tuple(caption))


def _draw_label(ax: plt.Axes, curves: ValueChangeCurves, label: str, individual: bool) -> None:
    """Mean curve of one label with its SE band; the band is omitted where the SE is NaN."""
    color = COLORS[label]
    if individual:
        for subject_means in curves.per_subject_mean.values():
            ax.plot(curves.x, subject_means[label], color="gray", lw=0.8, alpha=0.4, zorder=1)
    mean, se = curves.mean[label], curves.se[label]
    ax.plot(curves.x, mean, color=color, lw=2.2, label=LEGEND_LABELS[label], zorder=3)
    band = np.isfinite(mean) & np.isfinite(se)
    if band.any():
        ax.fill_between(
            curves.x, mean - se, mean + se, where=band, alpha=0.2, color=color, zorder=2
        )


def plot_value_change_panel(
    curves: ValueChangeCurves, ax: plt.Axes | None = None, *, show_individual_mice: bool = False
) -> plt.Axes:
    """Draw the choice-probability curves of one configuration.

    Args:
        curves (ValueChangeCurves): Output of ``summarize_windows``.
        ax (plt.Axes | None): Axis to draw into; a new figure is made if None.
        show_individual_mice (bool): Also draw each mouse's curve in grey.

    Returns:
        plt.Axes: The axis drawn into.
    """
    with axes_or_new(ax, figsize=(8, 5)) as ax:
        for label in curves.labels:
            _draw_label(ax, curves, label, show_individual_mice)
        ax.axvline(0, color="black", ls="--", lw=1, zorder=4)
        ax.axhline(1 / 3, color="gray", ls=":", lw=1, zorder=1)
        ax.set_xlim(curves.x[0], curves.x[-1])
        ax.set_ylim(0, 1)
        ax.set_xlabel(X_LABEL)
        ax.set_ylabel(Y_LABEL)
        despine(ax)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=10, frameon=False)
        set_title(ax, _title(curves, show_individual_mice), pad=12)
    return ax


def plot_value_change_choice_probs(
    curves: ValueChangeCurves,
    save_path: str | Path | None = None,
    *,
    show_individual_mice: bool = False,
) -> None:
    """Make the figure for one configuration and save it (with its caption), or show it.

    Args:
        curves (ValueChangeCurves): Output of ``summarize_windows``.
        save_path (str | Path | None): Base path without extension (parent folders are
            created); None or ``""`` shows the figure instead.
        show_individual_mice (bool): Also draw each mouse's curve in grey.
    """
    ax = plot_value_change_panel(curves, show_individual_mice=show_individual_mice)
    save_or_show(ax.figure, save_path or None)
