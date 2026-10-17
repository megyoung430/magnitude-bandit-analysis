"""Session figure for checking the best-decrease rank-swap windows event by event.

The top panel is the single-session reward-magnitude panel (see
:func:`~src.behavior_visualization.plot_single_session.plot_magnitudes_and_choices`) with the
x axis in session-local trial index, like ``ValueChangeEvent.trial_idx``. Choices that fall in
an event's window are recoloured by their label, so blue dots must sit on the arm that was
best one trial before t = 0. Below it a strip draws each event's pre and post window as its
own horizontal band (so overlapping events stay readable) and a text block names the arm
behind each label.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.behavior_analysis.single_session import chosen_magnitudes, reversal_event_indices
from src.behavior_analysis.value_change_diagnostic import (
    POST,
    PRE,
    EventDiagnostic,
    contiguous_runs,
    nearest_event_labels,
)
from src.behavior_analysis.value_change_windows import SWAP_LABELS
from src.behavior_visualization.plot_single_session import plot_magnitudes_and_choices
from src.behavior_visualization.plot_style import CHOICE_PROB_COLOR_MAP

# Light shading for the two windows, distinct from the blue / orange / green tower lines.
PRE_COLOR = "#D9C6F0"
POST_COLOR = "#F7D6D6"
LABEL_COLORS = {
    "prev_best": CHOICE_PROB_COLOR_MAP["prev_best"],
    "prev_second": CHOICE_PROB_COLOR_MAP["next_best"],
    "third": CHOICE_PROB_COLOR_MAP["third"],
}
LABEL_NAMES = {"prev_best": "prev best", "prev_second": "prev second", "third": "third"}
NO_EVENTS_TEXT = "no best-decrease rank-swap events"

_MAIN_HEIGHT = 4.5  # inches
_BAND_HEIGHT = 0.3
_STRIP_PAD = 0.5
_TEXT_LINE_HEIGHT = 0.26
_LEGEND_MARGIN = 0.82  # right edge of the axes, leaving room for the legend


def event_text(diagnostic: EventDiagnostic) -> str:
    """One line naming the event, the arm behind each label and the window lengths used."""
    arms = ", ".join(f"{LABEL_NAMES[label]} = {diagnostic.arms[label]}" for label in SWAP_LABELS)
    if diagnostic.truncated:
        truncation = f"truncated at swap (t=+{diagnostic.swap_offset})"
    else:
        truncation = "not truncated"
    return (
        f"#{diagnostic.number} t=0 at trial {diagnostic.trial_idx}: {arms} | "
        f"pre {diagnostic.pre_used} of {diagnostic.pre_requested} trials, "
        f"post {diagnostic.post_used} of {diagnostic.post_requested} trials, {truncation}"
    )


def _figure_layout(n_events: int, n_trials: int) -> tuple[Figure, tuple[Axes, Axes, Axes]]:
    """A figure with the session panel, the strip of event bands and the text block."""
    strip_height = _BAND_HEIGHT * max(n_events, 1) + _STRIP_PAD
    text_height = _TEXT_LINE_HEIGHT * (max(n_events, 1) + 1) + 0.5
    width = min(max(12.0, n_trials / 18), 32.0)
    fig = plt.figure(figsize=(width, _MAIN_HEIGHT + strip_height + text_height + 0.8))
    grid = fig.add_gridspec(
        3, 1, height_ratios=[_MAIN_HEIGHT, strip_height, text_height],
        hspace=0.35, left=0.07, right=_LEGEND_MARGIN, top=0.94, bottom=0.02,
    )  # fmt: skip
    main = fig.add_subplot(grid[0])
    strip = fig.add_subplot(grid[1], sharex=main)
    text = fig.add_subplot(grid[2])
    text.axis("off")
    return fig, (main, strip, text)


def _recolour_window_choices(
    ax: Axes,
    mags_by: Mapping,
    choices_by: Mapping,
    towers: Sequence[str],
    labels: Mapping[int, str],
) -> None:
    """Redraw the choices in event windows as larger dots coloured by their label."""
    chosen = chosen_magnitudes(mags_by, choices_by, towers=towers, N=len(mags_by[towers[0]]))
    for label in SWAP_LABELS:
        trials = [
            t for t, name in sorted(labels.items()) if name == label and np.isfinite(chosen[t])
        ]
        ax.scatter(
            trials, chosen[trials], s=70, color=LABEL_COLORS[label], edgecolor="black",
            linewidth=0.6, zorder=6, label=LABEL_NAMES[label],
        )  # fmt: skip


def _mark_t0(ax: Axes, diagnostics: Sequence[EventDiagnostic]) -> None:
    """Dotted line at each event's t = 0 on the session panel, numbered along the top."""
    for d in diagnostics:
        ax.axvline(d.trial_idx, color="black", linestyle=":", linewidth=1.2, zorder=4)
        # Alternate heights so the numbers of neighbouring events do not overprint.
        ax.annotate(
            f"#{d.number}", (d.trial_idx, 1.0), xycoords=("data", "axes fraction"),
            xytext=(0, 2 + 10 * (d.number % 2)), textcoords="offset points",
            ha="center", va="bottom", fontsize=8,
        )  # fmt: skip


def _draw_bands(ax: Axes, diagnostics: Sequence[EventDiagnostic]) -> None:
    """One horizontal band per event: pre window, post window, t = 0 and label-coloured dots."""
    for row, d in enumerate(diagnostics):
        y = len(diagnostics) - 1 - row  # event 1 on top
        for indices, color in ((d.pre_idx, PRE_COLOR), (d.post_idx, POST_COLOR)):
            runs = [(start - 0.5, length) for start, length in contiguous_runs(indices)]
            ax.broken_barh(runs, (y - 0.4, 0.8), facecolors=color, zorder=1)
        ax.plot([d.trial_idx] * 2, [y - 0.4, y + 0.4], color="black", linewidth=1.5, zorder=3)
        ax.annotate(
            "t=0", (d.trial_idx, y + 0.4), xytext=(0, 1), textcoords="offset points",
            ha="center", va="bottom", fontsize=6,
        )  # fmt: skip
        for trial, label in d.labels_by_trial.items():
            ax.scatter(trial, y, s=14, color=LABEL_COLORS[label], zorder=4, linewidth=0)
    ax.set_ylim(-0.6, max(len(diagnostics), 1) - 0.4 + 0.45)
    ax.set_yticks(range(len(diagnostics)))
    ax.set_yticklabels([f"#{d.number}" for d in reversed(diagnostics)], fontsize=8)
    ax.set_xlabel("Trial (session index, t=0 = first trial with the new value)")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _legend_handles() -> list:
    shading = [
        Patch(facecolor=PRE_COLOR, label="pre window (counted)"),
        Patch(facecolor=POST_COLOR, label="post window (counted)"),
        Line2D([], [], color="black", linestyle=":", label="t=0"),
    ]
    labels = [
        Line2D(
            [], [], marker="o", linestyle="", markerfacecolor=LABEL_COLORS[label],
            markeredgecolor="black", markersize=8, label=f"choice: {LABEL_NAMES[label]}",
        )
        for label in SWAP_LABELS
    ]  # fmt: skip
    return shading + labels


def _write_text(ax: Axes, diagnostics: Sequence[EventDiagnostic], pre: int, post: int) -> None:
    """Header and one line per event (or the no-events note) in the text axes."""
    header = (
        f"Best-value decreases, rank swap; pre {pre}, post {post}, post truncated at the first "
        "later best/second exchange. Labels name arms by their rank at t=-1."
    )
    lines = [event_text(d) for d in diagnostics] or [NO_EVENTS_TEXT]
    ax.text(0, 1, header, va="top", fontsize=9, style="italic", transform=ax.transAxes)
    for line, y in zip(lines, np.linspace(1, 0, len(lines) + 2)[1:-1], strict=True):
        ax.text(0, y, line, va="top", fontsize=9, family="monospace", transform=ax.transAxes)


def plot_value_change_diagnostic(
    session: Mapping[str, Any],
    diagnostics: Sequence[EventDiagnostic],
    *,
    title: str,
    pre: int = PRE,
    post: int = POST,
) -> Figure:
    """Session panel with every event's counted windows and label colours, plus arm labels.

    Args:
        session (Mapping[str, Any]): An imported session with ``reward_magnitudes_by_tower``
            and ``choices_by_tower`` (and optionally the reversal counters).
        diagnostics (Sequence[EventDiagnostic]): The session's events from
            :func:`~src.behavior_analysis.value_change_diagnostic.diagnose_session`; empty for
            a session without events, which is labelled as such.
        title (str): Title of the session panel.
        pre (int): Pre length the diagnostics were made with, for the header. Default ``PRE``.
        post (int): Post length. Default ``POST``.

    Returns:
        Figure: The new figure; the caller saves and closes it.
    """
    mags_by, choices_by = session["reward_magnitudes_by_tower"], session["choices_by_tower"]
    towers = list(mags_by)
    n_trials = len(mags_by[towers[0]])
    fig, (main, strip, text) = _figure_layout(len(diagnostics), n_trials)
    plot_magnitudes_and_choices(
        mags_by, choices_by, towers, reversal_event_indices(session, n_trials), title=title, ax=main
    )
    _recolour_window_choices(main, mags_by, choices_by, towers, nearest_event_labels(diagnostics))
    main.set_title(title, fontsize=14, pad=28)  # room for the event numbers above the panel
    _mark_t0(main, diagnostics)
    _draw_bands(strip, diagnostics)
    main.set_xlim(-0.5, n_trials - 0.5)
    _write_text(text, diagnostics, pre, post)
    strip.legend(
        handles=_legend_handles(), loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8
    )
    return fig
