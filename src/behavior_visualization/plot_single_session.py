"""Visualise a single behavioural session: reward magnitudes, EMA, and choice distributions.

Each panel function draws into an ``ax`` you pass (or a new figure if you
don't), so it can be one panel of a larger figure. :func:`plot_single_session`
composes the panels into the standard four-panel figure and saves or shows it.
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

import src.behavior_visualization.plot_style  # noqa: F401  # applies rcParams
from src.behavior_analysis.single_session import (
    choice_counts_by_tower,
    chosen_magnitudes,
    event_indices_from_cumulative,  # noqa: F401  # kept importable here; it was part of this module's public API
    has_ema_traces,
    latest_rank_counts,
    ordered_rank_counts,
    percent_of_total,
    reversal_event_indices,
    validate_choice_probabilities,
)

TOWER_COLORS = ["#A6CEE3", "#FDBF6F", "#B2DF8A"]
EMA_LABELS = {"ema_best_arm_choices": "Best", "ema_second_arm_choices": "Second"}
EMA_COLORS = {"ema_best_arm_choices": "#000000", "ema_second_arm_choices": "#858585"}
RANK_COLORS = {
    "best": EMA_COLORS["ema_best_arm_choices"],
    "second": EMA_COLORS["ema_second_arm_choices"],
    "third": "#CCCCCC",
}
GOOD_REVERSAL_COLOR = "#3A982E"
BAD_REVERSAL_COLOR = "#F97979"


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------


def _ensure_ax(ax):
    """Return *ax*, or the axes of a new figure when none is given."""
    if ax is None:
        _, ax = plt.subplots()
    return ax


def _vlines(ax, indices, color, label):
    """Dashed vertical line before each trial in *indices*; only the first is labelled."""
    for j, i in enumerate(indices):
        ax.axvline(
            i - 0.5, linestyle="--", color=color, linewidth=2.5, label=label if j == 0 else None
        )


def add_reversal_lines(ax, good_idx, bad_idx, block_idx):
    """Draw dashed lines for good/bad reversals, or for blocks when neither exists."""
    if good_idx or bad_idx:
        _vlines(ax, good_idx, GOOD_REVERSAL_COLOR, "Good Rev")
        _vlines(ax, bad_idx, BAD_REVERSAL_COLOR, "Bad Rev")
    elif block_idx:
        _vlines(ax, block_idx, GOOD_REVERSAL_COLOR, "Rev")


def plot_magnitudes_and_choices(mags_by, choices_by, towers, reversals, title=None, ax=None):
    """Per-tower reward magnitude traces with the chosen magnitude as black dots.

    Args:
        reversals: ``(good_idx, bad_idx, block_idx)`` as from
            :func:`reversal_event_indices`.
    """
    ax = _ensure_ax(ax)
    N = len(mags_by[towers[0]])
    x = np.arange(N)
    # Towers beyond the palette length (3) are not drawn, though their choices
    # still count in the bar chart.
    for tower, color in zip(towers, TOWER_COLORS, strict=False):
        ax.plot(
            x, np.asarray(mags_by[tower], dtype=float), linewidth=2.5, color=color, label=f"{tower}"
        )
    chosen = chosen_magnitudes(mags_by, choices_by, towers, N)
    ax.scatter(x, chosen, s=30, color="black", marker="o", label="Choice", zorder=5)
    add_reversal_lines(ax, *reversals)
    ax.set_ylabel("Reward Magnitude", fontsize=12)
    ax.set_yticks([0, 1, 4])
    ax.set_title(title or "Reward Magnitudes & Choices", fontsize=14)
    ax.legend(loc="upper left", fontsize=10)
    return ax


def plot_choice_probabilities(probabilities, towers, ax):
    """Dashed per-tower model choice probabilities on a right-hand twin of *ax*.

    Args:
        probabilities: ``(N, len(towers))`` array, one column per tower.
    """
    probability_axis = ax.twinx()
    x = np.arange(len(probabilities))
    for arm, (tower, color) in enumerate(zip(towers, TOWER_COLORS, strict=False)):
        probability_axis.plot(
            x,
            probabilities[:, arm],
            color=color,
            linestyle="--",
            linewidth=1.8,
            label=f"RNN P({tower})",
            alpha=0.95,
        )
    probability_axis.set_ylim(0, 1)
    probability_axis.set_ylabel("Vanilla RNN choice probability", fontsize=12)
    probability_axis.spines["top"].set_visible(False)
    probability_axis.legend(loc="upper right", fontsize=9)
    return probability_axis


def plot_ema_traces(session_data, ema_keys, reversals, N, ax=None):
    """EMA choice curves over *N* trials for the keys in *ema_keys* that the session holds."""
    ax = _ensure_ax(ax)
    x = np.arange(N)
    for k in ema_keys:
        if k in session_data:
            ax.plot(
                x,
                np.asarray(session_data[k], dtype=float),
                linewidth=3,
                color=EMA_COLORS.get(k),
                label=EMA_LABELS.get(k, k),
            )
    ax.axhline(0.7, linestyle="--", linewidth=2, color="#DCDCDC", alpha=0.7, label="Threshold")
    add_reversal_lines(ax, *reversals)
    ax.set_ylabel("EMA", fontsize=12)
    ax.set_xlabel("Trial", fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.legend(loc="upper left", fontsize=10)
    return ax


def _bar_with_percent_labels(ax, names, counts, colors, ylabel):
    """Bar chart of *counts* with each bar's share of the total written above it."""
    bars = ax.bar(names, counts, color=colors)
    ax.set_ylabel(ylabel, fontsize=12)
    for rect, pct in zip(bars, percent_of_total(counts), strict=True):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height(),
            f"{pct:.0f}%",
            fontsize=12,
            ha="center",
            va="bottom",
        )


def plot_choices_by_tower(choices_by, towers, N, ax=None):
    """Bar chart of how often each tower was chosen over the first *N* trials."""
    ax = _ensure_ax(ax)
    counts = choice_counts_by_tower(choices_by, towers, N)
    _bar_with_percent_labels(ax, towers, counts, TOWER_COLORS[: len(towers)], "Count by Tower")
    return ax


def plot_choices_by_rank(rank_counts, ax=None):
    """Bar chart of choices per rank; a placeholder text when *rank_counts* is empty/None."""
    ax = _ensure_ax(ax)
    if rank_counts:
        names, counts = ordered_rank_counts(rank_counts)
        colors = [RANK_COLORS.get(k, "#CCCCCC") for k in names]
        _bar_with_percent_labels(ax, names, counts, colors, "Count by Rank")
    else:
        ax.text(0.5, 0.5, "No rank data", ha="center", va="center")
        ax.set_xticks([])
    ax.set_ylabel("Count by Rank", fontsize=12)
    return ax


# ---------------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------------


def _make_axes(has_ema):
    """Figure and ``(ax_magnitudes, ax_ema | None, ax_towers, ax_ranks)``."""
    if has_ema:
        fig = plt.figure(figsize=(14, 11))
        gs = GridSpec(3, 2, height_ratios=[2.2, 1.8, 1.5], hspace=0.25, wspace=0.25)
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, :], sharex=ax1)
        return fig, (ax1, ax2, fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1]))
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 2, height_ratios=[2.5, 1.5], hspace=0.25, wspace=0.25)
    ax1 = fig.add_subplot(gs[0, :])
    return fig, (ax1, None, fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]))


def _style_spines(axes):
    for ax in axes:
        if ax is None:
            continue
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.2)
        ax.spines["bottom"].set_linewidth(1.2)


def _draw_session(
    session_data, mag_key, choice_key, ema_keys, title, choice_probabilities_by_tower
):
    """Build the four-panel figure for one session and return it.

    The figure is created before the inputs are fully validated, so an error
    raised afterwards (e.g. invalid probabilities) leaves it open in pyplot.
    """
    has_ema = has_ema_traces(session_data, ema_keys)
    mags_by = session_data[mag_key]
    choices_by = session_data[choice_key]
    towers = list(mags_by.keys())
    N = len(mags_by[towers[0]])

    fig, (ax1, ax2, ax3, ax4) = _make_axes(has_ema)
    reversals = reversal_event_indices(session_data, N)
    plot_magnitudes_and_choices(mags_by, choices_by, towers, reversals, title, ax=ax1)
    if choice_probabilities_by_tower is not None:
        probabilities = validate_choice_probabilities(choice_probabilities_by_tower, towers, N)
        plot_choice_probabilities(probabilities, towers, ax1)
    if has_ema:
        plot_ema_traces(session_data, ema_keys, reversals, N, ax=ax2)
    else:
        ax1.set_xlabel("Trial", fontsize=12)
    plot_choices_by_tower(choices_by, towers, N, ax=ax3)
    plot_choices_by_rank(latest_rank_counts(session_data), ax=ax4)
    _style_spines((ax1, ax2, ax3, ax4))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        plt.tight_layout()
    return fig


def plot_single_session(
    session_data,
    mag_key="reward_magnitudes_by_tower",
    choice_key="choices_by_tower",
    ema_keys=("ema_best_arm_choices", "ema_second_arm_choices"),
    title=None,
    save_path=None,
    choice_probabilities_by_tower=None,
):
    """Plot a four-panel summary of a single experimental session.

    Panels include reward magnitude traces (with chosen-arm scatter),
    exponential moving-average (EMA) choice curves, and bar charts of choices
    by tower and by rank.  Reversal events are shown as vertical dashed lines
    on the top two panels.

    Args:
        session_data: Session dict containing at minimum
            ``"reward_magnitudes_by_tower"`` and ``"choices_by_tower"``, and
            optionally ``"good_reversals"``, ``"bad_reversals"``, ``"blocks"``,
            ``"rank_counts"``, and EMA keys.
        mag_key: Key for the per-tower magnitude dict
            (default: ``"reward_magnitudes_by_tower"``).
        choice_key: Key for the per-tower one-hot choice dict
            (default: ``"choices_by_tower"``).
        ema_keys: Tuple of keys whose values are plotted as EMA traces
            (default: ``("ema_best_arm_choices", "ema_second_arm_choices")``).
        title: Custom figure title for the top panel (default: ``None``).
        choice_probabilities_by_tower: Optional normalized pre-choice probabilities
            by physical tower, aligned to session trials; dashed lines on a right axis.
        save_path: Base path (without extension) for saving ``.pdf`` and
            ``.png`` output.  If ``None``, the figure is shown interactively.

    Raises:
        KeyError: if *mag_key* or *choice_key* is missing from *session_data*.
        TypeError: if *session_data* is not a mapping.
        ValueError: if *choice_probabilities_by_tower* does not use the
            session's tower labels, or is not finite, within [0, 1], aligned
            to every trial and normalized on each trial; also from matplotlib
            when one EMA key holds data and another is ``None``.

    Notes:
        ``session_data["reversal_boundary_events"]`` (optional) is a list of
        ``{"kind": "good" | "bad", "index": int}`` dicts that add reversal lines
        the cumulative ``good_reversals`` / ``bad_reversals`` counters did not
        mark; any kind other than ``"good"`` is drawn as a bad reversal.

        The figure is closed on success but left open in pyplot when an error
        is raised while drawing or saving it.
    """
    fig = _draw_session(
        session_data, mag_key, choice_key, ema_keys, title, choice_probabilities_by_tower
    )
    if save_path:
        base = Path(save_path)
        base.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(base) + ".pdf", bbox_inches="tight")
        fig.savefig(str(base) + ".png", dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)
