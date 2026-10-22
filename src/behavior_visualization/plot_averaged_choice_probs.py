"""Utilities for averaging choice-probability curves across problems.

Functions here compute and plot choice-probability curves averaged across
multiple problems (≥ some cutoff), with a corrected cumulative fraction-removed
axis.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# The three curve functions are re-exported: notebooks import them from this module.
from src.behavior_analysis.average_choice_probs import (  # noqa: F401
    CHOICE_KEYS,
    average_across_problem_curves,
    compute_curve_from_windows,
    curves_by_problem_from_windows_dict,
)
from src.behavior_visualization.plot_helpers import (
    Title,
    axes_or_new,
    despine,
    error_drawn,
    save_or_show,
    set_title,
    stats_line,
)
from src.behavior_visualization.plot_style import CHOICE_PROB_COLOR_MAP as COLOR_MAP

_CURVE_LABELS = {"prev_best": "Previous Best", "next_best": "New Best", "third": "Third Arm"}
_FRACTION_COLOR = "#FF9100"


# ---------------------------------------------------------------------------
# Fraction removed
# ---------------------------------------------------------------------------


# This stays here (rather than in average_choice_probs) because it needs
# cumulative_total_events_over_post from the plotting module that defines it.
def fraction_removed_by_problem(
    windows_dict: dict,
    curves_by_problem: dict,
    x: np.ndarray,
    *,
    reversals_by_problem: dict,
    exclude_anchor: bool = True,
) -> tuple[np.ndarray, list]:
    """Compute the average cumulative fraction-removed curve across problems.

    Fraction removed is computed within each problem (never by merging windows
    across problems), then averaged across problems.

    Args:
        windows_dict (dict): ``{problem: reversal_windows}``; problems with empty windows are
            skipped.
        curves_by_problem (dict): ``{problem: {"x", "across", "subjects", "num_reversals"}}``
            as returned by :func:`curves_by_problem_from_windows_dict`. Problems absent from
            it are skipped.
        x (np.ndarray): Trial-offset array, shape ``(n_offsets,)``.
        reversals_by_problem (dict): ``{problem: reversals}`` where *reversals* is
            :func:`src.behavior_analysis.get_total_reversals.get_reversal_indices_by_kind`
            of that problem's trials (keys ``"good"``, ``"bad"``, ``"unclassified"``); a
            reversal of any kind cuts a window.
        exclude_anchor (bool): Passed to ``cumulative_total_events_over_post``.
            Default True.

    Returns:
        tuple[np.ndarray, list]: ``(frac_avg, probs_used)``. *frac_avg* is the fraction-removed
        curve averaged over problems (``nanmean``), shape ``(n_offsets,)``; *probs_used* lists
        the problems that contributed, in sorted order.

    Raises:
        KeyError: If a problem has windows and a curve but is missing from
            *reversals_by_problem*.

    Known issue: when no problem yields a curve, *frac_avg* is all NaN and no
    warning is raised.
    """
    from src.behavior_visualization.plot_choice_probs_around_good_reversals import (
        cumulative_total_events_over_post,
    )

    fracs: list[np.ndarray] = []
    probs_used: list = []
    for p in sorted(windows_dict.keys()):
        if p not in curves_by_problem:
            continue
        rw = windows_dict[p]
        if not rw:
            continue
        if p not in reversals_by_problem:
            raise KeyError(f"Problem {p!r} has windows but is missing from reversals_by_problem.")
        across_p = curves_by_problem[p]["across"]
        reversals = reversals_by_problem[p]
        _, _, frac = cumulative_total_events_over_post(
            rw,
            x,
            across_p,
            exclude_anchor=exclude_anchor,
            good_idx=reversals["good"],
            bad_idx=reversals["bad"],
            unclassified_idx=reversals["unclassified"],
        )
        frac = np.asarray(frac, float)
        if np.isnan(frac).all():  # no anchors: the problem contributes no data
            continue
        fracs.append(frac)
        probs_used.append(p)

    if not fracs:
        return np.full_like(x, float("nan"), dtype=float), probs_used

    F = np.vstack(fracs)
    frac_avg = np.nanmean(F, axis=0)
    return frac_avg, probs_used


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def _plot_choice_curves(ax: plt.Axes, x: np.ndarray, across: dict) -> None:
    """Mean choice probability per arm (line) with its SE band, finite points only."""
    for key in CHOICE_KEYS:
        y = np.asarray(across["mean"][key], float)
        s = np.asarray(across["se"][key], float)
        m = np.isfinite(x) & np.isfinite(y)
        if np.any(m):
            ax.plot(x[m], y[m], linewidth=2, color=COLOR_MAP[key], label=_CURVE_LABELS[key])
            # the band is left out where the SE is undefined; the mean line stays
            ax.fill_between(
                x, y - s, y + s, where=m & np.isfinite(s), alpha=0.2, color=COLOR_MAP[key]
            )


def _style_choice_axis(ax: plt.Axes, x: np.ndarray) -> None:
    """Reference lines, limits, labels and spines of the choice-probability axis."""
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.axhline(1 / 3, color="gray", linestyle=":", linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_xlim(x[0], x[-1])
    ax.set_xlabel("Trials from Good Reversal")
    ax.set_ylabel("Choice Probability")
    despine(ax)


def _plot_fraction_removed(ax: plt.Axes, x: np.ndarray, frac_removed: np.ndarray) -> plt.Axes:
    """Step curve of the fraction of data removed on a twin y-axis; returns that axis."""
    ax2 = ax.twinx()
    m = np.isfinite(x) & np.isfinite(frac_removed)
    if np.any(m):
        ax2.step(
            x[m],
            frac_removed[m],
            where="post",
            linewidth=2.0,
            color=_FRACTION_COLOR,
            alpha=0.7,
            label="Total Rev",
        )
    ax2.set_ylabel("Fraction of Data Removed", rotation=-90, labelpad=15)
    ax2.set_ylim(0, 1)
    ax2.spines["top"].set_visible(False)
    return ax2


def _figure_title(across: dict, title_extra: str) -> Title:
    """Heading and statistics on the figure; *title_extra* goes to the caption."""
    has_se = error_drawn(*across.get("se", {}).values())
    counts = [
        (across.get("num_problems", 0), "problem", "problems"),
        (across.get("num_subjects", 0), "mouse", "mice"),
        (across.get("num_reversals", 0), "reversal", "reversals"),
    ]
    stats = stats_line("mean ± SE" if has_se else "mean", "problems", counts)
    return Title("Reversal-Aligned Choices", stats, (str(title_extra),) if title_extra else ())


def plot_choice_probs_with_fraction_removed_panel(
    x: np.ndarray,
    across: dict,
    frac_removed: np.ndarray,
    *,
    ax: plt.Axes | None = None,
    title_extra: str = "",
) -> plt.Axes:
    """Draw averaged choice-probability curves with a fraction-removed twin axis.

    Args:
        x (np.ndarray): Trial-offset array, shape ``(n_offsets,)``.
        across (dict): Averaged across dict from :func:`average_across_problem_curves`, with
            ``"mean"`` and ``"se"`` (each ``{"prev_best", "next_best", "third"}`` to a curve of
            shape ``(n_offsets,)``) and the counts ``"num_subjects"``, ``"num_reversals"``,
            ``"num_problems"`` stated in the title.
        frac_removed (np.ndarray): Fraction-removed curve from
            :func:`fraction_removed_by_problem`, shape ``(n_offsets,)``.
        ax (plt.Axes | None): Axes to draw into; a new 10 x 6 inch figure is created if
            ``None``. Default None.
        title_extra (str): Optional extra line for the figure's caption. Default ``""``.

    Returns:
        plt.Axes: The left (choice-probability) axes; the twin axis shares its figure.
    """
    x = np.asarray(x, float)
    frac_removed = np.asarray(frac_removed, float)
    with axes_or_new(ax, figsize=(10, 6)) as ax:
        _plot_choice_curves(ax, x, across)
        _style_choice_axis(ax, x)
        ax2 = _plot_fraction_removed(ax, x, frac_removed)
        set_title(ax, _figure_title(across, title_extra), pad=20)
        handles, labels = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(handles + handles2, labels + labels2, loc="upper left", fontsize=10)
    return ax


def plot_choice_probs_with_fraction_removed_axis(
    x: np.ndarray,
    across: dict,
    frac_removed: np.ndarray,
    *,
    save_path: str | Path | None = None,
    title_extra: str = "",
) -> None:
    """Plot averaged choice-probability curves with a fraction-removed twin axis.

    Args:
        x (np.ndarray): Trial-offset array, shape ``(n_offsets,)``.
        across (dict): Averaged across dict from :func:`average_across_problem_curves`; see
            :func:`plot_choice_probs_with_fraction_removed_panel`.
        frac_removed (np.ndarray): Fraction-removed curve from
            :func:`fraction_removed_by_problem`, shape ``(n_offsets,)``.
        save_path (str | Path | None): If provided, saves ``.pdf`` and ``.png`` at this path
            (extension-less base). If ``None`` or empty, the figure is shown interactively.
            Default None.
        title_extra (str): Optional extra line for the figure's caption. Default ``""``.
    """
    ax = plot_choice_probs_with_fraction_removed_panel(
        x, across, frac_removed, title_extra=title_extra
    )
    fig = ax.figure
    fig.tight_layout()
    save_or_show(fig, save_path or None)  # an empty path means "show"
