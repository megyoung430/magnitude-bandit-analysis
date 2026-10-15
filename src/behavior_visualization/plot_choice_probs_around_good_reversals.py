"""Plotting functions for choice-probability curves around good reversals.

Functions here render reversal-aligned choice-probability curves for individual
sessions/subjects.  They also support optional twin axes showing the cumulative
fraction of data removed as subsequent reversals occur within the post window.

Two cumulative-axis modes are available:

- *total* mode (``use_total=True``): a single orange step-line showing the
  fraction of anchor windows that have been "cut short" by the next reversal of
  either kind (the good, bad and unclassified lines added together). Needs
  both ``all_good_idx`` and ``all_bad_idx``.
- *good/bad* mode (default): separate orange (good) and red (bad) step-lines
  that together sum to ≤ 1. Sessions that log no good/bad counters (blocks
  only) cut windows with reversals of unknown kind; when
  ``all_unclassified_idx`` lists any, a grey line shows them, so good + bad +
  unclassified = total.

The panel functions (:func:`plot_choice_probs`,
:func:`plot_choice_probs_broken_axis`, :func:`plot_choice_probs_with_removal`,
:func:`plot_cumulative_removed`) draw into axes you pass, or into a new figure
if you don't.  The removal curves themselves are computed in
:mod:`src.behavior_analysis.cumulative_reversal_removal`.
"""

import warnings
from collections.abc import Collection
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from numpy.typing import ArrayLike

from src.behavior_analysis.cumulative_reversal_removal import (
    cumulative_removal_series,
    good_bad_removal_curves,
    total_removal_curves,
)
from src.behavior_visualization.plot_style import CHOICE_PROB_COLOR_MAP as COLOR_MAP

CURVE_LABELS = [("prev_best", "Previous Best"), ("next_best", "New Best"), ("third", "Third Arm")]
# Removal series are keyed by kind in the analysis; these map a kind to what is drawn.
KIND_LABELS = {
    "total": "Total Rev",
    "good": "Good Rev",
    "bad": "Bad Rev",
    "unclassified": "Unclassified Rev",
}
KIND_COLORS = {"total": "#FF9100", "good": "#FF9100", "bad": "#CD0000", "unclassified": "#808080"}
PLAIN_YLIM = (-0.05, 1.05)
REMOVAL_YLIM = (0, 1)
X_LABEL = "Trials from Good Reversal"
Y_LABEL = "Choice Probability"


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------


def _title(across: dict) -> str:
    return (
        "Good Reversal-Aligned Choices\n"
        f"(mean ± se across subjects | "
        f"n={across.get('num_subjects', 0)} subjects, "
        f"n={across.get('num_reversals', 0)} reversals)\n"
    )


def _draw_choice_curves(ax: plt.Axes, x_disp, across: dict, xlim, ylim) -> None:
    """Mean ± SE curves for each arm on *ax*, restricted to finite bins within *xlim*."""
    mean = across.get("mean", {})
    se = across.get("se", {})
    ax.set_xlim(*xlim)

    for key, label in CURVE_LABELS:
        y = mean.get(key)
        s = se.get(key)
        if y is None or s is None:
            continue

        y = np.asarray(y, dtype=float)
        s = np.asarray(s, dtype=float)

        m = np.isfinite(x_disp) & np.isfinite(y) & np.isfinite(s)
        m &= (x_disp >= xlim[0]) & (x_disp <= xlim[1])
        if not np.any(m):
            continue

        ax.plot(x_disp[m], y[m], linewidth=2, color=COLOR_MAP[key], label=label)
        ax.fill_between(x_disp, y - s, y + s, where=m, alpha=0.2, color=COLOR_MAP[key])

    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.axhline(1 / 3, color="gray", linestyle=":", linewidth=1)

    ax.set_ylim(*ylim)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _full_xlim(x_disp) -> tuple[float, float]:
    return (np.nanmin(x_disp), np.nanmax(x_disp))


def plot_choice_probs(x: ArrayLike, across: dict, ax: plt.Axes | None = None) -> plt.Axes:
    """Mean ± SE choice-probability curves around good reversals on one axis.

    Args:
        x: Trial offsets relative to the reversal.
        across: Aggregated statistics dict (``mean``, ``se``, ``num_subjects``,
            ``num_reversals``).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))
    x = np.asarray(x, dtype=float)
    _draw_choice_curves(ax, x, across, _full_xlim(x), PLAIN_YLIM)
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    ax.set_title(_title(across), pad=20)
    ax.legend(loc="upper right", fontsize=10)
    return ax


def _skipped_offsets(x: ArrayLike, skip_n_trials_after_reversal: int) -> np.ndarray:
    """Offsets with the post-reversal ones (``>= 1``) shifted right by the skipped trials."""
    x_disp = np.asarray(x, dtype=float).copy()
    if skip_n_trials_after_reversal > 0:
        x_disp[x_disp >= 1] += skip_n_trials_after_reversal
    return x_disp


def _broken_xlims(x_disp, skip_n_trials_after_reversal: int):
    """x limits of the pre-reversal and post-skip segments of a broken axis."""
    left_xlim = (np.nanmin(x_disp), 0)
    right_xlim = (skip_n_trials_after_reversal + 1, np.nanmax(x_disp))
    return left_xlim, right_xlim


def _broken_axes(left_xlim, right_xlim):
    """Two axes sharing y, with widths proportional to the segments they show."""
    spans = [left_xlim[1] - left_xlim[0], right_xlim[1] - right_xlim[0]]
    _, axes = plt.subplots(
        1,
        2,
        sharey=True,
        figsize=(10, 4.5),
        gridspec_kw={"width_ratios": spans, "wspace": 0.15},
    )
    return axes


def _style_broken_axes(ax_left: plt.Axes, ax_right: plt.Axes, right_start: float) -> None:
    ax_left.spines["right"].set_visible(False)
    ax_right.spines["left"].set_visible(False)
    ax_right.spines["right"].set_visible(False)
    ax_right.tick_params(axis="y", left=False, right=False, labelleft=False, labelright=False)

    ax_right.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    rticks = [t for t in ax_right.get_xticks() if t >= right_start - 1e-9]
    if right_start not in rticks:
        rticks = sorted(set(rticks + [right_start]))
    ax_right.set_xticks(rticks)


def plot_choice_probs_broken_axis(
    x: ArrayLike,
    across: dict,
    skip_n_trials_after_reversal: int,
    axes: tuple[plt.Axes, plt.Axes] | None = None,
) -> tuple[plt.Axes, plt.Axes]:
    """Choice-probability curves on a broken x axis that omits the skipped trials.

    The post-reversal offsets (``>= 1``) of *x* are shifted right by the skipped
    trials here, so the right-hand segment starts just after the omitted gap.

    Args:
        x: Trial offsets relative to the reversal, unshifted.
        across: Aggregated statistics dict.
        skip_n_trials_after_reversal: Number of omitted trials (> 0).
        axes: ``(left, right)`` axes to draw into; a new figure if ``None``.
            The caller is responsible for their relative widths.

    Returns:
        The ``(left, right)`` axes.
    """
    x_disp = _skipped_offsets(x, skip_n_trials_after_reversal)
    left_xlim, right_xlim = _broken_xlims(x_disp, skip_n_trials_after_reversal)
    ax_left, ax_right = _broken_axes(left_xlim, right_xlim) if axes is None else axes

    _draw_choice_curves(ax_left, x_disp, across, left_xlim, PLAIN_YLIM)
    _draw_choice_curves(ax_right, x_disp, across, right_xlim, PLAIN_YLIM)
    _style_broken_axes(ax_left, ax_right, right_xlim[0])
    ax_left.set_ylabel(Y_LABEL)

    handles, labels = ax_left.get_legend_handles_labels()
    if handles:
        ax_right.legend(handles, labels, loc="upper right", fontsize=10)
    return ax_left, ax_right


def _finite_everywhere(x, series: dict[str, np.ndarray]) -> np.ndarray:
    """Bins where *x* and every series are finite (all curves share one mask)."""
    mask = np.isfinite(x)
    for values in series.values():
        mask &= np.isfinite(values)
    return mask


def _count_ylim(series: dict[str, np.ndarray], mask: np.ndarray) -> tuple[float, float]:
    """y limits for raw counts; the largest curve is taken over all series, even hidden ones."""
    ymax = max(float(np.nanmax(v[mask])) for v in series.values()) if np.any(mask) else 0.0
    return (0, ymax * 1.05 + 1)


def plot_cumulative_removed(
    x: ArrayLike,
    series: dict[str, np.ndarray],
    *,
    fraction: bool,
    hidden: Collection[str] = (),
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Step curves of data removed by later reversals on a twin (right-hand) y axis.

    Side effect on the host *ax*: it is raised above the twin (zorder 1 vs 0) and
    its background patch is hidden, so the twin's curves show behind the host's.

    Args:
        x: Trial offsets.
        series: ``{kind: values over x}`` (kinds ``"total"``, ``"good"``, ``"bad"``)
            as from
            :func:`src.behavior_analysis.cumulative_reversal_removal.cumulative_removal_series`.
        fraction: Whether *series* holds fractions (axis 0-1) or raw counts.
        hidden: Kinds in *series* to leave undrawn; they still determine which
            bins are finite and the count axis limits.
        ax: Axis to attach the twin to; a new figure if ``None``.

    Returns:
        The twin axis.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))
    x = np.asarray(x, dtype=float)
    ax2 = ax.twinx()
    ax2.set_zorder(0)
    ax.set_zorder(1)
    ax.patch.set_visible(False)

    mask = _finite_everywhere(x, series)
    for kind, values in series.items():
        if kind not in hidden:
            ax2.step(
                x[mask],
                values[mask],
                where="post",
                linewidth=2.0,
                color=KIND_COLORS[kind],
                alpha=0.7,
                label=KIND_LABELS[kind],
            )
    ax2.set_ylabel(
        "Fraction of Data Removed" if fraction else "Cumulative Reversals",
        rotation=-90,
        labelpad=15,
    )
    ax2.set_ylim(*((0, 1.05) if fraction else _count_ylim(series, mask)))
    ax2.spines["top"].set_visible(False)
    return ax2


def plot_choice_probs_with_removal(
    x: ArrayLike,
    across: dict,
    series: dict[str, np.ndarray],
    *,
    fraction: bool,
    hidden: Collection[str] = (),
    ax: plt.Axes | None = None,
) -> tuple[plt.Axes, plt.Axes]:
    """Choice-probability curves with the data-removed twin axis and a merged legend.

    Args:
        x: Trial offsets relative to the reversal.
        across: Aggregated statistics dict.
        series: ``{kind: values over x}`` removal curves; see
            :func:`plot_cumulative_removed`.
        fraction: Whether *series* holds fractions (twin axis 0-1) or raw counts.
        hidden: Kinds in *series* to leave undrawn (still used for the axis limits).
        ax: Axis to draw into; a new figure if ``None``. Its zorder and
            background are changed by :func:`plot_cumulative_removed`.

    Returns:
        ``(ax, ax2)``: the main axis and its twin.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))
    x = np.asarray(x, dtype=float)
    _draw_choice_curves(ax, x, across, _full_xlim(x), REMOVAL_YLIM)
    ax2 = plot_cumulative_removed(x, series, fraction=fraction, hidden=hidden, ax=ax)

    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    ax.set_title(_title(across), pad=20)

    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles + handles2, labels + labels2, loc="upper left", fontsize=10)
    return ax, ax2


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _save_or_show(fig: plt.Figure, save_path) -> None:
    """Save *fig* as ``<save_path>.pdf/.png``, or show it if there is no path; then close it."""
    if save_path:
        base = Path(save_path)
        base.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(base) + ".pdf", bbox_inches="tight")
        fig.savefig(str(base) + ".png", dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def _plain_figure(x: ArrayLike, across: dict) -> plt.Figure:
    ax = plot_choice_probs(x, across)
    ax.figure.tight_layout()
    return ax.figure


def _broken_axis_figure(
    x: ArrayLike, across: dict, skip_n_trials_after_reversal: int
) -> plt.Figure:
    ax_left, _ = plot_choice_probs_broken_axis(x, across, skip_n_trials_after_reversal)
    fig = ax_left.figure
    fig.supxlabel(X_LABEL)
    fig.suptitle(
        f"{_title(across)}(Skipping first {skip_n_trials_after_reversal} trials post-reversal)",
        y=1.03,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout()
    return fig


def plot_choice_probs_around_good_reversals(
    x: ArrayLike,
    across: dict,
    add_cumulative_axis: bool = True,
    only_good: bool = False,
    use_total: bool = False,
    windows_for_cumulative_axis: dict | None = None,
    all_good_idx: dict | None = None,
    all_bad_idx: dict | None = None,
    all_unclassified_idx: dict | None = None,
    skip_n_trials_after_reversal: int = 0,
    save_path: str | Path | None = None,
) -> None:
    """Plot reversal-aligned choice-probability curves for a single dataset.

    Mean ± SE curves for the Previous Best, New Best and Third Arm, optionally
    with a twin y axis of the fraction of data removed by later reversals.

    With ``skip_n_trials_after_reversal > 0`` the figure is a broken axis (two
    subplots sharing y) that omits the skipped post-reversal trials; that
    layout has no cumulative axis, so ``add_cumulative_axis`` is ignored.

    Args:
        x: Trial offsets relative to the reversal (negative = pre, else post).
        across: Aggregated statistics dict with keys ``"mean"``, ``"se"``,
            ``"num_subjects"``, and ``"num_reversals"`` as returned by
            :func:`src.behavior_analysis.get_choice_probs_around_good_reversals.get_choice_probs_around_good_reversals`.
        add_cumulative_axis: If ``True``, hand over to
            :func:`add_cumulative_rev_axis` (default: ``True``).
        only_good: In good/bad mode, draw only the good-reversal line
            (default: ``False``).
        use_total: If ``True``, use *total* mode (one line, every reversal
            kind) instead of good/bad mode (default: ``False``).
        windows_for_cumulative_axis: ``{subject: [reversal_dict]}`` windows
            dict for the cumulative axis; required when ``add_cumulative_axis``.
        all_good_idx: ``{subject: [int]}`` good-reversal indices (cumulative axis).
        all_bad_idx: ``{subject: [int]}`` bad-reversal indices (cumulative axis).
        all_unclassified_idx: ``{subject: [int]}`` block changes of sessions
            without good/bad counters
            (:func:`src.behavior_analysis.get_total_reversals.get_unclassified_reversal_indices`);
            ``None`` if every session logs counters.
        skip_n_trials_after_reversal: Post-reversal trials to omit (default: 0).
        save_path: Base path without extension for ``.pdf`` / ``.png`` output;
            if ``None``, the figure is shown.
    """
    if skip_n_trials_after_reversal > 0:
        fig = _broken_axis_figure(x, across, skip_n_trials_after_reversal)
    elif add_cumulative_axis:
        # Known issue: this plain figure is never saved or closed, so it is shown as a
        # duplicate alongside the cumulative figure (or, when saving, leaks one open
        # figure per call). Only the cumulative figure is output.
        _plain_figure(x, across)
        add_cumulative_rev_axis(
            x,
            across,
            windows_for_cumulative_axis,
            all_good_idx,
            all_bad_idx,
            save_path,
            only_good=only_good,
            use_total=use_total,
            all_unclassified_idx=all_unclassified_idx,
        )
        return
    else:
        fig = _plain_figure(x, across)
    _save_or_show(fig, save_path)


def add_cumulative_rev_axis(
    x: ArrayLike,
    across: dict,
    windows_for_cumulative_axis: dict | None = None,
    all_good_idx: dict | None = None,
    all_bad_idx: dict | None = None,
    save_path: str | Path | None = None,
    plot_fraction_removed: bool = True,
    only_good: bool = False,
    use_total: bool | None = None,
    all_unclassified_idx: dict | None = None,
) -> None:
    """Plot choice-probability curves with a cumulative fraction-removed twin axis.

    Creates a new figure with the reversal-aligned mean ± SE curves on the
    primary axis and a step-function showing how many anchor windows have been
    "cut short" by subsequent reversals on the secondary (right) axis.

    Args:
        x: 1-D array of trial offsets relative to the reversal.
        across: Aggregated statistics dict (same structure as in
            :func:`plot_choice_probs_around_good_reversals`).
        windows_for_cumulative_axis: ``{subject: [reversal_dict]}`` windows
            dict used to compute the cumulative axis.
        all_good_idx: ``{subject: [int]}`` good-reversal index lists.  Required.
        all_bad_idx: ``{subject: [int]}`` bad-reversal index lists.  Required.
        save_path: Base path for saving ``.pdf`` / ``.png`` files.  If
            ``None``, the figure is shown interactively.
        plot_fraction_removed: If ``True``, plot the fraction of data removed
            (range 0–1); if ``False``, plot raw cumulative reversal counts
            (default: ``True``).
        only_good: Suppress the bad-reversal line in good/bad mode (default:
            ``False``).
        use_total: If truthy, use total-mode cumulative axis (one line, every
            reversal kind); if falsy, use good/bad-mode (two lines, default:
            ``None`` → falsy).
        all_unclassified_idx: ``{subject: [int]}`` reversals of unknown kind
            (blocks-only sessions); adds an "unclassified" line in good/bad
            mode.

    Raises:
        ValueError: If ``all_good_idx`` or ``all_bad_idx`` is missing.
    """
    x = np.asarray(x, dtype=float)
    series = cumulative_removal_series(
        windows_for_cumulative_axis,
        all_good_idx,
        all_bad_idx,
        x,
        use_total=use_total,
        fraction=plot_fraction_removed,
        unclassified_idx=all_unclassified_idx,
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_choice_probs_with_removal(
        x,
        across,
        series,
        fraction=plot_fraction_removed,
        hidden=("bad",) if only_good else (),
        ax=ax,
    )
    fig.tight_layout()
    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Removal curves in the original return shape
# ---------------------------------------------------------------------------


def cumulative_reversal_events_over_post(
    good_windows: dict,
    all_good_idx: dict,
    all_bad_idx: dict,
    x: ArrayLike,
    across: dict | None,
    exclude_anchor_good: bool = True,
) -> tuple[np.ndarray, ...]:
    """Compute cumulative good/bad reversal removal curves over the post window.

    For each aligned good-reversal anchor *g*, finds the first subsequent
    reversal (good or bad) and marks that anchor as "removed" for all offsets
    ``t >= (next_rev - g)``.  The returned ``fraction_removed_good`` and
    ``fraction_removed_bad`` arrays therefore sum to ≤ 1 at every offset by
    construction.  The computation is
    :func:`src.behavior_analysis.cumulative_reversal_removal.good_bad_removal_curves`.

    Args:
        good_windows: ``{subject: [reversal_dict]}`` windows dict.
        all_good_idx: ``{subject: [int]}`` good-reversal trial indices.
        all_bad_idx: ``{subject: [int]}`` bad-reversal trial indices.
        x: 1-D float array of trial offsets.
        across: Aggregated statistics dict (unused, kept for API symmetry).
        exclude_anchor_good: If ``True``, only count reversals *after* anchor
            *g*; if ``False``, reversals at *g* itself are also eligible
            (default: ``True``).

    Returns:
        A five-tuple ``(x_disp, cumulative_good_full, cumulative_bad_full,
        fraction_removed_good, fraction_removed_bad)`` where each is a 1-D
        float array the same length as *x*.  Pre-window offsets are ``nan``.
    """
    x = np.asarray(x, dtype=float)
    curves = good_bad_removal_curves(
        good_windows, all_good_idx, all_bad_idx, x, exclude_anchor=exclude_anchor_good
    )
    return (x.copy(), *curves)


def cumulative_total_events_over_post(
    windows: dict,
    x: ArrayLike,
    across: dict | None,
    exclude_anchor: bool = True,
    *,
    good_idx: dict,
    bad_idx: dict,
    unclassified_idx: dict | None = None,
) -> tuple[np.ndarray, ...]:
    """Compute a single cumulative fraction-removed curve over the post window.

    For each aligned anchor reversal at index *g*, finds the first subsequent
    reversal after *g*, of any kind, and marks *g* as removed for all offsets
    ``t >= (next_rev - g)``.  The curve is the good and bad curves of
    :func:`cumulative_reversal_events_over_post` added together, plus the
    unclassified ones.

    This produces a monotonically non-decreasing step function bounded in
    ``[0, 1]``.  The computation is
    :func:`src.behavior_analysis.cumulative_reversal_removal.total_removal_curves`.

    Args:
        windows: ``{subject: [reversal_dict]}`` windows dict.
        x: 1-D float array of trial offsets.
        across: Aggregated statistics dict (unused, kept for API symmetry).
        exclude_anchor: If ``True``, only count reversals strictly after anchor
            *g*; if ``False``, reversals at *g* itself are also eligible
            (default: ``True``).
        good_idx: ``{subject: [int]}`` good-reversal trial indices.
        bad_idx: ``{subject: [int]}`` bad-reversal trial indices.
        unclassified_idx: ``{subject: [int]}`` reversals of unknown kind (block
            changes of sessions without good/bad counters); ``None`` if none.

    Returns:
        A three-tuple ``(x_disp, cum_total_full, frac_total_full)`` where each
        is a 1-D float array the same length as *x*.  Pre-window offsets are
        ``nan``.  ``frac_total_full`` is the fraction of anchor windows removed
        at each offset (in ``[0, 1]``).
    """
    x = np.asarray(x, dtype=float)
    curves = total_removal_curves(
        windows,
        x,
        exclude_anchor=exclude_anchor,
        good_idx=good_idx,
        bad_idx=bad_idx,
        unclassified_idx=unclassified_idx,
    )
    return (x.copy(), *curves)
