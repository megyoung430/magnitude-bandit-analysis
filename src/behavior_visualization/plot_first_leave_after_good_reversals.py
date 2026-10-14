"""Visualise the direction of the first leave from the previously-best arm.

After a good reversal the animal must eventually visit a different arm.  This
module plots a grouped bar chart showing what fraction of the time that first
post-reversal choice was the new best arm versus the third (lowest-reward) arm,
with individual subject lines overlaid.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from src.behavior_analysis.first_leave_fractions import first_leave_fractions
from src.behavior_visualization.plot_helpers import despine, save_or_show
from src.behavior_visualization.plot_style import MOUSE_COLORS

_LABELS = ["New Best", "Third Arm"]
_BAR_COLOR = "#999999"
_JITTER = 0.03


def _plot_mean_bars(ax: plt.Axes, x: np.ndarray, means: list, errs: list) -> None:
    """Two bars (mean ± SE across subjects), each labelled with its mean above the axes data."""
    ax.bar(
        x,
        means,
        yerr=errs,
        capsize=6,
        edgecolor="black",
        linewidth=1.5,
        alpha=0.55,
        color=[_BAR_COLOR, _BAR_COLOR],
    )
    for xi, m in zip(x, means, strict=False):
        ax.text(xi, 1.05, f"{m:.2f}", ha="center", va="bottom", fontsize=12)


def _plot_subject_lines(ax: plt.Axes, per_subject_counts: dict) -> list[Line2D]:
    """One jittered new-best -> third line per subject with reversals; returns legend handles.

    Known issue: the jitter draws from NumPy's global random state, so the x
    positions differ between runs unless the caller seeds it.
    """
    subjects = list(per_subject_counts.keys())
    handles = []
    for subj in subjects:
        counts = per_subject_counts[subj]
        fractions = first_leave_fractions(counts)
        if fractions is None:
            continue
        new_best, third = fractions
        # Known issue: a 17th subject raises IndexError; the palette is indexed, not cycled.
        c = MOUSE_COLORS[subjects.index(subj)]
        ax.plot(
            [0 + np.random.uniform(-_JITTER, _JITTER), 1 + np.random.uniform(-_JITTER, _JITTER)],
            [new_best, third],
            color=c,
            linewidth=2.5,
            alpha=0.9,
            marker="o",
            markersize=6,
        )
        handles.append(
            Line2D(
                [0],
                [0],
                color=c,
                lw=2.5,
                marker="o",
                label=f"{subj} (n={counts['total']} reversals)",
            )
        )
    return handles


def _style_axes(ax: plt.Axes, x: np.ndarray, mean: dict, n_subjects: int) -> None:
    # Known issue: n_subjects counts every subject in the input, including those
    # without reversals that are not drawn.
    despine(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(_LABELS, fontsize=12)
    ax.set_ylabel("Fraction of Good Reversals", fontsize=12)
    ax.set_ylim(0.0, 1.07)
    ax.set_title(
        "First Choice After Leaving Previous Best\n"
        f"(mean ± se across subjects | n={n_subjects} subjects "
        f"and n={mean['num_reversals']} reversals)",
        pad=20,
    )


def plot_first_leave_panel(
    mean: dict,
    se: dict,
    per_subject_counts: dict,
    p_value: float | None = None,
    *,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Draw first-leave proportions (new-best vs third arm) with per-subject lines.

    Args:
        mean: ``{"new_best", "third", "num_reversals"}``; see
            :func:`plot_first_leave_after_good_reversals`.
        se: ``{"new_best", "third"}`` standard errors across subjects.
        per_subject_counts: ``{subject: {"new_best", "third", "total"}}``.
        p_value: Annotated above the bars when given.
        ax: Axes to draw into; a new 6.5 x 5.5 inch figure is created if ``None``.

    Known issue: when the figure is created here, it stays open if drawing raises.
    """
    means = [mean["new_best"], mean["third"]]
    errs = [se["new_best"], se["third"]]
    x = np.arange(len(_LABELS))
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 5.5))
    _plot_mean_bars(ax, x, means, errs)
    handles = _plot_subject_lines(ax, per_subject_counts)
    ax.legend(handles=handles, fontsize=10, loc="upper right")
    _style_axes(ax, x, mean, len(per_subject_counts))
    if p_value is not None:
        ax.text(0.5, 1.02, f"p-value: {p_value:.3f}", ha="center", va="bottom", fontsize=12)
    return ax


def plot_first_leave_after_good_reversals(
    mean: dict,
    se: dict,
    per_subject_counts: dict,
    p_value: float | None = None,
    save_path: str | Path | None = None,
) -> None:
    """Plot a bar chart of first-leave proportions (new-best vs third arm).

    Draws two bars (New Best and Third Arm) with mean ± SE across subjects,
    overlays individual-subject lines with jitter, and optionally annotates
    with a p-value.

    Args:
        mean: Dict with keys ``"new_best"`` (float), ``"third"`` (float), and
            ``"num_reversals"`` (int) as returned by
            :func:`src.behavior_analysis.get_first_leave_after_good_reversals.average_first_leave_across_subjects`.
        se: Dict with keys ``"new_best"`` and ``"third"`` (floats), standard
            errors across subjects.
        per_subject_counts: ``{subject: {"new_best": int, "third": int,
            "total": int}}`` used to draw per-subject lines.
        p_value: Optional float p-value to annotate above the bars (default:
            ``None`` — no annotation).
        save_path: Base path (without extension) for saving ``.pdf`` and
            ``.png`` output.  If ``None``, the figure is shown interactively.
    """
    ax = plot_first_leave_panel(mean, se, per_subject_counts, p_value)
    fig = ax.figure
    fig.tight_layout()
    save_or_show(fig, save_path or None)
