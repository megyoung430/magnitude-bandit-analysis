"""Figure of choice probability and reward magnitude of the three arms around a reversal.

Choice probabilities (mean ± SE across subjects) are solid lines on the left axis; the mean
reward magnitude of the same three arms is a dashed step line of the same colour on a
secondary axis on the right. Two labelling schemes:

- ``prev``: arms labelled by their rank before the reversal (best blue, second green, third
  grey);
- ``new``: arms labelled by their rank after the reversal (best blue, second green, third
  black).

The analysis is in :mod:`src.behavior_analysis.reversal_aligned_arms`.
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.behavior_analysis.reversal_aligned_arms import ReversalAlignedArms
from src.behavior_visualization.plot_helpers import axes_or_new, despine, save_or_show

ROLE_COLORS = {
    "prev_best": "#5DA5DA",
    "prev_second": "#60BD68",
    "prev_third": "#7f7f7f",
    "new_best": "#5DA5DA",
    "new_second": "#60BD68",
    "new_third": "black",
}
ROLE_LABELS = {role: role.replace("_", " ").title() for role in ROLE_COLORS}
SCHEME_DESCRIPTIONS = {
    "prev": "arms labelled by their rank before the reversal",
    "new": "arms labelled by their rank after the reversal",
}
TYPE_LABELS = {
    "all": "All Reversals",
    "best_to_third": "Best to Third Reversals",
    "second_to_third": "Second to Third Reversals",
}
MIN_MAGNITUDE_YMAX = 4.0


def _title(result: ReversalAlignedArms, scope: str) -> str:
    left_out = result.counts.tied_labels + result.counts.tied_type
    detail = f"n={result.num_subjects} subjects, n={result.num_reversals} reversals"
    if left_out:
        detail += f", {left_out} left out for tied magnitudes"
    smoothing = (
        f"moving average over {result.moving_avg_window} trials ({result.moving_avg_mode})"
        if result.moving_avg_window > 1
        else "no moving average"
    )
    scope_text = f" [{scope}]" if scope else ""
    return (
        f"{TYPE_LABELS[result.reversal_type]}{scope_text}: {SCHEME_DESCRIPTIONS[result.scheme]}\n"
        "(type defined by the prev ranking)\n"
        f"mean ± se across subjects | {detail}\n{smoothing}"
    )


def _draw_choice_curves(ax: plt.Axes, result: ReversalAlignedArms) -> None:
    x = result.trial_offsets
    for role in result.roles:
        y, se = result.choice_mean[role], result.choice_se[role]
        shown = np.isfinite(y)
        ax.plot(
            x[shown], y[shown], lw=2, color=ROLE_COLORS[role], label=ROLE_LABELS[role], zorder=3
        )
        banded = shown & np.isfinite(se)
        if banded.any():
            ax.fill_between(
                x, y - se, y + se, where=banded, alpha=0.2, color=ROLE_COLORS[role], zorder=2
            )


def _magnitude_ylim(result: ReversalAlignedArms) -> tuple[float, float]:
    tops = [np.nanmax(m) for m in result.magnitude_mean.values() if np.isfinite(m).any()]
    return 0.0, max(MIN_MAGNITUDE_YMAX, math.ceil(max(tops)))


def _step_edges(x: np.ndarray, *ys: np.ndarray) -> tuple[np.ndarray, ...]:
    """*x* and each of *ys* with a last point one trial on, so the last step has a width."""
    return (np.append(x, x[-1] + 1), *(np.append(y, y[-1]) for y in ys))


def _draw_magnitude_curves(ax2: plt.Axes, result: ReversalAlignedArms) -> None:
    x = result.trial_offsets
    for role in result.roles:
        y, se = result.magnitude_mean[role], result.magnitude_se[role]
        shown = np.isfinite(y)
        xs, ys = _step_edges(x[shown], y[shown])
        ax2.step(
            xs,
            ys,
            where="post",
            lw=1.5,
            ls="--",
            color=ROLE_COLORS[role],
            alpha=0.85,
            label=f"{ROLE_LABELS[role]} magnitude",
        )
        banded = shown & np.isfinite(se)
        if banded.any():
            xe, lo, hi, band = _step_edges(x, y - se, y + se, banded)
            ax2.fill_between(
                xe, lo, hi, where=band, step="post", alpha=0.12, color=ROLE_COLORS[role]
            )
    ax2.set_ylim(*_magnitude_ylim(result))
    ax2.set_ylabel("Mean Reward Magnitude", rotation=-90, labelpad=15)
    ax2.spines["top"].set_visible(False)


def plot_reversal_aligned_arms(
    result: ReversalAlignedArms, ax: plt.Axes | None = None, *, scope: str = ""
) -> tuple[plt.Axes, plt.Axes]:
    """Choice probabilities and reward magnitudes of the three arms around the reversal.

    The SE band of a curve is left out where its SE is NaN (fewer than two subjects).

    Args:
        result: Output of
            :func:`~src.behavior_analysis.reversal_aligned_arms.compute_reversal_aligned_arms`.
        ax: Axes for the choice probabilities; a new figure if ``None``. Its zorder and
            background are changed so the twin's curves show behind it.
        scope: Text for the title naming the data, e.g. ``"problem-04"`` or ``"pooled"``.

    Returns:
        ``(ax, ax2)``: the choice-probability axes and the magnitude (right-hand) axes.

    Raises:
        ValueError: If the result holds no reversals.
    """
    if result.num_reversals == 0:
        raise ValueError(f"no reversals to plot for {result.scheme!r}/{result.reversal_type!r}")
    with axes_or_new(ax, figsize=(10, 5.5)) as ax:
        _draw_choice_curves(ax, result)
        ax.axvline(0, color="black", ls="--", lw=1, zorder=4)
        ax.axhline(1 / 3, color="gray", ls=":", lw=1, zorder=1)
        ax.set_xlim(result.trial_offsets[0], result.trial_offsets[-1] + 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Trials from Reversal")
        ax.set_ylabel("Choice Probability")
        despine(ax)

        ax2 = ax.twinx()
        ax2.set_zorder(0)
        ax.set_zorder(1)
        ax.patch.set_visible(False)
        _draw_magnitude_curves(ax2, result)

        handles, labels = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(
            handles + handles2,
            labels + labels2,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.14),
            ncol=3,
            fontsize=10,
            frameon=False,
        )
        ax.set_title(_title(result, scope), pad=12)
    return ax, ax2


def save_reversal_aligned_arms(
    result: ReversalAlignedArms, save_path: str | Path | None, *, scope: str = ""
) -> None:
    """Draw the figure and save it as ``<save_path>.png`` and ``.pdf``, or show it.

    Args:
        result: See :func:`plot_reversal_aligned_arms`.
        save_path: Base path without extension (parent folders are created); ``None`` or
            ``""`` shows the figure instead.
        scope: See :func:`plot_reversal_aligned_arms`.
    """
    ax, _ = plot_reversal_aligned_arms(result, scope=scope)
    ax.figure.tight_layout()
    save_or_show(ax.figure, save_path or None)
