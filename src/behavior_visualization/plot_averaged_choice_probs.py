"""Utilities for averaging choice-probability curves across problems.

Functions here compute and plot choice-probability curves averaged across
multiple problems (≥ some cutoff), with a corrected cumulative fraction-removed
axis.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.behavior_visualization.plot_style import CHOICE_PROB_COLOR_MAP as COLOR_MAP

# Keys used across the choice-probability analysis
_CHOICE_KEYS = ("prev_best", "next_best", "third")


# ---------------------------------------------------------------------------
# Curve computation helpers
# ---------------------------------------------------------------------------

def compute_curve_from_windows(
    reversal_windows: dict,
    *,
    pre: int,
    post: int,
    skip_n: int = 0,
    do_moving_avg: bool = False,
    moving_avg_window: int = 4,
) -> tuple:
    """Compute choice-probability curves from a single problem's reversal windows.

    Args:
        reversal_windows: ``{subject: list[reversal_dict]}`` for one problem.
        pre: Number of trials before the reversal to include.
        post: Number of trials after the reversal to include.
        skip_n: Number of trials immediately after the reversal to mark as
            missing (default: 0).
        do_moving_avg: If ``True``, apply a centred moving average to the
            curves (default: False).
        moving_avg_window: Window size for the moving average (default: 4).

    Returns:
        ``(x, per_subject, across)`` as returned by
        :func:`src.behavior_analysis.get_choice_probs_around_good_reversals.get_choice_probs_around_good_reversals`.
    """
    from src.behavior_analysis.get_choice_probs_around_good_reversals import (
        get_choice_probs_around_good_reversals,
        apply_moving_average_to_choice_probs,
    )

    x, per_subject, across = get_choice_probs_around_good_reversals(
        reversal_windows, pre=pre, post=post,
        skip_n_trials_after_reversal=skip_n,
    )
    if do_moving_avg:
        x, per_subject, across = apply_moving_average_to_choice_probs(
            x, per_subject, moving_avg_window=moving_avg_window, mode="centered"
        )
    return x, per_subject, across


def curves_by_problem_from_windows_dict(
    windows_dict: dict,
    *,
    pre: int,
    post: int,
    skip_n: int = 0,
    do_moving_avg: bool = False,
    moving_avg_window: int = 4,
) -> dict:
    """Compute per-problem choice-probability curves from a dict of windows.

    Args:
        windows_dict: ``{problem: reversal_windows}``
        pre: Trials before reversal.
        post: Trials after reversal.
        skip_n: Trials to skip after reversal (default: 0).
        do_moving_avg: Apply moving average (default: False).
        moving_avg_window: Moving-average window size (default: 4).

    Returns:
        ``{problem: {"x": ..., "across": ..., "subjects": set, "num_reversals": int}}``
        for each problem that had at least one subject contributing data.
    """
    out: dict = {}
    for p, rw in windows_dict.items():
        if not rw:
            continue
        x, per_subject, across = compute_curve_from_windows(
            rw, pre=pre, post=post, skip_n=skip_n,
            do_moving_avg=do_moving_avg, moving_avg_window=moving_avg_window,
        )
        if across.get("num_subjects", 0) == 0:
            continue
        out[p] = {
            "x": np.asarray(x, float),
            "across": across,
            "subjects": set(per_subject.keys()),
            "num_reversals": int(across.get("num_reversals", 0) or 0),
        }
    return out


def average_across_problem_curves(curves_by_problem: dict) -> tuple:
    """Average per-problem choice-probability curves across problems.

    Each problem contributes equally (one replicate) to the mean.

    Args:
        curves_by_problem: Dict as returned by
            :func:`curves_by_problem_from_windows_dict`.

    Returns:
        ``(x, across_avg)`` where *across_avg* contains keys
        ``"mean"``, ``"se"``, ``"num_subjects"``, ``"num_reversals"``,
        ``"num_problems"``, and ``"problems_used"``.

    Raises:
        ValueError: If *curves_by_problem* is empty or if the x-arrays do not
            match across problems.
    """
    probs = sorted(curves_by_problem.keys())
    if not probs:
        raise ValueError("No problems to average.")

    x0 = curves_by_problem[probs[0]]["x"]
    for p in probs[1:]:
        xp = curves_by_problem[p]["x"]
        if x0.shape != xp.shape or not np.allclose(x0, xp, equal_nan=True):
            raise ValueError(
                f"x-array mismatch between problems {probs[0]} and {p}. "
                "Ensure pre/post/skip_n are consistent."
            )

    mean_out: dict = {}
    se_out: dict = {}
    for k in _CHOICE_KEYS:
        mats = []
        for p in probs:
            y = curves_by_problem[p]["across"]["mean"].get(k)
            if y is None:
                continue
            mats.append(np.asarray(y, dtype=float))
        if not mats:
            mean_out[k] = None
            se_out[k] = None
            continue
        Y = np.vstack(mats)  # (n_problems, T)
        mean_out[k] = np.nanmean(Y, axis=0)
        se_out[k] = (
            np.nanstd(Y, axis=0, ddof=1) / np.sqrt(Y.shape[0])
            if Y.shape[0] > 1
            else np.zeros_like(mean_out[k])
        )

    subjects_union: set = set()
    total_reversals = 0
    for p in probs:
        subjects_union |= curves_by_problem[p]["subjects"]
        total_reversals += int(curves_by_problem[p]["num_reversals"] or 0)

    across_avg = {
        "mean": mean_out,
        "se": se_out,
        "num_subjects": len(subjects_union),
        "num_reversals": total_reversals,
        "num_problems": len(probs),
        "problems_used": probs,
    }
    return x0, across_avg


def fraction_removed_by_problem(
    windows_dict: dict,
    curves_by_problem: dict,
    x: np.ndarray,
    *,
    exclude_anchor: bool = True,
) -> tuple[np.ndarray, list]:
    """Compute the average cumulative fraction-removed curve across problems.

    Fraction removed is computed within each problem (never by merging windows
    across problems), then averaged across problems.

    Args:
        windows_dict: ``{problem: reversal_windows}``.
        curves_by_problem: As returned by
            :func:`curves_by_problem_from_windows_dict`.
        x: Trial-offset array.
        exclude_anchor: Passed to ``cumulative_total_events_over_post``
            (default: True).

    Returns:
        ``(frac_avg, probs_used)`` where *frac_avg* is the mean
        fraction-removed curve averaged over problems.
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
        across_p = curves_by_problem[p]["across"]
        _, _, frac = cumulative_total_events_over_post(
            rw, x, across_p, exclude_anchor=exclude_anchor
        )
        fracs.append(np.asarray(frac, float))
        probs_used.append(p)

    if not fracs:
        return np.full_like(x, float("nan"), dtype=float), probs_used

    F = np.vstack(fracs)
    frac_avg = np.nanmean(F, axis=0)
    return frac_avg, probs_used


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

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
        x: Trial-offset array.
        across: Averaged across dict from :func:`average_across_problem_curves`.
        frac_removed: Fraction-removed curve from
            :func:`fraction_removed_by_problem`.
        save_path: If provided, saves ``.pdf`` and ``.png`` at this path.
            If ``None``, the figure is shown interactively.
        title_extra: Optional extra string appended to the figure title.
    """
    x = np.asarray(x, float)
    frac_removed = np.asarray(frac_removed, float)

    fig, ax = plt.subplots(figsize=(10, 6))

    label_map = {
        "prev_best": "Previous Best",
        "next_best": "New Best",
        "third": "Third Arm",
    }

    for key in _CHOICE_KEYS:
        y = np.asarray(across["mean"][key], float)
        s = np.asarray(across["se"][key], float)
        m = np.isfinite(x) & np.isfinite(y) & np.isfinite(s)
        if np.any(m):
            ax.plot(x[m], y[m], linewidth=2, color=COLOR_MAP[key],
                    label=label_map[key])
            ax.fill_between(x, y - s, y + s, where=m, alpha=0.2,
                            color=COLOR_MAP[key])

    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.axhline(1 / 3, color="gray", linestyle=":", linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_xlim(x[0], x[-1])
    ax.set_xlabel("Trials from Good Reversal")
    ax.set_ylabel("Choice Probability")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax2 = ax.twinx()
    m2 = np.isfinite(x) & np.isfinite(frac_removed)
    if np.any(m2):
        ax2.step(x[m2], frac_removed[m2], where="post", linewidth=2.0,
                 color="#FF9100", alpha=0.7, label="Total Rev")
    ax2.set_ylabel("Fraction of Data Removed", rotation=-90, labelpad=15)
    ax2.set_ylim(0, 1)
    ax2.spines["top"].set_visible(False)

    title = (
        "Reversal-Aligned Choices\n"
        f"(mean ± se across problems | "
        f"n={across.get('num_subjects', 0)} subjects, "
        f"n={across.get('num_reversals', 0)} reversals, "
        f"n={across.get('num_problems', 0)} problems)\n"
    )
    if title_extra:
        title += str(title_extra)
    ax.set_title(title, pad=20)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=10)

    plt.tight_layout()

    if save_path is not None:
        base = Path(save_path)
        base.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(base) + ".pdf", bbox_inches="tight")
        fig.savefig(str(base) + ".png", dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close(fig)
