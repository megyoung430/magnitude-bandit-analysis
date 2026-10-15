"""Choice probability and reward magnitude of the three arms, aligned to block changes.

Pipeline (all pure; no files, no plots):

1. ``get_good_reversal_info`` windows each reversal; :func:`add_magnitude_windows` adds the
   reward magnitude of every arm over the same window.
2. :func:`compute_reversal_aligned_arms` labels the arms of each reversal by the ranking
   *before* the reversal (``"prev"``) or *after* it (``"new"``), keeps the reversals of one
   type, and averages one-hot choices and magnitudes per arm label: over reversals within a
   subject, then over subjects.

Alignment: every row has ``pre + post - skip`` bins. Pre values are right-aligned so the last
one sits at offset -1; post values start at offset 0. A window that is shorter than ``pre`` or
``post`` (near the start of the data, or cut by the next boundary) leaves NaN at the far end,
never shifts the other values.

Types and ties follow :mod:`src.behavior_analysis.reversal_arm_ranks`; the type is defined by
the prev ranking for both schemes.
"""

import warnings
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from src.behavior_analysis.get_choice_probs_around_good_reversals import (
    _keep_zero_then_skip,
    _smooth_pre_post,
    _trial_offsets,
)
from src.behavior_analysis.get_good_reversal_info import get_good_reversal_info
from src.behavior_analysis.get_variables_across_sessions import get_vars_across_all_sessions
from src.behavior_analysis.reversal_arm_ranks import (
    LABEL_SCHEMES,
    SCHEME_ROLES,
    SelectionCounts,
    arm_labels,
    select_reversals,
)
from src.behavior_analysis.stats import nan_standard_error

MOVING_AVG_MODES = ("centered", "trailing")
_SUM_TOLERANCE = 1e-6


class EmptySelectionWarning(UserWarning):
    """No reversal of the requested scheme and type is left, so there is nothing to average."""


class _Curves(NamedTuple):
    """Mean and SE over subjects for each role, plus the per-subject means."""

    mean: dict[str, np.ndarray]
    se: dict[str, np.ndarray]
    per_subject: dict[str, dict[str, np.ndarray]]


@dataclass(frozen=True)
class ReversalAlignedArms:
    """Aligned curves of one label scheme and reversal type.

    Attributes:
        scheme, reversal_type: What was computed (``"prev"``/``"new"``; ``"all"``,
            ``"best_to_third"``, ``"second_to_third"``).
        roles: Curve names, best to third, e.g. ``("prev_best", "prev_second", "prev_third")``.
        trial_offsets: Trials from the reversal of each bin. With a skip ``k``, bins after the
            reversal trial carry offsets ``k + 1`` upward.
        choice_mean, choice_se: ``{role: array}`` over ``trial_offsets``; the mean choice
            probability (after the moving average) and its standard error across subjects
            (NaN where fewer than two subjects have data).
        magnitude_mean, magnitude_se: Same for the reward magnitude of the arm with that
            label (not smoothed).
        per_subject_choice: ``{subject: {role: smoothed mean choice probability}}``.
        num_subjects, num_reversals: What the curves average over.
        counts: What the selection kept and why it left reversals out.
        pre, post, skip, moving_avg_window, moving_avg_mode: Parameters used.
    """

    scheme: str
    reversal_type: str
    roles: tuple[str, str, str]
    trial_offsets: np.ndarray
    choice_mean: dict[str, np.ndarray]
    choice_se: dict[str, np.ndarray]
    magnitude_mean: dict[str, np.ndarray]
    magnitude_se: dict[str, np.ndarray]
    per_subject_choice: dict[str, dict[str, np.ndarray]]
    num_subjects: int
    num_reversals: int
    counts: SelectionCounts
    pre: int
    post: int
    skip: int
    moving_avg_window: int
    moving_avg_mode: str


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------


def align_window_values(pre_values, post_values, *, pre: int, post: int, skip: int = 0):
    """One row of ``pre + post - skip`` bins from a reversal's pre and post values.

    Args:
        pre_values: Values of the trials before the reversal, oldest first (at most *pre*).
        post_values: Values from the reversal trial on (at most *post*).
        pre: Trials before the reversal in the row.
        post: Trials from the reversal on in the row, before skipping.
        skip: Post trial 0 is kept and post trials ``1 .. skip`` are dropped.

    Returns:
        Float array. The last pre value is in bin ``pre - 1`` (offset -1), the first post value
        in bin ``pre`` (offset 0); bins without a value are NaN.

    Raises:
        ValueError: If *skip* is negative or leaves no post window, or if a window is longer
            than *pre* or *post* (it was built with other settings).
    """
    n_bins = len(_trial_offsets(pre=pre, post=post, skip=skip))
    kept_post = _keep_zero_then_skip(list(post_values), skip)
    if len(pre_values) > pre:
        raise ValueError(f"pre window has {len(pre_values)} trials but pre={pre}")
    if len(kept_post) > post - skip:
        raise ValueError(
            f"post window has {len(kept_post)} trials after skipping; max {post - skip}"
        )
    row = np.full(n_bins, np.nan)
    row[pre - len(pre_values) : pre] = np.asarray(pre_values, dtype=float)
    row[pre : pre + len(kept_post)] = np.asarray(kept_post, dtype=float)
    return row


def true_trial_offsets(*, pre: int, post: int, skip: int = 0) -> np.ndarray:
    """Trials from the reversal of each bin: the post bins after the first sit *skip* later."""
    bins = _trial_offsets(pre=pre, post=post, skip=skip)
    return np.where(bins >= 1, bins + skip, bins)


# ---------------------------------------------------------------------------
# Magnitude windows
# ---------------------------------------------------------------------------


def _check_subject_has_sessions(windows: dict, subjects_trials: dict) -> None:
    for subject, reversals in windows.items():
        if reversals and not subjects_trials.get(subject):
            raise ValueError(f"{subject}: no sessions to read reward magnitudes from")


def _check_series_cover_windows(subject: str, reward_by_tower: dict, reversal: dict) -> None:
    window = reversal["trial_window_idx"]
    last = max([*window["pre"], *window["post"], reversal["reversal_idx"]], default=-1)
    short = [t for t, values in reward_by_tower.items() if len(values) <= last]
    if short:
        raise ValueError(f"{subject}: reward magnitudes of {short} end before trial {last}")


def _check_matches_stored_magnitudes(subject: str, reward_by_tower: dict, reversal: dict) -> None:
    """Raise unless the series gives the before/after magnitudes that the window recorded.

    The window and the series must come from the same merge (same sessions, same order);
    comparing the two values stored with the reversal catches any other pairing.
    """
    idx = reversal["reversal_idx"]
    for key, trial in (
        ("reward_magnitudes_by_tower_before", max(idx - 1, 0)),
        ("reward_magnitudes_by_tower_after", idx),
    ):
        stored = reversal[key]
        found = {t: reward_by_tower[t][trial] for t in stored}
        if found != stored:
            raise ValueError(
                f"{subject}: reward magnitudes at merged trial {trial} do not match the window "
                f"of the reversal at {idx} ({key}: {stored} vs {found}); the windows and the "
                "sessions must come from the same data."
            )


def _magnitude_windows(reward_by_tower: dict, reversal: dict) -> dict[str, dict[str, list]]:
    window = reversal["trial_window_idx"]
    return {
        tower: {
            "pre": [values[i] for i in window["pre"]],
            "post": [values[i] for i in window["post"]],
        }
        for tower, values in reward_by_tower.items()
    }


def add_magnitude_windows(reversal_windows: dict, subjects_trials: dict) -> dict:
    """Copy of the reversal windows with every arm's reward magnitude over each window.

    The magnitudes come from the same merge of sessions that ``get_good_reversal_info`` uses
    (``get_vars_across_all_sessions``: date-then-number session order, one entry per trial), so
    they sit on the windows' trial axis. The magnitudes at the reversal trial and the trial
    before it are checked against the ones stored in each reversal.

    Args:
        reversal_windows: ``{subject: [reversal dict]}`` from ``get_good_reversal_info``.
        subjects_trials: The ``{subject: {session: session dict}}`` the windows were made from.

    Returns:
        ``{subject: [reversal dict]}`` where each reversal dict has a new key
        ``"magnitudes_by_tower"``: ``{tower: {"pre": [...], "post": [...]}}``, the same shape
        as ``"choices_by_tower"``. The input is not modified.

    Raises:
        ValueError: If a subject has no sessions, a series is shorter than the window, or the
            magnitudes do not match those stored in the reversal.
    """
    _check_subject_has_sessions(reversal_windows, subjects_trials)
    subjects = [s for s, revs in reversal_windows.items() if revs]
    merged, _ = get_vars_across_all_sessions({s: subjects_trials[s] for s in subjects})
    enriched: dict[str, list[dict]] = {s: [] for s in reversal_windows}
    for subject in subjects:
        reward_by_tower = merged[subject]["reward_magnitudes_by_tower"]
        for reversal in reversal_windows[subject]:
            _check_series_cover_windows(subject, reward_by_tower, reversal)
            _check_matches_stored_magnitudes(subject, reward_by_tower, reversal)
            enriched[subject].append(
                {**reversal, "magnitudes_by_tower": _magnitude_windows(reward_by_tower, reversal)}
            )
    return enriched


def build_windows_with_magnitudes(subjects_trials: dict, *, pre: int, post: int) -> dict:
    """Reversal windows of one problem, without the first block, with magnitude windows.

    Args:
        subjects_trials: ``{subject: {session: session dict}}`` of one problem.
        pre, post: Trials before and from each reversal in the windows.

    Returns:
        ``{subject: [reversal dict]}`` as :func:`add_magnitude_windows` returns it.
    """
    windows = get_good_reversal_info(subjects_trials, pre=pre, post=post, include_first_block=False)
    return add_magnitude_windows(windows, subjects_trials)


# ---------------------------------------------------------------------------
# Averaging
# ---------------------------------------------------------------------------


def _nan_mean(stack: np.ndarray) -> np.ndarray:
    """Column mean over the non-NaN rows; NaN (without a warning) for all-NaN columns."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        return np.nanmean(stack, axis=0)


def _reversal_rows(reversal: dict, field: str, arms, *, pre: int, post: int, skip: int):
    """``(3, n_bins)`` rows of ``reversal[field]`` for *arms* (best, second, third order)."""
    return np.vstack(
        [
            align_window_values(
                reversal[field][arm]["pre"],
                reversal[field][arm]["post"],
                pre=pre,
                post=post,
                skip=skip,
            )
            for arm in arms
        ]
    )


def _subject_means(reversals: list[dict], scheme: str, field: str, **window) -> np.ndarray:
    """``(3, n_bins)`` mean over a subject's reversals of the rows of *field*."""
    rows = [
        _reversal_rows(r, field, arm_labels(r, scheme), **window) for r in reversals
    ]  # (n_reversals, 3, n_bins)
    return _nan_mean(np.stack(rows))


def _check_sum_to_one(means: dict[str, np.ndarray], where: str) -> None:
    total = sum(means.values())
    finite = total[np.isfinite(total)]
    if not np.allclose(finite, 1.0, atol=_SUM_TOLERANCE):
        raise ValueError(
            f"choice probabilities of the three arms do not sum to 1 ({where}); "
            f"sums range {finite.min():.4f} to {finite.max():.4f}. Each trial must have "
            "exactly one chosen arm in the windows."
        )


def _across_subjects(per_subject: dict[str, dict[str, np.ndarray]], roles) -> _Curves:
    mean, se = {}, {}
    for role in roles:
        stack = np.vstack([curves[role] for curves in per_subject.values()])
        mean[role], se[role] = _nan_mean(stack), nan_standard_error(stack)
    return _Curves(mean, se, per_subject)


def _nan_curves(roles, n_bins: int) -> _Curves:
    nan = {role: np.full(n_bins, np.nan) for role in roles}
    return _Curves(nan, {r: a.copy() for r, a in nan.items()}, {})


def _choice_curves(selected, scheme, roles, offsets, *, smooth, **window) -> _Curves:
    per_subject = {}
    for subject, reversals in selected.items():
        raw = dict(
            zip(roles, _subject_means(reversals, scheme, "choices_by_tower", **window), strict=True)
        )
        _check_sum_to_one(raw, f"subject {subject}")
        per_subject[subject] = {
            role: _smooth_pre_post(y, offsets < 0, smooth["window"], smooth["mode"])
            for role, y in raw.items()
        }
    curves = _across_subjects(per_subject, roles)
    _check_sum_to_one(curves.mean, "mean across subjects after smoothing")
    return curves


def _magnitude_curves(selected, scheme, roles, **window) -> _Curves:
    per_subject = {
        subject: dict(
            zip(
                roles,
                _subject_means(reversals, scheme, "magnitudes_by_tower", **window),
                strict=True,
            )
        )
        for subject, reversals in selected.items()
    }
    return _across_subjects(per_subject, roles)


def compute_reversal_aligned_arms(
    reversal_windows: dict,
    *,
    scheme: str,
    reversal_type: str = "all",
    pre: int = 10,
    post: int = 30,
    skip_n_trials_after_reversal: int = 0,
    moving_avg_window: int = 4,
    moving_avg_mode: str = "centered",
) -> ReversalAlignedArms:
    """Choice probability and reward magnitude of the three arms around reversals.

    Args:
        reversal_windows: ``{subject: [reversal dict]}`` from ``get_good_reversal_info`` after
            :func:`add_magnitude_windows`, built with the same *pre* and *post* and without the
            first-block record. Pooling across problems is concatenation per subject
            (``merge_windows_by_subject``).
        scheme: ``"prev"`` labels arms by their rank before the reversal, ``"new"`` by their
            rank after it.
        reversal_type: ``"all"``, ``"best_to_third"`` or ``"second_to_third"``; the type is
            always defined by the prev ranking. Reversals with tied magnitudes are left out as
            :func:`~src.behavior_analysis.reversal_arm_ranks.select_reversals` describes.
        pre, post: Trials before and from the reversal in each window.
        skip_n_trials_after_reversal: Post trial 0 is kept and the next *k* trials dropped.
        moving_avg_window: Width of the moving average of the per-subject choice probabilities
            (``<= 1`` for none), applied separately before and after the reversal. Magnitudes
            are not smoothed.
        moving_avg_mode: ``"centered"`` or ``"trailing"``; the two give different curves, so
            compare figures only if they use the same mode.

    Returns:
        A :class:`ReversalAlignedArms`. With no reversal left, its curves are all NaN and an
        :class:`EmptySelectionWarning` is raised.

    Raises:
        ValueError: For an unknown scheme, type or mode, a window longer than *pre*/*post*, a
            first-block record, or choice probabilities that do not sum to 1.
    """
    if scheme not in LABEL_SCHEMES:
        raise ValueError(f"scheme must be one of {LABEL_SCHEMES}, got {scheme!r}")
    if moving_avg_mode not in MOVING_AVG_MODES:
        raise ValueError(
            f"moving_avg_mode must be one of {MOVING_AVG_MODES}, got {moving_avg_mode!r}"
        )
    skip = skip_n_trials_after_reversal
    roles = SCHEME_ROLES[scheme]
    offsets = true_trial_offsets(pre=pre, post=post, skip=skip)
    selected, counts = select_reversals(reversal_windows, scheme, reversal_type)
    window = dict(pre=pre, post=post, skip=skip)
    smooth = {"window": moving_avg_window, "mode": moving_avg_mode}
    if selected:
        choice = _choice_curves(selected, scheme, roles, offsets, smooth=smooth, **window)
        magnitude = _magnitude_curves(selected, scheme, roles, **window)
    else:
        warnings.warn(
            f"no reversal left for scheme {scheme!r}, reversal_type {reversal_type!r}: {counts}",
            EmptySelectionWarning,
            stacklevel=2,
        )
        choice = magnitude = _nan_curves(roles, len(offsets))
    return ReversalAlignedArms(
        scheme=scheme,
        reversal_type=reversal_type,
        roles=roles,
        trial_offsets=offsets,
        choice_mean=choice.mean,
        choice_se=choice.se,
        magnitude_mean=magnitude.mean,
        magnitude_se=magnitude.se,
        per_subject_choice=choice.per_subject,
        num_subjects=len(selected),
        num_reversals=counts.selected,
        counts=counts,
        pre=pre,
        post=post,
        skip=skip,
        moving_avg_window=moving_avg_window,
        moving_avg_mode=moving_avg_mode,
    )
