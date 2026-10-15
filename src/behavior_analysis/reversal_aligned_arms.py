"""Choice probability and reward magnitude of the three arms, aligned to block changes.

Pipeline (all pure; no files, no plots):

1. ``get_good_reversal_info`` windows each reversal; :func:`add_magnitude_windows` adds the
   reward magnitude of every arm over the same window.
2. :func:`compute_reversal_aligned_arms` labels the arms of each reversal by the ranking
   *before* the reversal (``"prev"``) or *after* it (``"new"``), keeps the reversals of one
   type, and averages one-hot choices and magnitudes per arm label: over reversals within a
   subject, then over subjects.

Alignment: every row has ``pre + post - skip`` bins. Pre values are right-aligned so the last
one sits at offset -1; post values start at offset 0
(:mod:`src.behavior_analysis.reversal_alignment`). A window that is shorter than ``pre`` or
``post`` (the pre window stops at the previous block change or the first trial, the post window
at the next one) leaves NaN at the far end, never shifts the other values.

Types and ties follow :mod:`src.behavior_analysis.reversal_arm_ranks`; the type is defined by
the prev ranking for both schemes.
"""

import warnings
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from src.behavior_analysis.get_good_reversal_info import get_good_reversal_info
from src.behavior_analysis.get_variables_across_sessions import get_vars_across_all_sessions
from src.behavior_analysis.reversal_alignment import (
    align_window_values,
    smooth_pre_post,
    true_trial_offsets,
)
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


def _pre_room(blocks: list, idx: int) -> int:
    """Trials from the previous block change (or the first trial) up to the reversal at *idx*."""
    changes = [i for i in range(1, min(idx, len(blocks))) if blocks[i] != blocks[i - 1]]
    return idx - (changes[-1] if changes else 0)


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
        as ``"choices_by_tower"``, and ``"pre_room"``: trials from the previous block change.
        The input is not modified.

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
                {
                    **reversal,
                    "magnitudes_by_tower": _magnitude_windows(reward_by_tower, reversal),
                    "pre_room": _pre_room(merged[subject]["blocks"], reversal["reversal_idx"]),
                }
            )
    return enriched


def build_windows_with_magnitudes(subjects_trials: dict, *, pre: int, post: int) -> dict:
    """Reversal windows of one problem, without the first block, with magnitude windows.

    Args:
        subjects_trials: ``{subject: {session: session dict}}`` of one problem.
        pre, post: Trials before and from each reversal in the windows.

    Returns:
        ``{subject: [reversal dict]}`` as :func:`add_magnitude_windows` returns it, each
        reversal also recording the ``"window_pre"`` and ``"window_post"`` it was built with
        (see :func:`check_windows_built_with`).
    """
    windows = get_good_reversal_info(subjects_trials, pre=pre, post=post, include_first_block=False)
    enriched = add_magnitude_windows(windows, subjects_trials)
    return {
        subject: [{**r, "window_pre": pre, "window_post": post} for r in reversals]
        for subject, reversals in enriched.items()
    }


def check_windows_built_with(windows: dict, *, pre: int, post: int) -> None:
    """Raise unless every reversal was built with *pre* and *post* by the builder above.

    Windows built with a smaller pre or post would be NaN-padded to the requested length and
    look like ordinary data; each pre window must also be as long as the block before the
    reversal allows (the pre window stops at the previous block change).

    Raises:
        ValueError: Naming the first reversal whose settings or pre length do not fit.
    """
    for subject, reversals in windows.items():
        for r in reversals:
            built = (r.get("window_pre"), r.get("window_post"))
            if built != (pre, post):
                raise ValueError(
                    f"{subject}: reversal at {r.get('reversal_idx')} was windowed with "
                    f"(pre, post) = {built}, not ({pre}, {post}); rebuild it with "
                    "build_windows_with_magnitudes"
                )
            expected_pre = min(pre, r["pre_room"])
            if len(r["trial_window_idx"]["pre"]) != expected_pre:
                raise ValueError(
                    f"{subject}: reversal at {r['reversal_idx']} has a pre window of "
                    f"{len(r['trial_window_idx']['pre'])} trials, expected {expected_pre}"
                )


def check_moving_avg_window(moving_avg_window) -> None:
    """Raise ``ValueError`` unless *moving_avg_window* is a positive int (1 means no smoothing)."""
    if (
        isinstance(moving_avg_window, bool)
        or not isinstance(moving_avg_window, (int, np.integer))
        or moving_avg_window < 1
    ):
        raise ValueError(f"moving_avg_window must be a positive integer, got {moving_avg_window!r}")


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
            role: smooth_pre_post(y, offsets < 0, smooth["window"], smooth["mode"])
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
            (1 for none), applied separately before and after the reversal. Magnitudes
            are not smoothed.
        moving_avg_mode: ``"centered"`` or ``"trailing"``; the two give different curves, so
            compare figures only if they use the same mode.

    Returns:
        A :class:`ReversalAlignedArms`. With no reversal left, its curves are all NaN and an
        :class:`EmptySelectionWarning` is raised.

    Raises:
        ValueError: For an unknown scheme, type or mode, a *moving_avg_window* that is not a
            positive int, a window longer than *pre*/*post*, a first-block record, or choice
            probabilities that do not sum to 1.
    """
    if scheme not in LABEL_SCHEMES:
        raise ValueError(f"scheme must be one of {LABEL_SCHEMES}, got {scheme!r}")
    check_moving_avg_window(moving_avg_window)
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
