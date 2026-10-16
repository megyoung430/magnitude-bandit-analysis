"""Choice probability and reward magnitude of the three arms, aligned to block changes.

Pipeline (all pure; no files, no plots):

1. :func:`build_windows_with_magnitudes` windows each block change with ``get_good_reversal_info``
   (both windows stay inside the block of the block change) and adds the reward magnitude of
   every arm over the same window.
2. :func:`compute_reversal_aligned_arms` labels the arms of each block change by the ranking
   *before* the block change (``"prev"``) or *after* it (``"new"``), keeps the block changes of one
   type, and averages one-hot choices and magnitudes per arm label: over block changes within a
   subject, then over subjects.

Windows stay within one block: the pre window starts no earlier than the previous block
change and the post window ends at the next one, so every value of a row belongs to the block of
its block change or, before it, to the block it replaced.

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
    """No block change of the requested scheme and type is left, so there is nothing to average."""


class _Curves(NamedTuple):
    """Mean and SE over subjects for each role, plus the per-subject means."""

    mean: dict[str, np.ndarray]
    se: dict[str, np.ndarray]
    per_subject: dict[str, dict[str, np.ndarray]]


@dataclass(frozen=True)
class ReversalAlignedArms:
    """Aligned curves of one label scheme and block change type.

    Attributes:
        scheme (str): ``"prev"`` or ``"new"``: whether arms are labelled by their rank before
            or after the block change.
        reversal_type (str): ``"all"``, ``"best_to_third"`` or ``"second_to_third"``.
        roles (tuple[str, str, str]): Curve names, best to third, e.g.
            ``("prev_best", "prev_second", "prev_third")``.
        trial_offsets (np.ndarray): Trials from the block change of each bin, shape ``(n_bins,)``
            with ``n_bins = pre + post - skip``. With a skip ``k``, bins after the
            block change trial carry offsets ``k + 1`` upward.
        choice_mean (dict[str, np.ndarray]): ``{role: array of shape (n_bins,)}``; the mean
            choice probability across subjects (after the moving average).
        choice_se (dict[str, np.ndarray]): ``{role: array of shape (n_bins,)}``; its standard
            error across subjects (NaN where fewer than two subjects have data).
        magnitude_mean (dict[str, np.ndarray]): ``{role: array of shape (n_bins,)}``; mean reward
            magnitude of the arm with that label (not smoothed).
        magnitude_se (dict[str, np.ndarray]): ``{role: array of shape (n_bins,)}``; its standard
            error across subjects.
        per_subject_choice (dict[str, dict[str, np.ndarray]]): ``{subject: {role: smoothed mean
            choice probability, shape (n_bins,)}}``.
        num_subjects (int): Subjects the curves average over.
        num_reversals (int): Block changes the curves average over.
        counts (SelectionCounts): What the selection kept and why it left block changes out.
        pre (int): Trials before the block change in each window.
        post (int): Trials from the block change in each window.
        skip (int): Trials dropped after the block change trial.
        moving_avg_window (int): Moving-average width used (1 for none).
        moving_avg_mode (str): ``"centered"`` or ``"trailing"``.
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
    comparing the two values stored with the block change catches any other pairing.
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
                f"of the block change at {idx} ({key}: {stored} vs {found}); the windows and the "
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
    """Copy of the block change windows with every arm's reward magnitude over each window.

    The magnitudes come from the same merge of sessions that ``get_good_reversal_info`` uses
    (``get_vars_across_all_sessions``: date-then-number session order, one entry per trial), so
    they sit on the windows' trial axis. The magnitudes at the block change trial and the trial
    before it are checked against the ones stored in each block change.

    Args:
        reversal_windows (dict): ``{subject: [block change dict]}`` from ``get_good_reversal_info``.
        subjects_trials (dict): The ``{subject: {session: session dict}}`` the windows were made
            from.

    Returns:
        dict: ``{subject: [block change dict]}`` where each block change dict has a new key
        ``"magnitudes_by_tower"``: ``{tower: {"pre": [...], "post": [...]}}``, the same shape
        as ``"choices_by_tower"``. The input is not modified.

    Raises:
        ValueError: If a subject has no sessions, a series is shorter than the window, or the
            magnitudes do not match those stored in the block change.
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
    """Block-change windows of one problem, within one block, with magnitude windows.

    Both windows stay inside the block of the block change (``stop_at_block_changes=True``: the
    pre window starts at the previous block change, the post window ends at the next one),
    and the first block is not included.

    Args:
        subjects_trials (dict): ``{subject: {session: session dict}}`` of one problem.
        pre (int): Maximum trials before each block change in the windows.
        post (int): Maximum trials from each block change in the windows.

    Returns:
        dict: ``{subject: [block change dict]}`` as :func:`add_magnitude_windows` returns it, each
        block change also recording the ``"window_pre"`` and ``"window_post"`` it was built with
        (``"pre_start"`` and ``"post_end"``, the bounds, come from ``get_good_reversal_info``).
        :func:`check_windows_built_with` verifies all of these.

    Raises:
        ValueError: As ``get_good_reversal_info`` with ``stop_at_block_changes=True`` (a subject
            without a ``blocks`` series, or boundaries that are not block changes) and as
            :func:`add_magnitude_windows`.
    """
    windows = get_good_reversal_info(
        subjects_trials,
        pre=pre,
        post=post,
        include_first_block=False,
        stop_at_block_changes=True,
    )
    enriched = add_magnitude_windows(windows, subjects_trials)
    return {
        subject: [{**r, "window_pre": pre, "window_post": post} for r in reversals]
        for subject, reversals in enriched.items()
    }


_WINDOW_KEYS = ("window_pre", "window_post", "pre_start", "post_end", "magnitudes_by_tower")


def _check_window_lengths(subject: str, r: dict, pre: int, post: int) -> None:
    idx = r["reversal_idx"]
    for name, got, expected in (
        ("pre", len(r["trial_window_idx"]["pre"]), min(pre, idx - r["pre_start"])),
        ("post", len(r["trial_window_idx"]["post"]), min(post, r["post_end"] - idx)),
    ):
        if got != expected:
            raise ValueError(
                f"{subject}: block change at {idx} has a {name} window of {got} trials, expected "
                f"{expected} (the window stays inside its block and at most {pre}/{post} trials)"
            )


def check_windows_built_with(windows: dict, *, pre: int, post: int) -> None:
    """Raise unless every block change was built with *pre*, *post* by the builder.

    Windows built with a smaller pre or post would be NaN-padded to the requested length and
    look like ordinary data. Each window must also have the length its recorded bounds allow:
    ``min(pre, idx - pre_start)`` before the block change and ``min(post, post_end - idx)`` from it.

    Args:
        windows (dict): ``{subject: [block change dict]}`` from
            :func:`build_windows_with_magnitudes`.
        pre (int): The ``pre`` the windows must have been built with.
        post (int): The ``post`` the windows must have been built with.

    Raises:
        ValueError: Naming the first block change that lacks the recorded settings or magnitude
            windows, was built with other settings, or has a window of the wrong length. The
            message points to :func:`build_windows_with_magnitudes`.
    """
    for subject, reversals in windows.items():
        for r in reversals:
            missing = [k for k in _WINDOW_KEYS if k not in r]
            if missing:
                raise ValueError(
                    f"{subject}: block change at {r.get('reversal_idx')} lacks {missing}; build "
                    "the windows with build_windows_with_magnitudes"
                )
            if (r["window_pre"], r["window_post"]) != (pre, post):
                raise ValueError(
                    f"{subject}: block change at {r['reversal_idx']} was windowed with (pre, post) "
                    f"= {(r['window_pre'], r['window_post'])}, not ({pre}, {post}); rebuild it "
                    "with build_windows_with_magnitudes"
                )
            _check_window_lengths(subject, r, pre, post)


def check_moving_avg_window(moving_avg_window) -> None:
    """Raise ``ValueError`` unless *moving_avg_window* is a positive int.

    Args:
        moving_avg_window (int): Moving-average width; 1 means no smoothing.

    Raises:
        ValueError: If it is not an int (a bool or float is not), or is below 1.
    """
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
    """``(3, n_bins)`` mean over a subject's block changes of the rows of *field*."""
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
    """Choice probability and reward magnitude of the three arms around block changes.

    Args:
        reversal_windows (dict): ``{subject: [block change dict]}`` from
            :func:`build_windows_with_magnitudes` with the same *pre* and *post*. Pooling across
            problems is concatenation per subject (``merge_windows_by_subject``).
        scheme (str): ``"prev"`` labels arms by their rank before the block change, ``"new"`` by
            their rank after it.
        reversal_type (str): ``"all"``, ``"best_to_third"`` or ``"second_to_third"``; the type
            is always defined by the prev ranking. Block changes with tied magnitudes are left out
            as :func:`~src.behavior_analysis.reversal_arm_ranks.select_reversals` describes.
            Default ``"all"``.
        pre (int): Trials before the block change in each window. Default 10.
        post (int): Trials from the block change in each window. Default 30.
        skip_n_trials_after_reversal (int): Post trial 0 is kept and the next *k* trials
            dropped. Default 0.
        moving_avg_window (int): Width of the moving average of the per-subject choice
            probabilities (1 for none), applied separately before and after the block change.
            Magnitudes are not smoothed. Default 4.
        moving_avg_mode (str): ``"centered"`` or ``"trailing"``; the two give different curves,
            so compare figures only if they use the same mode. Default ``"centered"``.

    Returns:
        ReversalAlignedArms: The aligned curves. With no block change left, its curves are all NaN
        and an :class:`EmptySelectionWarning` is emitted.

    Raises:
        ValueError: For an unknown scheme, type or mode, a *moving_avg_window* that is not a
            positive int, windows not built by the builder with this *pre*/*post*, a first-block
            record, or choice probabilities that do not sum to 1.
    """
    if scheme not in LABEL_SCHEMES:
        raise ValueError(f"scheme must be one of {LABEL_SCHEMES}, got {scheme!r}")
    check_moving_avg_window(moving_avg_window)
    check_windows_built_with(reversal_windows, pre=pre, post=post)
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
            f"no block change left for scheme {scheme!r}, reversal_type {reversal_type!r}: "
            f"{counts}",
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
