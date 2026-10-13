"""Compute and smooth reversal-aligned choice-probability curves.

Functions here take the windowed reversal data produced by
:mod:`src.behavior_analysis.get_good_reversal_info` and compute mean
choice-probability curves aligned to each good reversal.  Optional NaN-aware
moving-average smoothing can be applied separately to the pre- and post-
reversal segments so there is no smoothing bleed-through across the boundary.
"""

from collections.abc import Callable, Sequence
from copy import deepcopy
from functools import partial

import numpy as np

from src.behavior_analysis.get_good_reversal_info import classify_towers_at_good_reversals

# Roles of the three towers around a good reversal, in the order returned by
# classify_towers_at_good_reversals.
_TOWER_ROLES = ("prev_best", "next_best", "third")
_MEAN_KEYS = {role: f"{role}_mean" for role in _TOWER_ROLES}


# ---------------------------------------------------------------------------
# Choice probabilities around good reversals
# ---------------------------------------------------------------------------


def _keep_zero_then_skip(post_list: list, skip: int) -> list:
    """Return *post_list* with index 0 kept and indices ``1 .. skip - 1`` dropped.

    That is ``skip - 1`` trials removed, not ``skip``.  Known issue: callers
    that read "skip n trials" literally get one trial too few skipped, and the
    result is one element longer than the shortened x axis, so the row's last
    post trial is cut by :func:`_pad_to_length`.
    """
    if skip <= 0:
        return post_list
    return post_list[:1] + post_list[skip:]


def _pad_to_length(arr: Sequence[float], length: int) -> np.ndarray:
    """Pad or truncate *arr* to *length*, filling missing entries with NaN."""
    out = np.full(length, np.nan, dtype=float)
    n = min(len(arr), length)
    out[:n] = arr[:n]
    return out


def _reversal_choice_rows(reversal: dict, *, skip: int, length: int) -> dict[str, np.ndarray]:
    """One-hot choice row per tower role (pre + post window) for one reversal.

    Args:
        reversal: A reversal dict from ``get_good_reversal_info``.
        skip: Post trial 0 (the reversal trial) is kept and post trials
            ``1 .. skip - 1`` are dropped, i.e. ``skip - 1`` trials (see
            :func:`_keep_zero_then_skip`).
        length: Row length; shorter windows are NaN-padded, longer truncated.

    Returns:
        ``{role: 1-D float array}`` for ``prev_best``, ``next_best``, ``third``.
    """
    towers = classify_towers_at_good_reversals(reversal)
    by_tower = reversal["choices_by_tower"]
    return {
        role: _pad_to_length(
            by_tower[tower]["pre"] + _keep_zero_then_skip(by_tower[tower]["post"], skip), length
        )
        for role, tower in zip(_TOWER_ROLES, towers, strict=True)
    }


def _subject_choice_matrices(
    reversals: list[dict], *, skip: int, length: int
) -> dict[str, np.ndarray]:
    """Stack the per-reversal rows of one subject into ``{role: (n_reversals, length)}``."""
    rows = [_reversal_choice_rows(reversal, skip=skip, length=length) for reversal in reversals]
    return {role: np.vstack([row[role] for row in rows]) for role in _TOWER_ROLES}


def _summarize_subject(matrices: dict[str, np.ndarray]) -> dict:
    """Raw matrices plus their column means and the reversal count for one subject."""
    summary = dict(matrices)
    summary.update({_MEAN_KEYS[role]: np.nanmean(matrices[role], axis=0) for role in _TOWER_ROLES})
    summary["num_reversals"] = matrices["prev_best"].shape[0]
    return summary


def _mean_and_se(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """NaN-aware column mean and standard error of a ``(n_rows, n_bins)`` stack.

    The standard error is zero when there is a single row.

    Known issue: it divides by ``sqrt(n_rows)`` even where some rows are NaN in
    a bin, so the SE is underestimated in bins where windows were truncated
    (it should divide by the number of finite rows in that bin).
    """
    n_rows, n_bins = stack.shape
    mean = np.nanmean(stack, axis=0)
    if n_rows > 1:
        return mean, np.nanstd(stack, axis=0, ddof=1) / np.sqrt(n_rows)
    return mean, np.zeros(n_bins)


def _across_subject_curves(per_subject: dict[str, dict], keys_by_role: dict[str, str]) -> dict:
    """Mean and SE across subjects of the curve stored under ``keys_by_role[role]``.

    Returns:
        ``{"mean": {role: array}, "se": {role: array}, "num_subjects": int,
        "num_reversals": int}``; subjects lacking ``"num_reversals"`` count as 0.

    Raises:
        ValueError: If *per_subject* is empty (nothing to stack).
    """
    across = {"mean": {}, "se": {}}
    for role, key in keys_by_role.items():
        stack = np.vstack([curves[key] for curves in per_subject.values()])
        across["mean"][role], across["se"][role] = _mean_and_se(stack)
    across["num_subjects"] = len(per_subject)
    # Curves supplied without a count (e.g. hand-built dicts) contribute 0 reversals.
    across["num_reversals"] = sum(curves.get("num_reversals", 0) for curves in per_subject.values())
    return across


def _assert_probabilities_sum_to_one(
    mean_by_role: dict[str, np.ndarray], *, label: str, allow_no_finite_bins: bool
) -> None:
    """Assert the three role means sum to 1 in every bin where the sum is finite.

    Args:
        mean_by_role: ``{role: array}`` as in ``across["mean"]``.
        label: Noun phrase for the failure message, e.g. ``"Choice probabilities"``.
        allow_no_finite_bins: If ``False``, a sum that is NaN everywhere also fails.

    Raises:
        AssertionError: If any finite bin does not sum to 1, or if no bin is
            finite and *allow_no_finite_bins* is ``False``.
    """
    total = mean_by_role["prev_best"] + mean_by_role["next_best"] + mean_by_role["third"]
    finite_sums = total[np.isfinite(total)]
    if finite_sums.size == 0:
        assert allow_no_finite_bins, "No finite bins to validate (all-NaN after padding/filtering)."
        return
    assert np.isclose(finite_sums, 1.0, atol=1e-6).all(), f"{label} do not sum to 1 (finite bins)."


def _trial_offsets(*, pre: int, post: int, skip: int) -> np.ndarray:
    """Trial offsets ``[-pre, 0)`` followed by ``[0, post - skip)``.

    Raises:
        ValueError: If *skip* is negative or leaves no post window (``skip >= post``).
    """
    if skip < 0:
        raise ValueError("skip_n_trials_after_reversal must be >= 0")
    if skip >= post:
        raise ValueError(
            "skip_n_trials_after_reversal must be < post (otherwise post window is empty)."
        )
    return np.concatenate([np.arange(-pre, 0), np.arange(0, post - skip)])


def get_choice_probs_around_good_reversals(
    reversal_windows, pre=10, post=40, skip_n_trials_after_reversal=0
):
    """Compute reversal-aligned choice-probability curves from reversal windows.

    For each subject, stacks one-hot choice arrays across all reversals to form
    per-reversal choice matrices, then averages within and across subjects.

    Args:
        reversal_windows: ``{subject: [reversal_dict]}`` as returned by
            :func:`src.behavior_analysis.get_good_reversal_info.get_good_reversal_info`.
        pre: Number of trials before the reversal to include (default: 10).
        post: Number of trials after the reversal to include (default: 40).
        skip_n_trials_after_reversal: ``n``; post trial 0 (the reversal trial)
            is always kept.  Known issue: this drops only post trials
            ``1 .. n - 1``, i.e. ``n - 1`` trials (``n = 15`` skips 14), and
            ``x`` is shortened by ``n``, so each row's last post trial is cut
            as well.  Default: 0 (nothing skipped).

    Returns:
        A three-tuple ``(x, per_subject, across)`` where:

        - ``x`` – 1-D int array of trial offsets, length
          ``pre + (post - skip_n_trials_after_reversal)``.
        - ``per_subject[subj]`` – dict with keys ``"prev_best"``,
          ``"next_best"``, ``"third"`` (raw matrices), ``"prev_best_mean"``,
          ``"next_best_mean"``, ``"third_mean"`` (per-subject means), and
          ``"num_reversals"``.
        - ``across`` – dict with keys ``"mean"``, ``"se"``, ``"num_subjects"``,
          and ``"num_reversals"``, where ``"mean"`` and ``"se"`` are themselves
          dicts keyed by ``"prev_best"``, ``"next_best"``, ``"third"``.

    Raises:
        ValueError: If ``skip_n_trials_after_reversal < 0`` or
            ``>= post``.
        AssertionError: If the mean choice probabilities do not sum to 1.0 at
            all finite time-bins.
    """
    skip = skip_n_trials_after_reversal
    x = _trial_offsets(pre=pre, post=post, skip=skip)
    n_bins = pre + (post - skip)

    per_subject = {}
    for subj, reversals in reversal_windows.items():
        if not reversals:
            print(f"[INFO] Skipping subject {subj} with no good reversals.")
            continue
        per_subject[subj] = _summarize_subject(
            _subject_choice_matrices(reversals, skip=skip, length=n_bins)
        )

    across = _across_subject_curves(per_subject, _MEAN_KEYS)
    _assert_probabilities_sum_to_one(
        across["mean"], label="Choice probabilities", allow_no_finite_bins=False
    )

    return x, per_subject, across


# ---------------------------------------------------------------------------
# Moving-average smoothing
# ---------------------------------------------------------------------------


def _nan_moving_average(
    y: Sequence[float], window: int | None, mode: str = "centered"
) -> np.ndarray:
    """Apply a NaN-aware moving average of *window* width to array *y*.

    Each output bin is the mean of the finite values inside its window; bins
    whose window holds no finite value are NaN.  ``mode="trailing"`` averages
    the current and previous ``window - 1`` bins, anything else is centred.
    A centred window of even width ``w`` covers bins ``[i - w/2, i + w/2 - 1]``
    (two before and one after for ``w = 4``).
    A *window* of ``None`` or ``<= 1`` returns *y* as a float array.
    """
    y = np.asarray(y, dtype=float)
    if window is None or window <= 1:
        return y

    valid = np.isfinite(y).astype(float)
    y0 = np.nan_to_num(y, nan=0.0)
    kernel = np.ones(int(window), dtype=float)

    if mode == "trailing":
        num = np.convolve(y0, kernel, mode="full")[: len(y0)]
        den = np.convolve(valid, kernel, mode="full")[: len(y0)]
    else:
        # Known issue: np.convolve(mode="same") returns max(len(y), len(kernel))
        # values, so a centred window longer than the segment makes the masked
        # assignment in _smooth_pre_post raise an IndexError.
        num = np.convolve(y0, kernel, mode="same")
        den = np.convolve(valid, kernel, mode="same")

    out = np.full_like(y0, np.nan, dtype=float)
    has_data = den > 0
    out[has_data] = num[has_data] / den[has_data]
    return out


def _smooth_pre_post(
    y: Sequence[float], pre_mask: np.ndarray, window: int | None, mode: str = "centered"
) -> np.ndarray:
    """Smooth *y* separately on the ``pre_mask`` and ``~pre_mask`` segments."""
    y = np.asarray(y, dtype=float)
    out = np.full_like(y, np.nan, dtype=float)
    for segment in (pre_mask, ~pre_mask):
        if segment.any():
            out[segment] = _nan_moving_average(y[segment], window, mode=mode)
    return out


def _add_smoothed_means(
    curves: dict,
    use_keys: Sequence[str],
    out_keys: Sequence[str],
    smooth: Callable[[np.ndarray], np.ndarray],
) -> dict:
    """Copy of one subject's dict with ``smooth(<src>)`` stored under each ``dst``.

    Sources are read from the copy being written, in order, so an output key
    that is also a *later* input key feeds its smoothed value into that later
    entry (chained smoothing).
    """
    smoothed = dict(curves)
    # Contract: use_keys and out_keys hold exactly three entries, in the order of
    # _TOWER_ROLES (prev_best, next_best, third); the caller reads out_keys[i] for
    # role i.  strict=False only avoids a ZipError here: a shorter use_keys makes
    # the later out_keys lookup raise KeyError, a shorter out_keys an IndexError.
    for src, dst in zip(use_keys, out_keys, strict=False):
        smoothed[dst] = smooth(smoothed[src])
    return smoothed


def _empty_across_summary() -> dict:
    """Across-subject summary for the case where no subject has curves to average."""
    return {
        "mean": dict.fromkeys(_TOWER_ROLES),
        "se": dict.fromkeys(_TOWER_ROLES),
        "num_subjects": 0,
        "num_reversals": 0,
    }


def apply_moving_average_to_choice_probs(
    x,
    per_subject,
    moving_avg_window=5,
    mode="centered",
    use_keys=("prev_best_mean", "next_best_mean", "third_mean"),
    out_keys=("prev_best_mean_sm", "next_best_mean_sm", "third_mean_sm"),
    split_at=0,
):
    """Apply a NaN-aware moving average to per-subject choice-probability curves.

    Smoothing is applied separately to the pre- (``x < split_at``) and
    post-reversal (``x >= split_at``) segments to prevent bleed-through across
    the boundary.  The smoothed means are stored under new keys in each
    subject's dict, and a new ``across`` dict with the smoothed averages is
    also returned.

    Args:
        x: 1-D float array of trial offsets (same *x* returned by
            :func:`get_choice_probs_around_good_reversals`).
        per_subject: Per-subject dict as returned by
            :func:`get_choice_probs_around_good_reversals`.
        moving_avg_window: Kernel width for the moving average (default: 5).
        mode: ``"centered"`` (default) or ``"trailing"``.  Controls whether
            the convolution kernel is centred or trailing.  An even centred
            window of width ``w`` covers bins ``[i - w/2, i + w/2 - 1]`` (for
            ``w = 4``: two before, one after).  A centred window longer than
            the pre or post segment raises ``IndexError``; use a shorter
            window or ``"trailing"``.
        use_keys: Exactly three source keys in each subject dict, one per
            tower role in the order prev-best, next-best, third (default:
            ``("prev_best_mean", "next_best_mean", "third_mean")``).  Subjects
            lacking any of them are dropped from the result.
        out_keys: Exactly three destination keys, in the same role order (default:
            ``("prev_best_mean_sm", "next_best_mean_sm", "third_mean_sm")``).
            Keys are processed in order, so an output key that is also a later
            input key chains the smoothing.
        split_at: Trial-offset value that separates pre and post segments
            (default: 0, i.e. the reversal trial itself begins the post
            segment).

    Returns:
        A three-tuple ``(x, per_subject_moving_avg, across_moving_avg)``
        with the same structure as :func:`get_choice_probs_around_good_reversals`
        but with smoothed means.
    """
    smooth = partial(_smooth_pre_post, pre_mask=x < split_at, window=moving_avg_window, mode=mode)
    per_subject_moving_avg = {
        subj: _add_smoothed_means(curves, use_keys, out_keys, smooth)
        for subj, curves in per_subject.items()
        if all(k in curves for k in use_keys)  # subjects without the requested curves are dropped
    }
    smoothing_params = {"moving_avg_window": moving_avg_window, "mode": mode, "split_at": split_at}

    if not per_subject_moving_avg:
        return x, per_subject_moving_avg, {**_empty_across_summary(), **smoothing_params}

    keys_by_role = {role: out_keys[i] for i, role in enumerate(_TOWER_ROLES)}
    across = _across_subject_curves(per_subject_moving_avg, keys_by_role)
    across.update(smoothing_params)
    _assert_probabilities_sum_to_one(
        across["mean"], label="Smoothed probabilities", allow_no_finite_bins=True
    )

    return x, per_subject_moving_avg, across


# ---------------------------------------------------------------------------
# Truncating post windows at the first intervening bad reversal
# ---------------------------------------------------------------------------


def _next_good_after(goods: Sequence[int], after: int) -> float:
    """First good-reversal index in *goods* strictly after *after* (``inf`` if none)."""
    return next((idx for idx in goods if idx > after), float("inf"))


def _first_bad_between(*, bads: Sequence[int], after: int, before: float) -> int | None:
    """First bad-reversal index strictly between *after* and *before* (``None`` if none)."""
    return next((idx for idx in bads if after < idx < before), None)


def _truncate_post(reversal: dict, cut_len: int) -> None:
    """Clip every ``"post"`` list in *reversal* to its first *cut_len* entries, in place."""
    for key in ("choices_by_tower", "choices_by_rank"):
        for entry in reversal.get(key, {}).values():
            if isinstance(entry, dict) and "post" in entry:
                entry["post"] = entry["post"][:cut_len]

    trial_window = reversal.get("trial_window_idx", {})
    if isinstance(trial_window, dict) and isinstance(trial_window.get("post", None), list):
        trial_window["post"] = trial_window["post"][:cut_len]


def _truncate_reversal_at_bad(
    reversal: dict, *, goods: Sequence[int], bads: Sequence[int], include_bad_trial: bool
) -> dict:
    """Copy of *reversal* with its post window cut at the first intervening bad reversal.

    The copy always carries ``"removed_bad_idx"`` and ``"post_len_after_removal"``
    (both ``None`` when there is nothing to cut).
    """
    out = deepcopy(reversal)
    # No reversal_idx means there is no interval to search: the window passes through uncut.
    reversal_idx = out.get("reversal_idx", None)
    bad_idx = (
        None
        if reversal_idx is None
        else _first_bad_between(
            bads=bads, after=reversal_idx, before=_next_good_after(goods, reversal_idx)
        )
    )

    if bad_idx is None:
        out["removed_bad_idx"] = None
        out["post_len_after_removal"] = None
        return out

    # bad_idx > reversal_idx, so cut_len >= 1 whether or not the bad trial is kept.
    cut_len = (bad_idx - reversal_idx) + (1 if include_bad_trial else 0)
    out["removed_bad_idx"] = bad_idx
    out["post_len_after_removal"] = cut_len
    _truncate_post(out, cut_len)
    return out


def _report_removal(subj: str, reversal: dict) -> None:
    """Print one line if *reversal*'s post window was actually cut."""
    if reversal["post_len_after_removal"] is not None:
        print(
            f"[REMOVE] {subj} good@{reversal['reversal_idx']}: "
            f"cut post at bad@{reversal['removed_bad_idx']} "
            f"(kept {reversal['post_len_after_removal']} post trials)"
        )


def remove_trials_after_bad_rev(good_windows, all_good_idx, all_bad_idx, include_bad_trial=True):
    """Truncate each good reversal's post window at the first intervening bad reversal.

    For each good reversal at index *g*, finds the first bad reversal that
    falls strictly between *g* and the next good reversal.  The post window is
    then clipped to end at (or just before) that bad reversal.

    Args:
        good_windows: ``{subject: [reversal_dict]}`` as returned by
            :func:`src.behavior_analysis.get_good_reversal_info.get_good_reversal_info`.
        all_good_idx: ``{subject: [int]}`` sorted good-reversal indices.
        all_bad_idx: ``{subject: [int]}`` sorted bad-reversal indices.
        include_bad_trial: If ``True``, the bad-reversal trial itself is
            included in the truncated post window; if ``False``, the window
            is cut just before it (default: ``True``).

    Returns:
        A new dict with the same structure as *good_windows* but with the
        ``"post"`` lists in ``"choices_by_tower"``, ``"choices_by_rank"``, and
        ``"trial_window_idx"`` truncated where a bad reversal intervenes.
        Each modified reversal dict gains two extra keys:
        ``"removed_bad_idx"`` (int | None) and
        ``"post_len_after_removal"`` (int | None).
    """
    out = {}
    for subj, good_revs in good_windows.items():
        # A reversal without "reversal_idx" sorts first (as 0) and is passed through
        # uncut; a subject absent from the index dicts has no known goods/bads, so
        # none of its windows are cut.
        revs_sorted = sorted(good_revs, key=lambda rev: rev.get("reversal_idx", 0))
        goods = sorted(all_good_idx.get(subj, []))
        bads = sorted(all_bad_idx.get(subj, []))

        out[subj] = []
        for rev in revs_sorted:
            truncated = _truncate_reversal_at_bad(
                rev, goods=goods, bads=bads, include_bad_trial=include_bad_trial
            )
            _report_removal(subj, truncated)
            out[subj].append(truncated)
    return out
