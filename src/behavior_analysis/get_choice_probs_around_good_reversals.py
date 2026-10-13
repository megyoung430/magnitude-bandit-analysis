"""Compute and smooth reversal-aligned choice-probability curves.

Functions here take the windowed reversal data produced by
:mod:`src.behavior_analysis.get_good_reversal_info` and compute mean
choice-probability curves aligned to each good reversal.  Optional NaN-aware
moving-average smoothing can be applied separately to the pre- and post-
reversal segments so there is no smoothing bleed-through across the boundary.
"""

from copy import deepcopy
from functools import partial

import numpy as np

from src.behavior_analysis.get_good_reversal_info import classify_towers_at_good_reversals

# Roles of the three towers around a good reversal, in the order returned by
# classify_towers_at_good_reversals.
TOWER_ROLES = ("prev_best", "next_best", "third")
MEAN_KEYS = {role: f"{role}_mean" for role in TOWER_ROLES}


# ---------------------------------------------------------------------------
# Choice probabilities around good reversals
# ---------------------------------------------------------------------------


def keep_zero_then_skip(post_list, skip):
    """Return *post_list* with index 0 kept and the next ``skip - 1`` elements dropped."""
    if skip <= 0:
        return post_list
    return post_list[:1] + post_list[skip:]


def pad_to_length(arr, length):
    """Pad or truncate *arr* to *length*, filling missing entries with NaN."""
    out = np.full(length, np.nan, dtype=float)
    n = min(len(arr), length)
    out[:n] = arr[:n]
    return out


def reversal_choice_rows(reversal, skip, length):
    """One-hot choice row per tower role (pre + post window) for one reversal.

    Args:
        reversal: A reversal dict from ``get_good_reversal_info``.
        skip: Post-window trials to skip after the reversal trial itself.
        length: Row length; shorter windows are NaN-padded, longer truncated.

    Returns:
        ``{role: 1-D float array}`` for ``prev_best``, ``next_best``, ``third``.
    """
    towers = classify_towers_at_good_reversals(reversal)
    by_tower = reversal["choices_by_tower"]
    return {
        role: pad_to_length(
            by_tower[t]["pre"] + keep_zero_then_skip(by_tower[t]["post"], skip), length
        )
        for role, t in zip(TOWER_ROLES, towers, strict=True)
    }


def subject_choice_matrices(revs, skip, length):
    """Stack the per-reversal rows of one subject into ``{role: (n_reversals, length)}``."""
    rows = [reversal_choice_rows(r, skip, length) for r in revs]
    return {role: np.vstack([row[role] for row in rows]) for role in TOWER_ROLES}


def summarize_subject(matrices):
    """Raw matrices plus their column means and the reversal count for one subject."""
    summary = dict(matrices)
    summary.update({MEAN_KEYS[role]: np.nanmean(matrices[role], axis=0) for role in TOWER_ROLES})
    summary["num_reversals"] = matrices["prev_best"].shape[0]
    return summary


def mean_and_se(stack):
    """NaN-aware column mean and standard error of a ``(n_rows, n_bins)`` stack.

    The standard error divides by ``sqrt(n_rows)`` (NaN bins still count as
    rows) and is zero when there is a single row.
    """
    n_rows, n_bins = stack.shape
    mean = np.nanmean(stack, axis=0)
    if n_rows > 1:
        return mean, np.nanstd(stack, axis=0, ddof=1) / np.sqrt(n_rows)
    return mean, np.zeros(n_bins)


def across_subject_curves(per_subject, keys_by_role):
    """Mean and SE across subjects of the curve stored under ``keys_by_role[role]``.

    Returns:
        ``{"mean": {role: array}, "se": {role: array}, "num_subjects": int,
        "num_reversals": int}``; subjects lacking ``"num_reversals"`` count as 0.

    Raises:
        ValueError: If *per_subject* is empty (nothing to stack).
    """
    across = {"mean": {}, "se": {}}
    for role, key in keys_by_role.items():
        stack = np.vstack([d[key] for d in per_subject.values()])
        across["mean"][role], across["se"][role] = mean_and_se(stack)
    across["num_subjects"] = len(per_subject)
    across["num_reversals"] = sum(d.get("num_reversals", 0) for d in per_subject.values())
    return across


def assert_probabilities_sum_to_one(mean_by_role, what, allow_no_finite_bins):
    """Assert the three role means sum to 1 in every bin where the sum is finite.

    Args:
        mean_by_role: ``{role: array}`` as in ``across["mean"]``.
        what: Noun phrase for the failure message, e.g. ``"Choice probabilities"``.
        allow_no_finite_bins: If ``False``, a sum that is NaN everywhere also fails.
    """
    total = mean_by_role["prev_best"] + mean_by_role["next_best"] + mean_by_role["third"]
    finite_sums = total[np.isfinite(total)]
    if finite_sums.size == 0:
        assert allow_no_finite_bins, "No finite bins to validate (all-NaN after padding/filtering)."
        return
    assert np.isclose(finite_sums, 1.0, atol=1e-6).all(), f"{what} do not sum to 1 (finite bins)."


def trial_offsets(pre, post, skip):
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
        skip_n_trials_after_reversal: Number of trials immediately after the
            reversal index to skip (index 0 of the post window is always kept
            as the reversal trial itself; default: 0).

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
    x = trial_offsets(pre, post, skip)
    n_bins = pre + (post - skip)

    per_subject = {}
    for subj, revs in reversal_windows.items():
        if not revs:
            print(f"[INFO] Skipping subject {subj} with no good reversals.")
            continue
        per_subject[subj] = summarize_subject(subject_choice_matrices(revs, skip, n_bins))

    across = across_subject_curves(per_subject, MEAN_KEYS)
    assert_probabilities_sum_to_one(across["mean"], "Choice probabilities", False)

    return x, per_subject, across


# ---------------------------------------------------------------------------
# Moving-average smoothing
# ---------------------------------------------------------------------------


def nan_moving_average(y, window, mode="centered"):
    """Apply a NaN-aware moving average of *window* width to array *y*.

    Each output bin is the mean of the finite values inside its window; bins
    whose window holds no finite value are NaN.  ``mode="trailing"`` averages
    the current and previous ``window - 1`` bins, anything else is centred.
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
        # Known issue, behaviour kept: np.convolve(mode="same") returns
        # max(len(y), len(kernel)) values, so a centred window longer than the
        # segment makes the caller's masked assignment raise an IndexError.
        num = np.convolve(y0, kernel, mode="same")
        den = np.convolve(valid, kernel, mode="same")

    out = np.full_like(y0, np.nan, dtype=float)
    has_data = den > 0
    out[has_data] = num[has_data] / den[has_data]
    return out


def smooth_pre_post(y, pre_mask, window, mode="centered"):
    """Smooth *y* separately on the ``pre_mask`` and ``~pre_mask`` segments."""
    y = np.asarray(y, dtype=float)
    out = np.full_like(y, np.nan, dtype=float)
    for segment in (pre_mask, ~pre_mask):
        if segment.any():
            out[segment] = nan_moving_average(y[segment], window, mode=mode)
    return out


def add_smoothed_means(curves, use_keys, out_keys, smooth):
    """Copy of one subject's dict with ``smooth(curves[src])`` stored under each ``dst``."""
    smoothed = dict(curves)
    for src, dst in zip(use_keys, out_keys, strict=False):  # extra keys on either side are ignored
        smoothed[dst] = smooth(curves[src])
    return smoothed


def empty_across_summary():
    """Across-subject summary for the case where no subject has curves to average."""
    return {
        "mean": dict.fromkeys(TOWER_ROLES),
        "se": dict.fromkeys(TOWER_ROLES),
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
            the convolution kernel is centred or trailing.
        use_keys: Source keys in each subject dict to smooth (default:
            ``("prev_best_mean", "next_best_mean", "third_mean")``).
        out_keys: Destination keys to write smoothed values to (default:
            ``("prev_best_mean_sm", "next_best_mean_sm",
            "third_mean_sm")``).
        split_at: Trial-offset value that separates pre and post segments
            (default: 0, i.e. the reversal trial itself begins the post
            segment).

    Returns:
        A three-tuple ``(x, per_subject_moving_avg, across_moving_avg)``
        with the same structure as :func:`get_choice_probs_around_good_reversals`
        but with smoothed means.
    """
    smooth = partial(smooth_pre_post, pre_mask=x < split_at, window=moving_avg_window, mode=mode)
    per_subject_moving_avg = {
        subj: add_smoothed_means(curves, use_keys, out_keys, smooth)
        for subj, curves in per_subject.items()
        if all(k in curves for k in use_keys)
    }
    smoothing_params = {"moving_avg_window": moving_avg_window, "mode": mode, "split_at": split_at}

    if not per_subject_moving_avg:
        return x, per_subject_moving_avg, {**empty_across_summary(), **smoothing_params}

    keys_by_role = {role: out_keys[i] for i, role in enumerate(TOWER_ROLES)}
    across = across_subject_curves(per_subject_moving_avg, keys_by_role)
    across.update(smoothing_params)
    assert_probabilities_sum_to_one(across["mean"], "Smoothed probabilities", True)

    return x, per_subject_moving_avg, across


# ---------------------------------------------------------------------------
# Truncating post windows at the first intervening bad reversal
# ---------------------------------------------------------------------------


def next_good_after(goods, g):
    """First good-reversal index in *goods* strictly after *g* (``inf`` if none)."""
    return next((gg for gg in goods if gg > g), float("inf"))


def first_bad_between(bads, g, next_g):
    """First bad-reversal index strictly between *g* and *next_g* (``None`` if none)."""
    return next((b for b in bads if g < b < next_g), None)


def truncate_post(reversal, cut_len):
    """Clip every ``"post"`` list in *reversal* to its first *cut_len* entries, in place."""
    for key in ("choices_by_tower", "choices_by_rank"):
        for entry in reversal.get(key, {}).values():
            if isinstance(entry, dict) and "post" in entry:
                entry["post"] = entry["post"][:cut_len]

    trial_window = reversal.get("trial_window_idx", {})
    if isinstance(trial_window, dict) and isinstance(trial_window.get("post", None), list):
        trial_window["post"] = trial_window["post"][:cut_len]


def truncate_reversal_at_bad(reversal, goods, bads, include_bad_trial):
    """Copy of *reversal* with its post window cut at the first intervening bad reversal.

    The copy always carries ``"removed_bad_idx"`` and ``"post_len_after_removal"``
    (both ``None`` when there is nothing to cut).
    """
    out = deepcopy(reversal)
    g = out.get("reversal_idx", None)
    cutoff = None if g is None else first_bad_between(bads, g, next_good_after(goods, g))

    if cutoff is None:
        out["removed_bad_idx"] = None
        out["post_len_after_removal"] = None
        return out

    cut_len = (cutoff - g) + (1 if include_bad_trial else 0)
    out["removed_bad_idx"] = cutoff
    out["post_len_after_removal"] = cut_len
    if cut_len >= 1:
        truncate_post(out, cut_len)
    return out


def report_removal(subj, reversal):
    """Print one line if *reversal*'s post window was actually cut."""
    cut_len = reversal["post_len_after_removal"]
    if cut_len is not None and cut_len >= 1:
        print(
            f"[REMOVE] {subj} good@{reversal['reversal_idx']}: "
            f"cut post at bad@{reversal['removed_bad_idx']} (kept {cut_len} post trials)"
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
        revs_sorted = sorted(good_revs, key=lambda r: r.get("reversal_idx", 0))
        goods = sorted(all_good_idx.get(subj, []))
        bads = sorted(all_bad_idx.get(subj, []))

        out[subj] = []
        for r in revs_sorted:
            truncated = truncate_reversal_at_bad(r, goods, bads, include_bad_trial)
            report_removal(subj, truncated)
            out[subj].append(truncated)
    return out
