"""How quickly reversal-aligned windows are cut short by the next reversal.

Each good reversal anchors a window of trials after it. Once another reversal
happens, the rest of that window no longer reflects the anchor's contingencies,
so the anchor counts as *removed* at every post-reversal offset from the next
reversal onwards. The functions here count removed anchors per offset:

- :func:`good_bad_removal_curves` splits removals by whether the next reversal
  was good or bad (the two fractions sum to at most 1).
- :func:`total_removal_curves` counts a removal by any later reversal in the windows.

All functions are pure; offsets before the reversal (``x < 0``) are ``nan``.
"""

import numpy as np

TOTAL_LABEL = "Total Rev"
GOOD_LABEL = "Good Rev"
BAD_LABEL = "Bad Rev"


def post_window_steps(x) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Locate the post-reversal offsets of *x* on a grid of integer trial steps.

    Returns:
        ``(post_mask, steps, step_idx)``: which entries of *x* are finite and
        ``>= 0``; the sorted unique rounded offsets of those entries; and, for
        each post entry, the index of its offset in ``steps``.
    """
    x = np.asarray(x, dtype=float)
    post_mask = np.isfinite(x) & (x >= 0)
    steps, step_idx = np.unique(np.rint(x[post_mask]).astype(int), return_inverse=True)
    return post_mask, steps, step_idx


def next_reversal_delays(
    anchors: np.ndarray, reversals: np.ndarray, exclude_anchor: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Trials from each anchor to the first reversal after it.

    Args:
        anchors: Trial indices of the anchor reversals.
        reversals: Trial indices of every reversal that can cut a window short.
        exclude_anchor: If ``True`` only reversals strictly after the anchor
            count; otherwise a reversal at the anchor's own index also does.

    Returns:
        ``(delays, next_reversals)`` for the anchors that have a later reversal.
    """
    reversals = np.unique(reversals)
    first_after = np.searchsorted(reversals, anchors, side="right" if exclude_anchor else "left")
    has_next = first_after < reversals.size
    next_reversals = reversals[first_after[has_next]]
    return next_reversals - np.asarray(anchors)[has_next], next_reversals


def cumulative_counts(delays: np.ndarray, steps: np.ndarray) -> np.ndarray:
    """Number of *delays* that have elapsed by each step (a non-decreasing staircase).

    A delay is counted from the first step ``>= delay`` onwards; delays beyond
    the last step never show up.
    """
    first_step = np.searchsorted(steps, delays, side="left")
    first_step = first_step[first_step < steps.size]
    return np.cumsum(np.bincount(first_step, minlength=steps.size)).astype(float)


def _anchors_by_subject(windows: dict) -> dict[str, np.ndarray]:
    """``{subject: reversal indices}`` of the windows that have a ``reversal_idx``."""
    return {
        subject: np.array([r["reversal_idx"] for r in revs if r.get("reversal_idx") is not None])
        for subject, revs in windows.items()
    }


def _over_x(per_step: np.ndarray, post_mask: np.ndarray, step_idx: np.ndarray) -> np.ndarray:
    """Spread per-step values back onto the entries of ``x``; pre-window entries are ``nan``."""
    full = np.full(post_mask.shape, np.nan)
    full[post_mask] = per_step[step_idx]
    return full


def _check_fraction(fraction: np.ndarray) -> None:
    if np.nanmax(fraction) > 1.0 + 1e-6:
        raise AssertionError("Fraction removed exceeds 1.0")


def good_bad_removal_curves(
    windows: dict, good_idx: dict, bad_idx: dict, x, exclude_anchor: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Anchors removed by a later good / bad reversal, per offset of *x*.

    Each anchor in *windows* is removed once, by the first later reversal in
    ``good_idx`` or ``bad_idx`` of its subject, so the good and bad fractions
    sum to at most 1.

    Args:
        windows: ``{subject: [reversal_dict]}``; each dict's ``reversal_idx`` is an anchor.
        good_idx: ``{subject: [trial index]}`` of good reversals.
        bad_idx: ``{subject: [trial index]}`` of bad reversals.
        x: Trial offsets relative to the anchor.
        exclude_anchor: See :func:`next_reversal_delays`.

    Returns:
        ``(cumulative_good, cumulative_bad, fraction_good, fraction_bad)``,
        each the length of *x* (fractions are counts divided by the number of anchors).
    """
    post_mask, steps, step_idx = post_window_steps(x)
    if not post_mask.any():
        return tuple(np.full(post_mask.shape, np.nan) for _ in range(4))

    removed_good = np.zeros(steps.size)
    removed_bad = np.zeros(steps.size)
    anchors = _anchors_by_subject(windows)
    for subject, subject_anchors in anchors.items():
        goods = np.unique(good_idx.get(subject, []))
        bads = np.unique(bad_idx.get(subject, []))
        delays, next_reversals = next_reversal_delays(
            subject_anchors, np.union1d(goods, bads), exclude_anchor
        )
        # every reversal comes from one of the two lists, so "not good" means bad
        next_is_good = np.isin(next_reversals, goods)
        removed_good += cumulative_counts(delays[next_is_good], steps)
        removed_bad += cumulative_counts(delays[~next_is_good], steps)

    cumulative_good = _over_x(removed_good, post_mask, step_idx)
    cumulative_bad = _over_x(removed_bad, post_mask, step_idx)
    n_anchors = max(sum(a.size for a in anchors.values()), 1)
    fraction_good, fraction_bad = cumulative_good / n_anchors, cumulative_bad / n_anchors
    _check_fraction(fraction_good + fraction_bad)
    return cumulative_good, cumulative_bad, fraction_good, fraction_bad


def total_removal_curves(
    windows: dict, x, exclude_anchor: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Anchors removed by any later reversal in *windows*, per offset of *x*.

    Returns:
        ``(cumulative, fraction)``, each the length of *x*.
    """
    post_mask, steps, step_idx = post_window_steps(x)
    if not post_mask.any():
        return np.full(post_mask.shape, np.nan), np.full(post_mask.shape, np.nan)

    removed = np.zeros(steps.size)
    anchors = _anchors_by_subject(windows)
    for subject_anchors in anchors.values():
        delays, _ = next_reversal_delays(subject_anchors, subject_anchors, exclude_anchor)
        removed += cumulative_counts(delays, steps)

    cumulative = _over_x(removed, post_mask, step_idx)
    fraction = cumulative / max(sum(a.size for a in anchors.values()), 1)
    _check_fraction(fraction)
    return cumulative, fraction


def cumulative_removal_series(
    x, windows: dict, good_idx, bad_idx, *, use_total, fraction: bool
) -> dict[str, np.ndarray]:
    """The removal curves to draw, ``{label: values over x}``.

    Total mode (*use_total* truthy) has one entry; good/bad mode has ``Good Rev``
    and ``Bad Rev`` and needs both index dicts. *fraction* picks fractions over counts.
    Anchors are always the strictly-later-reversal kind (``exclude_anchor=True``).

    Raises:
        ValueError: In good/bad mode if *good_idx* or *bad_idx* is missing.
    """
    if use_total:
        cumulative, frac = total_removal_curves(windows, x, exclude_anchor=True)
        return {TOTAL_LABEL: frac if fraction else cumulative}
    if good_idx is None or bad_idx is None:
        raise ValueError("For mode='goodbad', provide all_good_idx and all_bad_idx.")
    cum_good, cum_bad, frac_good, frac_bad = good_bad_removal_curves(
        windows, good_idx, bad_idx, x, exclude_anchor=True
    )
    if fraction:
        return {GOOD_LABEL: frac_good, BAD_LABEL: frac_bad}
    return {GOOD_LABEL: cum_good, BAD_LABEL: cum_bad}
