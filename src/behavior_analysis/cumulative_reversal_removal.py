"""How quickly reversal-aligned windows are cut short by the next reversal.

Each good reversal anchors a window of trials after it. Once another reversal
happens, the rest of that window no longer reflects the anchor's contingencies,
so the anchor counts as *removed* at every post-reversal offset from the next
reversal onwards. The functions here count removed anchors per offset:

- :func:`good_bad_removal_curves` splits removals by whether the next reversal
  was good or bad (the two fractions sum to at most 1).
- :func:`total_removal_curves` counts a removal by the next reversal of either
  kind, so at every offset total = good + bad.

Windows are built from good reversals only, so the reversals that can cut a
window come from the per-subject ``good_idx`` / ``bad_idx`` index dicts, never
from the windows themselves.

All functions are pure; offsets before the reversal (``x < 0``) are ``nan``.
Fractions are ``nan`` when there are no anchors (no data, not 0% removed).
Series are keyed by kind (``"good"``, ``"bad"``, ``"total"``); display labels
belong to the plotting code.
"""

import numpy as np
from numpy.typing import ArrayLike


def post_window_steps(x: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    finite = fraction[np.isfinite(fraction)]
    if finite.size and finite.max() > 1.0 + 1e-6:
        raise AssertionError("Fraction removed exceeds 1.0")


def _fraction(cumulative: np.ndarray, anchors: dict[str, np.ndarray]) -> np.ndarray:
    """*cumulative* counts over the number of anchors; all ``nan`` when there are none."""
    n_anchors = sum(a.size for a in anchors.values())
    if n_anchors == 0:
        return np.full(cumulative.shape, np.nan)
    return cumulative / n_anchors


def _reversals_of(index: dict, subject: str, name: str) -> ArrayLike:
    """``index[subject]``, or a ``KeyError`` naming the subject if it is absent."""
    if subject not in index:
        raise KeyError(f"Subject {subject!r} has windows but is missing from {name}.")
    return index[subject]


def _subject_removals(
    anchors: np.ndarray, good: ArrayLike, bad: ArrayLike, steps: np.ndarray, exclude_anchor: bool
) -> np.ndarray:
    """``[good, bad]`` cumulative removal counts over *steps* for one subject's anchors."""
    goods, bads = np.unique(good), np.unique(bad)
    delays, next_reversals = next_reversal_delays(anchors, np.union1d(goods, bads), exclude_anchor)
    # every reversal comes from one of the two lists, so "not good" means bad
    next_is_good = np.isin(next_reversals, goods)
    return np.stack(
        [
            cumulative_counts(delays[next_is_good], steps),
            cumulative_counts(delays[~next_is_good], steps),
        ]
    )


def good_bad_removal_curves(
    windows: dict, good_idx: dict, bad_idx: dict, x: ArrayLike, exclude_anchor: bool = True
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
        each the length of *x* (fractions are counts divided by the number of
        anchors, ``nan`` if there are none).

    Raises:
        KeyError: If a subject in *windows* is missing from *good_idx* or *bad_idx*.
    """
    post_mask, steps, step_idx = post_window_steps(x)
    if not post_mask.any():
        return tuple(np.full(post_mask.shape, np.nan) for _ in range(4))

    anchors = _anchors_by_subject(windows)
    per_subject = (
        _subject_removals(
            a,
            _reversals_of(good_idx, subject, "good_idx"),
            _reversals_of(bad_idx, subject, "bad_idx"),
            steps,
            exclude_anchor,
        )
        for subject, a in anchors.items()
    )
    removed_good, removed_bad = sum(per_subject, np.zeros((2, steps.size)))

    cumulative_good = _over_x(removed_good, post_mask, step_idx)
    cumulative_bad = _over_x(removed_bad, post_mask, step_idx)
    fraction_good, fraction_bad = (
        _fraction(cumulative_good, anchors),
        _fraction(cumulative_bad, anchors),
    )
    _check_fraction(fraction_good + fraction_bad)
    return cumulative_good, cumulative_bad, fraction_good, fraction_bad


def total_removal_curves(
    windows: dict,
    x: ArrayLike,
    exclude_anchor: bool = True,
    *,
    good_idx: dict | None = None,
    bad_idx: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Anchors removed by the next reversal of either kind, per offset of *x*.

    Uses the same reversals as :func:`good_bad_removal_curves`, so the result is
    exactly its good plus bad curves at every offset.

    Args:
        windows, x, exclude_anchor: See :func:`good_bad_removal_curves`.
        good_idx: ``{subject: [trial index]}`` of good reversals; required.
        bad_idx: ``{subject: [trial index]}`` of bad reversals; required.

    Returns:
        ``(cumulative, fraction)``, each the length of *x*.

    Raises:
        ValueError: If *good_idx* or *bad_idx* is ``None``; without both, bad
            reversals would silently be ignored.
        KeyError: If a subject in *windows* is missing from *good_idx* or *bad_idx*.
    """
    if good_idx is None or bad_idx is None:
        raise ValueError("Total removal needs both good_idx and bad_idx.")
    cum_good, cum_bad, _, _ = good_bad_removal_curves(windows, good_idx, bad_idx, x, exclude_anchor)
    cumulative = cum_good + cum_bad
    return cumulative, _fraction(cumulative, _anchors_by_subject(windows))


def cumulative_removal_series(
    windows: dict,
    good_idx: dict | None,
    bad_idx: dict | None,
    x: ArrayLike,
    *,
    use_total: bool | None,
    fraction: bool,
) -> dict[str, np.ndarray]:
    """The removal curves to draw, ``{kind: values over x}``.

    Total mode (*use_total* truthy) has one entry, ``"total"`` (good plus bad);
    good/bad mode has ``"good"`` and ``"bad"``. Both modes need both index
    dicts. *fraction* picks fractions over counts. Only reversals strictly after
    the anchor count (``exclude_anchor=True``).

    Raises:
        ValueError: If *good_idx* or *bad_idx* is ``None``.
    """
    if use_total:
        cumulative, frac = total_removal_curves(
            windows, x, exclude_anchor=True, good_idx=good_idx, bad_idx=bad_idx
        )
        return {"total": frac if fraction else cumulative}
    if good_idx is None or bad_idx is None:
        raise ValueError("For mode='goodbad', provide all_good_idx and all_bad_idx.")
    cum_good, cum_bad, frac_good, frac_bad = good_bad_removal_curves(
        windows, good_idx, bad_idx, x, exclude_anchor=True
    )
    if fraction:
        return {"good": frac_good, "bad": frac_bad}
    return {"good": cum_good, "bad": cum_bad}
