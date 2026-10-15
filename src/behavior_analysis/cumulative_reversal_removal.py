"""How quickly reversal-aligned windows are cut short by the next reversal.

Each reversal in the windows anchors a window of trials after it. Once another
reversal happens, the rest of that window no longer reflects the anchor's
contingencies, so the anchor counts as *removed* at every post-reversal offset
from the next reversal onwards. The next reversal can be of three kinds:
``good``, ``bad``, or ``unclassified`` (a block change in a session that logs no
good/bad counters, where the kind is unknown). The functions here count removed
anchors per offset:

- :func:`removal_curves_by_kind` splits removals by the kind of the next reversal.
- :func:`good_bad_removal_curves` keeps the good and bad parts of that split.
- :func:`total_removal_curves` counts a removal by the next reversal of any
  kind, so at every offset total = good + bad + unclassified.

The anchors are the windows' ``reversal_idx`` (good reversals when the session
logs a good counter, otherwise block changes). The reversals that can cut a
window come from the per-subject ``good_idx`` / ``bad_idx`` / ``unclassified_idx``
index dicts, not from the windows, because a bad reversal is never an anchor.

All functions are pure; offsets before the reversal (``x < 0``) are ``nan``.
Fractions are ``nan`` when there are no anchors (no data, not 0% removed).
Series are keyed by kind (``"good"``, ``"bad"``, ``"unclassified"``,
``"total"``); display labels belong to the plotting code.
"""

import numpy as np
from numpy.typing import ArrayLike

KINDS = ("good", "bad", "unclassified")
MISSING_INDEX_MESSAGE = (
    "Cumulative removal needs all_good_idx, all_bad_idx and all_unclassified_idx."
)


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
    anchors: np.ndarray, reversals: dict[str, ArrayLike], steps: np.ndarray, exclude_anchor: bool
) -> np.ndarray:
    """Cumulative removal counts over *steps*, one row per kind in :data:`KINDS`.

    *reversals* is ``{kind: trial indices}`` for one subject. Each anchor is
    counted once, under the kind of its first later reversal.
    """
    by_kind = {kind: np.unique(reversals[kind]) for kind in KINDS}
    every = np.unique(np.concatenate(list(by_kind.values())))
    delays, next_reversals = next_reversal_delays(anchors, every, exclude_anchor)
    # kinds are disjoint (a trial is one kind of reversal), so membership picks the row
    kind_of_next = np.select(
        [np.isin(next_reversals, by_kind[kind]) for kind in KINDS[:-1]], range(len(KINDS) - 1), -1
    )
    kind_of_next[kind_of_next < 0] = len(KINDS) - 1
    return np.stack(
        [cumulative_counts(delays[kind_of_next == row], steps) for row in range(len(KINDS))]
    )


def _cumulative_by_kind(
    windows: dict,
    x: ArrayLike,
    exclude_anchor: bool,
    good_idx: dict,
    bad_idx: dict,
    unclassified_idx: dict,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """``(anchors, {kind: cumulative removed over x})``; ``nan`` before the reversal."""
    post_mask, steps, step_idx = post_window_steps(x)
    anchors = _anchors_by_subject(windows)
    if not post_mask.any():
        return anchors, {kind: np.full(post_mask.shape, np.nan) for kind in KINDS}

    def reversals(subject: str) -> dict[str, ArrayLike]:
        return {
            "good": _reversals_of(good_idx, subject, "good_idx"),
            "bad": _reversals_of(bad_idx, subject, "bad_idx"),
            "unclassified": _reversals_of(unclassified_idx, subject, "unclassified_idx"),
        }

    removed = sum(
        (
            _subject_removals(a, reversals(subject), steps, exclude_anchor)
            for subject, a in anchors.items()
        ),
        np.zeros((len(KINDS), steps.size)),
    )
    return anchors, {
        kind: _over_x(row, post_mask, step_idx) for kind, row in zip(KINDS, removed, strict=True)
    }


def removal_curves_by_kind(
    windows: dict,
    x: ArrayLike,
    exclude_anchor: bool = True,
    *,
    good_idx: dict,
    bad_idx: dict,
    unclassified_idx: dict,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Anchors removed by the next reversal, split by its kind, per offset of *x*.

    Each anchor in *windows* is removed once, by the first later reversal of its
    subject, whatever its kind. Kinds are ``"good"``, ``"bad"`` and
    ``"unclassified"``: a reversal of a session that logs no good/bad counters
    (blocks only), where every block change is a reversal of unknown kind. The
    three fractions therefore sum to at most 1, and to the total curve.

    Args:
        windows: ``{subject: [reversal_dict]}``; each dict's ``reversal_idx`` is an anchor.
        x: Trial offsets relative to the anchor.
        exclude_anchor: See :func:`next_reversal_delays`.
        good_idx: ``{subject: [trial index]}`` of good reversals.
        bad_idx: ``{subject: [trial index]}`` of bad reversals.
        unclassified_idx: ``{subject: [trial index]}`` of reversals of unknown
            kind (block changes of sessions without good/bad counters); required,
            ``{subject: []}`` when every session logs counters.

    Returns:
        ``{kind: (cumulative, fraction)}``, each the length of *x*. Fractions
        are counts divided by the number of anchors, ``nan`` if there are none.

    Raises:
        KeyError: If a subject in *windows* is missing from a given index dict.
    """
    anchors, cumulative = _cumulative_by_kind(
        windows, x, exclude_anchor, good_idx, bad_idx, unclassified_idx
    )
    fractions = {kind: _fraction(values, anchors) for kind, values in cumulative.items()}
    _check_fraction(sum(fractions.values()))
    return {kind: (cumulative[kind], fractions[kind]) for kind in KINDS}


def good_bad_removal_curves(
    windows: dict,
    good_idx: dict,
    bad_idx: dict,
    x: ArrayLike,
    exclude_anchor: bool = True,
    *,
    unclassified_idx: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Anchors removed by a later good / bad reversal, per offset of *x*.

    As :func:`removal_curves_by_kind`, keeping only the good and bad curves.
    Anchors first cut by an unclassified reversal are in neither, so the good
    and bad fractions sum to at most 1 and to less than the total when
    *unclassified_idx* has reversals; read them from
    :func:`removal_curves_by_kind`.

    Returns:
        ``(cumulative_good, cumulative_bad, fraction_good, fraction_bad)``,
        each the length of *x*.
    """
    curves = removal_curves_by_kind(
        windows,
        x,
        exclude_anchor,
        good_idx=good_idx,
        bad_idx=bad_idx,
        unclassified_idx=unclassified_idx,
    )
    (cum_good, frac_good), (cum_bad, frac_bad) = curves["good"], curves["bad"]
    return cum_good, cum_bad, frac_good, frac_bad


def total_removal_curves(
    windows: dict,
    x: ArrayLike,
    exclude_anchor: bool = True,
    *,
    good_idx: dict,
    bad_idx: dict,
    unclassified_idx: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Anchors removed by the next reversal of any kind, per offset of *x*.

    Uses the same reversals as :func:`removal_curves_by_kind`, so at every offset
    total = good + bad + unclassified. Arguments are those of
    :func:`removal_curves_by_kind`; the index dicts are required, since without
    bad reversals the curve would silently understate removal.

    Returns:
        ``(cumulative, fraction)``, each the length of *x*.
    """
    anchors, cumulative = _cumulative_by_kind(
        windows, x, exclude_anchor, good_idx, bad_idx, unclassified_idx
    )
    total = sum(cumulative.values())
    fraction = _fraction(total, anchors)
    _check_fraction(fraction)
    return total, fraction


def cumulative_removal_series(
    windows: dict,
    good_idx: dict | None,
    bad_idx: dict | None,
    x: ArrayLike,
    *,
    use_total: bool | None,
    fraction: bool,
    unclassified_idx: dict | None,
) -> dict[str, np.ndarray]:
    """The removal curves to draw, ``{kind: values over x}``.

    Total mode (*use_total* truthy) has one entry, ``"total"`` (every reversal
    kind). Good/bad mode has ``"good"`` and ``"bad"``, plus ``"unclassified"``
    when *unclassified_idx* holds any reversal, so removals by reversals of
    unknown kind (blocks-only sessions) stay visible and good + bad +
    unclassified = total. *fraction* picks fractions over counts. Only
    reversals strictly after the anchor count (``exclude_anchor=True``).

    Raises:
        ValueError: If *good_idx*, *bad_idx* or *unclassified_idx* is ``None``.
    """
    if good_idx is None or bad_idx is None or unclassified_idx is None:
        raise ValueError(MISSING_INDEX_MESSAGE)
    if use_total:
        cumulative, frac = total_removal_curves(
            windows,
            x,
            exclude_anchor=True,
            good_idx=good_idx,
            bad_idx=bad_idx,
            unclassified_idx=unclassified_idx,
        )
        return {"total": frac if fraction else cumulative}
    curves = removal_curves_by_kind(
        windows,
        x,
        exclude_anchor=True,
        good_idx=good_idx,
        bad_idx=bad_idx,
        unclassified_idx=unclassified_idx,
    )
    shown = ("good", "bad")
    if unclassified_idx and any(len(v) for v in unclassified_idx.values()):
        shown += ("unclassified",)
    return {kind: curves[kind][1 if fraction else 0] for kind in shown}
