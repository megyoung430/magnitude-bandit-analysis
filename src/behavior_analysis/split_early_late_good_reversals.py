"""Split good-reversal windows into early and late subsets per subject.

This allows analyses to compare behaviour early in training (when the animal
is still learning the task structure) versus late in training (when it is more
experienced).
"""


def _as_count(n):
    """*n* as a non-negative int (negative counts mean zero)."""
    return max(0, int(n))


def _split_sorted(revs_sorted, *, first_n, last_n):
    """``(early, late)`` slices of an ordered, non-empty reversal list."""
    n = len(revs_sorted)
    if first_n is None and last_n is None:
        mid = n // 2
        return revs_sorted[:mid], revs_sorted[mid:]

    if first_n is not None:
        first_n = _as_count(first_n)
    if last_n is not None:
        last_n = _as_count(last_n)
    # A split that is not given is the remainder of the one that is, so the
    # two cut points coincide; given both, the windows may overlap or leave a gap.
    early_end = min(first_n, n) if first_n is not None else max(0, n - last_n)
    late_start = max(0, n - last_n) if last_n is not None else min(first_n, n)
    return revs_sorted[:early_end], revs_sorted[late_start:]


def _split_subject(revs, *, first_n, last_n):
    """``(early, late)`` for one subject's reversals; both empty if it has none."""
    if not revs:
        return [], []
    revs_sorted = sorted(revs, key=lambda r: r.get("good_reversal_number", 0))
    return _split_sorted(revs_sorted, first_n=first_n, last_n=last_n)


def split_good_reversals_early_late(good_reversal_info, first_n=None, last_n=None):
    """Split good-reversal windows into early and late subsets per subject.

    Reversals are sorted by ``"good_reversal_number"`` within each subject
    before splitting.

    The windows come from :func:`get_good_reversal_info`, which has already
    dropped reversals whose best tower did not change. The split therefore
    counts valid reversals only: "the first N" means the first N valid ones.

    Splitting modes:

    - **Default** (both *first_n* and *last_n* are ``None``): splits at the
      midpoint — early = first half, late = second half.
    - **Fixed first_n**: early = first *first_n* reversals, late = remainder.
    - **Fixed last_n**: late = last *last_n* reversals, early = remainder.
    - **Both specified**: early = first *first_n*, late = last *last_n*
      (may overlap if the subject has too few reversals).

    Args:
        good_reversal_info (dict): ``{subject: list[reversal_dict]}`` as returned by
            :func:`src.behavior_analysis.get_good_reversal_info.get_good_reversal_info`.
        first_n (int | None): Number of reversals to include in the early split, or
            ``None`` to use the midpoint / complement of *last_n*. Negative counts mean zero.
            Default ``None``.
        last_n (int | None): Number of reversals to include in the late split, or
            ``None`` to use the midpoint / complement of *first_n*. Negative counts mean zero.
            Default ``None``.

    Returns:
        tuple[dict, dict]: A two-tuple ``(early_info, late_info)`` where each element is a
        ``{subject: list[reversal_dict]}`` dict with the same subjects as the input.
    """
    early, late = {}, {}
    for subj, revs in good_reversal_info.items():
        early[subj], late[subj] = _split_subject(revs, first_n=first_n, last_n=last_n)
    return early, late
