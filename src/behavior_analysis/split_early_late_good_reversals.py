"""Split good-reversal windows into early and late subsets per subject.

This allows analyses to compare behaviour early in training (when the animal
is still learning the task structure) versus late in training (when it is more
experienced).
"""


def split_good_reversals_early_late(good_reversal_info, first_n=None, last_n=None):
    """Split good-reversal windows into early and late subsets per subject.

    Reversals are sorted by ``"good_reversal_number"`` within each subject
    before splitting.

    Splitting modes:

    - **Default** (both *first_n* and *last_n* are ``None``): splits at the
      midpoint — early = first half, late = second half.
    - **Fixed first_n**: early = first *first_n* reversals, late = remainder.
    - **Fixed last_n**: late = last *last_n* reversals, early = remainder.
    - **Both specified**: early = first *first_n*, late = last *last_n*
      (may overlap if the subject has too few reversals).

    Args:
        good_reversal_info: ``{subject: list[reversal_dict]}`` as returned by
            :func:`src.behavior_analysis.get_good_reversal_info.get_good_reversal_info`.
        first_n: Number of reversals to include in the early split, or
            ``None`` to use the midpoint / complement of *last_n*
            (default: ``None``).
        last_n: Number of reversals to include in the late split, or
            ``None`` to use the midpoint / complement of *first_n*
            (default: ``None``).

    Returns:
        A two-tuple ``(early_info, late_info)`` where each element is a
        ``{subject: list[reversal_dict]}`` dict.
    """
    early, late = {}, {}

    for subj, revs in good_reversal_info.items():
        if not revs:
            early[subj] = []
            late[subj] = []
            continue

        revs_sorted = sorted(revs, key=lambda r: r.get("good_reversal_number", 0))
        n = len(revs_sorted)

        # Default: split into halves
        if first_n is None and last_n is None:
            mid = n // 2
            early[subj] = revs_sorted[:mid]
            late[subj] = revs_sorted[mid:]
            continue

        # Custom: first_n / last_n
        early_revs = revs_sorted[:]
        late_revs = revs_sorted[:]

        if first_n is not None:
            first_n = max(0, int(first_n))
            early_revs = revs_sorted[:min(first_n, n)]
        else:
            early_revs = []  # if not specified, define it by remainder below

        if last_n is not None:
            last_n = max(0, int(last_n))
            late_revs = revs_sorted[max(0, n - last_n):]
        else:
            late_revs = []  # if not specified, define it by remainder below

        # If only one of first_n/last_n specified, define the other as "remainder"
        if first_n is not None and last_n is None:
            late_revs = revs_sorted[min(first_n, n):]
        if first_n is None and last_n is not None:
            early_revs = revs_sorted[:max(0, n - last_n)]

        early[subj] = early_revs
        late[subj] = late_revs

    return early, late
