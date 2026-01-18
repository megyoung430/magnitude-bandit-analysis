def split_good_reversals_early_late(good_reversal_info, first_n=None, last_n=None):
    """
    Split good reversals into early vs late per subject.

    Default (first_n=None and last_n=None):
      - early = first half
      - late  = second half

    If first_n and/or last_n are provided:
      - early = first `first_n` reversals (if first_n is not None)
      - late  = last  `last_n` reversals  (if last_n is not None)

    Notes:
      - You can set only first_n (early fixed, late = remaining)
      - You can set only last_n (late fixed, early = remaining)
      - You can set both first_n and last_n (could overlap if too few reversals)

    Returns:
      early_info, late_info: dict[subj] -> list[reversal_dict]
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
        e = revs_sorted[:]
        l = revs_sorted[:]

        if first_n is not None:
            first_n = max(0, int(first_n))
            e = revs_sorted[:min(first_n, n)]
        else:
            e = []  # if not specified, define it by remainder below

        if last_n is not None:
            last_n = max(0, int(last_n))
            l = revs_sorted[max(0, n - last_n):]
        else:
            l = []  # if not specified, define it by remainder below

        # If only one of first_n/last_n specified, define the other as "remainder"
        if first_n is not None and last_n is None:
            l = revs_sorted[min(first_n, n):]
        if first_n is None and last_n is not None:
            e = revs_sorted[:max(0, n - last_n)]

        early[subj] = e
        late[subj] = l

    return early, late