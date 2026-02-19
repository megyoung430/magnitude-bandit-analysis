from src.behavior_analysis.get_variables_across_sessions import *

def get_total_reversals(subject_sessions):
    """
    If any session has has_good/has_bad True, use existing good/bad logic.
    If no session has either, fall back to (#blocks - 1), using the last block id
    from the last session (or max across sessions) as the number of blocks.
    """

    total_good = 0
    total_bad = 0

    any_good = any(bool(sess.get("has_good", False)) for sess in subject_sessions.values())
    any_bad  = any(bool(sess.get("has_bad",  False)) for sess in subject_sessions.values())

    if any_good or any_bad:
        for sess in subject_sessions.values():
            if sess.get("has_good", False) and "good_reversals" in sess and sess["good_reversals"]:
                total_good += (sess["good_reversals"][-1] or 0)
            if sess.get("has_bad", False) and "bad_reversals" in sess and sess["bad_reversals"]:
                total_bad += (sess["bad_reversals"][-1] or 0)

        total = total_good + total_bad
        return {"total_reversals": total, "good_reversals": total_good, "bad_reversals": total_bad}

    total = 0
    for sess in subject_sessions.values():
        blocks = sess.get("blocks", [])
        if blocks:
            try:
                total += int(max(b for b in blocks if b is not None)) - 1
            except ValueError:
                pass
            except TypeError:
                pass

    return {"total_reversals": total}

def get_all_reversal_indices(subjects_trials):
    """
    Returns:
      all_good[subj] -> sorted list of good reversal indices
      all_bad[subj]  -> sorted list of bad reversal indices
      blocks[subj]   -> merged blocks list (for optional block logic)
    """
    merged, _ = get_vars_across_all_sessions(subjects_trials)

    all_good = {}
    all_bad = {}
    all_blocks = {}

    for subj, d in merged.items():
        good = d.get("good_reversals", []) or []
        bad  = d.get("bad_reversals", []) or []
        blocks = d.get("blocks", []) or []

        all_good[subj] = find_increment_indices(good)
        all_bad[subj]  = find_increment_indices(bad)
        all_blocks[subj] = blocks

    return all_good, all_bad, all_blocks

def find_increment_indices(cumulative_list):
    """Indices i where cumulative_list[i] > cumulative_list[i-1], ignoring None."""
    inc = []
    prev = None
    for i, v in enumerate(cumulative_list):
        if i == 0:
            prev = v
            continue
        if v is None or prev is None:
            prev = v
            continue
        if v > prev:
            inc.append(i)
        prev = v
    return inc