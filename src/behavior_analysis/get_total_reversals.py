"""Functions for counting and locating reversal events in session data.

Provides helpers to count total good/bad reversals across sessions per subject,
retrieve all reversal indices, and identify trials where cumulative reversal
counters increment.
"""
from src.behavior_analysis.get_variables_across_sessions import get_vars_across_all_sessions


def get_total_reversals(subject_sessions):
    """Count total good, bad, and overall reversals for a single subject.

    If any session has ``has_good`` or ``has_bad`` set to ``True``, reads the
    final value of the ``good_reversals`` / ``bad_reversals`` cumulative counter
    from each session.  Otherwise falls back to summing
    ``max(blocks) - 1`` across sessions.

    Args:
        subject_sessions: ``{session_key: session_dict}`` for a single subject.

    Returns:
        Dict with key ``"total_reversals"`` (int) and, when good/bad data are
        available, also ``"good_reversals"`` (int) and ``"bad_reversals"``
        (int).
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
    """Return per-subject lists of good and bad reversal trial indices.

    Merges sessions for each subject (via
    :func:`src.behavior_analysis.get_variables_across_sessions.get_vars_across_all_sessions`)
    and then identifies the trial indices where the cumulative ``good_reversals``
    and ``bad_reversals`` counters increment.

    Args:
        subjects_trials: Nested dict ``{subject: {session_key: session_dict}}``.

    Returns:
        A three-tuple ``(all_good, all_bad, all_blocks)`` where:

        - ``all_good[subj]`` – sorted list of good-reversal trial indices.
        - ``all_bad[subj]`` – sorted list of bad-reversal trial indices.
        - ``all_blocks[subj]`` – merged blocks list for optional block logic.
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
    """Return trial indices where a cumulative counter increments.

    Iterates through *cumulative_list* and collects every index ``i`` where the
    value is strictly greater than the previous non-``None`` value.

    Args:
        cumulative_list: Sequence of numeric values (or ``None``) representing a
            cumulative counter, e.g. ``[0, 0, 1, 1, 2, None, 2, 3]``.

    Returns:
        List of integer indices where an increment occurred.
    """
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
