"""Functions for counting and locating reversal events in session data.

Provides helpers to count total good/bad reversals across sessions per subject,
retrieve all reversal indices, and identify trials where cumulative reversal
counters increment.
"""
from src.behavior_analysis.get_variables_across_sessions import get_vars_across_all_sessions
from src.behavior_analysis.reversal_format import check_formats_agree
from src.behavior_analysis.suspect_reversals import count_suspect_good_reversals
from src.behavior_import.session_order import ordered_session_keys


def get_total_reversals(subject_sessions):
    """Count total good, bad, and overall reversals for a single subject.

    If any session has ``has_good`` or ``has_bad`` set to ``True``, reads the
    reconciled session totals (including terminal and incoming boundary events),
    falling back to the final trial counter for legacy extracted data.  Otherwise falls back to summing
    ``max(blocks) - 1`` across sessions.

    Args:
        subject_sessions: ``{session_key: session_dict}`` for a single subject.

    Returns:
        Dict with key ``"total_reversals"`` (int) and, when good/bad data are
        available, also ``"good_reversals"`` (int) and ``"bad_reversals"``
        (int), as logged by the task. Beside them, ``"suspect_good_reversals"``
        counts the logged good reversals whose best tower does not change, and
        ``"uncheckable_good_reversals"`` those with no trial to check (between
        runs, after the last trial, or a session without reward magnitudes);
        see :mod:`src.behavior_analysis.suspect_reversals`.
    """

    total_good = 0
    total_bad = 0

    any_good = any(bool(sess.get("has_good", False)) for sess in subject_sessions.values())
    any_bad  = any(bool(sess.get("has_bad",  False)) for sess in subject_sessions.values())

    if any_good or any_bad:
        for sess in subject_sessions.values():
            for kind in ('good', 'bad'):
                values = sess.get(kind + '_reversals', [])
                fallback = next((v for v in reversed(values) if v is not None), 0)
                count = sess.get('reversal_totals', {}).get(kind, fallback)
                if kind == 'good':
                    total_good += count
                else:
                    total_bad += count

        total = total_good + total_bad
        suspects = [count_suspect_good_reversals(sess) for sess in subject_sessions.values()]
        return {
            "total_reversals": total,
            "good_reversals": total_good,
            "bad_reversals": total_bad,
            "suspect_good_reversals": sum(c["suspect"] for c in suspects),
            "uncheckable_good_reversals": sum(c["uncheckable"] for c in suspects),
        }

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

def get_unclassified_reversal_indices(subjects_trials):
    """Block-change trial indices of sessions that log no good/bad counters.

    A session with neither ``has_good`` nor ``has_bad`` cannot say whether a
    reversal was good or bad, so each of its block changes is a reversal of
    unknown kind. Indices are on the merged-trial axis that
    :func:`get_all_reversal_indices` uses, and sessions that do log counters
    contribute none (their reversals are already good or bad).

    Args:
        subjects_trials: Nested dict ``{subject: {session_key: session_dict}}``.

    Returns:
        ``{subject: sorted list of trial indices}``; every subject is a key.

    Raises:
        ValueError: If a session's ``blocks`` and ``trial`` differ in length
            (missing counts as empty); the merged block axis would no longer
            line up with the merged trial axis. Also if a subject mixes
            good-counter and blocks-only sessions
            (:func:`src.behavior_analysis.reversal_format.check_formats_agree`), which keeps
            the unclassified indices disjoint from the good ones.
    """
    for subject, sessions in subjects_trials.items():
        check_formats_agree(subject, sessions)
        for key, sess in sessions.items():
            n_trials, n_blocks = len(sess.get("trial", [])), len(sess.get("blocks", []))
            if n_trials != n_blocks:
                raise ValueError(
                    f"{subject} {key}: blocks has {n_blocks} entries but trial has {n_trials}."
                )
    merged, _ = get_vars_across_all_sessions(subjects_trials)
    unclassified = {}
    for subject, sessions in subjects_trials.items():
        blocks = merged[subject]["blocks"]
        indices, start = [], 0
        for key in ordered_session_keys(subject, sessions):
            sess = sessions[key]
            n_trials = len(sess.get("trial", []))
            if n_trials and not (sess.get("has_good") or sess.get("has_bad")):
                first = max(start, 1)
                indices += [i for i in range(first, start + n_trials) if blocks[i] != blocks[i - 1]]
            start += n_trials
        unclassified[subject] = indices
    return unclassified


def get_reversal_indices_by_kind(subjects_trials):
    """Good, bad and unclassified reversal indices per subject, for the removal curves.

    Returns:
        ``{"good": ..., "bad": ..., "unclassified": ...}``, each ``{subject: [index]}``
        (see :func:`get_all_reversal_indices` and :func:`get_unclassified_reversal_indices`).
    """
    good, bad, _ = get_all_reversal_indices(subjects_trials)
    return {
        "good": good,
        "bad": bad,
        "unclassified": get_unclassified_reversal_indices(subjects_trials),
    }


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
