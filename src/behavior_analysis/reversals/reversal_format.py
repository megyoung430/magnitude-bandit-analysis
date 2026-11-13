"""Which way a subject's sessions record reversals, and the check that they agree.

A session with trials records reversals in one of two formats:

- **counter format**: ``has_good`` or ``has_bad`` is set. The cumulative ``good_reversals`` and
  ``bad_reversals`` counters say which block changes are good or bad reversals. A bad-only
  session (``has_bad`` without ``has_good``) is counter format too: its bad counter marks
  the bad reversals, but its other block changes have no good counter to label them.
- **blocks-only format**: neither is set (sessions from before the counters existed, or
  without any). Only the ``blocks`` series is there, and every block change is a reversal of
  unknown kind.

Sessions without trials are ignored.

Analyses that treat block changes as reversals (the windows, the unclassified indices, the
good/bad indices, the cumulative reversal series) read counter sessions from the counters and
blocks-only sessions from the blocks, and merge both on one trial axis. A subject with both
formats would count the same kind of event two different ways, so
:func:`check_formats_agree` refuses that mix.
"""

from src.behavior_import.session_order import ordered_session_keys


def _has_trials(session):
    return bool(session.get("trial"))


def is_counter_session(session):
    """True for a session with trials that logs the good or the bad reversal counter.

    Args:
        session (dict): A session dict with ``"trial"`` and optionally ``"has_good"`` and
            ``"has_bad"``.

    Returns:
        bool: Whether the session is counter format.
    """
    return _has_trials(session) and bool(session.get("has_good") or session.get("has_bad"))


def is_blocks_only_session(session):
    """True for a session with trials that logs neither the good nor the bad counter.

    Args:
        session (dict): A session dict with ``"trial"`` and optionally ``"has_good"`` and
            ``"has_bad"``.

    Returns:
        bool: Whether the session is blocks-only format.
    """
    return _has_trials(session) and not session.get("has_good") and not session.get("has_bad")


def check_formats_agree(subject, sessions):
    """Raise if *subject* has both counter and blocks-only sessions.

    Args:
        subject (str): Subject id, named in the error.
        sessions (dict): ``{session_key: session_dict}`` of that subject.

    Returns:
        None: Nothing; the function only validates.

    Raises:
        ValueError: Naming the subject and the sessions of each format, in session order.
    """
    keys = ordered_session_keys(subject, sessions)
    counter = [k for k in keys if is_counter_session(sessions[k])]
    blocks_only = [k for k in keys if is_blocks_only_session(sessions[k])]
    if counter and blocks_only:
        raise ValueError(
            f"{subject}: sessions mix the counter format with the blocks-only format "
            f"(counter sessions {counter}, blocks-only sessions {blocks_only}). Reversals "
            "would be classified two ways within one subject, so analyse the two kinds of "
            "session separately."
        )


_DRIFTING_ONLY_FACT = (
    "The drifting task has only block changes, no good or bad reversals, so this analysis "
    "runs on drifting-task data only."
)


def check_blocks_only_sessions(subjects_trials_by_problem, analysis):
    """Raise if any session of any problem is counter format.

    The drifting-only analyses treat every block change alike; a session with good/bad
    reversal counters comes from a task that has reversals, so it is the wrong data.

    Args:
        subjects_trials_by_problem (Mapping): ``{problem: {subject: {session_key: session}}}``.
        analysis (str): Name of the calling analysis, for the error message.

    Returns:
        None: Nothing; the function only validates.

    Raises:
        ValueError: Naming the analysis, the problem, the mouse and the first counter-format
            session, and saying the drifting task has only block changes.
    """
    for problem, subjects in subjects_trials_by_problem.items():
        for subject, sessions in subjects.items():
            for key, session in sessions.items():
                if is_counter_session(session):
                    raise ValueError(
                        f"{analysis}: mouse {subject!r}, session {key!r} (problem {problem!r}) "
                        f"is counter format (it logs good/bad reversal counters). "
                        f"{_DRIFTING_ONLY_FACT}"
                    )


def check_block_change_windows(windows, analysis):
    """Raise unless every window record was bounded by the ``blocks`` series.

    Windows built from counter-format sessions record ``boundary_source`` ``"good_reversals"``
    (or ``"num_blocks"``); only ``"blocks"`` means plain block changes.

    Args:
        windows (Mapping): ``{subject: [window record]}`` as ``get_good_reversal_info`` returns.
        analysis (str): Name of the calling analysis, for the error message.

    Returns:
        None: Nothing; the function only validates.

    Raises:
        ValueError: Naming the analysis, the mouse and the reversal index of the first record
            whose ``boundary_source`` is missing or is not ``"blocks"``.
    """
    for subject, records in windows.items():
        for record in records:
            source = record.get("boundary_source")
            if source != "blocks":
                found = "no boundary_source" if source is None else f"boundary_source {source!r}"
                raise ValueError(
                    f"{analysis}: mouse {subject!r}, window at trial "
                    f"{record.get('reversal_idx')} has {found}, not 'blocks'. "
                    f"{_DRIFTING_ONLY_FACT}"
                )
