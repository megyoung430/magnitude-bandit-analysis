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
    """True for a session with trials that logs the good or the bad reversal counter."""
    return _has_trials(session) and bool(session.get("has_good") or session.get("has_bad"))


def is_blocks_only_session(session):
    """True for a session with trials that logs neither the good nor the bad counter."""
    return _has_trials(session) and not session.get("has_good") and not session.get("has_bad")


def check_formats_agree(subject, sessions):
    """Raise if *subject* has both counter and blocks-only sessions.

    Args:
        subject: Subject id, named in the error.
        sessions: ``{session_key: session_dict}`` of that subject.

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
