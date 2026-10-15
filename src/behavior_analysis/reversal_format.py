"""Which way a subject's sessions record reversals, and the check that they agree.

A session with trials records reversals in one of two formats:

- **good-counter format**: ``has_good`` is set, so the cumulative ``good_reversals`` counter
  says which block changes are good reversals.
- **blocks-only format**: neither ``has_good`` nor ``has_bad`` is set (sessions from before the
  counters existed, or without any). Only the ``blocks`` series is there, and every block
  change is a reversal of unknown kind.

A bad-only session (``has_bad`` without ``has_good``) is neither: its block changes are bad
reversals, and next to a blocks-only session the windows and the unclassified indices both fall
back to block changes and agree, so it is not part of the check. Sessions without trials are
ignored.

When one subject has both formats, the good-reversal windows would follow the good counter and
drop the block changes of the blocks-only sessions, while the unclassified indices would count
them. :func:`check_formats_agree` refuses that mix.
"""

from src.behavior_import.session_order import ordered_session_keys


def _has_trials(session):
    return bool(session.get("trial"))


def is_good_counter_session(session):
    """True for a session with trials that logs the good-reversal counter."""
    return _has_trials(session) and bool(session.get("has_good"))


def is_blocks_only_session(session):
    """True for a session with trials that logs neither the good nor the bad counter."""
    return _has_trials(session) and not session.get("has_good") and not session.get("has_bad")


def check_formats_agree(subject, sessions):
    """Raise if *subject* has both good-counter and blocks-only sessions.

    Args:
        subject: Subject id, named in the error.
        sessions: ``{session_key: session_dict}`` of that subject.

    Raises:
        ValueError: Naming the subject and the sessions of each format, in session order.
    """
    keys = ordered_session_keys(subject, sessions)
    good = [k for k in keys if is_good_counter_session(sessions[k])]
    blocks_only = [k for k in keys if is_blocks_only_session(sessions[k])]
    if good and blocks_only:
        raise ValueError(
            f"{subject}: sessions mix the good-counter format with the blocks-only format "
            f"(good-counter sessions {good}, blocks-only sessions {blocks_only}). Windows "
            "would follow the good counter and lose the block changes of the blocks-only "
            "sessions, so analyse the two kinds of session separately."
        )
