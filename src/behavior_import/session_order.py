"""The one rule for ordering a subject's sessions, used wherever sessions are ordered.

Session keys look like ``ses-3_date-20260102`` (a recording split by
``split_recording_configurations`` adds ``_part-01``). Counters, blocks and trials are merged
across a subject's sessions in this order, and problems are numbered in it, so every step must
agree on it.

- :func:`session_sort_key` orders by date, then session number, then the key itself (so
  ``_part-NN`` segments keep recording order).
- Fallback, deterministic: a key without a parseable ``date-YYYYMMDD`` sorts after every
  dated key, and one without ``ses-<n>`` after every numbered key of the same date; keys that
  tie on both fall back to string order.
- :func:`check_session_order` raises ``ValueError`` when session-number order and date order
  disagree for a subject (a later date with a smaller session number). That means a
  mislabelled session, and merging by either order would put trials on the wrong days. Only
  keys with both a date and a number are compared; equal dates (several sessions on a day) and
  equal numbers (segments) never conflict.
"""

import re

_DATE_RE = re.compile(r"date-(\d{8})")
_NUMBER_RE = re.compile(r"ses-(\d+)")


def session_date(key):
    """The ``date-YYYYMMDD`` of a session key as an int, or ``None``."""
    match = _DATE_RE.search(key)
    return int(match[1]) if match else None


def session_number(key):
    """The ``ses-<n>`` number of a session key as an int, or ``None``."""
    match = _NUMBER_RE.search(key)
    return int(match[1]) if match else None


def session_sort_key(session_id):
    """Chronological sort key: date, then session number, then the key (see the module doc)."""
    date, number = session_date(session_id), session_number(session_id)
    return (
        float("inf") if date is None else date,
        float("inf") if number is None else number,
        session_id,
    )


def check_session_order(subject, keys):
    """Raise if the session numbers and the dates of *keys* order the sessions differently.

    Raises:
        ValueError: Naming *subject* and the first two sessions whose number order and date
            order are opposite.
    """
    dated = sorted(
        (k for k in keys if session_date(k) is not None and session_number(k) is not None),
        key=session_sort_key,
    )
    for earlier, later in zip(dated, dated[1:], strict=False):
        if session_number(later) < session_number(earlier):
            raise ValueError(
                f"{subject}: session number order and date order disagree: {later!r} is "
                f"dated after {earlier!r} but has a smaller session number. Fix the session "
                "keys; the sessions cannot be ordered unambiguously."
            )


def ordered_session_keys(subject, keys):
    """*keys* of one subject in :func:`session_sort_key` order, after checking them.

    Raises ``ValueError`` as :func:`check_session_order` does.
    """
    keys = list(keys)
    check_session_order(subject, keys)
    return sorted(keys, key=session_sort_key)
