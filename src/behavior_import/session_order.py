"""The one rule for ordering a subject's sessions, used wherever sessions are ordered.

Session keys look like ``ses-3_date-20260102`` (a recording split by
``split_recording_configurations`` adds ``_part-01``). Counters, blocks and trials are merged
across a subject's sessions in this order, and problems are numbered in it, so every step must
agree on it.

- :func:`session_sort_key` orders by date, then session number, then the key itself (so
  ``_part-NN`` segments keep recording order).
- :func:`ordered_session_keys` requires ``ses-<n>`` in every key and raises ``ValueError``
  otherwise. The date is optional: a bare key such as ``ses-5`` (used by many synthetic
  fixtures; imported keys always have both) sorts after every dated key, by number.
- :func:`check_session_order` raises ``ValueError`` when a subject's sessions are mislabelled:
  session-number order and date order disagree (a later date with a smaller number), or one
  number appears on two dates. Equal numbers are allowed only on equal dates (split segments).
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
    """Raise if the session keys of *subject* cannot be ordered unambiguously.

    Raises:
        ValueError: Naming *subject* and the sessions, if a key has no ``ses-<n>``, one number
            is on more than one date, or the numbers and dates order two sessions oppositely.
            Keys without a date are not compared by date.
    """
    for key in keys:
        if session_number(key) is None:
            raise ValueError(
                f"{subject}: session key {key!r} has no 'ses-<n>' number; keys look like "
                "'ses-<n>_date-<YYYYMMDD>'."
            )
    dated = sorted((k for k in keys if session_date(k) is not None), key=session_sort_key)
    dates_of = {}
    for key in dated:
        dates_of.setdefault(session_number(key), {})[session_date(key)] = key
    for number, by_date in dates_of.items():
        if len(by_date) > 1:
            raise ValueError(
                f"{subject}: session number {number} is on more than one date: "
                f"{sorted(by_date.values())}. Fix the session keys."
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
