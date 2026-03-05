"""Utilities for sorting and labelling session keys.

Session keys follow the format ``ses-XX_date-YYYYMMDD``.  The helpers here
extract the numeric ``ses-XX`` index for chronological ordering and provide
a short display label for x-tick annotations.
"""
import re

def sort_sessions_chronologically(session_name):
    """Return a sort key that orders session keys by their ``ses-XX`` index.

    Extracts the integer from the ``ses-XX`` part of the session name and
    returns it as an ``int``.  Falls back to the raw string if the pattern is
    not found, preserving relative order with standard string comparison.

    Args:
        session_name: Session key string, e.g. ``"ses-03_date-20240115"``.

    Returns:
        ``int`` session number when the ``ses-XX`` pattern is found; the raw
        string otherwise.
    """
    m = re.search(r"ses-(\d+)", str(session_name))
    return int(m.group(1)) if m else str(session_name)

def get_short_session_label(session_name):
    """Extract the short ``ses-XX`` label from a full session key.

    Args:
        session_name: Session key string, e.g. ``"ses-03_date-20240115"``.

    Returns:
        The matched ``"ses-XX"`` substring (e.g. ``"ses-03"``), or the full
        *session_name* string if the pattern is not found.
    """
    m = re.search(r"(ses-\d+)", str(session_name))
    return m.group(1) if m else str(session_name)