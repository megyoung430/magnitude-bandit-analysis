import re

def sort_sessions_chronologically(session_name):
    """
    Sort sessions by numeric index in 'ses-XX' (so ses-01..ses-30).
    Falls back to the raw string if not found.
    """
    m = re.search(r"ses-(\d+)", str(session_name))
    return int(m.group(1)) if m else str(session_name)

def get_short_session_label(session_name):
    """
    Display-only label: extract 'ses-XX' from a session key.
    """
    m = re.search(r"(ses-\d+)", str(session_name))
    return m.group(1) if m else str(session_name)