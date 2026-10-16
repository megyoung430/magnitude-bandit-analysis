"""Utilities for filtering subject/session trial data by session or subject."""


def filter_trials_by_session(subjects_trials, sessions_to_keep):
    """Return a copy of *subjects_trials* containing only specified sessions.

    Args:
        subjects_trials (dict): Nested dict ``{subject: {session_key: session_data}}``.
        sessions_to_keep (Iterable[str]): Session-key strings to retain.

    Returns:
        dict: A new ``{subject: {session_key: session_data}}`` dict with the same subject keys
        (a subject with no matching session maps to ``{}``) but only the sessions whose
        keys appear in *sessions_to_keep*. Session data are not copied.
    """
    sessions_to_keep = set(sessions_to_keep)
    return {
        subj: {
            session: trials
            for session, trials in sessions.items()
            if session in sessions_to_keep
        }
        for subj, sessions in subjects_trials.items()
    }