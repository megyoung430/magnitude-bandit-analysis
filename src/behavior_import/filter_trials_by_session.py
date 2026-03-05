"""Utilities for filtering subject/session trial data by session or subject."""


def filter_trials_by_session(subjects_trials, sessions_to_keep):
    """Return a copy of *subjects_trials* containing only specified sessions.

    Args:
        subjects_trials: Nested dict ``{subject: {session_key: session_data}}``.
        sessions_to_keep: Iterable of session-key strings to retain.

    Returns:
        A new dict with the same subject keys but only the sessions whose
        keys appear in *sessions_to_keep*.
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