"""Utility for filtering subject/session trial data to a subset of subjects."""


def filter_trials_by_subject(subjects_trials, subjects_to_keep):
    """Return a copy of *subjects_trials* containing only specified subjects.

    Args:
        subjects_trials: Dict ``{subject: subject_data}`` where *subject_data*
            can be any value (e.g. a session dict).
        subjects_to_keep: Iterable of subject-key strings to retain.

    Returns:
        A new dict with only the subjects whose keys appear in
        *subjects_to_keep*.
    """
    subjects_to_keep = set(subjects_to_keep)
    return {
        subj: trials
        for subj, trials in subjects_trials.items()
        if subj in subjects_to_keep
    }