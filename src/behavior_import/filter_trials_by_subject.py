def filter_trials_by_subject(subjects_trials, subjects_to_keep):
    subjects_to_keep = set(subjects_to_keep)
    return {
        subj: trials
        for subj, trials in subjects_trials.items()
        if subj in subjects_to_keep
    }