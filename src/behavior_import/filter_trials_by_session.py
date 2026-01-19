def filter_trials_by_session(subjects_trials, sessions_to_keep):
    sessions_to_keep = set(sessions_to_keep)
    return {
        subj: {
            session: trials
            for session, trials in sessions.items()
            if session in sessions_to_keep
        }
        for subj, sessions in subjects_trials.items()
    }