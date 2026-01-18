def get_total_reversals(subject_sessions):
    """Get the number of good, bad, and total reversals for a subject across all sessions."""
    all_sessions = list(subject_sessions.keys())
    total_reversals = 0
    good_reversals = 0
    bad_reversals = 0
    for current_session in all_sessions:
        if "good_reversals" in subject_sessions[current_session] and "bad_reversals" in subject_sessions[current_session]:
            good_reversals += subject_sessions[current_session]["good_reversals"][-1]
            bad_reversals += subject_sessions[current_session]["bad_reversals"][-1]
    total_reversals = good_reversals + bad_reversals
    return {"total_reversals": total_reversals, "good_reversals": good_reversals, "bad_reversals": bad_reversals}