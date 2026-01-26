from src.behavior_analysis.get_variables_across_sessions import *

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

def get_all_reversal_indices(subjects_trials):
    """
    Returns:
      all_good[subj] -> sorted list of good reversal indices
      all_bad[subj]  -> sorted list of bad reversal indices
      blocks[subj]   -> merged blocks list (for optional block logic)
    """
    merged, _ = get_vars_across_all_sessions(subjects_trials)

    all_good = {}
    all_bad = {}
    all_blocks = {}

    for subj, d in merged.items():
        good = d.get("good_reversals", []) or []
        bad  = d.get("bad_reversals", []) or []
        blocks = d.get("blocks", []) or []

        all_good[subj] = find_increment_indices(good)
        all_bad[subj]  = find_increment_indices(bad)
        all_blocks[subj] = blocks

    return all_good, all_bad, all_blocks

def find_increment_indices(cumulative_list):
    """Indices i where cumulative_list[i] > cumulative_list[i-1], ignoring None."""
    inc = []
    prev = None
    for i, v in enumerate(cumulative_list):
        if i == 0:
            prev = v
            continue
        if v is None or prev is None:
            prev = v
            continue
        if v > prev:
            inc.append(i)
        prev = v
    return inc