from copy import deepcopy

def fix_grid_maze_cohort_02_problems(subjects_trials_by_problem, fix_subjects=None, start_problem=2, in_place=False):
    """
    For specified subjects:
      - problem start_problem gets replaced by start_problem+1
      - problem start_problem+1 gets replaced by start_problem+2
      - ...
      - last problem for that subject is removed

    Example (start_problem=2):
      P2 <- P3, P3 <- P4, ..., Plast removed

    Parameters
    ----------
    subjects_trials_by_problem : dict
        problems[problem][subject][session] = session_data
    fix_subjects : list[str]
        subjects to fix (default: ["MY_04_R", "MY_05_R"])
    start_problem : int
        first problem to shift (default 2)
    in_place : bool
        if False, returns a deep-copied new dict; if True, mutates original

    Returns
    -------
    dict
        fixed structure (same type as input keys/values)
    """
    if fix_subjects is None:
        fix_subjects = ["MY_04_R", "MY_05_R"]

    problems = subjects_trials_by_problem if in_place else deepcopy(subjects_trials_by_problem)

    # sort problem keys numerically if possible
    prob_keys = sorted(problems.keys())

    for subj in fix_subjects:
        # find which problem numbers this subject actually exists in
        subj_probs = [p for p in prob_keys if subj in problems.get(p, {})]
        if not subj_probs:
            continue

        # only shift starting at start_problem
        shift_from = [p for p in subj_probs if p >= start_problem]
        if len(shift_from) <= 1:
            # nothing to shift (need at least start_problem and a "next" one)
            continue

        shift_from_sorted = sorted(shift_from)

        # perform: p <- next_p for all but the last
        for i in range(len(shift_from_sorted) - 1):
            p = shift_from_sorted[i]
            p_next = shift_from_sorted[i + 1]
            problems.setdefault(p, {})
            problems[p][subj] = problems[p_next][subj]

        # delete subject from last problem
        last_p = shift_from_sorted[-1]
        if subj in problems.get(last_p, {}):
            del problems[last_p][subj]
            # optional cleanup: remove empty problem dict
            if isinstance(problems.get(last_p, None), dict) and len(problems[last_p]) == 0:
                del problems[last_p]

    return problems