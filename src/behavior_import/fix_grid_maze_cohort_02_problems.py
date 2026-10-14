from copy import deepcopy
from itertools import pairwise

DEFAULT_FIX_SUBJECTS = ("MY_04_R", "MY_05_R")


def _problems_to_shift(problems, prob_keys, subject, start_problem):
    """Problems at or after *start_problem* in which *subject* has data, ascending.

    *prob_keys* is already sorted, so the result is too.
    """
    # problems.get(p, {}): a problem removed while fixing an earlier subject has no data left
    with_subject = [p for p in prob_keys if subject in problems.get(p, {})]
    return [p for p in with_subject if p >= start_problem]


def _shift_subject_down(problems, subject, shift_from):
    """Move *subject*'s data one problem earlier and drop the last problem.

    Each problem in *shift_from* takes the data of the next one; the last problem
    then loses the subject, and is removed from *problems* if no subject is left.
    """
    for problem, next_problem in pairwise(shift_from):
        problems[problem][subject] = problems[next_problem][subject]

    last_problem = shift_from[-1]
    del problems[last_problem][subject]
    if isinstance(problems[last_problem], dict) and not problems[last_problem]:
        del problems[last_problem]


def fix_grid_maze_cohort_02_problems(
    subjects_trials_by_problem, fix_subjects=None, start_problem=2, in_place=False
):
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
    # Known issue: a bare string for fix_subjects is iterated character by character,
    # so it silently fixes nothing instead of raising.
    if fix_subjects is None:
        fix_subjects = DEFAULT_FIX_SUBJECTS

    problems = subjects_trials_by_problem if in_place else deepcopy(subjects_trials_by_problem)
    prob_keys = sorted(problems.keys())  # raises TypeError for mixed key types

    for subject in fix_subjects:
        shift_from = _problems_to_shift(problems, prob_keys, subject, start_problem)
        # need the start problem and a later one to take its data
        if len(shift_from) > 1:
            _shift_subject_down(problems, subject, shift_from)

    return problems
