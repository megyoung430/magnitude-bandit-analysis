"""A tiny imported cohort for the runner tests: one blocked problem, then two drifting ones.

Every session carries raw pyControl-like frames (so the poke analysis really parses them) and
the importer's trial variables, task name and overall problem number. Each problem has its own
choice ports, so the red outlines of "the previous problem" can be told apart.
"""

from src.behavior_import.extract_trials import extract_session_trials
from tests.poke_fixtures import Recording, full_trial, session_record

BLOCKED_NAME = "Meg\\3x3_field_blocked_value_bandit"
DRIFTING_NAME = "Meg\\3x3_field_drifting_value_bandit"

# overall problem -> (task name, choice ports); B2 is the initiation port
PROBLEMS = {
    1: (BLOCKED_NAME, ["A1", "B1", "C1"]),
    2: (DRIFTING_NAME, ["A3", "B3", "C3"]),
    3: (DRIFTING_NAME, ["A1", "B3", "C2"]),
}
MICE = ("m1", "m2")


def recording_frame(towers, n_trials=4):
    rec = Recording(towers=towers)
    for i in range(n_trials):
        full_trial(rec, choice=towers[i % 3], after_choice=("A2",) if i == 0 else ())
    return rec.to_frame()


def imported_session(problem, mouse, number):
    """One session of *mouse* in the overall *problem*, ``number`` days after the first."""
    name, towers = PROBLEMS[problem]
    session = session_record(
        [recording_frame(towers)], problem=problem, task_name=name, choice_towers=set(towers)
    )
    extract_session_trials(session, mouse, f"ses-{number}_date-202601{number:02d}")
    return session


def blocked_then_two_drifting():
    """``{overall problem: {mouse: {session_key: session}}}``, two sessions per mouse, problem."""
    cohort = {}
    for problem in PROBLEMS:
        cohort[problem] = {
            mouse: {
                f"ses-{n}_date-202601{n:02d}": imported_session(problem, mouse, n)
                for n in (2 * problem - 1, 2 * problem)
            }
            for mouse in MICE
        }
    return cohort
