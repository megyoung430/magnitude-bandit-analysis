"""Sessions whose starting magnitudes differ from the previous session's ending magnitudes."""

from src.behavior_analysis import starting_magnitudes as sm
from src.behavior_analysis.starting_magnitudes import COLUMNS, starting_magnitudes_table
from src.behavior_import import import_log
from src.behavior_import.import_log import SkipRecord
from src.behavior_import.run_boundaries import describe_boundary
from tests.review_fixtures import BLOCKED_NAME, DRIFTING_NAME, cohort, recording, session

S1, S2, S3 = "ses-1_date-20260101", "ses-2_date-20260102", "ses-4_date-20260104"
ZERO_B3 = {"B1": 4, "B3": 0, "C2": 2}
ZERO_B3_DRIFTED = {"B1": 3, "B3": 0, "C2": 2}
ZERO_B3_JUMPED = {"B1": 1, "B3": 0, "C2": 2}
ZERO_B1 = {"B1": 0, "B3": 4, "C2": 3}
ALL_TWO = {"B1": 2, "B3": 2, "C2": 2}
B1_BEST = {"B1": 4, "B3": 1, "C2": 0}
B3_BEST = {"B1": 0, "B3": 4, "C2": 1}
SECOND_SWAPPED = {"B1": 4, "B3": 0, "C2": 1}


def side(magnitudes, **totals):
    return {"magnitudes": magnitudes, "totals": totals, "counters": {}}


def boundary(before, after, mode="drifting", **totals):
    """``describe_boundary`` with the totals ``block=n`` etc. logged on both sides."""
    return describe_boundary(side(before, **dict.fromkeys(totals, 0)), side(after, **totals), mode)


def two_sessions(first_boundary, task=DRIFTING_NAME, later=S2):
    first = session(task_name=task, boundaries=[None])
    second = session(task_name=task, boundaries=[first_boundary])
    return cohort({1: {"M1": {S1: first, later: second}}})


def table_for(first_boundary, skips=(), **kw):
    _, index, grouped, _ = two_sessions(first_boundary, **kw)
    return starting_magnitudes_table(grouped, list(skips), index)


def only_row(table):
    (row,) = table.to_dict("records")
    return row


def test_identical_magnitudes_are_not_listed_and_the_first_session_has_no_row():
    assert table_for(boundary(ZERO_B3, ZERO_B3)).empty


def test_a_block_change_logged_between_sessions_and_shown_by_the_magnitudes():
    row = only_row(table_for(boundary(ZERO_B3, ZERO_B1, block=1)))
    assert row["likely_cause"] == sm.LOGGED_AND_CONSISTENT
    assert (row["mouse"], row["session"], row["previous_session"]) == ("M1", S2, S1)
    assert row["magnitudes_end_previous"] == "B1=4, B3=0, C2=2"
    assert row["magnitudes_start"] == "B1=0, B3=4, C2=3"
    assert row["towers_differing"] == "B1;B3;C2"
    assert (row["task_type"], row["problem_in_type"], row["overall_problem"]) == ("drifting", 1, 1)
    assert row["verdict"] == "consistent"
    assert row["block_total_change"] == 1


def test_a_good_reversal_logged_in_the_blocked_task_is_consistent_when_the_best_changes():
    row = only_row(
        table_for(boundary(B1_BEST, B3_BEST, mode="blocked", good=1, bad=0), task=BLOCKED_NAME)
    )
    assert row["likely_cause"] == sm.LOGGED_AND_CONSISTENT
    assert (row["good_total_change"], row["bad_total_change"]) == (1, 0)


def test_a_logged_change_the_magnitudes_do_not_show_is_inconsistent():
    row = only_row(
        table_for(boundary(B1_BEST, SECOND_SWAPPED, mode="blocked", good=1), task=BLOCKED_NAME)
    )
    assert row["likely_cause"] == sm.LOGGED_BUT_INCONSISTENT
    assert row["verdict"] == "inconsistent"


def test_a_logged_change_with_unmoved_drifting_zero_arm_is_inconsistent():
    row = only_row(table_for(boundary(ZERO_B3, ZERO_B3_DRIFTED, block=1)))
    assert row["likely_cause"] == sm.LOGGED_BUT_INCONSISTENT


def test_one_step_of_drift_with_the_zero_arm_unchanged_is_normal_in_the_drifting_task():
    row = only_row(table_for(boundary(ZERO_B3, ZERO_B3_DRIFTED, block=0)))
    assert row["likely_cause"] == sm.DRIFT


def test_a_jump_beyond_one_trial_of_drift_is_a_change_with_no_logged_change():
    row = only_row(table_for(boundary(ZERO_B3, ZERO_B3_JUMPED, block=0)))
    assert row["likely_cause"] == sm.NO_LOGGED_CHANGE


def test_a_moved_zero_arm_with_nothing_logged_is_a_change_with_no_logged_change():
    row = only_row(table_for(boundary(ZERO_B3, ZERO_B1, block=0)))
    assert row["likely_cause"] == sm.NO_LOGGED_CHANGE


def test_blocked_magnitudes_that_differ_without_a_logged_change_are_listed():
    row = only_row(
        table_for(boundary(B1_BEST, B3_BEST, mode="blocked", good=0, bad=0), task=BLOCKED_NAME)
    )
    assert row["likely_cause"] == sm.NO_LOGGED_CHANGE


def test_magnitudes_back_at_the_initial_values_are_a_reset():
    row = only_row(table_for(boundary(ZERO_B3, ALL_TWO, block=0)))
    assert row["likely_cause"] == sm.RESET
    assert "all 2" in sm.RESET


def test_a_logged_consistent_change_wins_over_a_reset():
    row = only_row(table_for(boundary(ZERO_B3, ALL_TWO, block=1)))
    assert row["likely_cause"] == sm.LOGGED_AND_CONSISTENT


def test_a_session_without_trials_between_the_two_is_the_cause_when_nothing_else_fits():
    skips = [SkipRecord("M1", "ses-3_date-20260103", None, import_log.SESSION, "x", "no trials")]
    row = only_row(table_for(boundary(ZERO_B3, ZERO_B1, block=0), skips=skips, later=S3))
    assert row["likely_cause"] == sm.PREVIOUS_SESSION_NO_TRIALS


def test_a_skipped_session_elsewhere_or_of_another_mouse_is_not_the_cause():
    skips = [
        SkipRecord("M2", "ses-3_date-20260103", None, import_log.SESSION, "x", "r"),
        SkipRecord("M1", "ses-9_date-20260109", None, import_log.SESSION, "x", "r"),
        SkipRecord("M1", "ses-3_date-20260103", "f.tsv", import_log.RECORDING, "x", "r"),
    ]
    row = only_row(table_for(boundary(ZERO_B3, ZERO_B1, block=0), skips=skips, later=S3))
    assert row["likely_cause"] == sm.NO_LOGGED_CHANGE


def test_missing_magnitudes_cannot_be_compared_and_are_listed_as_such():
    row = only_row(table_for(boundary(None, ZERO_B1, block=0)))
    assert row["likely_cause"] == sm.CANNOT_COMPARE


def test_sessions_that_start_a_problem_are_left_out():
    first = session(boundaries=[boundary(ZERO_B3, ZERO_B1, block=1)])
    _, index, grouped, _ = cohort({1: {"M1": {S1: first}}})
    assert starting_magnitudes_table(grouped, [], index).empty


def test_the_first_recording_decides_not_the_later_ones():
    later_recording = boundary(ZERO_B3, ZERO_B1, block=1)
    second = session(
        recordings=[recording(), recording()],
        boundaries=[boundary(ZERO_B3, ZERO_B3), later_recording],
    )
    first = session()
    _, index, grouped, _ = cohort({1: {"M1": {S1: first, S2: second}}})
    assert starting_magnitudes_table(grouped, [], index).empty


def test_causes_are_listed_from_most_to_least_likely():
    assert sm.CAUSES == (
        sm.LOGGED_AND_CONSISTENT,
        sm.LOGGED_BUT_INCONSISTENT,
        sm.DRIFT,
        sm.RESET,
        sm.PREVIOUS_SESSION_NO_TRIALS,
        sm.NO_LOGGED_CHANGE,
        sm.CANNOT_COMPARE,
    )


def test_empty_cohort_gives_the_columns():
    _, index, grouped, _ = cohort({})
    table = starting_magnitudes_table(grouped, [], index)
    assert table.empty
    assert list(table.columns) == list(COLUMNS)


def test_a_later_session_the_importer_could_not_describe_is_listed_as_not_comparable():
    first = session()
    second = session(boundaries=[None])
    _, index, grouped, _ = cohort({1: {"M1": {S1: first, S2: second}}})
    (row,) = starting_magnitudes_table(grouped, [], index).to_dict("records")
    assert row["likely_cause"] == sm.CANNOT_COMPARE
    assert row["session"] == S2 and row["magnitudes_start"] is None
