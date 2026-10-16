"""Sessions whose starting magnitudes differ from the previous session's ending magnitudes."""

import pandas as pd

from src.behavior_analysis import starting_magnitudes as sm
from src.behavior_analysis.starting_magnitudes import COLUMNS, starting_magnitudes_table
from src.behavior_import import import_log
from src.behavior_import.import_log import SkipRecord
from src.behavior_import.run_boundaries import describe_boundary
from tests.review_fixtures import (
    BLOCKED_NAME,
    DRIFTING_NAME,
    cohort,
    edge_sides,
    recording,
    session,
    side,
)

S1, S2, S3 = "ses-1_date-20260101", "ses-2_date-20260102", "ses-4_date-20260104"
ZERO_B3 = {"B1": 4, "B3": 0, "C2": 2}
ZERO_B3_DRIFTED = {"B1": 3, "B3": 0, "C2": 2}
ZERO_B3_TWO_STEPS = {"B1": 2, "B3": 0, "C2": 2}
ZERO_B3_JUMPED = {"B1": 1, "B3": 0, "C2": 2}
ZERO_B1 = {"B1": 0, "B3": 4, "C2": 3}
ALL_TWO = {"B1": 2, "B3": 2, "C2": 2}
B1_BEST = {"B1": 4, "B3": 1, "C2": 0}
B3_BEST = {"B1": 0, "B3": 4, "C2": 1}
SECOND_SWAPPED = {"B1": 4, "B3": 0, "C2": 1}


def boundary(before, after, mode="drifting", **totals):
    """``describe_boundary`` with the totals ``block=n`` etc. logged on both sides."""
    return describe_boundary(side(before, **dict.fromkeys(totals, 0)), side(after, **totals), mode)


def two_sessions(before, after, *, task=DRIFTING_NAME, later=S2, totals_before=None, **totals):
    """Two sessions of a problem; the first ends on *before*, the second starts on *after*.

    The session totals are ``totals`` at the second session's start and ``totals_before`` (default
    0 for each kind) at the first's end.
    """
    totals_before = dict.fromkeys(totals, 0) if totals_before is None else totals_before
    first = session(
        task_name=task,
        edges=edge_sides(before, before, first_totals=totals_before, last_totals=totals_before),
    )
    second = session(
        task_name=task,
        edges=edge_sides(after, after, first_totals=totals, last_totals=totals),
    )
    return cohort({1: {"M1": {S1: first, later: second}}})


def table_for(before, after, skips=(), **kw):
    _, index, grouped, _ = two_sessions(before, after, **kw)
    return starting_magnitudes_table(grouped, list(skips), index)


def only_row(table):
    (row,) = table.to_dict("records")
    return row


def cause(before, after, *, gap=False, mode="drifting", **totals):
    return sm.starting_magnitude_cause(
        boundary(before, after, mode, **totals), gap_without_trials=gap
    )


def test_identical_magnitudes_are_not_listed_and_the_first_session_has_no_row():
    assert table_for(ZERO_B3, ZERO_B3, block=0).empty


def test_a_block_change_logged_between_sessions_and_shown_by_the_magnitudes():
    row = only_row(table_for(ZERO_B3, ZERO_B1, block=1))
    assert row["likely_cause"] == sm.LOGGED_AND_CONSISTENT
    assert (row["mouse"], row["session"], row["previous_session"]) == ("M1", S2, S1)
    assert row["magnitudes_end_previous"] == "B1=4, B3=0, C2=2"
    assert row["magnitudes_start"] == "B1=0, B3=4, C2=3"
    assert row["towers_differing"] == "B1;B3;C2"
    assert (row["task_type"], row["problem_in_type"], row["overall_problem"]) == ("drifting", 1, 1)
    assert row["verdict"] == "consistent"
    assert row["block_total_change"] == 1


def test_a_good_reversal_logged_in_the_blocked_task_is_consistent_when_the_best_changes():
    row = only_row(table_for(B1_BEST, B3_BEST, task=BLOCKED_NAME, good=1, bad=0))
    assert row["likely_cause"] == sm.LOGGED_AND_CONSISTENT
    assert (row["good_total_change"], row["bad_total_change"]) == (1, 0)


def test_a_logged_change_the_magnitudes_do_not_show_is_inconsistent():
    row = only_row(table_for(B1_BEST, SECOND_SWAPPED, task=BLOCKED_NAME, good=1))
    assert row["likely_cause"] == sm.LOGGED_BUT_INCONSISTENT
    assert row["verdict"] == "inconsistent"


def test_a_logged_change_with_unmoved_drifting_zero_arm_is_inconsistent():
    row = only_row(table_for(ZERO_B3, ZERO_B3_DRIFTED, block=1))
    assert row["likely_cause"] == sm.LOGGED_BUT_INCONSISTENT


def test_one_step_of_drift_with_the_zero_arm_unchanged_is_normal_in_the_drifting_task():
    row = only_row(table_for(ZERO_B3, ZERO_B3_DRIFTED, block=0))
    assert row["likely_cause"] == sm.DRIFT


def test_a_step_of_exactly_the_drift_limit_is_still_drift_and_one_more_is_not():
    assert sm.MAX_DRIFT_STEP == 2
    assert cause(ZERO_B3, ZERO_B3_TWO_STEPS, block=0) == sm.DRIFT
    assert cause(ZERO_B3, ZERO_B3_JUMPED, block=0) == sm.NO_LOGGED_CHANGE


def test_a_jump_beyond_one_trial_of_drift_is_a_change_with_no_logged_change():
    row = only_row(table_for(ZERO_B3, ZERO_B3_JUMPED, block=0))
    assert row["likely_cause"] == sm.NO_LOGGED_CHANGE


def test_a_moved_zero_arm_with_nothing_logged_is_a_change_with_no_logged_change():
    row = only_row(table_for(ZERO_B3, ZERO_B1, block=0))
    assert row["likely_cause"] == sm.NO_LOGGED_CHANGE


def test_blocked_magnitudes_that_differ_without_a_logged_change_are_listed():
    row = only_row(table_for(B1_BEST, B3_BEST, task=BLOCKED_NAME, good=0, bad=0))
    assert row["likely_cause"] == sm.NO_LOGGED_CHANGE


def test_magnitudes_back_at_the_initial_values_are_a_reset():
    row = only_row(table_for(ZERO_B3, ALL_TWO, block=0))
    assert row["likely_cause"] == sm.RESET
    assert "all 2" in sm.RESET


def test_a_reset_needs_the_previous_session_not_to_have_ended_on_the_initial_values():
    assert cause(ALL_TWO, ALL_TWO, mode="blocked", good=0, bad=0) != sm.RESET


def test_a_logged_consistent_change_wins_over_a_reset():
    row = only_row(table_for(ZERO_B3, ALL_TWO, block=1))
    assert row["likely_cause"] == sm.LOGGED_AND_CONSISTENT


def test_a_task_restart_with_totals_back_at_zero_is_a_reset_in_the_drifting_task():
    row = only_row(table_for(ZERO_B3, ALL_TWO, totals_before={"block": 7}, block=0))
    assert row["likely_cause"] == sm.RESET
    assert row["block_total_change"] == -7


def test_a_task_restart_with_totals_back_at_zero_is_a_reset_in_the_blocked_task():
    row = only_row(
        table_for(
            B1_BEST,
            ALL_TWO,
            task=BLOCKED_NAME,
            totals_before={"good": 3, "bad": 2},
            good=0,
            bad=0,
        )
    )
    assert row["likely_cause"] == sm.RESET


def test_a_total_that_went_down_without_a_restart_is_a_logged_but_inconsistent_change():
    row = only_row(table_for(ZERO_B3, ZERO_B1, totals_before={"block": 7}, block=3))
    assert row["likely_cause"] == sm.LOGGED_BUT_INCONSISTENT
    assert row["verdict"] == "inconsistent"


def test_a_session_without_trials_between_the_two_is_the_cause_when_nothing_else_fits():
    skips = [SkipRecord("M1", "ses-3_date-20260103", None, import_log.SESSION, "x", "no trials")]
    row = only_row(table_for(ZERO_B3, ZERO_B1, skips=skips, later=S3, block=0))
    assert row["likely_cause"] == sm.PREVIOUS_SESSION_NO_TRIALS


def test_a_skipped_session_elsewhere_or_of_another_mouse_is_not_the_cause():
    skips = [
        SkipRecord("M2", "ses-3_date-20260103", None, import_log.SESSION, "x", "r"),
        SkipRecord("M1", "ses-9_date-20260109", None, import_log.SESSION, "x", "r"),
        SkipRecord("M1", "ses-3_date-20260103", "f.tsv", import_log.RECORDING, "x", "r"),
    ]
    row = only_row(table_for(ZERO_B3, ZERO_B1, skips=skips, later=S3, block=0))
    assert row["likely_cause"] == sm.NO_LOGGED_CHANGE


def test_missing_magnitudes_cannot_be_compared_and_are_listed_as_such():
    row = only_row(table_for(None, ZERO_B1, block=0))
    assert row["likely_cause"] == sm.CANNOT_COMPARE


def test_one_missing_tower_on_either_side_cannot_be_compared():
    one_missing = {"B1": 4, "B3": None, "C2": 2}
    for before, after in ((one_missing, ZERO_B1), (ZERO_B1, one_missing), (one_missing, ALL_TWO)):
        row = only_row(table_for(before, after, block=1))
        assert row["likely_cause"] == sm.CANNOT_COMPARE, (before, after)
        assert row["verdict"] == "unverifiable"
    row = only_row(table_for(one_missing, ZERO_B1, block=1))
    assert "B3=missing" in row["magnitudes_end_previous"]


def test_sides_with_different_towers_cannot_be_compared():
    other_towers = {"B1": 4, "B3": 0, "A1": 2}
    assert cause(ZERO_B3, other_towers, block=1) == sm.CANNOT_COMPARE


def test_a_moved_zero_arm_with_no_totals_logged_is_a_change_inferred_from_the_magnitudes():
    row = only_row(table_for(ZERO_B3, ZERO_B1))
    assert row["likely_cause"] == sm.CHANGE_FROM_MAGNITUDES
    assert "no totals logged" in sm.CHANGE_FROM_MAGNITUDES
    assert row["verdict"] == "consistent"
    assert pd.isna(row["block_total_change"])


def test_blocked_magnitudes_that_differ_with_no_totals_logged_are_a_change_inferred():
    assert cause(B1_BEST, B3_BEST, mode="blocked") == sm.CHANGE_FROM_MAGNITUDES


def test_no_totals_and_a_return_to_the_initial_values_is_a_reset():
    assert cause(ZERO_B3, ALL_TWO) == sm.RESET


def test_no_totals_and_a_drift_beyond_one_trial_is_listed_without_a_logged_change():
    assert cause(ZERO_B3, ZERO_B3_JUMPED) == sm.NO_LOGGED_CHANGE


def test_the_cause_and_the_verdict_never_contradict_each_other():
    cases = [
        (ZERO_B3, ZERO_B1, {}),
        (ZERO_B3, ZERO_B1, {"block": 0}),
        (ZERO_B3, ZERO_B1, {"block": 1}),
        (ZERO_B3, ZERO_B3_DRIFTED, {"block": 1}),
        (ZERO_B3, ZERO_B3_JUMPED, {}),
    ]
    for before, after, totals in cases:
        found = boundary(before, after, **totals)
        text = sm.starting_magnitude_cause(found, gap_without_trials=False)
        if found["source"] == "magnitudes" and found["verdict"] == "consistent":
            assert text != sm.NO_LOGGED_CHANGE, (before, after, totals)


def test_sessions_that_start_a_problem_are_left_out():
    only = session()
    _, index, grouped, _ = cohort({1: {"M1": {S1: only}}})
    assert starting_magnitudes_table(grouped, [], index).empty


def test_the_last_trial_of_the_previous_session_meets_the_first_trial_of_this_one():
    first = session(edges=edge_sides(ZERO_B3_DRIFTED, ZERO_B3, last_totals={"block": 0}))
    second = session(
        edges=edge_sides(ZERO_B1, ZERO_B3_DRIFTED, first_totals={"block": 1}),
    )
    _, index, grouped, _ = cohort({1: {"M1": {S1: first, S2: second}}})
    row = only_row(starting_magnitudes_table(grouped, [], index))
    assert row["magnitudes_end_previous"] == "B1=4, B3=0, C2=2"
    assert row["magnitudes_start"] == "B1=0, B3=4, C2=3"
    assert row["block_total_change"] == 1


def test_the_session_before_the_previous_one_is_not_compared():
    third = "ses-5_date-20260105"
    one = session(edges=edge_sides(ZERO_B1, ZERO_B1))
    two = session(edges=edge_sides(ZERO_B3, ZERO_B3))
    three = session(edges=edge_sides(ZERO_B3, ZERO_B3))
    _, index, grouped, _ = cohort({1: {"M1": {S1: one, S2: two, third: three}}})
    (row,) = starting_magnitudes_table(grouped, [], index).to_dict("records")
    assert (row["previous_session"], row["session"]) == (S1, S2)


def test_causes_are_listed_from_most_to_least_likely():
    assert sm.CAUSES == (
        sm.LOGGED_AND_CONSISTENT,
        sm.LOGGED_BUT_INCONSISTENT,
        sm.DRIFT,
        sm.RESET,
        sm.CHANGE_FROM_MAGNITUDES,
        sm.PREVIOUS_SESSION_NO_TRIALS,
        sm.NO_LOGGED_CHANGE,
        sm.CANNOT_COMPARE,
    )


def test_empty_cohort_gives_the_columns():
    _, index, grouped, _ = cohort({})
    table = starting_magnitudes_table(grouped, [], index)
    assert table.empty
    assert list(table.columns) == list(COLUMNS)


def test_a_session_whose_edges_the_importer_could_not_describe_is_listed_as_not_comparable():
    first = session()
    second = session()
    second["session_edges"] = None
    _, index, grouped, _ = cohort({1: {"M1": {S1: first, S2: second}}})
    (row,) = starting_magnitudes_table(grouped, [], index).to_dict("records")
    assert row["likely_cause"] == sm.CANNOT_COMPARE
    assert row["session"] == S2 and row["magnitudes_start"] is None


def test_the_file_is_the_first_recording_of_the_session():
    second = session(
        recordings=[recording(path="M1-a.tsv"), recording(path="M1-b.tsv")],
        edges=edge_sides(ZERO_B1, ZERO_B1, first_totals={"block": 1}),
    )
    first = session(edges=edge_sides(ZERO_B3, ZERO_B3, last_totals={"block": 0}))
    _, index, grouped, _ = cohort({1: {"M1": {S1: first, S2: second}}})
    assert only_row(starting_magnitudes_table(grouped, [], index))["file"] == "M1-a.tsv"
