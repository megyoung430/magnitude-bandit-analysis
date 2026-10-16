"""Counts of suspect good reversals beside the logged totals."""

import pandas as pd
import pytest

from src.behavior_analysis.best_change import best_change_problem
from src.behavior_analysis.get_total_reversals import get_total_reversals
from src.behavior_analysis.problem_summary import (
    compute_problem_goodrev_summary,
    compute_suspect_good_reversals_per_session,
)
from src.behavior_analysis.reversals_by_session import reversal_counts_by_session
from src.behavior_analysis.suspect_reversals import count_suspect_good_reversals


def rewards(best_per_trial):
    """4/1/0 magnitudes with the given best tower on each trial (others in alphabetical order)."""
    out = {t: [] for t in "ABC"}
    for best in best_per_trial:
        others = [t for t in "ABC" if t != best]
        for tower, value in zip([best, *others], (4, 1, 0), strict=True):
            out[tower].append(value)
    return out


def session(good, best_per_trial, **extra):
    return {
        "has_good": True,
        "good_reversals": good,
        "reward_magnitudes_by_tower": rewards(best_per_trial),
        **extra,
    }


# --- the rule ---


@pytest.mark.parametrize(
    "before, after, expected",
    [
        ({"A": 4, "B": 1, "C": 0}, {"A": 0, "B": 4, "C": 1}, None),
        ({"A": 4, "B": 1, "C": 0}, {"A": 4, "B": 0, "C": 1}, "best tower unchanged"),
        ({"A": 4, "B": 1, "C": 0}, {"A": 4, "B": 4, "C": 1}, "tie for best"),
        ({"A": 4, "B": 1, "C": 0}, {"A": 4, "B": None, "C": 1}, "missing magnitude"),
        ({"A": 4, "B": 1, "C": 0}, {"A": None, "B": None, "C": None}, "missing magnitude"),
    ],
)
def test_best_change_problem(before, after, expected):
    assert best_change_problem(before, after) == expected


# --- one session ---


def test_increments_where_the_best_tower_is_unchanged_are_suspect():
    # Counter rises at trials 2 (A to B: fine) and 4 (B to B: suspect).
    sess = session([0, 0, 1, 1, 2, 2], "AABBBB")
    assert count_suspect_good_reversals(sess) == {"suspect": 1, "uncheckable": 0}


def test_a_jump_of_two_counts_both_reversals():
    sess = session([0, 0, 2, 2], "AAAA")
    assert count_suspect_good_reversals(sess) == {"suspect": 2, "uncheckable": 0}


def test_missing_magnitude_at_an_increment_is_uncheckable_not_suspect():
    sess = session([0, 0, 1, 1], "AABB")
    sess["reward_magnitudes_by_tower"]["C"][2] = None
    assert count_suspect_good_reversals(sess) == {"suspect": 0, "uncheckable": 1}


def test_a_between_run_event_is_checked_like_any_other():
    # The counter jumps at trial 3 because a reversal happened while nothing was recording;
    # the recordings form one session, so trials 2 and 3 are compared and the unchanged best
    # tower makes it suspect.
    events = [dict(kind="good", index=3, count=1, source="between_runs")]
    sess = session([0, 0, 0, 1], "AAAA", reversal_boundary_events=events)
    assert count_suspect_good_reversals(sess) == {"suspect": 1, "uncheckable": 0}


def test_a_between_run_event_that_changes_the_best_tower_is_fine():
    events = [dict(kind="good", index=2, count=1, source="between_runs")]
    sess = session([0, 0, 1, 1], "AABB", reversal_boundary_events=events)
    assert count_suspect_good_reversals(sess) == {"suspect": 0, "uncheckable": 0}


def test_between_run_event_and_a_reversal_on_the_same_trial_are_both_checked():
    events = [dict(kind="good", index=2, count=1, source="between_runs")]
    sess = session([0, 0, 2, 2], "AAAA", reversal_boundary_events=events)
    assert count_suspect_good_reversals(sess) == {"suspect": 2, "uncheckable": 0}


def test_terminal_event_of_an_earlier_run_is_checked_like_any_other():
    # Run 1 ends after trial 2 with one reversal after its last trial; the counter of run 2
    # steps at trial 2, the first trial after it: one reversal, checked once.
    events = [dict(kind="good", index=2, count=1, source="run_end")]
    sess = session([0, 0, 1, 1], "AAAA", reversal_boundary_events=events)
    assert count_suspect_good_reversals(sess) == {"suspect": 1, "uncheckable": 0}


def test_an_event_after_the_last_trial_with_no_check_has_no_trial_to_compare_with():
    events = [dict(kind="good", index=4, count=2, source="run_end")]
    sess = session([0, 0, 1, 1], "AABB", reversal_boundary_events=events)
    assert count_suspect_good_reversals(sess) == {"suspect": 0, "uncheckable": 2}


def test_an_event_after_the_last_trial_that_the_next_session_shows_needs_no_check():
    events = [dict(kind="good", index=4, count=1, source="run_end", session_check="consistent")]
    sess = session([0, 0, 1, 1], "AABB", reversal_boundary_events=events)
    assert count_suspect_good_reversals(sess) == {"suspect": 0, "uncheckable": 0}


def test_an_event_after_the_last_trial_that_the_next_session_does_not_show_is_uncheckable():
    events = [dict(kind="good", index=4, count=1, source="run_end", session_check="inconsistent")]
    sess = session([0, 0, 1, 1], "AABB", reversal_boundary_events=events)
    assert count_suspect_good_reversals(sess) == {"suspect": 0, "uncheckable": 1}


@pytest.mark.parametrize(
    "check, expected",
    [
        ("consistent", {"suspect": 0, "uncheckable": 0}),
        ("inconsistent", {"suspect": 1, "uncheckable": 0}),
        (None, {"suspect": 0, "uncheckable": 1}),
    ],
)
def test_an_event_in_the_first_run_start_is_judged_by_its_check_against_the_previous_session(
    check, expected
):
    event = dict(kind="good", index=0, count=1, source="between_runs")
    if check:
        event["session_check"] = check
    sess = session([1, 1, 1, 1], "AAAA", reversal_boundary_events=[event])
    assert count_suspect_good_reversals(sess) == expected


def test_bad_events_are_not_counted():
    events = [dict(kind="bad", index=4, count=2, source="run_end")]
    sess = session([0, 0, 1, 1], "AABB", reversal_boundary_events=events)
    assert count_suspect_good_reversals(sess) == {"suspect": 0, "uncheckable": 0}


def test_session_without_reward_magnitudes_cannot_be_checked():
    sess = session([0, 0, 1, 1], "AABB")
    sess["reward_magnitudes_by_tower"] = {}
    assert count_suspect_good_reversals(sess) == {"suspect": 0, "uncheckable": 1}


def test_session_without_a_good_counter_has_none():
    assert count_suspect_good_reversals({"blocks": [1, 1, 2]}) == {"suspect": 0, "uncheckable": 0}


# --- beside the logged totals ---


def test_logged_totals_are_unchanged_and_the_counts_sit_beside_them():
    first = session([0, 0, 1, 1, 2, 2], "AABBBB", reversal_totals={"good": 2, "bad": 0})
    second = session([0, 1, 1], "ABB", reversal_totals={"good": 1, "bad": 0})
    totals = get_total_reversals({"ses-1": first, "ses-2": second})
    assert totals["good_reversals"] == 3 and totals["total_reversals"] == 3
    assert totals["suspect_good_reversals"] == 1
    assert totals["uncheckable_good_reversals"] == 0


def test_totals_without_counters_have_no_suspect_fields():
    assert get_total_reversals({"ses-1": {"blocks": [1, 1, 2]}}) == {"total_reversals": 1}


def test_reversal_counts_by_session_has_the_new_columns():
    sess = session([0, 0, 1, 1, 2, 2], "AABBBB", reversal_totals={"good": 2, "bad": 0})
    counts = reversal_counts_by_session({"m": {"ses-1_date-20260101": sess}})
    assert isinstance(counts, pd.DataFrame)
    row = counts.iloc[0]
    assert (row["good"], row["suspect_good_reversals"], row["uncheckable_good_reversals"]) == (
        2,
        1,
        0,
    )


def test_problem_summary_averages_over_sessions():
    one = session([0, 0, 1, 1, 2, 2], "AABBBB", reversal_totals={"good": 2, "bad": 0})
    two = session([0, 1, 1], "ABB", reversal_totals={"good": 1, "bad": 0})
    out = compute_suspect_good_reversals_per_session({"m": {"ses-1": one, "ses-2": two}})
    assert out == {"m": {"suspect": 0.5, "uncheckable": 0.0}}


def test_problem_summary_reports_suspect_counts_beside_the_logged_ones():
    logged = {1: {"a": 2.0, "b": 4.0}, 2: {"a": 3.0}}
    suspect = {
        1: {"a": {"suspect": 1.0, "uncheckable": 0.0}, "b": {"suspect": 0.0, "uncheckable": 2.0}},
        2: {"a": {"suspect": 0.5, "uncheckable": 0.0}},
    }
    summary = compute_problem_goodrev_summary(logged, suspect)
    assert summary["means"] == compute_problem_goodrev_summary(logged)["means"]  # unchanged
    assert summary["suspect"]["means"] == pytest.approx([0.5, 0.5])
    assert summary["uncheckable"]["means"] == pytest.approx([1.0, 0.0])
    assert summary["suspect"]["blocks"] == [1, 2]


def test_problem_summary_without_suspect_counts_has_no_suspect_keys():
    assert set(compute_problem_goodrev_summary({1: {"a": 2.0}})) == {
        "blocks",
        "means",
        "ses",
        "ns",
        "per_mouse_blocklens",
        "mice",
    }
