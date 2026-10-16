"""Block changes read from the magnitudes, and changes logged at the end of a session.

Covers the cases where the magnitudes decide (files without session totals), the check of a change
logged after a session's last trial against the next session that has trials, two changes at one
boundary, crashed recordings and the task-type resolution. Every test builds raw recordings and
runs the import.
"""

import copy
from collections import Counter

import pytest

from src.behavior_analysis.get_total_reversals import get_total_reversals
from src.behavior_analysis.get_variables_across_sessions import get_vars_across_all_sessions
from src.behavior_analysis.suspect_reversals import count_suspect_good_reversals, good_event_counts
from src.behavior_import.extract_trials import extract_trials
from src.behavior_import.problem_numbering import known_task_type
from src.behavior_import.reconcile_reversals import uncarried_counts
from src.behavior_import.session_order import ordered_session_keys
from tests.boundary_fixtures import (
    B1_BEST,
    B3_BEST,
    BLOCKED_TASK,
    DRIFTING_TASK,
    PORT_FINDING_MAGS,
    ZERO_B1,
    ZERO_B3,
    ZERO_B3_DRIFTED,
    ZERO_C2,
    port_finding_prints,
    port_finding_start,
    prints,
    recording,
    session,
    snapshot,
)

S1, S2, S3 = "ses-1_date-20260101", "ses-2_date-20260102", "ses-3_date-20260103"


def imported(sessions, task=DRIFTING_TASK):
    """``{key: session}`` after ``extract_trials``; *sessions* maps keys to lists of recordings."""
    data = {"m": {key: session(frames, task) for key, frames in sessions.items()}}
    return extract_trials(data)["m"]


def changes(values):
    return [i for i in range(1, len(values)) if values[i] != values[i - 1]]


def sources(diagnostics):
    return [d["source"] for d in diagnostics]


# --- the magnitude fallback compares trials, not snapshots ---

TASKS = {
    "drifting": (DRIFTING_TASK, ZERO_B3, {"blocks": 1}),
    "blocked": (BLOCKED_TASK, B1_BEST, {"blocks": 1, "good": 0, "bad": 0}),
}


def port_finding_recording(task, *, totals=None):
    """A recording that finds the ports, then runs 2 trials on the unchanged magnitudes."""
    _, mags, counters = TASKS[task]
    extra = {} if totals is None else totals
    trials = port_finding_prints(2) + prints(mags, 2, **counters)
    return recording(
        port_finding_start(**counters, **extra),
        trials,
        snapshot(mags, **counters, **extra),
    )


def plain_recording(task, *, totals=None):
    _, mags, counters = TASKS[task]
    extra = {} if totals is None else totals
    return recording(
        snapshot(mags, **counters, **extra),
        prints(mags, 3, **counters),
        snapshot(mags, **counters, **extra),
    )


@pytest.mark.parametrize("task", ["drifting", "blocked"])
@pytest.mark.parametrize("layout", ["same session", "next session"])
def test_a_port_finding_run_start_does_not_create_a_block_change(task, layout):
    frames = [plain_recording(task), port_finding_recording(task)]
    sessions = {S1: frames} if layout == "same session" else {S1: frames[:1], S2: frames[1:]}
    out = imported(sessions, TASKS[task][0])
    for sess in out.values():
        assert set(sess["blocks"]) == {1}
        assert sess["reversal_boundary_events"] == []
        assert sess["reversal_diagnostics"] == []


@pytest.mark.parametrize("task", ["drifting", "blocked"])
def test_a_port_finding_run_start_with_totals_logs_no_diagnostic(task):
    totals = {"tot_blocks": 1} | ({"tot_good": 0, "tot_bad": 0} if task == "blocked" else {})
    frames = [plain_recording(task, totals=totals), port_finding_recording(task, totals=totals)]
    sess = imported({S1: frames}, TASKS[task][0])[S1]
    assert sess["reversal_boundary_events"] == []
    assert sess["reversal_diagnostics"] == []


def test_a_block_change_is_still_read_across_a_port_finding_recording():
    # The trials around the port-finding recording show the zero arm moving.
    first = recording(snapshot(ZERO_B3, blocks=1), prints(ZERO_B3, 3, blocks=1), snapshot(ZERO_B3))
    finding = recording(
        port_finding_start(blocks=1), port_finding_prints(2), snapshot(PORT_FINDING_MAGS)
    )
    last = recording(snapshot(ZERO_B1, blocks=1), prints(ZERO_B1, 2, blocks=1), snapshot(ZERO_B1))
    sess = imported({S1: [first, finding, last]})[S1]
    assert sess["blocks"] == [1, 1, 1, 2, 2]
    assert [(e["source"], e["index"]) for e in sess["reversal_boundary_events"]] == [
        ("magnitudes", 3)
    ]


def test_a_change_logged_in_run_end_without_magnitudes_is_counted_once():
    first = recording(
        snapshot(ZERO_B3, blocks=1), prints(ZERO_B3, 3, blocks=1), snapshot(None, blocks=2)
    )
    second = recording(snapshot(ZERO_B1, blocks=1), prints(ZERO_B1, 2, blocks=1), snapshot(ZERO_B1))
    sess = imported({S1: [first, second]})[S1]
    assert sess["blocks"] == [1, 1, 1, 2, 2]
    assert [e["source"] for e in sess["reversal_boundary_events"]] == ["run_end"]


def test_a_rejected_session_end_change_does_not_come_back_at_the_next_session_start():
    ses_1 = recording(
        snapshot(ZERO_B3, blocks=1), prints(ZERO_B3, 3, blocks=1), snapshot(ZERO_B1, blocks=2)
    )
    ses_2 = recording(
        snapshot(ZERO_B3_DRIFTED, blocks=1),
        prints(ZERO_B3_DRIFTED, 2, blocks=1),
        snapshot(ZERO_B3_DRIFTED, blocks=1),
    )
    out = imported({S1: [ses_1], S2: [ses_2]})
    (event,) = out[S1]["reversal_boundary_events"]
    assert event["session_check"] == "inconsistent"
    assert out[S2]["reversal_boundary_events"] == []
    merged, _ = get_vars_across_all_sessions({"m": out})
    assert merged["m"]["blocks"] == [1] * 5


def test_totals_of_only_one_kind_still_stop_the_magnitude_fallback():
    # The good total is logged on both sides and shows no reversal: the counters are the record.
    def counters(total_good):
        return dict(blocks=1, good=0, bad=0, tot_good=total_good)

    first = recording(
        snapshot(B1_BEST, **counters(0)),
        prints(B1_BEST, 3, blocks=1, good=0, bad=0),
        snapshot(B1_BEST, **counters(0)),
    )
    second = recording(
        snapshot(B3_BEST, **counters(0)),
        prints(B3_BEST, 2, blocks=1, good=0, bad=0),
        snapshot(B3_BEST, **counters(0)),
    )
    sess = imported({S1: [first, second]}, BLOCKED_TASK)[S1]
    assert sess["reversal_boundary_events"] == []
    assert sess["blocks"] == [1] * 5
    assert sources(sess["reversal_diagnostics"]) == ["unclassified_magnitude_change"]


def test_drift_of_the_other_arms_with_totals_logged_is_not_flagged():
    first = recording(
        snapshot(ZERO_B3, blocks=1, tot_blocks=1),
        prints(ZERO_B3, 3, blocks=1),
        snapshot(ZERO_B3, blocks=1, tot_blocks=1),
    )
    second = recording(
        snapshot(ZERO_B3_DRIFTED, blocks=1, tot_blocks=1),
        prints(ZERO_B3_DRIFTED, 2, blocks=1),
        snapshot(ZERO_B3_DRIFTED, blocks=1, tot_blocks=1),
    )
    sess = imported({S1: [first, second]})[S1]
    assert sess["reversal_diagnostics"] == []


def test_a_moved_zero_arm_with_totals_logged_but_no_change_is_flagged():
    first = recording(
        snapshot(ZERO_B3, blocks=1, tot_blocks=1),
        prints(ZERO_B3, 3, blocks=1),
        snapshot(ZERO_B3, blocks=1, tot_blocks=1),
    )
    second = recording(
        snapshot(ZERO_B1, blocks=1, tot_blocks=1),
        prints(ZERO_B1, 2, blocks=1),
        snapshot(ZERO_B1, blocks=1, tot_blocks=1),
    )
    sess = imported({S1: [first, second]})[S1]
    assert sess["blocks"] == [1] * 5
    assert sources(sess["reversal_diagnostics"]) == ["unclassified_magnitude_change"]


# --- the first block number ---


def test_a_session_starts_from_the_block_number_of_its_first_run_start():
    rec = recording(
        snapshot(ZERO_B3, blocks=3, tot_blocks=3),
        prints(ZERO_B3, 2, blocks=3) + prints(ZERO_B1, 2, first=3, blocks=4),
        snapshot(ZERO_B1, blocks=4, tot_blocks=4),
    )
    assert imported({S1: [rec]})[S1]["blocks"] == [3, 3, 4, 4]


# --- drifting totals count the changes the merged blocks show ---


def three_sessions(*, second_start):
    """A change after session 1, one inside session 2 (starting at *second_start*), session 3."""
    ses_1 = recording(
        snapshot(ZERO_B3, blocks=1, tot_blocks=1),
        prints(ZERO_B3, 3, blocks=1),
        snapshot(ZERO_B1, blocks=2, tot_blocks=2),
    )
    ses_2 = recording(
        snapshot(second_start, blocks=1, tot_blocks=2),
        prints(second_start, 2, blocks=1) + prints(ZERO_C2, 2, first=3, blocks=2),
        snapshot(ZERO_C2, blocks=2, tot_blocks=3),
    )
    ses_3 = recording(
        snapshot(ZERO_C2, blocks=1, tot_blocks=3),
        prints(ZERO_C2, 2, blocks=1),
        snapshot(ZERO_C2, blocks=1, tot_blocks=3),
    )
    return {S1: [ses_1], S2: [ses_2], S3: [ses_3]}


def test_drifting_total_counts_the_changes_logged_at_the_end_of_a_session():
    out = imported(three_sessions(second_start=ZERO_B1))
    merged, _ = get_vars_across_all_sessions({"m": out})
    assert changes(merged["m"]["blocks"]) == [3, 5]
    assert get_total_reversals(out) == {"total_reversals": 2}


def test_drifting_total_leaves_out_a_session_end_change_that_the_next_session_does_not_show():
    out = imported(three_sessions(second_start=ZERO_B3))
    merged, _ = get_vars_across_all_sessions({"m": out})
    assert changes(merged["m"]["blocks"]) == [5]
    assert get_total_reversals(out) == {"total_reversals": 1}


# --- a crashed recording with no trial ---


def crashed_between(task):
    """Recordings of 3 trials, then one that crashed with no trial and no run_end, then 2 trials.

    The change is logged in the third recording's ``run_start`` totals.
    """
    if task == "drifting":
        old, new = ZERO_B3, ZERO_B1
        keys = dict(blocks=1)
        totals = lambda n: dict(tot_blocks=n)  # noqa: E731
    else:
        old, new = B1_BEST, B3_BEST
        keys = dict(blocks=1, good=0, bad=0)
        totals = lambda n: dict(tot_blocks=n, tot_good=n - 1, tot_bad=0)  # noqa: E731
    first = recording(
        snapshot(old, **keys, **totals(1)),
        prints(old, 3, **keys),
        snapshot(old, **keys, **totals(1)),
    )
    crashed = recording(snapshot(old, **keys, **totals(1)), [])
    third = recording(
        snapshot(new, **keys, **totals(2)),
        prints(new, 2, **keys),
        snapshot(new, **keys, **totals(2)),
    )
    return [first, crashed, third]


def test_a_crashed_recording_with_no_trial_keeps_the_drifting_change_logged_across_it():
    sess = imported({S1: crashed_between("drifting")})[S1]
    assert sess["blocks"] == [1, 1, 1, 2, 2]
    assert [(e["kind"], e["index"]) for e in sess["reversal_boundary_events"]] == [("block", 3)]


def test_a_crashed_recording_with_no_trial_keeps_the_good_reversal_logged_across_it():
    sess = imported({S1: crashed_between("blocked")}, BLOCKED_TASK)[S1]
    assert sess["good_reversals"] == [0, 0, 0, 1, 1]
    assert sess["reversal_totals"] == {"good": 1, "bad": 0}


# --- the check of a session-end change waits for a session with trials ---


def end_change_then_port_finding(next_with_trials):
    ses_1 = recording(
        snapshot(ZERO_B3, blocks=1, tot_blocks=1),
        prints(ZERO_B3, 3, blocks=1),
        snapshot(ZERO_B1, blocks=2, tot_blocks=2),
    )
    finding = recording(
        port_finding_start(blocks=1, tot_blocks=2),
        port_finding_prints(3),
        snapshot(PORT_FINDING_MAGS, blocks=1, tot_blocks=2, found_ports=False),
    )
    ses_3 = recording(
        snapshot(next_with_trials, blocks=1, tot_blocks=2),
        prints(next_with_trials, 2, blocks=1),
        snapshot(next_with_trials, blocks=1, tot_blocks=2),
    )
    return {S1: [ses_1], S2: [finding], S3: [ses_3]}


def test_a_session_end_change_is_checked_against_the_next_session_with_trials():
    out = imported(end_change_then_port_finding(ZERO_B3))
    (event,) = out[S1]["reversal_boundary_events"]
    assert event["session_check"] == "inconsistent"  # the zero arm is back where it was
    assert "session_end_change_unmatched" in sources(out[S1]["reversal_diagnostics"])
    merged, _ = get_vars_across_all_sessions({"m": out})
    assert merged["m"]["blocks"] == [1] * 5


def test_a_session_end_change_that_the_session_after_a_port_finding_one_shows_is_carried():
    out = imported(end_change_then_port_finding(ZERO_C2))
    (event,) = out[S1]["reversal_boundary_events"]
    assert event["session_check"] == "consistent"
    merged, _ = get_vars_across_all_sessions({"m": out})
    assert merged["m"]["blocks"] == [1, 1, 1, 2, 2]


def test_checking_session_ends_leaves_the_reconciled_sessions_unchanged():
    from src.behavior_import.reconcile_reversals import _check_session_ends, _reconcile_session

    data = {
        "m": {
            key: session(frames, DRIFTING_TASK)
            for key, frames in end_change_then_port_finding(ZERO_B3).items()
        }
    }
    extract_trials(data)
    results, previous = [], None
    for key in ordered_session_keys("m", data["m"]):
        result, previous = _reconcile_session(data["m"][key], previous)
        results.append(result)
    before = copy.deepcopy([(r.events, r.diagnostics) for r in results])
    checked = _check_session_ends(results)
    assert [(r.events, r.diagnostics) for r in results] == before
    assert checked[0][0][0]["session_check"] == "inconsistent"
    assert "session_check" not in results[0].events[0]


# --- two changes at one boundary ---


def test_two_drifting_changes_that_return_the_zero_arm_are_consistent_with_a_diagnostic():
    ses_1 = recording(
        snapshot(ZERO_B3, blocks=1, tot_blocks=1),
        prints(ZERO_B3, 3, blocks=1),
        snapshot(ZERO_B3, blocks=3, tot_blocks=3),
    )
    ses_2 = recording(
        snapshot(ZERO_B3_DRIFTED, blocks=1, tot_blocks=3),
        prints(ZERO_B3_DRIFTED, 2, blocks=1),
        snapshot(ZERO_B3_DRIFTED, blocks=1, tot_blocks=3),
    )
    out = imported({S1: [ses_1], S2: [ses_2]})
    (event,) = out[S1]["reversal_boundary_events"]
    assert (event["count"], event["session_check"]) == (2, "consistent")
    assert uncarried_counts(out[S1]) == {}
    merged, _ = get_vars_across_all_sessions({"m": out})
    assert merged["m"]["blocks"] == [1, 1, 1, 3, 3]
    (stacked,) = [
        d for d in out[S2]["reversal_diagnostics"] if d["source"] == "stacked_changes_net_unchanged"
    ]
    assert (stacked["kind"], stacked["count"]) == ("block", 2)


def test_two_good_reversals_that_return_the_best_tower_are_not_suspect():
    def counters(total_good):
        return dict(blocks=1, tot_blocks=1 + total_good, good=0, tot_good=total_good)

    ses_1 = recording(
        snapshot(B1_BEST, **counters(0)),
        prints(B1_BEST, 3, blocks=1, good=0),
        snapshot(B1_BEST, **counters(0)),
    )
    ses_2 = recording(
        snapshot(B1_BEST, **counters(2)),
        prints(B1_BEST, 2, blocks=1, good=0),
        snapshot(B1_BEST, **counters(2)),
    )
    out = imported({S1: [ses_1], S2: [ses_2]}, BLOCKED_TASK)
    (event,) = out[S2]["reversal_boundary_events"]
    assert (event["count"], event["session_check"]) == (2, "consistent")
    assert count_suspect_good_reversals(out[S2]) == {"suspect": 0, "uncheckable": 0}
    assert "stacked_changes_net_unchanged" in sources(out[S2]["reversal_diagnostics"])


def test_one_change_that_leaves_the_best_tower_in_place_stays_inconsistent():
    ses_1 = recording(
        snapshot(B1_BEST, blocks=1, good=0, tot_good=0),
        prints(B1_BEST, 3, blocks=1, good=0),
        snapshot(B1_BEST, blocks=1, good=0, tot_good=0),
    )
    ses_2 = recording(
        snapshot(B1_BEST, blocks=1, good=0, tot_good=1),
        prints(B1_BEST, 2, blocks=1, good=0),
        snapshot(B1_BEST, blocks=1, good=0, tot_good=1),
    )
    out = imported({S1: [ses_1], S2: [ses_2]}, BLOCKED_TASK)
    (event,) = out[S2]["reversal_boundary_events"]
    assert event["session_check"] == "inconsistent"
    assert "stacked_changes_net_unchanged" not in sources(out[S2]["reversal_diagnostics"])


# --- the diagnostic about the end of a session belongs to the outgoing events only ---


def test_a_change_in_the_next_run_start_that_is_not_shown_is_marked_but_not_reported_unmatched():
    def counters(total_good):
        return dict(blocks=1, good=0, tot_good=total_good)

    ses_1 = recording(
        snapshot(B1_BEST, **counters(0)),
        prints(B1_BEST, 3, blocks=1, good=0),
        snapshot(B1_BEST, **counters(0)),
    )
    ses_2 = recording(
        snapshot(B1_BEST, **counters(1)),
        prints(B1_BEST, 2, blocks=1, good=0),
        snapshot(B1_BEST, **counters(1)),
    )
    out = imported({S1: [ses_1], S2: [ses_2]}, BLOCKED_TASK)
    (event,) = out[S2]["reversal_boundary_events"]
    assert event["session_check"] == "inconsistent"
    assert [d for s in out.values() for d in s["reversal_diagnostics"]] == []


# --- good event counts ---


def test_good_event_counts_leave_out_only_rejected_events_after_the_last_trial():
    sess = {
        "trial": [1, 2, 3],
        "reversal_boundary_events": [
            dict(kind="good", index=3, count=2, source="run_end", session_check="inconsistent"),
            dict(kind="good", index=3, count=1, source="run_end", session_check="consistent"),
            dict(
                kind="good", index=0, count=1, source="between_runs", session_check="inconsistent"
            ),
            dict(kind="good", index=1, count=4, source="between_runs"),
            dict(kind="bad", index=1, count=7, source="between_runs"),
        ],
    }
    assert good_event_counts(sess) == Counter({3: 1, 0: 1, 1: 4})


# --- the task type of a session ---


def moved_zero_arm_session(task_name, **extra):
    first = recording(snapshot(ZERO_B3, blocks=1), prints(ZERO_B3, 3, blocks=1), snapshot(ZERO_B3))
    second = recording(snapshot(ZERO_B1, blocks=1), prints(ZERO_B1, 2, blocks=1), snapshot(ZERO_B1))
    return {"data": [first, second], "task_name": task_name, **extra}


def blocks_of(sess):
    return extract_trials({"m": {S1: sess}})["m"][S1]["blocks"]


def test_a_task_name_that_matches_both_types_raises_in_both_resolvers():
    both = "Meg\\drifting_blocked_bandit"
    with pytest.raises(ValueError, match="matches both"):
        known_task_type({"task_name": both})
    with pytest.raises(ValueError, match=r"m ses-1.*matches both"):
        blocks_of(moved_zero_arm_session(both))


def test_the_task_type_of_the_session_decides_before_its_name():
    sess = moved_zero_arm_session("Meg\\something_else", task_type="drifting")
    assert blocks_of(sess) == [1, 1, 1, 2, 2]


@pytest.mark.parametrize("task_name", ["Meg\\something_else", None])
def test_an_unrecognised_or_missing_task_name_reads_no_block_change_from_the_magnitudes(task_name):
    assert blocks_of(moved_zero_arm_session(task_name)) == [1] * 5


def test_a_recognised_task_name_reads_the_zero_arm():
    assert blocks_of(moved_zero_arm_session(DRIFTING_TASK)) == [1, 1, 1, 2, 2]
