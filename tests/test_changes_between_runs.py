"""Block changes and reversals logged while nothing was recording.

Recordings of a session form one continuous session: a change that the task logs between two
recordings (in a ``run_end``, or in the next ``run_start`` totals) but not in a trial print makes
the next recording's first trial the first trial of the new block. Every test builds raw
recordings, runs the import and checks the result.
"""

import warnings

import pytest

from src.behavior_analysis.get_good_reversal_info import ReversalWarning, get_good_reversal_info
from src.behavior_analysis.get_total_reversals import get_total_reversals
from src.behavior_analysis.get_variables_across_sessions import get_vars_across_all_sessions
from src.behavior_analysis.suspect_reversals import count_suspect_good_reversals
from src.behavior_import.extract_trials import extract_trials
from src.behavior_import.reconcile_reversals import uncarried_counts
from tests.boundary_fixtures import (
    B1_BEST,
    B3_BEST,
    BLOCKED_TASK,
    DRIFTING_TASK,
    SECOND_SWAPPED,
    ZERO_B1,
    ZERO_B3,
    ZERO_B3_DRIFTED,
    ZERO_C2,
    prints,
    recording,
    session,
    snapshot,
    trial_print,
)


def imported(sessions, task=DRIFTING_TASK):
    """``{key: session}`` after ``extract_trials``; *sessions* maps keys to lists of recordings."""
    data = {"m": {key: session(frames, task) for key, frames in sessions.items()}}
    return extract_trials(data)["m"]


def changes(values):
    """Indices where a per-trial series changes."""
    return [i for i in range(1, len(values)) if values[i] != values[i - 1]]


def zero_arm_changes(sess):
    rewards = sess["reward_magnitudes_by_tower"]
    zero = [min(rewards, key=lambda t: rewards[t][i]) for i in range(len(sess["blocks"]))]
    return changes(zero)


# --- drifting: a block change between recordings ---


def two_recordings_with_a_change_at_the_last_trial(totals=True):
    def counters(blocks, total):
        return {"blocks": blocks, **({"tot_blocks": total} if totals else {})}

    first = recording(
        snapshot(ZERO_B3, **counters(1, 1)),
        prints(ZERO_B3, 3, blocks=1),
        snapshot(ZERO_B1, **counters(2, 2)),
    )
    second = recording(
        snapshot(ZERO_B1, **counters(1, 2)),
        prints(ZERO_B1, 2, blocks=1),
        snapshot(ZERO_B1, **counters(1, 2)),
    )
    return [first, second]


def test_a_change_at_a_recordings_last_trial_starts_a_new_block():
    sess = imported({"ses-1": two_recordings_with_a_change_at_the_last_trial()})["ses-1"]
    assert sess["blocks"] == [1, 1, 1, 2, 2]
    assert zero_arm_changes(sess) == [3]
    assert sess["reversal_boundary_events"] == [
        dict(kind="block", index=3, count=1, source="run_end")
    ]
    assert sess["reversal_diagnostics"] == []


def test_a_change_logged_only_in_the_next_run_start_totals_starts_a_new_block():
    first = recording(
        snapshot(ZERO_B3, blocks=1, tot_blocks=1),
        prints(ZERO_B3, 3, blocks=1),
        snapshot(ZERO_B3, blocks=1, tot_blocks=1),
    )
    second = recording(
        snapshot(ZERO_B1, blocks=1, tot_blocks=2),
        prints(ZERO_B1, 2, blocks=1),
        snapshot(ZERO_B1, blocks=1, tot_blocks=2),
    )
    sess = imported({"ses-1": [first, second]})["ses-1"]
    assert sess["blocks"] == [1, 1, 1, 2, 2]
    assert sess["reversal_boundary_events"] == [
        dict(kind="block", index=3, count=1, source="between_runs")
    ]


def test_a_recording_that_restarts_its_counter_with_no_change_continues_the_block():
    first = recording(
        snapshot(ZERO_B3, blocks=1, tot_blocks=1),
        prints(ZERO_B3, 3, blocks=1),
        snapshot(ZERO_B3_DRIFTED, blocks=1, tot_blocks=1),
    )
    second = recording(
        snapshot(ZERO_B3_DRIFTED, blocks=1, tot_blocks=1),
        prints(ZERO_B3_DRIFTED, 2, blocks=1),
        snapshot(ZERO_B3_DRIFTED, blocks=1, tot_blocks=1),
    )
    sess = imported({"ses-1": [first, second]})["ses-1"]
    assert sess["blocks"] == [1, 1, 1, 1, 1]
    assert sess["reversal_boundary_events"] == []
    assert sess["reversal_diagnostics"] == []


def test_a_restart_after_a_change_inside_the_recording_continues_the_new_block():
    trials = prints(ZERO_B3, 2, blocks=1) + prints(ZERO_B1, 2, first=3, blocks=2)
    first = recording(
        snapshot(ZERO_B3, blocks=1, tot_blocks=1), trials, snapshot(ZERO_B1, blocks=2, tot_blocks=2)
    )
    second = recording(
        snapshot(ZERO_B1, blocks=1, tot_blocks=2),
        prints(ZERO_B1, 2, blocks=1),
        snapshot(ZERO_B1, blocks=1, tot_blocks=2),
    )
    sess = imported({"ses-1": [first, second]})["ses-1"]
    assert sess["blocks"] == [1, 1, 2, 2, 2, 2]


def test_four_recordings_give_the_block_changes_at_the_end_of_one_and_inside_another():
    """Recordings of 25, 8, 20 and 30 trials: the change after trial 33, then one at trial 64."""
    r1 = recording(
        snapshot(ZERO_B3, blocks=1, tot_blocks=1),
        prints(ZERO_B3, 25, blocks=1),
        snapshot(ZERO_B3, blocks=1, tot_blocks=1),
    )
    r2 = recording(
        snapshot(ZERO_B3, blocks=1, tot_blocks=1),
        prints(ZERO_B3, 8, blocks=1),
        snapshot(ZERO_B1, blocks=2, tot_blocks=2),
    )
    r3 = recording(
        snapshot(ZERO_B1, blocks=1, tot_blocks=2),
        prints(ZERO_B1, 20, blocks=1),
        snapshot(ZERO_B1, blocks=1, tot_blocks=2),
    )
    r4 = recording(
        snapshot(ZERO_B1, blocks=1, tot_blocks=2),
        prints(ZERO_B1, 11, blocks=1) + prints(ZERO_C2, 19, first=12, blocks=2),
        snapshot(ZERO_C2, blocks=2, tot_blocks=3),
    )
    sess = imported({"ses-1": [r1, r2, r3, r4]})["ses-1"]
    assert len(sess["blocks"]) == 83
    assert changes(sess["blocks"]) == [33, 64]
    assert zero_arm_changes(sess) == [33, 64]
    assert sess["blocks"][-1] == 3


def test_a_recording_that_crashed_before_its_run_end_still_shows_the_change():
    first = recording(snapshot(ZERO_B3, blocks=1, tot_blocks=1), prints(ZERO_B3, 3, blocks=1))
    second = recording(
        snapshot(ZERO_B1, blocks=1, tot_blocks=2),
        prints(ZERO_B1, 2, blocks=1),
        snapshot(ZERO_B1, blocks=1, tot_blocks=2),
    )
    sess = imported({"ses-1": [first, second]})["ses-1"]
    assert sess["blocks"] == [1, 1, 1, 2, 2]


def test_a_change_after_the_last_trial_and_another_in_the_next_run_start_are_two_blocks():
    first = recording(
        snapshot(ZERO_B3, blocks=1, tot_blocks=1),
        prints(ZERO_B3, 3, blocks=1),
        snapshot(ZERO_B1, blocks=2, tot_blocks=2),
    )
    second = recording(
        snapshot(ZERO_C2, blocks=1, tot_blocks=3),
        prints(ZERO_C2, 2, blocks=1),
        snapshot(ZERO_C2, blocks=1, tot_blocks=3),
    )
    sess = imported({"ses-1": [first, second]})["ses-1"]
    assert sess["blocks"] == [1, 1, 1, 3, 3]
    assert [(e["source"], e["count"]) for e in sess["reversal_boundary_events"]] == [
        ("run_end", 1),
        ("between_runs", 1),
    ]


def test_uncarried_counts_are_the_inconsistent_events_after_the_last_trial():
    sess = {
        "trial": [1, 2],
        "reversal_boundary_events": [
            dict(kind="good", index=2, count=2, source="run_end", session_check="inconsistent"),
            dict(kind="bad", index=2, count=1, source="run_end", session_check="consistent"),
            dict(
                kind="good", index=1, count=1, source="between_runs", session_check="inconsistent"
            ),
            dict(kind="block", index=2, count=1, source="run_end"),
        ],
    }
    assert uncarried_counts(sess) == {"good": 2}
    assert uncarried_counts({}) == {}


def test_a_change_inside_a_recording_is_counted_once():
    trials = prints(ZERO_B3, 2, blocks=1) + prints(ZERO_B1, 2, first=3, blocks=2)
    rec = recording(
        snapshot(ZERO_B3, blocks=1, tot_blocks=1), trials, snapshot(ZERO_B1, blocks=2, tot_blocks=2)
    )
    sess = imported({"ses-1": [rec]})["ses-1"]
    assert sess["blocks"] == [1, 1, 2, 2]
    assert sess["reversal_boundary_events"] == []


# --- drifting: old files without session totals ---


def test_old_files_use_the_per_recording_counter_for_a_change_at_the_last_trial():
    sess = imported({"ses-1": two_recordings_with_a_change_at_the_last_trial(totals=False)})
    assert sess["ses-1"]["blocks"] == [1, 1, 1, 2, 2]


def test_old_files_without_counters_in_run_end_use_the_zero_arm():
    first = recording(snapshot(ZERO_B3, blocks=1), prints(ZERO_B3, 3, blocks=1), snapshot(ZERO_B1))
    second = recording(snapshot(ZERO_B1, blocks=1), prints(ZERO_B1, 2, blocks=1), snapshot(ZERO_B1))
    sess = imported({"ses-1": [first, second]})["ses-1"]
    assert sess["blocks"] == [1, 1, 1, 2, 2]
    assert sess["reversal_boundary_events"] == [
        dict(kind="block", index=3, count=1, source="magnitudes")
    ]


def test_old_files_a_zero_arm_move_between_run_end_and_run_start_is_a_block_change():
    first = recording(snapshot(ZERO_B3, blocks=1), prints(ZERO_B3, 3, blocks=1))
    second = recording(snapshot(ZERO_B1, blocks=1), prints(ZERO_B1, 2, blocks=1), snapshot(ZERO_B1))
    sess = imported({"ses-1": [first, second]})["ses-1"]
    assert sess["blocks"] == [1, 1, 1, 2, 2]


def test_old_files_drift_of_the_other_arms_is_not_a_block_change():
    first = recording(
        snapshot(ZERO_B3, blocks=1), prints(ZERO_B3, 3, blocks=1), snapshot(ZERO_B3_DRIFTED)
    )
    second = recording(
        snapshot(ZERO_B3_DRIFTED, blocks=1),
        prints(ZERO_B3_DRIFTED, 2, blocks=1),
        snapshot(ZERO_B3_DRIFTED),
    )
    sess = imported({"ses-1": [first, second]})["ses-1"]
    assert sess["blocks"] == [1, 1, 1, 1, 1]
    assert sess["reversal_boundary_events"] == []
    assert sess["reversal_diagnostics"] == []


def test_totals_that_show_no_change_win_over_a_moved_zero_arm():
    # The counters are the record; a magnitude move without a logged change is only flagged.
    first = recording(
        snapshot(ZERO_B3, blocks=1, tot_blocks=1),
        prints(ZERO_B3, 3, blocks=1),
        snapshot(ZERO_B1, blocks=1, tot_blocks=1),
    )
    second = recording(
        snapshot(ZERO_B1, blocks=1, tot_blocks=1),
        prints(ZERO_B1, 2, blocks=1),
        snapshot(ZERO_B1, blocks=1, tot_blocks=1),
    )
    sess = imported({"ses-1": [first, second]})["ses-1"]
    assert sess["blocks"] == [1, 1, 1, 1, 1]
    assert [d["source"] for d in sess["reversal_diagnostics"]] == [
        "unclassified_terminal_magnitude_change"
    ]


# --- the description of each boundary ---


def test_each_recording_boundary_is_described_with_a_verdict():
    sess = imported({"ses-1": two_recordings_with_a_change_at_the_last_trial()})["ses-1"]
    first, second = sess["run_boundaries"]
    assert first is None  # nothing precedes the session's first recording
    assert second["verdict"] == "consistent"
    assert second["magnitudes_before"] == ZERO_B3 and second["magnitudes_after"] == ZERO_B1
    assert second["total_delta"]["block"] == 1
    assert second["counter_delta"]["block"] == 0  # the per-recording counter restarted at 1


def test_the_first_recording_of_a_session_is_described_against_the_previous_session():
    ses_1 = [
        recording(
            snapshot(ZERO_B3, blocks=1, tot_blocks=1),
            prints(ZERO_B3, 3, blocks=1),
            snapshot(ZERO_B3, blocks=1, tot_blocks=1),
        )
    ]
    ses_2 = [
        recording(
            snapshot(ZERO_B1, blocks=1, tot_blocks=2),
            prints(ZERO_B1, 2, blocks=1),
            snapshot(ZERO_B1, blocks=1, tot_blocks=2),
        )
    ]
    out = imported({"ses-1_date-20260101": ses_1, "ses-2_date-20260102": ses_2})
    (boundary,) = out["ses-2_date-20260102"]["run_boundaries"]
    assert boundary["verdict"] == "consistent"
    assert boundary["total_delta"]["block"] == 1


# --- blocked: good and bad reversals between recordings ---


def blocked_pair(second_mags, *, tot_good=0, tot_bad=0, end_good=0, end_bad=0):
    """Two recordings; the second starts from *second_mags* and the given session totals."""
    first = recording(
        snapshot(B1_BEST, blocks=1, tot_blocks=1, good=0, tot_good=0, bad=0, tot_bad=0),
        prints(B1_BEST, 3, blocks=1, good=0, bad=0),
        snapshot(
            B1_BEST,
            blocks=1,
            tot_blocks=1,
            good=end_good,
            tot_good=end_good,
            bad=end_bad,
            tot_bad=end_bad,
        ),
    )
    n_changes = tot_good + tot_bad
    second = recording(
        snapshot(
            second_mags,
            blocks=1,
            tot_blocks=1 + n_changes,
            good=0,
            tot_good=tot_good,
            bad=0,
            tot_bad=tot_bad,
        ),
        prints(second_mags, 2, blocks=1, good=0, bad=0),
        snapshot(
            second_mags,
            blocks=1,
            tot_blocks=1 + n_changes,
            good=0,
            tot_good=tot_good,
            bad=0,
            tot_bad=tot_bad,
        ),
    )
    return [first, second]


def test_a_good_reversal_between_recordings_is_added_and_checked():
    frames = blocked_pair(B3_BEST, tot_good=1)
    sess = imported({"ses-1": frames}, BLOCKED_TASK)["ses-1"]
    assert sess["good_reversals"] == [0, 0, 0, 1, 1]
    assert sess["blocks"] == [1, 1, 1, 2, 2]
    assert sess["reversal_boundary_events"] == [
        dict(kind="good", index=3, count=1, source="between_runs")
    ]
    assert sess["reversal_totals"] == {"good": 1, "bad": 0}
    # checked against the trials around it: the best tower changed, so nothing is suspect
    assert count_suspect_good_reversals(sess) == {"suspect": 0, "uncheckable": 0}
    windows = get_good_reversal_info({"m": {"ses-1": sess}}, pre=1)
    assert [w["reversal_idx"] for w in windows["m"]] == [3]


def test_a_good_reversal_between_recordings_with_an_unchanged_best_is_suspect_not_uncheckable():
    frames = blocked_pair(SECOND_SWAPPED, tot_good=1)
    sess = imported({"ses-1": frames}, BLOCKED_TASK)["ses-1"]
    assert count_suspect_good_reversals(sess) == {"suspect": 1, "uncheckable": 0}
    with pytest.warns(ReversalWarning, match="best tower unchanged"):
        get_good_reversal_info({"m": {"ses-1": sess}}, pre=1)


def test_a_bad_reversal_between_recordings_is_added_and_the_best_tower_may_stay():
    frames = blocked_pair(SECOND_SWAPPED, tot_bad=1)
    sess = imported({"ses-1": frames}, BLOCKED_TASK)["ses-1"]
    assert sess["bad_reversals"] == [0, 0, 0, 1, 1]
    assert sess["good_reversals"] == [0, 0, 0, 0, 0]
    assert sess["blocks"] == [1, 1, 1, 2, 2]
    assert sess["reversal_totals"] == {"good": 0, "bad": 1}
    assert sess["reversal_boundary_events"] == [
        dict(kind="bad", index=3, count=1, source="between_runs")
    ]
    assert count_suspect_good_reversals(sess) == {"suspect": 0, "uncheckable": 0}


def test_a_good_reversal_logged_in_run_end_is_checked_like_any_other():
    frames = blocked_pair(B3_BEST, tot_good=1, end_good=1)
    sess = imported({"ses-1": frames}, BLOCKED_TASK)["ses-1"]
    assert sess["reversal_boundary_events"][0]["source"] == "run_end"
    assert count_suspect_good_reversals(sess) == {"suspect": 0, "uncheckable": 0}


def test_blocked_old_files_without_totals_take_the_block_change_from_the_permutation():
    first = recording(
        snapshot(B1_BEST, blocks=1, good=0, bad=0),
        prints(B1_BEST, 3, blocks=1, good=0, bad=0),
        snapshot(B1_BEST, blocks=1, good=0, bad=0),
    )
    second = recording(
        snapshot(B3_BEST, blocks=1, good=0, bad=0),
        prints(B3_BEST, 2, blocks=1, good=0, bad=0),
        snapshot(B3_BEST, blocks=1, good=0, bad=0),
    )
    sess = imported({"ses-1": [first, second]}, BLOCKED_TASK)["ses-1"]
    assert sess["blocks"] == [1, 1, 1, 2, 2]
    assert sess["reversal_boundary_events"] == [
        dict(kind="block", index=3, count=1, source="magnitudes")
    ]
    # the kind of the reversal is unknown, so the good and bad counts are not touched
    assert sess["good_reversals"] == [0] * 5 and sess["bad_reversals"] == [0] * 5
    assert sess["reversal_totals"] == {"good": 0, "bad": 0}


def test_blocked_restart_with_no_change_is_not_a_reversal():
    frames = blocked_pair(B1_BEST)
    sess = imported({"ses-1": frames}, BLOCKED_TASK)["ses-1"]
    assert sess["blocks"] == [1] * 5
    assert sess["reversal_boundary_events"] == []


# --- a change logged at the end of a session ---


def blocked_sessions(next_start_mags, *, end_mags):
    """ses-1 logs a good reversal after its last trial; ses-2 starts from *next_start_mags*."""
    ses_1 = recording(
        snapshot(B1_BEST, blocks=1, tot_blocks=1, good=0, tot_good=0, bad=0, tot_bad=0),
        prints(B1_BEST, 3, blocks=1, good=0, bad=0),
        snapshot(end_mags, blocks=1, tot_blocks=2, good=1, tot_good=1, bad=0, tot_bad=0),
    )
    ses_2 = recording(
        snapshot(next_start_mags, blocks=1, tot_blocks=2, good=0, tot_good=1, bad=0, tot_bad=0),
        prints(next_start_mags, 2, blocks=1, good=0, bad=0),
        snapshot(next_start_mags, blocks=1, tot_blocks=2, good=0, tot_good=1, bad=0, tot_bad=0),
    )
    return {"ses-1_date-20260101": [ses_1], "ses-2_date-20260102": [ses_2]}


def test_a_session_end_reversal_consistent_with_the_next_start_begins_the_new_block():
    out = imported(blocked_sessions(B3_BEST, end_mags=B3_BEST), BLOCKED_TASK)
    first = out["ses-1_date-20260101"]
    (event,) = first["reversal_boundary_events"]
    assert (event["kind"], event["index"], event["source"]) == ("good", 3, "run_end")
    assert event["session_check"] == "consistent"
    merged, _ = get_vars_across_all_sessions({"m": out})
    assert merged["m"]["good_reversals"] == [0, 0, 0, 1, 1]
    totals = get_total_reversals(out)
    assert (totals["suspect_good_reversals"], totals["uncheckable_good_reversals"]) == (0, 0)
    windows = get_good_reversal_info({"m": out}, pre=1)
    assert [w["reversal_idx"] for w in windows["m"]] == [3]


def test_a_session_end_reversal_that_the_next_start_does_not_show_stays_uncheckable():
    out = imported(blocked_sessions(B1_BEST, end_mags=B1_BEST), BLOCKED_TASK)
    first = out["ses-1_date-20260101"]
    (event,) = first["reversal_boundary_events"]
    assert event["session_check"] == "inconsistent"
    merged, _ = get_vars_across_all_sessions({"m": out})
    assert merged["m"]["good_reversals"] == [0] * 5  # the next start is not the new block
    assert first["reversal_totals"]["good"] == 1  # the logged total is unchanged
    totals = get_total_reversals(out)
    assert totals["good_reversals"] == 1
    assert (totals["suspect_good_reversals"], totals["uncheckable_good_reversals"]) == (0, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ReversalWarning)
        assert get_good_reversal_info({"m": out}, pre=1)["m"] == []
    assert any(d["source"] == "session_end_change_unmatched" for d in first["reversal_diagnostics"])


def test_a_session_end_reversal_with_no_later_session_is_uncheckable():
    out = imported(
        {"ses-1_date-20260101": blocked_sessions(B3_BEST, end_mags=B3_BEST)["ses-1_date-20260101"]},
        BLOCKED_TASK,
    )
    (event,) = out["ses-1_date-20260101"]["reversal_boundary_events"]
    assert "session_check" not in event
    assert count_suspect_good_reversals(out["ses-1_date-20260101"]) == {
        "suspect": 0,
        "uncheckable": 1,
    }


def test_a_session_end_reversal_is_not_matched_to_a_session_of_another_problem():
    sessions = blocked_sessions(B3_BEST, end_mags=B3_BEST)
    other = recording(
        snapshot({"A1": 4, "A2": 1, "A3": 0}, blocks=1, good=0, tot_good=1, bad=0, tot_bad=0),
        [trial_print(n, {"A1": 4, "A2": 1, "A3": 0}, blocks=1, good=0, bad=0) for n in (1, 2)],
    )
    sessions["ses-2_date-20260102"] = [other]
    out = imported(sessions, BLOCKED_TASK)
    (event,) = out["ses-1_date-20260101"]["reversal_boundary_events"]
    assert "session_check" not in event


def test_a_session_end_block_change_consistent_with_the_next_start_begins_a_new_block():
    ses_1 = recording(
        snapshot(ZERO_B3, blocks=1, tot_blocks=1),
        prints(ZERO_B3, 3, blocks=1),
        snapshot(ZERO_B1, blocks=2, tot_blocks=2),
    )
    ses_2 = recording(
        snapshot(ZERO_B1, blocks=1, tot_blocks=2),
        prints(ZERO_B1, 2, blocks=1),
        snapshot(ZERO_B1, blocks=1, tot_blocks=2),
    )
    out = imported({"ses-1_date-20260101": [ses_1], "ses-2_date-20260102": [ses_2]})
    (event,) = out["ses-1_date-20260101"]["reversal_boundary_events"]
    assert (event["kind"], event["index"], event["session_check"]) == ("block", 3, "consistent")
    merged, _ = get_vars_across_all_sessions({"m": out})
    assert merged["m"]["blocks"] == [1, 1, 1, 2, 2]


def test_a_session_end_block_change_that_the_next_start_does_not_show_is_not_a_new_block():
    ses_1 = recording(
        snapshot(ZERO_B3, blocks=1, tot_blocks=1),
        prints(ZERO_B3, 3, blocks=1),
        snapshot(ZERO_B3_DRIFTED, blocks=2, tot_blocks=2),
    )
    ses_2 = recording(
        snapshot(ZERO_B3_DRIFTED, blocks=1, tot_blocks=2),
        prints(ZERO_B3_DRIFTED, 2, blocks=1),
        snapshot(ZERO_B3_DRIFTED, blocks=1, tot_blocks=2),
    )
    out = imported({"ses-1_date-20260101": [ses_1], "ses-2_date-20260102": [ses_2]})
    first = out["ses-1_date-20260101"]
    (event,) = first["reversal_boundary_events"]
    assert event["session_check"] == "inconsistent"
    merged, _ = get_vars_across_all_sessions({"m": out})
    assert merged["m"]["blocks"] == [1] * 5
    (unmatched,) = [
        d for d in first["reversal_diagnostics"] if d["source"] == "session_end_change_unmatched"
    ]
    assert unmatched["kind"] == "block" and "zero arm" in unmatched["reason"]


def test_reconciling_twice_gives_the_same_flags_and_diagnostics():
    from src.behavior_import.reconcile_reversals import reconcile_reversals

    data = {
        "m": {
            key: session(frames, BLOCKED_TASK)
            for key, frames in blocked_sessions(B1_BEST, end_mags=B1_BEST).items()
        }
    }
    extract_trials(data)
    once = {
        key: (s["reversal_boundary_events"], s["reversal_diagnostics"])
        for key, s in data["m"].items()
    }
    reconcile_reversals(data)
    again = {
        key: (s["reversal_boundary_events"], s["reversal_diagnostics"])
        for key, s in data["m"].items()
    }
    assert again == once
