from copy import deepcopy

import pytest

from src.behavior_analysis.matched_trial_budget import match_trial_budget
from tests.test_reversal_boundaries import session, snapshot


def recorded(n, good=None, bad=None, **extra):
    """A session of ``n`` trials; counters default to all zero."""
    s = session(
        snapshot(),
        snapshot(),
        good=good if good is not None else (0,) * n,
        bad=bad if bad is not None else (0,) * n,
    )
    s.update(extra)
    return s


def numbered(subjects):
    return {m: int(m[1:]) for m in subjects}


def trials(matched, mouse):
    """``{session key: retained trial numbers}``."""
    return {k: s["trial"] for k, s in matched[mouse].items()}


def test_fixed_budget_fills_sessions_in_numeric_order_and_cuts_the_last():
    sessions = {"ses-10": recorded(3), "ses-2": recorded(3), "ses-9": recorded(3)}
    matched, _ = match_trial_budget({"m1": sessions}, {"m1": 1}, trial_budget=7)
    assert list(matched["m1"]) == ["ses-2", "ses-9", "ses-10"]
    assert [len(s["trial"]) for s in matched["m1"].values()] == [3, 3, 1]


def test_sessions_after_the_budget_is_spent_are_dropped():
    subjects = {"m1": {"ses-1": recorded(4), "ses-2": recorded(4)}}
    matched, _ = match_trial_budget(subjects, {"m1": 1}, trial_budget=4)
    assert list(matched["m1"]) == ["ses-1"]


def test_sessions_without_trials_are_skipped():
    subjects = {"m1": {"ses-1": recorded(0), "ses-2": recorded(2), "ses-3": {}}}
    matched, _ = match_trial_budget(subjects, {"m1": 1}, trial_budget=5)
    assert list(matched["m1"]) == ["ses-2"]


def test_reference_mice_set_the_cap_as_the_floored_mean():
    subjects = {
        "m5": {"ses-1": recorded(3)},
        "m6": {"ses-1": recorded(4)},
        "m1": {"ses-1": recorded(9)},
    }
    matched, report = match_trial_budget(subjects, numbered(subjects), (5, 6))
    assert report["trial_budget"] == 3  # floor(3.5)
    assert report["reference_mean"] == 3.5
    assert report["reference_trial_counts"] == {"m5": 3, "m6": 4}
    assert [len(matched[m]["ses-1"]["trial"]) for m in ("m5", "m6", "m1")] == [3, 3, 3]


def test_default_reference_numbers_are_five_to_eight():
    subjects = {f"m{i}": {"ses-1": recorded(4)} for i in (5, 6, 7, 8)}
    subjects["m1"] = {"ses-1": recorded(10)}
    _, report = match_trial_budget(subjects, numbered(subjects))
    assert report["trial_budget"] == 4
    assert set(report["reference_trial_counts"]) == {"m5", "m6", "m7", "m8"}


@pytest.mark.parametrize("reference", [(5, 9), ()])
def test_missing_or_empty_reference_raises(reference):
    subjects = {"m5": {"ses-1": recorded(3)}}
    with pytest.raises(ValueError, match="Reference mice"):
        match_trial_budget(subjects, numbered(subjects), reference)


def test_reference_error_names_the_missing_subject_numbers():
    subjects = {"m5": {"ses-1": recorded(3)}}
    with pytest.raises(ValueError, match=r"\[9\]"):
        match_trial_budget(subjects, numbered(subjects), (5, 9))


def test_reference_mouse_without_trials_raises():
    subjects = {"m5": {"ses-1": recorded(0)}, "m6": {"ses-1": recorded(4)}}
    with pytest.raises(ValueError, match="Reference mice"):
        match_trial_budget(subjects, numbered(subjects), (5, 6))


@pytest.mark.parametrize("budget", [0, -2, 2.5, "3", True])
def test_fixed_budget_must_be_a_positive_int(budget):
    with pytest.raises(ValueError):
        match_trial_budget({"m1": {"ses-1": recorded(3)}}, {"m1": 1}, trial_budget=budget)


def test_report_has_one_row_per_mouse_including_those_below_budget():
    subjects = {"m1": {"ses-1": recorded(2)}, "m2": {"ses-1": recorded(9)}, "m3": {}}
    _, report = match_trial_budget(subjects, numbered(subjects), trial_budget=5)
    assert report["mice"] == [
        dict(mouse="m1", subject_number=1, available_trials=2, retained_trials=2,
             below_budget=True, reference_mouse=False),
        dict(mouse="m2", subject_number=2, available_trials=9, retained_trials=5,
             below_budget=False, reference_mouse=False),
        dict(mouse="m3", subject_number=3, available_trials=0, retained_trials=0,
             below_budget=True, reference_mouse=False),
    ]  # fmt: skip
    assert report["reference_mean"] is None and report["reference_trial_counts"] == {}


def test_mice_without_trials_are_in_the_report_but_not_the_result():
    subjects = {"m1": {"ses-1": recorded(2)}, "m2": {}}
    matched, report = match_trial_budget(subjects, numbered(subjects), trial_budget=5)
    assert list(matched) == ["m1"]
    assert [r["mouse"] for r in report["mice"]] == ["m1", "m2"]


def test_reference_mice_are_flagged_in_the_report():
    subjects = {"m5": {"ses-1": recorded(3)}, "m1": {"ses-1": recorded(3)}}
    _, report = match_trial_budget(subjects, numbered(subjects), (5,))
    assert {r["mouse"]: r["reference_mouse"] for r in report["mice"]} == {"m5": True, "m1": False}


def test_per_trial_and_per_tower_fields_are_cut_to_the_budget():
    subjects = {"m1": {"ses-1": recorded(5, good=(0, 0, 1, 1, 2))}}
    matched, _ = match_trial_budget(subjects, {"m1": 1}, trial_budget=3)
    cut = matched["m1"]["ses-1"]
    assert cut["trial"] == [1, 2, 3]
    assert cut["good_reversals"] == [0, 0, 1]
    assert cut["blocks"] == [1, 1, 1]
    assert cut["trials_in_block"] == [1, 2, 3]
    assert all(len(v) == 3 for v in cut["reward_magnitudes_by_tower"].values())
    assert all(len(v) == 3 for v in cut["choices_by_tower"].values())
    assert cut["choices_by_rank"] == {"best": [1, 1, 1]}
    assert cut["trial_info"] is True


def test_only_analysis_fields_are_copied():
    subjects = {"m1": {"ses-1": recorded(2, unrelated="x")}}
    matched, _ = match_trial_budget(subjects, {"m1": 1}, trial_budget=5)
    assert "data" not in matched["m1"]["ses-1"]
    assert "unrelated" not in matched["m1"]["ses-1"]
    assert matched["m1"]["ses-1"]["has_good"] is True


def test_partial_session_totals_use_the_counter_at_the_next_trial():
    sess = recorded(6, good=(0, 0, 1, 1, 2, 3), bad=(0, 1, 1, 1, 1, 2))
    matched, _ = match_trial_budget({"m1": {"ses-1": sess}}, {"m1": 1}, trial_budget=2)
    # cut after 2 trials: the good reversal on trial 3 counts, the bad one on trial 2 too
    assert matched["m1"]["ses-1"]["reversal_totals"] == {"good": 1, "bad": 1}


def test_partial_totals_skip_unrecorded_trials_and_missing_counters():
    sess = recorded(4, good=(0, 1, None, None), bad=())
    matched, _ = match_trial_budget({"m1": {"ses-1": sess}}, {"m1": 1}, trial_budget=3)
    assert matched["m1"]["ses-1"]["reversal_totals"] == {"good": 1}  # empty bad counter: no total


def test_boundary_events_after_the_cut_are_dropped():
    events = [{"index": 0}, {"index": 2}, {"index": 3}]
    sess = recorded(5, reversal_boundary_events=events, reversal_totals={"good": 4, "bad": 4})
    matched, _ = match_trial_budget({"m1": {"ses-1": sess}}, {"m1": 1}, trial_budget=2)
    assert matched["m1"]["ses-1"]["reversal_boundary_events"] == [{"index": 0}, {"index": 2}]


def test_complete_sessions_keep_recorded_totals_and_events():
    events = [{"index": 3}]
    sess = recorded(2, reversal_boundary_events=events, reversal_totals={"good": 4, "bad": 5})
    matched, _ = match_trial_budget({"m1": {"ses-1": sess}}, {"m1": 1}, trial_budget=2)
    kept = matched["m1"]["ses-1"]
    assert kept["reversal_totals"] == {"good": 4, "bad": 5}
    assert kept["reversal_boundary_events"] == [{"index": 3}]


def test_input_is_not_modified_and_result_does_not_share_with_it():
    subjects = {"m1": {"ses-1": recorded(4, reversal_totals={"good": 1, "bad": 0})}}
    before = deepcopy(subjects)
    matched, _ = match_trial_budget(subjects, {"m1": 1}, trial_budget=2)
    matched["m1"]["ses-1"]["choices_by_rank"]["best"].append(99)
    matched["m1"]["ses-1"]["reversal_totals"]["good"] = 7
    assert subjects["m1"]["ses-1"]["choices_by_rank"] == before["m1"]["ses-1"]["choices_by_rank"]
    assert subjects["m1"]["ses-1"]["reversal_totals"] == {"good": 1, "bad": 0}


def test_keys_without_a_session_number_come_last():
    sessions = {"first": recorded(2), "ses-3": recorded(2)}
    matched, _ = match_trial_budget({"m1": sessions}, {"m1": 1}, trial_budget=4)
    assert list(matched["m1"]) == ["ses-3", "first"]
