"""Unit tests for the pure helpers behind ``reconcile_reversals``."""

import copy

import pandas as pd
import pytest

from src.behavior_import.reconcile_reversals import (
    _between_runs,
    _combine_runs,
    _correct_blocks,
    _counter_observed,
    _end_state,
    _reconcile_run,
    _Run,
    _run_total,
    _running_counts,
    _session_order,
    _terminal_magnitude_diagnostics,
    reconcile_reversals,
)

MAGS = {"A": 4, "B": 1, "C": 0}
OTHER_MAGS = {"A": 0, "B": 4, "C": 1}


def state(**kw):
    return {"magnitudes": MAGS, "problem": 1, **kw}


def frame(start, end):
    return pd.DataFrame(
        [
            {"type": "variable", "subtype": "run_start", "content": start},
            {"type": "variable", "subtype": "run_end", "content": end},
        ]
    )


def run_variables(good, bad):
    n = len(good)
    return {
        "trial": list(range(1, n + 1)),
        "good_reversals": list(good),
        "bad_reversals": list(bad),
        "reward_magnitudes": [MAGS] * n,
    }


def test_session_order_sorts_numerically_and_puts_unnumbered_last():
    keys = ["ses-10", "other", "ses-2_date-1", "ses-2"]
    assert sorted(keys, key=_session_order) == ["ses-2", "ses-2_date-1", "ses-10", "other"]


def test_running_counts_subtracts_baseline_and_never_decreases():
    assert _running_counts([2, 3, 2, 4], 4, baseline=2) == [0, 1, 1, 2]


def test_running_counts_carries_over_missing_and_short_samples():
    assert _running_counts([None, 1, None], 5, baseline=0) == [0, 1, 1, 1, 1]


def test_running_counts_empty_run():
    assert _running_counts([], 0, baseline=3) == []


def test_run_total_prefers_largest_evidence():
    start = {"good": 1, "total_good": 5}
    assert _run_total(0, start, {"good": 4}, "good", baseline=1) == 3
    assert _run_total(0, start, {"good": 2, "total_good": 9}, "good", baseline=1) == 4
    assert _run_total(2, start, {}, "good", baseline=1) == 2


def test_run_total_ignores_decreasing_total_counter():
    assert _run_total(1, {"total_bad": 5}, {"total_bad": 3}, "bad", baseline=0) == 1


def test_counter_observed_from_trials_or_start_snapshot():
    assert _counter_observed([None, 1], {}, "good")
    assert _counter_observed([], {"good": 0}, "good")
    assert _counter_observed([], {"total_good": 0}, "good")
    assert not _counter_observed([None], {"bad": 0}, "good")


def test_between_runs_positive_delta_is_an_event_at_the_run_start():
    previous = state(total_good=1, total_bad=0)
    start = state(total_good=3, total_bad=0)
    incoming, events, diagnostics = _between_runs(previous, start, trial_offset=7, run_index=1)
    assert incoming == {"good": 2, "bad": 0}
    assert events == [dict(kind="good", index=7, count=2, source="between_runs")]
    assert diagnostics == []


def test_between_runs_rollback_is_a_diagnostic_not_an_event():
    incoming, events, diagnostics = _between_runs(
        state(total_bad=5), state(total_bad=4), trial_offset=0, run_index=2
    )
    assert incoming == {"good": 0, "bad": 0}
    assert events == []
    assert diagnostics == [dict(kind="bad", source="counter_rollback", delta=-1, run=2)]


def test_between_runs_magnitude_change_without_events_is_flagged():
    _, events, diagnostics = _between_runs(
        state(total_good=1), state(total_good=1, magnitudes=OTHER_MAGS), 0, 1
    )
    assert events == []
    assert [d["source"] for d in diagnostics] == ["unclassified_magnitude_change"]


def test_between_runs_does_not_bridge_different_problems():
    incoming, events, diagnostics = _between_runs(
        state(total_good=1), state(total_good=9, problem=2), 0, 1
    )
    assert (incoming, events, diagnostics) == ({"good": 0, "bad": 0}, [], [])


def test_between_runs_first_run_has_nothing_incoming():
    assert _between_runs(None, state(total_good=9), 0, 0)[0] == {"good": 0, "bad": 0}


def test_terminal_magnitude_change_flagged_unless_a_terminal_reversal_explains_it():
    end = {"magnitudes": OTHER_MAGS}
    flagged = _terminal_magnitude_diagnostics(MAGS, end, 2, 0, [], run_index=0)
    assert [d["source"] for d in flagged] == ["unclassified_terminal_magnitude_change"]
    explained = [dict(kind="good", index=2, count=1, source="run_end")]
    assert _terminal_magnitude_diagnostics(MAGS, end, 2, 0, explained, run_index=0) == []
    assert _terminal_magnitude_diagnostics(MAGS, {"magnitudes": MAGS}, 2, 0, [], 0) == []
    assert _terminal_magnitude_diagnostics(MAGS, {}, 2, 0, [], 0) == []


def test_end_state_fills_missing_total_magnitudes_and_problem():
    start = {"total_good": 2, "problem": 3}
    filled = _end_state(start, {}, MAGS, {"good": 1, "bad": 0})
    assert filled["total_good"] == 3
    assert filled.get("total_bad") is None  # no start total to extend
    assert filled["magnitudes"] == MAGS
    assert filled["problem"] == 3


def test_end_state_keeps_recorded_values():
    end = {"total_good": 9, "magnitudes": OTHER_MAGS, "problem": 2}
    filled = _end_state({"total_good": 2, "problem": 3}, end, None, {"good": 1, "bad": 0})
    assert filled == {**end, "total_good": 9}


def test_correct_blocks_adds_events_at_or_before_each_trial():
    events = [dict(index=0, count=1), dict(index=2, count=2)]
    assert _correct_blocks([1, 1, 1, 1], events) == [2, 2, 4, 4]


def test_correct_blocks_with_no_blocks():
    assert _correct_blocks([], [dict(index=0, count=1)]) == []


def test_correct_blocks_raises_type_error_on_none_block():
    with pytest.raises(TypeError):
        _correct_blocks([None, None], [])


def test_reconcile_run_offsets_trials_and_reports_terminal_event():
    start = {"num_good_rev": 0, "curr_rew_mag": MAGS, "num_problem": 1}
    end = {"num_good_rev": 2, "curr_rew_mag": MAGS, "num_problem": 1}
    variables = run_variables((0, 1), ())
    carried = {"good": 5, "bad": 0}
    run = _reconcile_run(frame(start, end), variables, None, carried, trial_offset=10, run_index=0)
    assert run.n_trials == 2
    assert run.corrected["good"] == [5, 6]
    assert run.added == {"good": 2, "bad": 0}
    assert run.events == [dict(kind="good", index=12, count=1, source="run_end")]
    assert run.observed == {"good": True, "bad": False}


def two_run_session():
    """Run 2 starts with a cumulative good total 2 above where run 1 ended."""
    first = frame(
        {"num_good_rev": 0, "num_bad_rev": 0, "tot_num_good_rev": 0, "tot_num_bad_rev": 0}
        | {"curr_rew_mag": MAGS, "num_problem": 1},
        {"num_good_rev": 1, "num_bad_rev": 0, "tot_num_good_rev": 1, "tot_num_bad_rev": 0}
        | {"curr_rew_mag": MAGS, "num_problem": 1},
    )
    second = frame(
        {"num_good_rev": 1, "num_bad_rev": 0, "tot_num_good_rev": 3, "tot_num_bad_rev": 0}
        | {"curr_rew_mag": MAGS, "num_problem": 1},
        {"num_good_rev": 2, "num_bad_rev": 0, "tot_num_good_rev": 4, "tot_num_bad_rev": 0}
        | {"curr_rew_mag": MAGS, "num_problem": 1},
    )
    return {
        "data": [first, second],
        "trial_variables": [run_variables((0, 1), (0, 0)), run_variables((1, 2), (0, 0))],
        "blocks": [1, 1, 1, 1],
    }


def test_positive_change_between_runs_is_added_at_the_second_run_start():
    sess = two_run_session()
    reconcile_reversals({"mouse": {"ses-1": sess}})
    assert sess["good_reversals"] == [0, 1, 3, 4]
    assert sess["bad_reversals"] == [0, 0, 0, 0]
    assert sess["reversal_totals"] == {"good": 4, "bad": 0}
    assert sess["blocks"] == [1, 1, 3, 3]
    assert sess["reversal_boundary_events"] == [
        dict(kind="good", index=2, count=2, source="between_runs")
    ]
    assert sess["reversal_diagnostics"] == []


def test_reconcile_reversals_twice_leaves_the_result_unchanged():
    data = {"mouse": {"ses-1": two_run_session()}}
    reconcile_reversals(data)
    once = copy.deepcopy(data)
    assert reconcile_reversals(data) is data
    for key, sess in data["mouse"]["ses-1"].items():
        if key == "data":  # DataFrames do not compare with ==
            continue
        assert sess == once["mouse"]["ses-1"][key], key


def test_session_with_empty_trial_variables_has_no_trials_but_keeps_terminal_events():
    start = {"num_good_rev": 0, "num_bad_rev": 0, "curr_rew_mag": MAGS, "num_problem": 1}
    end = start | {"num_good_rev": 1}
    sess = {"data": [frame(start, end)], "trial_variables": {}, "blocks": []}
    reconcile_reversals({"mouse": {"ses-1": sess}})
    assert sess["good_reversals"] == []
    assert sess["blocks"] == []
    assert sess["reversal_totals"] == {"good": 1, "bad": 0}
    assert sess["reversal_boundary_events"] == [
        dict(kind="good", index=0, count=1, source="run_end")
    ]


def make_run(**overrides):
    fields = dict(
        n_trials=0,
        corrected={"good": [], "bad": []},
        added={"good": 0, "bad": 0},
        observed={"good": False, "bad": False},
        events=[],
        diagnostics=[],
        end_state={},
    )
    return _Run(**(fields | overrides))


def test_combine_runs_with_no_runs_is_empty_and_unobserved():
    arrays, totals, observed, events, diagnostics = _combine_runs([])
    assert arrays == {"good": [], "bad": []}
    assert totals == {"good": 0, "bad": 0}
    assert observed == {"good": False, "bad": False}
    assert (events, diagnostics) == ([], [])


def test_combine_runs_concatenates_arrays_events_and_diagnostics_in_run_order():
    first = make_run(
        corrected={"good": [0, 1], "bad": [0, 0]},
        added={"good": 1, "bad": 0},
        observed={"good": True, "bad": False},
        events=[dict(kind="good", index=2, count=1, source="run_end")],
        diagnostics=[dict(source="a")],
    )
    second = make_run(
        corrected={"good": [3], "bad": [0]},
        added={"good": 2, "bad": 0},
        observed={"good": False, "bad": False},
        events=[dict(kind="good", index=2, count=2, source="between_runs")],
        diagnostics=[dict(source="b")],
    )
    arrays, totals, observed, events, diagnostics = _combine_runs([first, second])
    assert arrays == {"good": [0, 1, 3], "bad": [0, 0, 0]}
    assert totals == {"good": 3, "bad": 0}
    assert observed == {"good": True, "bad": False}
    assert [e["source"] for e in events] == ["run_end", "between_runs"]
    assert diagnostics == [dict(source="a"), dict(source="b")]


def test_combine_runs_does_not_mutate_its_input():
    run = make_run(corrected={"good": [0], "bad": [0]}, added={"good": 1, "bad": 0})
    before = copy.deepcopy(run)
    _combine_runs([run, run])
    assert run == before


def test_unparseable_run_start_raises_naming_the_session():
    # Skipping the snapshot would make the baseline 0 and count earlier events again.
    sess = {
        "data": frame("{not json", state(num_good_reversals=1)),
        "trial_variables": [run_variables([1], [0])],
    }
    with pytest.raises(ValueError, match=r"M1.*ses-1_date-20260101.*run 0.*run_start"):
        reconcile_reversals({"M1": {"ses-1_date-20260101": sess}})


def test_the_run_index_is_named_when_a_later_run_start_is_unparseable():
    good = frame(state(), state(num_good_reversals=0))
    bad = frame("{not json", "{}")
    sess = {
        "data": [good, bad],
        "trial_variables": [run_variables([0], [0]), run_variables([0], [0])],
    }
    with pytest.raises(ValueError, match=r"M1 ses-1.*run 1"):
        reconcile_reversals({"M1": {"ses-1": sess}})


def test_unparseable_run_end_is_skipped_and_totals_fall_back_to_the_trials():
    # A crashed recording can end with a truncated run_end; the trial counters still hold
    # the events, so the missing end snapshot only loses extras after the last trial.
    sess = {
        "data": frame(state(num_good_reversals=0), "{truncated"),
        "trial_variables": [run_variables([0, 1, 2], [0, 0, 0])],
    }
    reconcile_reversals({"M1": {"ses-1": sess}})
    assert sess["reversal_totals"] == {"good": 2, "bad": 0}
    assert sess["good_reversals"] == [0, 1, 2]


def test_non_string_run_start_content_raises():
    sess = {"data": frame(float("nan"), "{}"), "trial_variables": [run_variables([], [])]}
    with pytest.raises(ValueError, match="run_start"):
        reconcile_reversals({"M1": {"ses-1": sess}})
