"""Unit tests for the pure helpers behind ``reconcile_reversals``."""

import pandas as pd
import pytest

from src.behavior_import.reconcile_reversals import (
    _between_runs,
    _correct_blocks,
    _counter_observed,
    _end_state,
    _reconcile_run,
    _run_total,
    _running_counts,
    _session_order,
    _terminal_magnitude_diagnostics,
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
    tv = {"reward_magnitudes": [MAGS, MAGS]}
    end = {"magnitudes": OTHER_MAGS}
    flagged = _terminal_magnitude_diagnostics(tv, end, 2, 0, [], run_index=0)
    assert [d["source"] for d in flagged] == ["unclassified_terminal_magnitude_change"]
    explained = [dict(kind="good", index=2, count=1, source="run_end")]
    assert _terminal_magnitude_diagnostics(tv, end, 2, 0, explained, run_index=0) == []
    assert _terminal_magnitude_diagnostics(tv, {"magnitudes": MAGS}, 2, 0, [], 0) == []
    assert _terminal_magnitude_diagnostics(tv, {}, 2, 0, [], 0) == []


def test_end_state_fills_missing_total_magnitudes_and_problem():
    start = {"total_good": 2, "problem": 3}
    tv = {"reward_magnitudes": [OTHER_MAGS, MAGS]}
    filled = _end_state(start, {}, tv, 2, {"good": 1, "bad": 0})
    assert filled["total_good"] == 3
    assert filled.get("total_bad") is None  # no start total to extend
    assert filled["magnitudes"] == MAGS
    assert filled["problem"] == 3


def test_end_state_keeps_recorded_values():
    end = {"total_good": 9, "magnitudes": OTHER_MAGS, "problem": 2}
    filled = _end_state({"total_good": 2, "problem": 3}, end, {}, 0, {"good": 1, "bad": 0})
    assert filled == {**end, "total_good": 9}


def test_correct_blocks_adds_events_at_or_before_each_trial():
    events = [dict(index=0, count=1), dict(index=2, count=2)]
    assert _correct_blocks([1, 1, 1, 1], events) == [2, 2, 4, 4]


def test_correct_blocks_with_no_blocks():
    assert _correct_blocks([], [dict(index=0, count=1)]) == []


def test_correct_blocks_with_missing_values_fails_loudly():
    with pytest.raises(TypeError):
        _correct_blocks([None, None], [])


def test_reconcile_run_offsets_trials_and_reports_terminal_event():
    start = {"num_good_rev": 0, "curr_rew_mag": MAGS, "num_problem": 1}
    end = {"num_good_rev": 2, "curr_rew_mag": MAGS, "num_problem": 1}
    tv = {"trial": [1, 2], "good_reversals": [0, 1], "reward_magnitudes": [MAGS, MAGS]}
    carried = {"good": 5, "bad": 0}
    run = _reconcile_run(frame(start, end), tv, None, carried, trial_offset=10, run_index=0)
    assert run.n_trials == 2
    assert run.corrected["good"] == [5, 6]
    assert run.added == {"good": 2, "bad": 0}
    assert run.events == [dict(kind="good", index=12, count=1, source="run_end")]
    assert run.observed == {"good": True, "bad": False}
