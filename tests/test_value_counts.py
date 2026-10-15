"""Tie handling, out-of-grid reporting and belief scope in the per-cell choice counts."""

import numpy as np
import pandas as pd
import pytest

from src.behavior_analysis.get_choice_probs_by_value import (
    accumulate_choice_probs,
    count_choice_probs,
    init_matrix,
)


def session(magnitudes, ranks, choices):
    return {
        "trial_variables": [
            {"reward_magnitudes": magnitudes, "chosen_rank": ranks, "choice": choices}
        ]
    }


def labelled_session(offer, choice_seq, ranks):
    return session([offer] * len(choice_seq), ranks, list(choice_seq))


def one_session(sess, subject="m1", problem=1):
    return {problem: {subject: {"ses-1": sess}}}


def assert_only_cells(df, expected):
    full = init_matrix(fill=np.nan)
    for (b, s), value in expected.items():
        full.loc[b, s] = value
    pd.testing.assert_frame_equal(df, full)


# --- ties split evenly among the tied arms -------------------------------------------------


@pytest.mark.parametrize("names", [("A", "B", "C"), ("C", "B", "A")])
def test_believed_second_third_tie_does_not_depend_on_arm_names(names):
    top, tie1, tie2 = names
    offer = {top: 4, tie1: 2, tie2: 2}
    seq = [top, tie1, tie2, tie1, tie1, tie1, tie1]  # only ever the first tied arm
    sess = labelled_session(offer, seq, ["best"] * len(seq))
    _, _, per_best, per_second = accumulate_choice_probs(one_session(sess), use_believed=True)
    # after all arms are seen, four trials choose one tied arm: half "second" whichever it is
    assert_only_cells(per_second["m1"], {(4, 2): 0.5})
    assert_only_cells(per_best["m1"], {(4, 2): 0.0})


def test_believed_three_way_tie_gives_each_arm_a_third():
    offer = {"A": 3, "B": 3, "C": 3}
    sess = labelled_session(offer, ["A", "B", "C", "A", "A", "A"], ["best"] * 6)
    _, _, per_best, per_second = accumulate_choice_probs(one_session(sess), use_believed=True)
    assert_only_cells(per_best["m1"], {(3, 3): 1 / 3})
    assert_only_cells(per_second["m1"], {(3, 3): 1 / 3})


def test_true_best_second_tie_ignores_the_arbitrary_rank_label():
    offer = {"A": 3, "B": 3, "C": 1}
    sess = labelled_session(offer, ["A", "A"], ["best", "best"])
    _, _, per_best, per_second = accumulate_choice_probs(one_session(sess))
    assert_only_cells(per_best["m1"], {(3, 3): 0.5})
    assert_only_cells(per_second["m1"], {(3, 3): 0.5})


def test_true_second_third_tie_splits_second_between_the_tied_arms():
    offer = {"A": 4, "B": 2, "C": 2}
    sess = labelled_session(offer, ["B", "B"], ["second", "second"])
    _, _, per_best, per_second = accumulate_choice_probs(one_session(sess))
    assert_only_cells(per_second["m1"], {(4, 2): 0.5})
    assert_only_cells(per_best["m1"], {(4, 2): 0.0})


def test_true_three_way_tie_gives_each_arm_a_third():
    offer = {"A": 3, "B": 3, "C": 3}
    sess = labelled_session(offer, ["A", "B", "C"], ["best", "best", "best"])
    _, _, per_best, per_second = accumulate_choice_probs(one_session(sess))
    assert_only_cells(per_best["m1"], {(3, 3): 1 / 3})
    assert_only_cells(per_second["m1"], {(3, 3): 1 / 3})


def test_true_tie_with_a_choice_not_among_the_arms_raises():
    sess = labelled_session({"A": 3, "B": 3, "C": 1}, ["Z"], ["best"])
    with pytest.raises(ValueError, match="'Z'"):
        accumulate_choice_probs(one_session(sess))


# --- out-of-grid trials are counted and reported -------------------------------------------


def test_out_of_grid_trials_are_counted_by_cell_and_warned_per_mouse(caplog):
    in_grid = {"A": 4, "B": 3, "C": 0}
    low_best = {"A": 1, "B": 1, "C": 0}  # best 1 is outside BEST_VALS
    zero_second = {"A": 4, "B": 0, "C": 0}  # second 0 is outside SECOND_VALS
    sess_a = session([in_grid, low_best, low_best, zero_second], ["best"] * 4, list("AAAA"))
    sess_b = session([in_grid], ["best"], ["A"])
    data = {1: {"m1": {"ses-1": sess_a}, "m2": {"ses-1": sess_b}}}
    with caplog.at_level("WARNING"):
        counts = count_choice_probs(data)
    assert counts["m1"].out_of_grid == {(1, 1): 2, (4, 0): 1}
    assert counts["m2"].out_of_grid == {}
    warned = [r.getMessage() for r in caplog.records]
    assert len(warned) == 1 and "m1" in warned[0] and "3 of 4" in warned[0]
    assert "outside" in warned[0]


def test_other_skipped_trials_are_counted_by_reason_without_warning(caplog):
    mags = {"A": 4, "B": 3, "C": 0}
    sess = session([mags, mags, {"A": 4}], ["weird", "best", "best"], list("AAA"))
    with caplog.at_level("WARNING"):
        counts = count_choice_probs(one_session(sess))
    assert counts["m1"].dropped == {"unrecognised_rank": 1, "too_few_arms": 1}
    assert counts["m1"].denom.loc[4, 3] == 1
    assert caplog.records == []


# --- believed values carry across a problem's sessions and reset per problem ---------------


def test_believed_values_carry_across_sessions_of_a_problem_in_session_order():
    mags = {"A": 4, "B": 3, "C": 0}
    first = session([mags] * 3, ["best"] * 3, ["A", "B", "C"])
    second = session([mags] * 2, ["best"] * 2, ["A", "B"])
    # session 2 is listed first: dict order must not matter
    data = {1: {"m1": {"ses-2_date-20260102": second, "ses-1_date-20260101": first}}}
    counts = count_choice_probs(data, use_believed=True)
    # all three arms were seen before session 2, so both its trials are binned
    assert counts["m1"].denom.loc[4, 3] == 2


def test_believed_values_reset_at_each_problem_boundary():
    mags = {"A": 4, "B": 3, "C": 0}
    seen = session([mags] * 3, ["best"] * 3, ["A", "B", "C"])
    fresh = session([mags] * 2, ["best"] * 2, ["A", "B"])
    data = {1: {"m1": {"ses-1": seen}}, 2: {"m1": {"ses-1": fresh}}}
    counts = count_choice_probs(data, use_believed=True)
    assert counts["m1"].denom.to_numpy().sum() == 0  # problem 2 starts with nothing seen
