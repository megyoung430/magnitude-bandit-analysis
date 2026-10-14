import numpy as np
import pytest

from src.behavior_analysis.single_session import (
    choice_counts_by_tower,
    chosen_magnitudes,
    event_indices_from_cumulative,
    has_ema_traces,
    latest_rank_counts,
    ordered_rank_counts,
    percent_of_total,
    reversal_event_indices,
    choice_probability_matrix,
)

TOWERS = ["A1", "B1", "C1"]


# ---------------------------------------------------------------------------
# event_indices_from_cumulative / reversal_event_indices
# ---------------------------------------------------------------------------


def test_event_indices_marks_trials_where_counter_increments():
    assert event_indices_from_cumulative([0, 0, 1, 1, 2, 2]) == [2, 4]


@pytest.mark.parametrize("counter", [None, [], [3]])
def test_event_indices_empty_for_missing_or_too_short_counter(counter):
    assert event_indices_from_cumulative(counter) == []


def test_event_indices_truncates_to_n_and_ignores_decrements():
    assert event_indices_from_cumulative([0, 1, 2, 3, 4], N=3) == [1, 2]
    assert event_indices_from_cumulative([2, 1, 0, 3]) == [3]
    assert event_indices_from_cumulative([0, 1], N=1) == []


def test_reversal_event_indices_reads_cumulative_counters():
    session = {
        "good_reversals": [0, 1, 1, 1],
        "bad_reversals": [0, 0, 0, 1],
        "blocks": [1, 1, 2, 2],
    }
    assert reversal_event_indices(session, N=4) == ([1], [3], [2])


def test_reversal_event_indices_missing_keys_give_empty_lists():
    assert reversal_event_indices({}, N=5) == ([], [], [])


def test_reversal_event_indices_merges_boundary_events_without_duplicates():
    session = {
        "good_reversals": [0, 1, 1, 1],
        "reversal_boundary_events": [
            {"kind": "good", "index": 1},  # already present
            {"kind": "good", "index": 3},
            {"kind": "bad", "index": 2},
            {"kind": "bad", "index": 2},  # duplicate event
        ],
    }
    good, bad, blocks = reversal_event_indices(session, N=4)
    assert good == [1, 3]
    assert bad == [2]
    assert blocks == []


def test_reversal_event_indices_does_not_mutate_session():
    session = {"good_reversals": [0, 1], "reversal_boundary_events": [{"kind": "good", "index": 5}]}
    reversal_event_indices(session, N=2)
    assert session["reversal_boundary_events"] == [{"kind": "good", "index": 5}]


def test_reversal_event_indices_boundary_event_missing_index_raises():
    with pytest.raises(KeyError):
        reversal_event_indices({"reversal_boundary_events": [{"kind": "good"}]}, N=3)


# ---------------------------------------------------------------------------
# chosen_magnitudes / choice_counts_by_tower
# ---------------------------------------------------------------------------


def test_chosen_magnitudes_picks_magnitude_of_chosen_tower():
    mags = {"A1": [1, 4, 0], "B1": [4, 0, 1], "C1": [0, 1, 4]}
    choices = {"A1": [1, 0, 0], "B1": [0, 1, 0], "C1": [0, 0, 1]}
    np.testing.assert_array_equal(chosen_magnitudes(mags, choices, towers=TOWERS, N=3), [1, 0, 4])


def test_chosen_magnitudes_nan_when_no_choice_and_first_tower_wins_ties():
    mags = {"A1": [1, 1, 1], "B1": [4, 4, 4], "C1": [0, 0, 0]}
    choices = {"A1": [0, 0, 1], "B1": [0, 0, 1], "C1": [0, 0, 0]}
    result = chosen_magnitudes(mags, choices, towers=TOWERS, N=3)
    assert np.isnan(result[:2]).all()
    assert result[2] == 1


def test_chosen_magnitudes_short_choice_list_leaves_trailing_nan():
    mags = {"A1": [1, 1, 1]}
    choices = {"A1": [1]}
    result = chosen_magnitudes(mags, choices, towers=["A1"], N=3)
    assert result[0] == 1
    assert np.isnan(result[1:]).all()


def test_chosen_magnitudes_single_trial_without_choice():
    result = chosen_magnitudes({"A1": [4]}, {"A1": [0]}, towers=["A1"], N=1)
    assert result.shape == (1,) and np.isnan(result).all()


def test_choice_counts_by_tower_sums_each_tower():
    choices = {"A1": [1, 0, 1], "B1": [0, 1, 0], "C1": [0, 0, 0]}
    assert choice_counts_by_tower(choices, TOWERS, 3) == [2, 1, 0]


def test_choice_counts_by_tower_truncates_to_n_and_ignores_nan():
    choices = {"A1": [1, 1, 1, 1], "B1": [np.nan, 1, 0, 0]}
    assert choice_counts_by_tower(choices, ["A1", "B1"], 2) == [2, 1]


def test_choice_counts_by_tower_missing_tower_raises():
    with pytest.raises(KeyError):
        choice_counts_by_tower({"A1": [1]}, ["A1", "B1"], 1)


# ---------------------------------------------------------------------------
# has_ema_traces
# ---------------------------------------------------------------------------

EMA_KEYS = ("ema_best_arm_choices", "ema_second_arm_choices")


def test_has_ema_traces_true_when_any_value_present():
    assert has_ema_traces({"ema_best_arm_choices": [None, 0.5]}, EMA_KEYS)


@pytest.mark.parametrize(
    "session",
    [
        {},
        {"ema_best_arm_choices": None},
        {"ema_best_arm_choices": [None, None]},
        {"ema_best_arm_choices": []},
    ],
)
def test_has_ema_traces_false_when_absent_or_all_none(session):
    assert not has_ema_traces(session, EMA_KEYS)


def test_has_ema_traces_false_for_empty_key_tuple():
    assert not has_ema_traces({"ema_best_arm_choices": [0.1]}, ())


# ---------------------------------------------------------------------------
# choice_probability_matrix
# ---------------------------------------------------------------------------


def good_probs(n=4):
    p = np.full((n, 3), 1 / 3)
    return {t: p[:, j].tolist() for j, t in enumerate(TOWERS)}


def test_choice_probability_matrix_returns_trial_by_tower_matrix():
    probs = {"A1": [0.2, 1.0], "B1": [0.3, 0.0], "C1": [0.5, 0.0]}
    result = choice_probability_matrix(probs, TOWERS, 2)
    np.testing.assert_allclose(result, [[0.2, 0.3, 0.5], [1.0, 0.0, 0.0]])


def test_choice_probability_matrix_follows_tower_order_not_dict_order():
    probs = {"C1": [0.5], "A1": [0.2], "B1": [0.3]}
    np.testing.assert_allclose(choice_probability_matrix(probs, TOWERS, 1), [[0.2, 0.3, 0.5]])


def test_choice_probability_matrix_wrong_tower_labels():
    probs = good_probs()
    probs["X"] = probs.pop("A1")
    with pytest.raises(ValueError, match="physical tower labels"):
        choice_probability_matrix(probs, TOWERS, 4)


def test_choice_probability_matrix_wrong_length():
    with pytest.raises(ValueError):
        choice_probability_matrix(good_probs(3), TOWERS, 4)


def test_choice_probability_matrix_not_normalised():
    probs = good_probs()
    probs["A1"] = [0.9] * 4
    with pytest.raises(ValueError, match="normalized"):
        choice_probability_matrix(probs, TOWERS, 4)


def test_choice_probability_matrix_nan_and_negative():
    nan_probs = good_probs()
    nan_probs["A1"][0] = np.nan
    with pytest.raises(ValueError, match="normalized"):
        choice_probability_matrix(nan_probs, TOWERS, 4)
    negative = {"A1": [-0.1], "B1": [0.6], "C1": [0.5]}
    with pytest.raises(ValueError, match="normalized"):
        choice_probability_matrix(negative, TOWERS, 1)


# ---------------------------------------------------------------------------
# latest_rank_counts / ordered_rank_counts / percent_of_total
# ---------------------------------------------------------------------------


def test_latest_rank_counts_returns_last_entry():
    session = {"rank_counts": [{"best": 1}, {"best": 3, "second": 2}]}
    assert latest_rank_counts(session) == {"best": 3, "second": 2}


@pytest.mark.parametrize(
    "session",
    [{}, {"rank_counts": []}, {"rank_counts": None}, {"rank_counts": [{"best": 1}, [1, 2]]}],
)
def test_latest_rank_counts_none_when_missing_empty_or_not_a_dict(session):
    assert latest_rank_counts(session) is None


def test_latest_rank_counts_empty_dict_is_returned_as_is():
    assert latest_rank_counts({"rank_counts": [{}]}) == {}


def test_ordered_rank_counts_puts_standard_ranks_first():
    keys, vals = ordered_rank_counts({"extra": 2, "third": 5, "best": 10, "second": 7})
    assert keys == ["best", "second", "third", "extra"]
    assert vals == [10, 7, 5, 2]


def test_ordered_rank_counts_skips_absent_standard_ranks_and_casts_to_int():
    keys, vals = ordered_rank_counts({"zzz": 3.0, "third": 1})
    assert keys == ["third", "zzz"]
    assert vals == [1, 3] and all(isinstance(v, int) for v in vals)


def test_percent_of_total_sums_to_100():
    assert percent_of_total([1, 1, 2]) == [25.0, 25.0, 50.0]


def test_percent_of_total_all_zero_does_not_divide_by_zero():
    assert percent_of_total([0, 0, 0]) == [0.0, 0.0, 0.0]


def test_percent_of_total_empty():
    assert percent_of_total([]) == []
