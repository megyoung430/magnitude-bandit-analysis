"""Unit tests for the GLM design-matrix builders, on tiny synthetic sessions."""

import numpy as np
import pytest

from src.behavior_analysis import glm_utils as g

NAN = np.nan


def session(choices, rewards):
    """One session dict from per-trial chosen towers and per-trial {tower: magnitude}."""
    towers = list(rewards[0])
    n = len(choices)
    return {
        "trial": list(range(1, n + 1)),
        "blocks": [1] * n,
        "trials_in_block": list(range(1, n + 1)),
        "choices_by_tower": {t: [int(c == t) for c in choices] for t in towers},
        "reward_magnitudes_by_tower": {t: [r[t] for r in rewards] for t in towers},
        "choices_by_rank": {},
    }


def subjects(choices, rewards):
    return {"m": {"ses-1": session(choices, rewards)}}


# ---- shared helpers -------------------------------------------------------


def test_design_labels_follow_column_layout():
    labels = g._design_labels(["c1", "c2", "c3"], ["i1", "i2", "i3"], n_lags=2)
    assert labels == (
        ["Bias"]
        + ["c1 t-1", "c2 t-1", "c3 t-1", "c1 t-2", "c2 t-2", "c3 t-2"]
        + ["Reward t-1", "Reward t-2"]
        + ["i1 t-1", "i2 t-1", "i3 t-1", "i1 t-2", "i2 t-2", "i3 t-2"]
    )


def test_keep_mask_drops_labels_containing_any_pattern():
    mask = g._keep_mask(["Bias", "x Arm3", "y"], ("Arm3",))
    assert mask.tolist() == [True, False, True]


def test_lagged_shifts_down_and_fills_leading_rows():
    values = np.array([[1.0], [2.0], [3.0]])
    assert g._lagged(values, 1, NAN)[:, 0].tolist() == [pytest.approx(NAN, nan_ok=True), 1.0, 2.0]
    assert g._lagged(values, 2, 0.0)[:, 0].tolist() == [0.0, 0.0, 1.0]


@pytest.mark.parametrize("lag", [3, 10])
def test_lagged_with_lag_beyond_length_is_all_fill(lag):
    assert np.all(g._lagged(np.ones(3), lag, 0.0) == 0.0)


def test_design_matrix_hand_computed_one_lag():
    choices = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    rewards = np.array([2.0, 3.0, 4.0])
    x = g._design_matrix(choices, rewards, n_lags=1, fill=0.0)
    expected = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 2, 2, 0, 0],
            [1, 0, 1, 0, 3, 0, 3, 0],
        ]
    )
    np.testing.assert_array_equal(x, expected)


def test_design_matrix_zero_lags_is_bias_only():
    x = g._design_matrix(np.ones((4, 3)), np.ones(4), n_lags=0, fill=NAN)
    assert x.shape == (4, 1)


def test_design_matrix_of_no_trials_has_full_width():
    x = g._design_matrix(np.empty((0, 3)), np.empty(0), n_lags=2, fill=NAN)
    assert x.shape == (0, 1 + 7 * 2)


# ---- rank helpers ---------------------------------------------------------


def test_ranks_from_magnitudes_breaks_ties_by_tower_key():
    ranks = g._ranks_from_magnitudes({"C": 3.0, "A": 3.0, "B": 1.0})
    assert ranks == {"A": "best", "C": "second", "B": "third"}


def test_ranks_beyond_third_are_labelled_third():
    ranks = g._ranks_from_magnitudes({"A": 4, "B": 3, "C": 2, "D": 1})
    assert ranks["D"] == "third"


def test_chosen_tower_per_trial_is_none_unless_exactly_one():
    choices = {"A": [1, 0, 1, 0], "B": [0, 1, 1, 0]}
    assert g._chosen_tower_per_trial(choices, ["A", "B"], 4) == ["A", "B", None, None]


def test_chosen_rewards_is_nan_without_a_chosen_tower():
    rewards = {"A": [5.0, 6.0], "B": [7.0, 8.0]}
    out = g._chosen_rewards(rewards, ["B", None])
    assert out[0] == 7.0 and np.isnan(out[1])


def test_chose_rank_indicators_one_hot_and_nan():
    ranks = [{"A": "best", "B": "second"}, None, {"A": "best", "B": "second"}]
    out = g._chose_rank_indicators(["B", "A", None], ranks)
    np.testing.assert_array_equal(out[0], [0, 1, 0])
    assert np.isnan(out[1]).all() and np.isnan(out[2]).all()


def test_chose_rank_indicators_of_no_trials_is_empty_three_column():
    assert g._chose_rank_indicators([], []).shape == (0, 3)


def test_complete_history_rows_skips_first_trial_and_nan_rows():
    x = np.array([[1.0, 1], [1, NAN], [1, 1], [1, 1]])
    assert g._complete_history_rows(x).tolist() == [2, 3]


# ---- format_glm_input -----------------------------------------------------


def test_format_glm_input_values_and_labels():
    rewards = [{"A": 1.0, "B": 2.0, "C": 3.0}] * 3
    x, y, labels = g.format_glm_input(subjects(["A", "B", "C"], rewards), n_lags=1)
    assert labels == [
        "Bias", "Choice Arm1 t-1", "Choice Arm2 t-1", "Reward t-1",
        "Chx Rew Arm1 t-1", "Chx Rew Arm2 t-1",
    ]  # fmt: skip
    # Trial 0 has no history and is dropped; Arm3 columns are removed.
    np.testing.assert_array_equal(x["m"], [[1, 1, 0, 1.0, 1.0, 0], [1, 0, 1, 2.0, 0, 2.0]])
    np.testing.assert_array_equal(y["m"], [[0, 1, 0], [0, 0, 1]])


def test_format_glm_input_keeps_arm3_columns_when_asked():
    rewards = [{"A": 1.0, "B": 2.0, "C": 3.0}] * 3
    x, _, _ = g.format_glm_input(subjects(["A", "B", "C"], rewards), 2, drop_redundant_cols=False)
    assert x["m"].shape == (2, 1 + 7 * 2)


def test_format_glm_input_pins_labels_to_dropped_set_even_when_columns_kept():
    # Existing behaviour: the labels always omit Arm3 while the matrix may keep it.
    rewards = [{"A": 1.0, "B": 2.0, "C": 3.0}] * 3
    x, _, labels = g.format_glm_input(
        subjects(["A", "B", "C"], rewards), 1, drop_redundant_cols=False
    )
    assert len(labels) == 6 and x["m"].shape[1] == 8


def test_format_glm_input_session_with_one_trial_is_empty():
    x, y, _ = g.format_glm_input(subjects(["A"], [{"A": 1.0, "B": 2.0, "C": 3.0}]), n_lags=2)
    assert x["m"].shape == (0, 1 + 5 * 2) and y["m"].shape == (0, 3)


def test_format_glm_input_rejects_other_than_three_arms():
    rewards = [{"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}] * 3
    with pytest.raises(ValueError, match="Expected 3"):
        g.format_glm_input(subjects(["A", "B", "C"], rewards))


def test_format_glm_input_rejects_trial_without_single_choice():
    data = subjects(["A", "B", "C"], [{"A": 1.0, "B": 2.0, "C": 3.0}] * 3)
    data["m"]["ses-1"]["choices_by_tower"]["A"] = [1, 1, 0]
    with pytest.raises(ValueError, match="exactly one"):
        g.format_glm_input(data)


# ---- format_glm_input_by_rank --------------------------------------------


def rank_data():
    rewards = [{"A": 3.0, "B": 2.0, "C": 1.0}] * 4
    return subjects(["A", "B", "C", "A"], rewards)


def test_by_rank_true_rank_values_and_shapes():
    x, y, labels = g.format_glm_input_by_rank(rank_data(), n_lags=1)
    assert labels == [
        "Bias", "Chose Best t-1", "Chose Second t-1", "Reward t-1",
        "Best*Reward t-1", "Second*Reward t-1",
    ]  # fmt: skip
    np.testing.assert_array_equal(
        x["m"], [[1, 1, 0, 3.0, 3.0, 0], [1, 0, 1, 2.0, 0, 2.0], [1, 0, 0, 1.0, 0, 0]]
    )
    np.testing.assert_array_equal(y["m"], [[0, 1, 0], [0, 0, 1], [1, 0, 0]])


def test_by_rank_drops_first_n_lags_trials_for_missing_history():
    x, _, _ = g.format_glm_input_by_rank(rank_data(), n_lags=3)
    assert x["m"].shape[0] == 1


def test_by_rank_believed_waits_until_every_arm_was_visited():
    rewards = [{"A": 3.0, "B": 2.0, "C": 1.0}] * 5
    data = subjects(["A", "B", "C", "A", "B"], rewards)
    x, y, _ = g.format_glm_input_by_rank(data, use_believed=True, n_lags=1)
    # Beliefs exist from trial 3 on, so only trial 4 has a believed rank at both t and t-1.
    np.testing.assert_array_equal(x["m"], [[1, 1, 0, 3.0, 3.0, 0]])
    np.testing.assert_array_equal(y["m"], [[0, 1, 0]])


def test_by_rank_unchosen_trial_poisons_only_the_rows_that_use_it_as_history():
    data = rank_data()
    for tower in "ABC":
        data["m"]["ses-1"]["choices_by_tower"][tower][1] = 0
    x, y, _ = g.format_glm_input_by_rank(data, n_lags=1)
    # Row 2 uses trial 1 as history and is dropped. Row 1 stays although its own
    # outcome is unknown, so its true choice is NaN.
    assert x["m"].shape[0] == 2
    assert np.isnan(y["m"][0]).all() and y["m"][1].tolist() == [1.0, 0.0, 0.0]


def test_by_rank_keeps_all_columns_when_asked():
    x, _, labels = g.format_glm_input_by_rank(rank_data(), n_lags=1, drop_redundant_cols=False)
    assert x["m"].shape[1] == len(labels) == 1 + 7


def test_by_rank_zero_lags_keeps_rows_with_unavailable_beliefs():
    # Without history columns nothing is NaN, so the believed-rank NaN outcome rows survive.
    x, y, _ = g.format_glm_input_by_rank(rank_data(), use_believed=True, n_lags=0)
    assert x["m"].shape == (3, 1) and np.isnan(y["m"]).any()


def test_by_rank_without_towers_fails_loudly():
    data = {"m": {"ses-1": {"trial": [1], "blocks": [1], "choices_by_tower": {},
                            "reward_magnitudes_by_tower": {}, "choices_by_rank": {}}}}  # fmt: skip
    with pytest.raises(ValueError):
        g.format_glm_input_by_rank(data)
