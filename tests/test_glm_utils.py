"""Unit tests for the GLM design-matrix builders, on tiny synthetic sessions."""

import numpy as np
import pytest

from src.behavior_analysis import glm_utils as g

NAN = np.nan
CONST = {"A": 1.0, "B": 2.0, "C": 3.0}


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


def constant_subjects(choices, magnitudes=CONST):
    return subjects(choices, [magnitudes] * len(choices))


def columns(labels, row):
    return dict(zip(labels, row, strict=True))


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
    mask = g._keep_mask(["Bias", "x Arm3", "y"], redundant_patterns=("Arm3",))
    assert mask.tolist() == [True, False, True]


def test_lagged_shifts_down_and_fills_leading_rows():
    values = np.array([[1.0], [2.0], [3.0]])
    assert g._lagged(values, lag=1, fill=NAN)[:, 0].tolist() == [
        pytest.approx(NAN, nan_ok=True),
        1.0,
        2.0,
    ]
    assert g._lagged(values, lag=2, fill=0.0)[:, 0].tolist() == [0.0, 0.0, 1.0]


@pytest.mark.parametrize("lag", [3, 10])
def test_lagged_with_lag_beyond_length_is_all_fill(lag):
    assert np.all(g._lagged(np.ones(3), lag=lag, fill=0.0) == 0.0)


ONE_HOT = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def test_history_terms_multiply_each_reward_into_its_choice_row():
    rewards, interactions = g._history_terms(ONE_HOT, [2.0, 3.0, 4.0], n_lags=1)
    # The last trial is never part of any row's history, so its entries are NaN padding.
    np.testing.assert_array_equal(rewards[:2], [2.0, 3.0])
    np.testing.assert_array_equal(interactions[:2], [[2, 0, 0], [0, 3, 0]])
    assert np.isnan(rewards[2]) and np.isnan(interactions[2]).all()


def test_history_terms_ignore_a_missing_reward_on_the_last_trial():
    rewards, _ = g._history_terms(ONE_HOT, [2.0, 3.0, None], n_lags=1)
    assert rewards[:2].tolist() == [2.0, 3.0]


def test_history_terms_raise_on_a_missing_reward_that_enters_history():
    with pytest.raises(
        TypeError, match=r"unsupported operand type\(s\) for \*: 'int' and 'NoneType'"
    ):
        g._history_terms(ONE_HOT, [2.0, None, 4.0], n_lags=1)


def test_history_terms_do_not_read_rewards_without_lags():
    g._history_terms(ONE_HOT, [None, None, None], n_lags=0)


def test_history_terms_raise_when_rewards_are_too_short():
    with pytest.raises(IndexError):
        g._history_terms(ONE_HOT, [2.0], n_lags=1)


def test_design_matrix_hand_computed_one_lag():
    choices = np.array(ONE_HOT, dtype=float)
    rewards, interactions = g._history_terms(ONE_HOT, [2.0, 3.0, 4.0], n_lags=1)
    x = g._design_matrix(choices, rewards, interactions, n_lags=1, fill=0.0)
    expected = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 2, 2, 0, 0],
            [1, 0, 1, 0, 3, 0, 3, 0],
        ]
    )
    np.testing.assert_array_equal(x, expected)


def test_design_matrix_zero_lags_is_bias_only():
    x = g._design_matrix(np.ones((4, 3)), np.ones(4), np.ones((4, 3)), n_lags=0, fill=NAN)
    assert x.shape == (4, 1)


def test_design_matrix_of_no_trials_has_full_width():
    x = g._design_matrix(np.empty((0, 3)), np.empty(0), np.empty((0, 3)), n_lags=2, fill=NAN)
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
    assert g._chosen_tower_per_trial(choices, ["A", "B"], n_trials=4) == ["A", "B", None, None]


def test_chosen_rewards_is_nan_without_a_chosen_tower():
    rewards = {"A": [5.0, 6.0], "B": [7.0, 8.0]}
    out = g._chosen_rewards(rewards, ["B", None])
    assert out[0] == 7.0 and np.isnan(out[1])


def test_chosen_rewards_keeps_a_missing_reward_as_none():
    assert g._chosen_rewards({"A": [None, 6.0]}, ["A", "A"]) == [None, 6.0]


def test_chose_rank_indicators_one_hot_and_nan():
    ranks = [{"A": "best", "B": "second"}, None, {"A": "best", "B": "second"}]
    out = np.array(g._chose_rank_indicators(["B", "A", None], ranks), dtype=float)
    np.testing.assert_array_equal(out[0], [0, 1, 0])
    assert np.isnan(out[1]).all() and np.isnan(out[2]).all()


def test_chose_rank_indicators_of_no_trials_is_empty():
    assert g._chose_rank_indicators([], []) == []


def test_complete_history_rows_skips_first_trial_and_nan_rows():
    x = np.array([[1.0, 1], [1, NAN], [1, 1], [1, 1]])
    assert g._complete_history_rows(x).tolist() == [2, 3]


# ---- format_glm_input -----------------------------------------------------


def test_format_glm_input_values_and_labels():
    x, y, labels = g.format_glm_input(constant_subjects(["A", "B", "C"]), n_lags=1)
    assert labels == [
        "Bias", "Choice Arm1 t-1", "Choice Arm2 t-1", "Reward t-1",
        "Chx Rew Arm1 t-1", "Chx Rew Arm2 t-1",
    ]  # fmt: skip
    # Trial 0 has no history and is dropped; Arm3 columns are removed.
    np.testing.assert_array_equal(x["m"], [[1, 1, 0, 1.0, 1.0, 0], [1, 0, 1, 2.0, 0, 2.0]])
    np.testing.assert_array_equal(y["m"], [[0, 1, 0], [0, 0, 1]])


def test_format_glm_input_keeps_arm3_columns_when_asked():
    data = constant_subjects(["A", "B", "C"])
    x, _, _ = g.format_glm_input(data, 2, drop_redundant_cols=False)
    assert x["m"].shape == (2, 1 + 7 * 2)


def test_format_glm_input_labels_omit_arm3_even_when_columns_are_kept():
    # Known issue: the labels always omit Arm3 while the matrix may keep it.
    data = constant_subjects(["A", "B", "C"])
    x, _, labels = g.format_glm_input(data, 1, drop_redundant_cols=False)
    assert len(labels) == 6 and x["m"].shape[1] == 8


def test_format_glm_input_session_with_one_trial_is_empty():
    x, y, _ = g.format_glm_input(constant_subjects(["A"]), n_lags=2)
    assert x["m"].shape == (0, 1 + 5 * 2) and y["m"].shape == (0, 3)


def test_format_glm_input_missing_history_before_the_first_trial_is_zero():
    # Four trials, three lags: rows for trials 1..3 reach back before trial 0.
    x, _, labels = g.format_glm_input(constant_subjects(["A", "B", "C", "A"]), n_lags=3)
    assert x["m"].shape == (3, 1 + 5 * 3) and not np.isnan(x["m"]).any()
    first = columns(labels, x["m"][0])
    assert first["Choice Arm1 t-1"] == 1 and first["Reward t-1"] == 1.0
    for lag in (2, 3):
        assert first[f"Choice Arm1 t-{lag}"] == 0 and first[f"Reward t-{lag}"] == 0
    assert columns(labels, x["m"][2])["Choice Arm1 t-3"] == 1


def test_format_glm_input_with_more_lags_than_trials_is_still_a_full_matrix():
    x, _, _ = g.format_glm_input(constant_subjects(["A", "B", "C", "A"]), n_lags=10)
    assert x["m"].shape == (3, 1 + 5 * 10) and not np.isnan(x["m"]).any()


def test_format_glm_input_orders_arms_by_sorted_tower_key():
    magnitudes = {"Z": 1.0, "B": 2.0, "M": 3.0}  # arms: B=Arm1, M=Arm2, Z=Arm3
    data = constant_subjects(["Z", "B", "M", "B"], magnitudes)
    x, y, _ = g.format_glm_input(data, n_lags=1)
    np.testing.assert_array_equal(
        x["m"],
        [[1, 0, 0, 1.0, 0, 0], [1, 1, 0, 2.0, 2.0, 0], [1, 0, 1, 3.0, 0, 3.0]],
    )
    np.testing.assert_array_equal(y["m"], [[1, 0, 0], [0, 1, 0], [1, 0, 0]])


def two_session_subject():
    """ses-2 (A, B, C) is merged before ses-10 (C, A, B)."""
    return {
        "m": {
            "ses-10": session(["C", "A", "B"], [CONST] * 3),
            "ses-2": session(["A", "B", "C"], [CONST] * 3),
        }
    }


def test_format_glm_input_lags_cross_session_boundaries():
    # Known issue: sessions are concatenated, so the first trial of the second
    # session sees the last trial of the first (trial 3: chose C, reward 3) as history.
    x, y, _ = g.format_glm_input(two_session_subject(), n_lags=1)
    np.testing.assert_array_equal(x["m"][2], [1, 0, 0, 3.0, 0, 0])
    np.testing.assert_array_equal(y["m"][2], [0, 0, 1])
    assert x["m"].shape[0] == 5


def test_format_glm_input_rejects_trial_without_single_choice():
    data = constant_subjects(["A", "B", "C"])
    data["m"]["ses-1"]["choices_by_tower"]["A"] = [1, 1, 0]
    with pytest.raises(ValueError, match="exactly one"):
        g.format_glm_input(data)


# ---- format_glm_input: malformed input fails as it always has --------------


def test_format_glm_input_missing_chosen_reward_raises_type_error():
    # A missing magnitude must not turn into a NaN regressor.
    data = constant_subjects(["A", "B", "C", "A"])
    data["m"]["ses-1"]["reward_magnitudes_by_tower"]["A"][0] = None
    with pytest.raises(
        TypeError, match=r"unsupported operand type\(s\) for \*: 'int' and 'NoneType'"
    ):
        g.format_glm_input(data, n_lags=1)


def test_format_glm_input_missing_reward_is_harmless_without_lags_or_on_last_trial():
    data = constant_subjects(["A", "B", "C", "A"])
    data["m"]["ses-1"]["reward_magnitudes_by_tower"]["A"][0] = None
    x, _, _ = g.format_glm_input(data, n_lags=0)
    assert x["m"].shape == (3, 1)
    data = constant_subjects(["A", "B", "C", "A"])
    data["m"]["ses-1"]["reward_magnitudes_by_tower"]["A"][3] = None
    x, _, _ = g.format_glm_input(data, n_lags=1)
    assert x["m"].shape[0] == 3


def test_format_glm_input_missing_reward_on_unchosen_tower_is_ignored():
    data = constant_subjects(["A", "B", "C", "A"])
    data["m"]["ses-1"]["reward_magnitudes_by_tower"]["C"][0] = None
    x, _, _ = g.format_glm_input(data, n_lags=1)
    assert not np.isnan(x["m"]).any()


def test_format_glm_input_reward_list_one_short_returns_the_matrix():
    data = constant_subjects(["A", "B", "C", "A"])
    expected, _, _ = g.format_glm_input(data, n_lags=2)
    for rewards in data["m"]["ses-1"]["reward_magnitudes_by_tower"].values():
        rewards.pop()  # the last reward is never part of any history
    x, _, _ = g.format_glm_input(data, n_lags=2)
    np.testing.assert_array_equal(x["m"], expected["m"])


def test_format_glm_input_short_reward_list_in_middle_session_shifts_later_rewards():
    # Known issue: sessions are merged by concatenation, so one missing reward in
    # an early session moves every later reward one trial earlier. Merged trial 2
    # (ses-2, chose C, reward 3) therefore receives the reward 30 of ses-3's first
    # trial.
    big = {t: 10 * v for t, v in CONST.items()}
    data = {"m": {"ses-2": session(["A", "B", "C"], [CONST] * 3),
                  "ses-3": session(["A", "B", "C"], [big] * 3)}}  # fmt: skip
    for rewards in data["m"]["ses-2"]["reward_magnitudes_by_tower"].values():
        rewards.pop()
    x, _, labels = g.format_glm_input(data, n_lags=1)
    reward_col = labels.index("Reward t-1")
    assert x["m"][:, reward_col].tolist() == [1.0, 2.0, 30.0, 10.0, 20.0]


def test_format_glm_input_four_towers_fail_with_a_broadcast_error():
    # Known issue: only exactly three towers are supported; other counts are not checked.
    towers = {**CONST, "D": 4.0}
    with pytest.raises(
        ValueError, match=r"could not broadcast input array from shape \(3,\) into shape \(4,\)"
    ):
        g.format_glm_input(constant_subjects(["A", "B", "C"], towers))


def test_format_glm_input_two_towers_fail_with_a_key_error():
    with pytest.raises(KeyError, match="Arm3"):
        g.format_glm_input(constant_subjects(["A", "B", "A"], {"A": 1.0, "B": 2.0}))


def test_format_glm_input_choice_series_shorter_than_trials_raises_index_error():
    data = constant_subjects(["A", "B", "C", "A"])
    for choices in data["m"]["ses-1"]["choices_by_tower"].values():
        choices.pop()
    with pytest.raises(IndexError, match="list index out of range"):
        g.format_glm_input(data, n_lags=1)


# ---- format_glm_input_by_rank --------------------------------------------


def rank_data():
    return constant_subjects(["A", "B", "C", "A"], {"A": 3.0, "B": 2.0, "C": 1.0})


def drifting_data():
    """Magnitudes that change every trial, so believed and true ranks disagree."""
    rewards = [
        {"A": 1.0, "B": 2.0, "C": 3.0},
        {"A": 3.0, "B": 1.0, "C": 2.0},
        {"A": 2.0, "B": 3.0, "C": 1.0},
        {"A": 3.0, "B": 2.0, "C": 1.0},
        {"A": 1.0, "B": 3.0, "C": 2.0},
        {"A": 2.0, "B": 1.0, "C": 3.0},
    ]
    return subjects(["A", "B", "C", "A", "B", "C"], rewards)


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


def test_by_rank_drops_trials_whose_history_reaches_before_the_first_trial():
    x, _, _ = g.format_glm_input_by_rank(rank_data(), n_lags=3)
    assert x["m"].shape[0] == 1 and not np.isnan(x["m"]).any()


def test_by_rank_with_more_lags_than_trials_has_no_rows():
    x, y, _ = g.format_glm_input_by_rank(rank_data(), n_lags=10)
    assert x["m"].shape == (0, 1 + 5 * 10) and y["m"].shape == (0, 3)


def test_by_rank_believed_waits_until_every_arm_was_visited():
    rewards = [{"A": 3.0, "B": 2.0, "C": 1.0}] * 5
    data = subjects(["A", "B", "C", "A", "B"], rewards)
    x, y, _ = g.format_glm_input_by_rank(data, use_believed=True, n_lags=1)
    # Beliefs exist from trial 3 on, so only trial 4 has a believed rank at both t and t-1.
    np.testing.assert_array_equal(x["m"], [[1, 1, 0, 3.0, 3.0, 0]])
    np.testing.assert_array_equal(y["m"], [[0, 1, 0]])


def test_by_rank_believed_rank_differs_from_true_rank_when_magnitudes_drift():
    # Believed ranks use the last magnitude seen at each tower (beliefs exist from
    # trial 3), true ranks use the current magnitudes.
    x_true, y_true, _ = g.format_glm_input_by_rank(drifting_data(), n_lags=1)
    x_bel, y_bel, _ = g.format_glm_input_by_rank(drifting_data(), use_believed=True, n_lags=1)
    assert x_true["m"].shape[0] == 5 and x_bel["m"].shape[0] == 2
    # Trials 4 and 5 chose B (true best, believed second) and C (true best, believed third).
    np.testing.assert_array_equal(y_true["m"][-2:], [[1, 0, 0], [1, 0, 0]])
    np.testing.assert_array_equal(y_bel["m"], [[0, 1, 0], [0, 0, 1]])
    # History at trial 5 is trial 4 (chose B, reward 3): best vs second.
    np.testing.assert_array_equal(x_true["m"][-1], [1, 1, 0, 3.0, 3.0, 0])
    np.testing.assert_array_equal(x_bel["m"][-1], [1, 0, 1, 3.0, 0, 3.0])


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
    # Known issue: without history columns nothing is NaN, so rows whose own
    # believed-rank outcome is NaN survive.
    x, y, _ = g.format_glm_input_by_rank(rank_data(), use_believed=True, n_lags=0)
    assert x["m"].shape == (3, 1) and np.isnan(y["m"]).any()


def test_by_rank_lags_and_beliefs_cross_session_boundaries():
    # Known issue: sessions are concatenated, so trial 3 (first of ses-10) has the
    # last trial of ses-2 as history, and believed ranks carry over: with beliefs
    # from ses-2, ses-10 has a believed rank from its first trial.
    x, y, _ = g.format_glm_input_by_rank(two_session_subject(), n_lags=1)
    assert x["m"].shape[0] == 5
    np.testing.assert_array_equal(x["m"][2], [1, 1, 0, 3.0, 3.0, 0])  # C is best
    np.testing.assert_array_equal(y["m"][2], [1, 0, 0])
    x, _, _ = g.format_glm_input_by_rank(two_session_subject(), use_believed=True, n_lags=1)
    assert x["m"].shape[0] == 2  # merged trials 4 and 5 are the first with beliefs


def test_by_rank_without_towers_fails_loudly():
    data = {"m": {"ses-1": {"trial": [1], "blocks": [1], "choices_by_tower": {},
                            "reward_magnitudes_by_tower": {}, "choices_by_rank": {}}}}  # fmt: skip
    with pytest.raises(ValueError):
        g.format_glm_input_by_rank(data)


def test_by_rank_missing_chosen_reward_raises_when_beliefs_never_form():
    # Three trials never visit every arm again, so no rank sorts the missing value
    # and the reward reaches the history product.
    data = constant_subjects(["A", "B", "C"], {"A": 3.0, "B": 2.0, "C": 1.0})
    data["m"]["ses-1"]["reward_magnitudes_by_tower"]["A"][0] = None
    with pytest.raises(
        TypeError, match=r"unsupported operand type\(s\) for \*: 'float' and 'NoneType'"
    ):
        g.format_glm_input_by_rank(data, use_believed=True, n_lags=1)


def test_by_rank_missing_reward_on_any_tower_fails_when_ranking_true_magnitudes():
    data = rank_data()
    data["m"]["ses-1"]["reward_magnitudes_by_tower"]["C"][0] = None
    with pytest.raises(TypeError, match=r"bad operand type for unary -: 'NoneType'"):
        g.format_glm_input_by_rank(data, n_lags=1)
