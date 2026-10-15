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
    labels = g._design_labels(["c1", "c2", "c3"], interaction_terms=["i1", "i2", "i3"], n_lags=2)
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


def test_history_terms_do_not_read_rewards_with_negative_lags():
    g._history_terms(ONE_HOT, [None, None, None], n_lags=-1)


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


def test_positions_in_session_restart_at_every_session():
    assert g._positions_in_session([3, 2], 5).tolist() == [0, 1, 2, 0, 1]


def test_positions_in_session_continue_in_the_last_session_past_its_length():
    assert g._positions_in_session([2, 2], 6).tolist() == [0, 1, 0, 1, 2, 3]
    assert g._positions_in_session([], 3).tolist() == [0, 1, 2]


def test_history_ready_with_reset_needs_n_lags_trials_in_the_same_session():
    positions = g._positions_in_session([3, 3], 6)
    ready = g._history_ready(positions, n_lags=2, reset=True, first_row=1)
    assert ready.tolist() == [False, False, True, False, False, True]


def test_history_ready_without_reset_only_skips_the_leading_rows():
    positions = g._positions_in_session([3, 3], 6)
    ready = g._history_ready(positions, n_lags=2, reset=False, first_row=2)
    assert ready.tolist() == [False, False, True, True, True, True]


def test_select_complete_rows_drops_unready_and_nan_rows_and_counts_them():
    x = np.array([[1.0, 1], [1, NAN], [1, 1], [1, 1], [1, 1]])
    y = np.array([[0.0], [0], [0], [NAN], [1]])
    ready = np.array([False, True, True, True, True])
    kept_x, kept_y, counts = g._select_complete_rows(x, y, ready)
    assert kept_x.tolist() == [[1, 1], [1, 1]] and kept_y.tolist() == [[0.0], [1.0]]
    assert counts == g.RowCounts(n_trials=5, n_dropped_no_history=1, n_dropped_nan=2, n_rows=2)


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
    x, _, _ = g.format_glm_input(
        data, 2, drop_redundant_cols=False, reset_history_per_session=False
    )
    assert x["m"].shape == (2, 1 + 7 * 2)


def test_format_glm_input_labels_include_arm3_when_columns_are_kept():
    data = constant_subjects(["A", "B", "C"])
    x, _, labels = g.format_glm_input(data, 1, drop_redundant_cols=False)
    assert labels == [
        "Bias", "Choice Arm1 t-1", "Choice Arm2 t-1", "Choice Arm3 t-1", "Reward t-1",
        "Chx Rew Arm1 t-1", "Chx Rew Arm2 t-1", "Chx Rew Arm3 t-1",
    ]  # fmt: skip
    assert x["m"].shape[1] == len(labels)


@pytest.mark.parametrize("n_lags", [0, 1, 3])
@pytest.mark.parametrize("drop", [True, False])
@pytest.mark.parametrize("by_rank", [False, True])
def test_labels_always_match_matrix_columns(by_rank, drop, n_lags):
    fmt = g.format_glm_input_by_rank if by_rank else g.format_glm_input
    data = constant_subjects(["A", "B", "C"] * 3)
    x, _, labels = fmt(data, n_lags=n_lags, drop_redundant_cols=drop)
    assert x["m"].shape[1] == len(labels)
    assert len(set(labels)) == len(labels)


def test_format_glm_input_session_with_one_trial_is_empty():
    x, y, _ = g.format_glm_input(constant_subjects(["A"]), n_lags=2)
    assert x["m"].shape == (0, 1 + 5 * 2) and y["m"].shape == (0, 3)


def test_format_glm_input_negative_lags_mean_bias_only():
    # Negative lags behave like zero lags, even with missing rewards or too few trials.
    data = constant_subjects(["A", "B", "C", "A"])
    data["m"]["ses-1"]["reward_magnitudes_by_tower"]["A"][0] = None
    x, y, labels = g.format_glm_input(data, n_lags=-1, reset_history_per_session=False)
    assert labels == ["Bias"] and x["m"].tolist() == [[1.0]] * 3 and y["m"].shape == (3, 3)
    x, y, _ = g.format_glm_input(
        constant_subjects(["A"]), n_lags=-1, reset_history_per_session=False
    )
    assert x["m"].shape == (0, 1) and y["m"].shape == (0, 3)
    # With a reset, no lags means no trial lacks history.
    x, y, _ = g.format_glm_input(data, n_lags=-1)
    assert x["m"].shape == (4, 1) and y["m"].shape == (4, 3)


def test_format_glm_input_missing_history_before_the_first_trial_is_zero():
    # Four trials, three lags: rows for trials 1..3 reach back before trial 0.
    data = constant_subjects(["A", "B", "C", "A"])
    x, _, labels = g.format_glm_input(data, n_lags=3, reset_history_per_session=False)
    assert x["m"].shape == (3, 1 + 5 * 3) and not np.isnan(x["m"]).any()
    first = columns(labels, x["m"][0])
    assert first["Choice Arm1 t-1"] == 1 and first["Reward t-1"] == 1.0
    for lag in (2, 3):
        assert first[f"Choice Arm1 t-{lag}"] == 0 and first[f"Reward t-{lag}"] == 0
    assert columns(labels, x["m"][2])["Choice Arm1 t-3"] == 1


def test_format_glm_input_with_more_lags_than_trials_is_still_a_full_matrix():
    data = constant_subjects(["A", "B", "C", "A"])
    x, _, _ = g.format_glm_input(data, n_lags=10, reset_history_per_session=False)
    assert x["m"].shape == (3, 1 + 5 * 10) and not np.isnan(x["m"]).any()
    x, _, _ = g.format_glm_input(data, n_lags=10)  # every trial lacks a full history
    assert x["m"].shape == (0, 1 + 5 * 10)


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


def test_format_glm_input_without_reset_lags_cross_session_boundaries():
    # The first trial of the second session sees the last trial of the first
    # (trial 3: chose C, reward 3) as history.
    x, y, _ = g.format_glm_input(two_session_subject(), n_lags=1, reset_history_per_session=False)
    np.testing.assert_array_equal(x["m"][2], [1, 0, 0, 3.0, 0, 0])
    np.testing.assert_array_equal(y["m"][2], [0, 0, 1])
    assert x["m"].shape[0] == 5


def test_format_glm_input_reset_drops_the_first_n_lags_trials_of_every_session():
    x, y, _ = g.format_glm_input(two_session_subject(), n_lags=1, reset_history_per_session=True)
    # Merged trials 2, 3 (ses-2: B after A, C after B) and 5, 6 (ses-10: A after C, B after A).
    np.testing.assert_array_equal(
        x["m"],
        [
            [1, 1, 0, 1.0, 1.0, 0],
            [1, 0, 1, 2.0, 0, 2.0],
            [1, 0, 0, 3.0, 0, 0],
            [1, 1, 0, 1.0, 1.0, 0],
        ],
    )
    np.testing.assert_array_equal(y["m"], [[0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]])


def test_format_glm_input_reset_is_the_default():
    default = g.format_glm_input(two_session_subject(), n_lags=1)
    explicit = g.format_glm_input(two_session_subject(), n_lags=1, reset_history_per_session=True)
    np.testing.assert_array_equal(default[0]["m"], explicit[0]["m"])


def test_format_glm_input_reset_with_two_lags_keeps_one_row_per_session():
    x, y, labels = g.format_glm_input(two_session_subject(), n_lags=2)
    assert x["m"].shape[0] == 2 and not np.isnan(x["m"]).any()
    # Merged trial 3 (C) sees trials 2 and 1; merged trial 6 (B) sees trials 5 and 4.
    second = columns(labels, x["m"][1])
    assert second["Choice Arm1 t-1"] == 1  # trial 5 chose A, not trial 3's C
    assert second["Reward t-2"] == 3.0  # trial 4 chose C
    np.testing.assert_array_equal(y["m"], [[0, 0, 1], [0, 1, 0]])


def test_format_glm_input_reset_drops_the_zero_filled_start_rows_instead_of_keeping_them():
    data = constant_subjects(["A", "B", "C", "A"])
    kept, _, _ = g.format_glm_input(data, n_lags=3, reset_history_per_session=False)
    reset, _, _ = g.format_glm_input(data, n_lags=3, reset_history_per_session=True)
    assert kept["m"].shape[0] == 3 and reset["m"].shape[0] == 1
    np.testing.assert_array_equal(reset["m"][0], kept["m"][-1])


def test_format_glm_input_reset_without_lags_keeps_every_trial():
    x, _, _ = g.format_glm_input(two_session_subject(), n_lags=0, reset_history_per_session=True)
    assert x["m"].shape[0] == 6
    x, _, _ = g.format_glm_input(two_session_subject(), n_lags=0, reset_history_per_session=False)
    assert x["m"].shape[0] == 5


@pytest.mark.parametrize("reset", [True, False])
def test_with_info_records_the_option_and_the_row_counts(reset):
    design = g.format_glm_input_with_info(
        two_session_subject(), n_lags=1, reset_history_per_session=reset
    )
    assert design.reset_history_per_session is reset and design.n_lags == 1
    counts = design.row_counts["m"]
    assert counts.n_trials == 6
    assert counts.n_dropped_no_history == (2 if reset else 1)
    assert counts.n_dropped_nan == 0
    assert counts.n_rows == design.regressors["m"].shape[0] == (4 if reset else 5)


@pytest.mark.parametrize("reset", [True, False])
def test_arm_rows_with_a_nan_reward_in_their_history_are_dropped_and_counted(reset):
    data = constant_subjects(["A", "B", "C", "A", "B"])
    data["m"]["ses-1"]["reward_magnitudes_by_tower"]["B"][1] = NAN  # trial 1 chose B
    design = g.format_glm_input_with_info(data, n_lags=1, reset_history_per_session=reset)
    assert not np.isnan(design.regressors["m"]).any()
    assert design.row_counts["m"].n_dropped_nan == 1  # trial 2 uses trial 1 as history
    assert design.regressors["m"].shape[0] == 3


def test_format_glm_input_rejects_trial_without_single_choice():
    data = constant_subjects(["A", "B", "C"])
    data["m"]["ses-1"]["choices_by_tower"]["A"] = [1, 1, 0]
    with pytest.raises(ValueError, match="exactly one"):
        g.format_glm_input(data)


def test_trial_without_single_choice_names_subject_session_and_local_trial():
    data = two_scaled_sessions()
    data["m"]["ses-3"]["choices_by_tower"]["A"][1] = 1  # chosen twice: B and A
    with pytest.raises(ValueError, match=r"subject 'm': session 2 of 2, trial 2 .*exactly one"):
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
    x, _, _ = g.format_glm_input(data, n_lags=0, reset_history_per_session=False)
    assert x["m"].shape == (3, 1)
    x, _, _ = g.format_glm_input(data, n_lags=0)
    assert x["m"].shape == (4, 1)
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


def two_scaled_sessions():
    big = {t: 10 * v for t, v in CONST.items()}
    return {"m": {"ses-2": session(["A", "B", "C"], [CONST] * 3),
                  "ses-3": session(["A", "B", "C"], [big] * 3)}}  # fmt: skip


@pytest.mark.parametrize("fmt", ["format_glm_input", "format_glm_input_by_rank"])
@pytest.mark.parametrize("reset", [True, False])
def test_short_reward_list_in_a_non_final_session_is_rejected(fmt, reset):
    # Concatenation would move every later session's rewards one trial earlier.
    data = two_scaled_sessions()
    for rewards in data["m"]["ses-2"]["reward_magnitudes_by_tower"].values():
        rewards.pop()
    with pytest.raises(ValueError, match=r"subject 'm'.*session 1 of 2.*2 rewards.*3 trials"):
        getattr(g, fmt)(data, n_lags=1, reset_history_per_session=reset)


@pytest.mark.parametrize("reset", [True, False])
def test_short_choice_list_in_a_non_final_session_is_rejected(reset):
    data = two_scaled_sessions()
    for choices in data["m"]["ses-2"]["choices_by_tower"].values():
        choices.pop()
    with pytest.raises(ValueError, match=r"session 1 of 2.*2 choices.*3 trials"):
        g.format_glm_input(data, n_lags=1, reset_history_per_session=reset)


def test_aligned_sessions_are_not_shifted():
    x, _, labels = g.format_glm_input(
        two_scaled_sessions(), n_lags=1, reset_history_per_session=False
    )
    assert x["m"][:, labels.index("Reward t-1")].tolist() == [1.0, 2.0, 3.0, 10.0, 20.0]


@pytest.mark.parametrize(
    ("towers", "choices"),
    [
        ({**CONST, "D": 4.0}, ["A", "B", "C"]),
        ({"A": 1.0, "B": 2.0}, ["A", "B", "A"]),
        ({**CONST, "D": 4.0}, ["A"]),  # a single trial is no excuse
    ],
)
def test_format_glm_input_rejects_other_than_three_towers(towers, choices):
    with pytest.raises(ValueError, match=r"exactly 3 towers.*subject 'm'"):
        g.format_glm_input(constant_subjects(choices, towers))


def test_format_glm_input_choice_series_shorter_than_trials_raises_value_error():
    data = constant_subjects(["A", "B", "C", "A"])
    for choices in data["m"]["ses-1"]["choices_by_tower"].values():
        choices.pop()
    with pytest.raises(ValueError, match=r"subject 'm'.*3 choices.*4 trials"):
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
    # Row 2 uses trial 1 as history and is dropped. Row 1 is dropped too: its own
    # outcome is unknown.
    assert x["m"].shape[0] == 1
    assert y["m"].tolist() == [[1.0, 0.0, 0.0]]


def test_by_rank_keeps_all_columns_when_asked():
    x, _, labels = g.format_glm_input_by_rank(rank_data(), n_lags=1, drop_redundant_cols=False)
    assert x["m"].shape[1] == len(labels) == 1 + 7


@pytest.mark.parametrize("reset", [True, False])
def test_by_rank_zero_lags_drops_rows_with_unavailable_beliefs(reset):
    design = g.format_glm_input_by_rank_with_info(
        rank_data(), use_believed=True, n_lags=0, reset_history_per_session=reset
    )
    # Beliefs exist from the fourth trial only, so only that trial has an outcome.
    assert design.regressors["m"].shape == (1, 1)
    assert not np.isnan(design.true_choices["m"]).any()
    counts = design.row_counts["m"]
    assert counts.n_dropped_nan == (3 if reset else 2)
    assert counts.n_dropped_no_history == (0 if reset else 1)


def test_by_rank_without_reset_lags_and_beliefs_cross_session_boundaries():
    # Trial 4 (first of ses-10) has the last trial of ses-2 as history, and believed
    # ranks carry over: with beliefs from ses-2, ses-10 has a believed rank from its
    # first trial.
    kwargs = {
        "n_lags": 1,
        "reset_history_per_session": False,
        "reset_beliefs_per_session": False,
    }
    x, y, _ = g.format_glm_input_by_rank(two_session_subject(), **kwargs)
    assert x["m"].shape[0] == 5
    np.testing.assert_array_equal(x["m"][2], [1, 1, 0, 3.0, 3.0, 0])  # C is best
    np.testing.assert_array_equal(y["m"][2], [1, 0, 0])
    x, _, _ = g.format_glm_input_by_rank(two_session_subject(), use_believed=True, **kwargs)
    assert x["m"].shape[0] == 2  # merged trials 4 and 5 are the first with beliefs


def test_by_rank_reset_drops_session_starts():
    x, y, _ = g.format_glm_input_by_rank(two_session_subject(), n_lags=1)
    # Merged trials 2, 3, 5, 6; none of them uses the other session as history.
    assert x["m"].shape[0] == 4
    np.testing.assert_array_equal(x["m"][2], [1, 1, 0, 3.0, 3.0, 0])  # trial 5 follows C (best)
    np.testing.assert_array_equal(y["m"][2], [0, 0, 1])  # trial 5 chose A, the worst


def test_by_rank_reset_forgets_beliefs_at_a_session_start():
    # Each 3-trial session visits every arm only at its end, so no believed rank forms.
    x, _, _ = g.format_glm_input_by_rank(two_session_subject(), use_believed=True, n_lags=1)
    assert x["m"].shape[0] == 0


def test_by_rank_reset_keeps_beliefs_within_a_session():
    one = session(["A", "B", "C", "A", "B"], [{"A": 3.0, "B": 2.0, "C": 1.0}] * 5)
    data = {"m": {"ses-1": one, "ses-2": one}}
    # Beliefs form at trial 4 of a session, so only its trial 5 has beliefs at t and t-1.
    x, _, _ = g.format_glm_input_by_rank(data, use_believed=True, n_lags=1)
    assert x["m"].shape[0] == 2
    x, _, _ = g.format_glm_input_by_rank(
        data,
        use_believed=True,
        n_lags=1,
        reset_history_per_session=False,
        reset_beliefs_per_session=False,
    )
    assert x["m"].shape[0] == 6  # beliefs and history run on through trials 5..10


@pytest.mark.parametrize("reset", [True, False])
def test_by_rank_with_info_records_the_option(reset):
    design = g.format_glm_input_by_rank_with_info(
        two_session_subject(), n_lags=1, reset_history_per_session=reset
    )
    assert design.reset_history_per_session is reset
    assert design.col_labels == [
        "Bias", "Chose Best t-1", "Chose Second t-1", "Reward t-1",
        "Best*Reward t-1", "Second*Reward t-1",
    ]  # fmt: skip
    assert design.row_counts["m"].n_rows == design.regressors["m"].shape[0] == (4 if reset else 5)


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


# ---- history and belief resets are independent -----------------------------


def belief_sessions():
    """ses-1 (A,B,C,A,B) has beliefs from its fourth trial; ses-2 (C,A,B) pays other magnitudes.

    Merged trials 1..5 are ses-1, trials 6..8 ses-2. With beliefs carried over, ses-2's
    first trial is ranked by ses-1's last-seen magnitudes (A=3, B=2, C=1), so the C it
    chooses is believed worst although it pays the best magnitude (3) in ses-2.
    """
    return {
        "m": {
            "ses-1": session(list("ABCAB"), [{"A": 3.0, "B": 2.0, "C": 1.0}] * 5),
            "ses-2": session(list("CAB"), [{"A": 1.0, "B": 2.0, "C": 3.0}] * 3),
        }
    }


# (reset_history, reset_beliefs) -> rows as (merged trial, regressors, true choice).
BELIEF_ROWS = {
    # Every later trial has a believed rank at t and t-1, the session boundary included.
    (False, False): [
        (5, [1, 1, 0, 3.0, 3.0, 0], [0, 1, 0]),
        (6, [1, 0, 1, 2.0, 0, 2.0], [0, 0, 1]),  # ses-2's first trial: C believed third
        (7, [1, 0, 0, 3.0, 0, 0], [1, 0, 0]),
        (8, [1, 1, 0, 1.0, 1.0, 0], [0, 1, 0]),
    ],
    # The boundary row is dropped for lack of in-session history; beliefs still carry.
    (True, False): [
        (5, [1, 1, 0, 3.0, 3.0, 0], [0, 1, 0]),
        (7, [1, 0, 0, 3.0, 0, 0], [1, 0, 0]),
        (8, [1, 1, 0, 1.0, 1.0, 0], [0, 1, 0]),
    ],
    # ses-2 forms no belief within its 3 trials, so only ses-1's row remains.
    (False, True): [(5, [1, 1, 0, 3.0, 3.0, 0], [0, 1, 0])],
    (True, True): [(5, [1, 1, 0, 3.0, 3.0, 0], [0, 1, 0])],
}


@pytest.mark.parametrize(("reset_history", "reset_beliefs"), list(BELIEF_ROWS))
def test_by_rank_history_and_belief_resets_combine_independently(reset_history, reset_beliefs):
    design = g.format_glm_input_by_rank_with_info(
        belief_sessions(),
        use_believed=True,
        n_lags=1,
        reset_history_per_session=reset_history,
        reset_beliefs_per_session=reset_beliefs,
    )
    expected = BELIEF_ROWS[(reset_history, reset_beliefs)]
    np.testing.assert_array_equal(design.regressors["m"], [x for _, x, _ in expected])
    np.testing.assert_array_equal(design.true_choices["m"], [y for _, _, y in expected])
    assert design.reset_history_per_session is reset_history
    assert design.reset_beliefs_per_session is reset_beliefs
    assert design.row_counts["m"].n_rows == len(expected)


def test_by_rank_both_resets_false_equal_one_concatenated_session():
    two = belief_sessions()
    merged_session = session(list("ABCABCAB"), [{"A": 3.0, "B": 2.0, "C": 1.0}] * 5 + [
        {"A": 1.0, "B": 2.0, "C": 3.0}
    ] * 3)  # fmt: skip
    kwargs = {"use_believed": True, "n_lags": 1}
    x, y, _ = g.format_glm_input_by_rank(
        two, reset_history_per_session=False, reset_beliefs_per_session=False, **kwargs
    )
    x_one, y_one, _ = g.format_glm_input_by_rank(
        {"m": {"ses-1": merged_session}}, reset_history_per_session=False, **kwargs
    )
    np.testing.assert_array_equal(x["m"], x_one["m"])
    np.testing.assert_array_equal(y["m"], y_one["m"])


def test_beliefs_reset_is_the_default_and_is_ignored_without_believed_ranks():
    default = g.format_glm_input_by_rank_with_info(belief_sessions(), use_believed=True, n_lags=1)
    assert default.reset_beliefs_per_session is True
    a = g.format_glm_input_by_rank(belief_sessions(), n_lags=1, reset_beliefs_per_session=True)
    b = g.format_glm_input_by_rank(belief_sessions(), n_lags=1, reset_beliefs_per_session=False)
    np.testing.assert_array_equal(a[0]["m"], b[0]["m"])


def test_arm_design_has_no_belief_option_to_record():
    assert (
        g.format_glm_input_with_info(belief_sessions(), n_lags=1).reset_beliefs_per_session is None
    )


# ---- errors added with the belief option -------------------------------------


@pytest.mark.parametrize("by_rank", [False, True])
def test_other_than_three_towers_are_rejected_by_both_designs(by_rank):
    fmt = g.format_glm_input_by_rank if by_rank else g.format_glm_input
    for towers in ({**CONST, "D": 4.0}, {"A": 1.0, "B": 2.0}):
        with pytest.raises(ValueError, match=r"exactly 3 towers.*subject 'm'"):
            fmt(constant_subjects(["A", "B", "A"], towers))


def test_reward_list_two_short_in_the_only_session_is_a_value_error():
    data = constant_subjects(["A", "B", "C", "A"])
    for rewards in data["m"]["ses-1"]["reward_magnitudes_by_tower"].values():
        del rewards[-2:]
    with pytest.raises(ValueError, match=r"subject 'm'.*session 1 of 1 has 2 rewards.*4 trials"):
        g.format_glm_input(data, n_lags=1)


def test_reward_list_two_short_in_the_last_of_two_sessions_names_that_session():
    data = two_scaled_sessions()
    for rewards in data["m"]["ses-3"]["reward_magnitudes_by_tower"].values():
        del rewards[-2:]
    with pytest.raises(ValueError, match=r"session 2 of 2 has 1 rewards.*3 trials"):
        g.format_glm_input(data, n_lags=1)


# ---- sessions that do not fit their trial list or each other ----------------

BOTH_DESIGNS = ["format_glm_input", "format_glm_input_by_rank"]


def shorten(data, session_key, field, by):
    for series in data["m"][session_key][field].values():
        del series[len(series) - by :]


@pytest.mark.parametrize("fmt", BOTH_DESIGNS)
def test_last_session_reward_list_three_short_is_rejected(fmt):
    data = two_scaled_sessions()
    shorten(data, "ses-3", "reward_magnitudes_by_tower", 3)
    with pytest.raises(ValueError, match=r"session 2 of 2 has 0 rewards.*3 trials"):
        getattr(g, fmt)(data, n_lags=1)


def test_rank_design_needs_every_reward_of_the_last_session():
    # The own rank of the last trial is computed from all towers' magnitudes there.
    data = two_scaled_sessions()
    shorten(data, "ses-3", "reward_magnitudes_by_tower", 1)
    with pytest.raises(ValueError, match=r"session 2 of 2 has 2 rewards.*3 trials"):
        g.format_glm_input_by_rank(data, n_lags=1)
    g.format_glm_input(data, n_lags=1)  # the arm design never reads the final reward


@pytest.mark.parametrize("fmt", BOTH_DESIGNS)
def test_trial_list_shorter_than_the_choice_lists_is_rejected(fmt):
    data = constant_subjects(["A", "B", "C", "A", "B", "C"])
    data["m"]["ses-1"]["trial"] = [1, 2, 3]
    with pytest.raises(ValueError, match=r"session 1 of 1 has 6 choices.*3 trials"):
        getattr(g, fmt)(data, n_lags=1)


@pytest.mark.parametrize("fmt", BOTH_DESIGNS)
def test_last_session_with_fewer_towers_is_rejected_naming_both_sets(fmt):
    data = two_scaled_sessions()
    for field in ("choices_by_tower", "reward_magnitudes_by_tower"):
        del data["m"]["ses-3"][field]["C"]
    with pytest.raises(
        ValueError,
        match=r"subject 'm': session 2 of 2 has towers \{'A', 'B'\}, expected \{'A', 'B', 'C'\}",
    ):
        getattr(g, fmt)(data, n_lags=1)


@pytest.mark.parametrize("fmt", BOTH_DESIGNS)
def test_first_session_with_fewer_towers_is_rejected(fmt):
    data = two_scaled_sessions()
    for field in ("choices_by_tower", "reward_magnitudes_by_tower"):
        del data["m"]["ses-2"][field]["C"]
    with pytest.raises(ValueError, match=r"session 2 of 2 has towers"):
        getattr(g, fmt)(data, n_lags=1)


def test_reward_towers_that_differ_from_the_choice_towers_are_rejected():
    data = two_scaled_sessions()
    del data["m"]["ses-3"]["reward_magnitudes_by_tower"]["C"]
    with pytest.raises(ValueError, match=r"session 2 of 2 has towers .*reward"):
        g.format_glm_input(data, n_lags=1)


@pytest.mark.parametrize("fmt", BOTH_DESIGNS)
def test_row_counts_account_for_every_trial_of_the_trial_list(fmt):
    design = getattr(g, f"{fmt}_with_info")(two_session_subject(), n_lags=1)
    counts = design.row_counts["m"]
    assert counts.n_trials == 6
    assert counts.n_trials == counts.n_dropped_no_history + counts.n_dropped_nan + counts.n_rows


def test_by_rank_nan_magnitude_on_an_unchosen_tower_drops_the_row():
    # Without a magnitude the ranking of that trial is unknown, so it has no rank.
    rewards = [{"A": 3.0, "B": 2.0, "C": 1.0}] * 6
    kwargs = {"n_lags": 1, "reset_history_per_session": False}
    clean = g.format_glm_input_by_rank_with_info(subjects(list("ABCABC"), rewards), **kwargs)
    data = subjects(list("ABCABC"), rewards)
    data["m"]["ses-1"]["reward_magnitudes_by_tower"]["C"][4] = NAN  # trial 5 chose B
    design = g.format_glm_input_by_rank_with_info(data, **kwargs)
    assert not np.isnan(design.regressors["m"]).any()
    assert not np.isnan(design.true_choices["m"]).any()
    # The row of trial 5 has no rank, and the row after it uses trial 5 as history.
    assert clean.row_counts["m"].n_rows == 5 and design.row_counts["m"].n_rows == 3


def test_by_rank_nan_magnitude_of_a_chosen_tower_leaves_no_believed_rank():
    rewards = [{"A": 3.0, "B": 2.0, "C": 1.0}] * 8
    data = subjects(list("ABCABCAB"), rewards)
    data["m"]["ses-1"]["reward_magnitudes_by_tower"]["C"][2] = NAN  # trial 3 chose C
    design = g.format_glm_input_by_rank_with_info(
        data, use_believed=True, n_lags=0, reset_history_per_session=False
    )
    # C's last-seen magnitude is NaN from trial 3 until C is chosen again at trial 6,
    # so beliefs are only usable from trial 7 on.
    assert design.regressors["m"].shape[0] == 2
    assert not np.isnan(design.true_choices["m"]).any()


def test_nan_row_drops_are_logged_as_warnings(caplog):
    data = constant_subjects(["A", "B", "C", "A", "B"])
    data["m"]["ses-1"]["reward_magnitudes_by_tower"]["B"][1] = NAN
    with caplog.at_level("INFO", logger=g.logger.name):
        g.format_glm_input(data, n_lags=1)
    assert [r.levelname for r in caplog.records] == ["WARNING"]
    assert "subject m: dropped 1 of 5" in caplog.records[0].getMessage()
