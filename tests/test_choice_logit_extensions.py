"""Model variants, held-out scores and the four extension analyses of the choice logit."""

import numpy as np
import pandas as pd
import pytest

from src.behavior_analysis.choice_logit import choice_logit as cl
from src.behavior_analysis.choice_logit import choice_logit_cv as cv
from src.behavior_analysis.choice_logit import choice_logit_extensions as ext
from tests.choice_logit_fixtures import simulate_logit_trials

N_LAGS = 2


def comparison_of(variants: dict, mouse: str = "m1") -> pd.DataFrame:
    """A ``model_comparison``-shaped table with the given nats per choice."""
    return pd.DataFrame(
        [(mouse, name, 1000, nats) for name, nats in variants.items()],
        columns=["mouse", "model", "n_trials", "nats_per_choice"],
    )


def evaluate(trials: pd.DataFrame, variants=None) -> ext.VariantCV:
    design = cl.design_matrix(trials, N_LAGS, None, cl.EXTRAS)
    return ext.cross_validate_variants(design, variants or ext.VARIANTS, cl.DEFAULT_L2)


def nats_of(result: ext.VariantCV, names) -> dict:
    table = cv.model_comparison(result.scores, models=list(names))
    return table.set_index("model")["nats_per_choice"].to_dict()


@pytest.fixture(scope="module")
def pulse_mouse():
    return evaluate(simulate_logit_trials("m1", 1, pulse_weight=0.9, choice_weight=1.0))


@pytest.fixture(scope="module")
def binary_mouse():
    return evaluate(simulate_logit_trials("m1", 2, binary_weight=2.0, choice_weight=1.0))


@pytest.fixture(scope="module")
def noise_tower_mouse():
    """Reward and choice history drive choices; the towers carry no signal."""
    trials = simulate_logit_trials("m1", 3, pulse_weight=0.7, choice_weight=1.5)
    return evaluate(trials)


# --- held-out scores ---------------------------------------------------------------------------


def test_variants_have_the_models_the_analyses_need():
    needed = {
        "full", "full + believed", "choice + tower + believed", "binary", "binary + pulse",
        "no reward", "no choice", "no tower", "reward only", "choice only", "tower only",
    }  # fmt: skip
    assert needed <= set(ext.VARIANTS)


def test_scores_have_the_layout_of_the_standard_validation(pulse_mouse):
    scores = pulse_mouse.scores
    assert list(scores.columns) == [
        "mouse", "problem", "model", "n_trials", "held_out_log_likelihood",
    ]  # fmt: skip
    assert set(scores["model"]) == set(ext.VARIANTS)
    assert scores["problem"].nunique() == 6
    assert (scores["held_out_log_likelihood"] < 0).all()


def test_held_out_probabilities_are_distributions_that_reproduce_the_scores(pulse_mouse):
    trials = simulate_logit_trials("m1", 1, pulse_weight=0.9, choice_weight=1.0)
    design = cl.design_matrix(trials, N_LAGS, None, cl.EXTRAS)
    probabilities = pulse_mouse.probabilities["full"]
    assert probabilities.shape == (len(trials), 3)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    total = np.log(probabilities[np.arange(len(trials)), design.chosen]).sum()
    full = pulse_mouse.scores.query("model == 'full'")["held_out_log_likelihood"].sum()
    assert total == pytest.approx(full)


def test_full_variant_matches_the_standard_cross_validation():
    trials = simulate_logit_trials("m1", 4, pulse_weight=0.5, choice_weight=1.0, n_problems=4)
    design = cl.design_matrix(trials, N_LAGS, None, cl.EXTRAS)
    mine = ext.cross_validate_variants(design, {"full": cv.MODELS["full"]}, cl.DEFAULT_L2)
    standard = cv.cross_validate_mouse(cl.design_matrix(trials, N_LAGS), cl.DEFAULT_L2)
    reference = standard.scores.query("model == 'full'").reset_index(drop=True)
    np.testing.assert_allclose(
        mine.scores["held_out_log_likelihood"], reference["held_out_log_likelihood"]
    )


def test_cross_validation_needs_two_problems():
    trials = simulate_logit_trials("m1", 5, n_problems=1)
    with pytest.raises(ValueError, match="at least two problems"):
        evaluate(trials, {"full": cv.MODELS["full"]})


def test_cross_validation_names_a_group_the_design_lacks():
    trials = simulate_logit_trials("m1", 6, n_problems=4, trials_per_session=60)
    design = cl.design_matrix(trials, N_LAGS)  # built without extras
    with pytest.raises(ValueError, match="believed"):
        ext.cross_validate_variants(design, {"b": (cl.BELIEVED,)}, cl.DEFAULT_L2)


def test_model_comparison_orders_models_as_asked():
    scores = pd.DataFrame(
        [
            ("m1", 1, "b", 10, -7.0),
            ("m1", 1, "a", 10, -8.0),
            ("m1", 2, "b", 10, -7.0),
            ("m1", 2, "a", 10, -8.0),
        ],
        columns=["mouse", "problem", "model", "n_trials", "held_out_log_likelihood"],
    )
    table = cv.model_comparison(scores, models=["a", "b"])
    assert list(table["model"]) == ["a", "b"]
    assert table["nats_per_choice"].tolist() == pytest.approx([0.8, 0.7])


# --- 2. binary against pulse -------------------------------------------------------------------


def test_binary_wins_on_a_mouse_that_only_sees_whether_it_was_rewarded(binary_mouse):
    table = ext.binary_pulse_table(
        cv.model_comparison(binary_mouse.scores, models=list(ext.VARIANTS))
    )
    row = table.iloc[0]
    assert row["preferred"] == "binary"
    assert row["gain_binary"] > 0.01  # binary beats pulse by a clear margin in nats per choice


def test_pulse_wins_on_a_mouse_that_weights_magnitude(pulse_mouse):
    table = ext.binary_pulse_table(
        cv.model_comparison(pulse_mouse.scores, models=list(ext.VARIANTS))
    )
    row = table.iloc[0]
    assert row["preferred"] == "pulse"
    assert row["gain_binary"] < -0.01


def test_binary_pulse_table_values_are_differences_of_nats():
    comparison = comparison_of({"full": 0.80, "binary": 0.75, "binary + pulse": 0.78})
    row = ext.binary_pulse_table(comparison).iloc[0]
    assert row["gain_binary"] == pytest.approx(0.05)  # pulse minus binary: positive, binary better
    assert row["gain_binary_pulse"] == pytest.approx(0.02)


def test_binary_pulse_table_needs_its_models():
    with pytest.raises(ValueError, match="binary"):
        ext.binary_pulse_table(comparison_of({"full": 0.8}))


# --- 1. believed value -------------------------------------------------------------------------


def test_believed_table_reports_weights_and_the_gain_over_the_full_model():
    comparison = comparison_of(
        {"full": 0.80, "full + believed": 0.78, "choice + tower + believed": 0.90}
    )
    weights = cl.weights_table(
        "m1",
        ("reward_lag1", "believed_value", "believed_unvisited"),
        np.array([0.2, 0.15, -0.4]),
    )
    row = ext.believed_value_table(comparison, weights).iloc[0]
    assert row["believed_weight"] == pytest.approx(0.15)
    assert row["unvisited_weight"] == pytest.approx(-0.4)
    assert row["gain_believed"] == pytest.approx(0.02)  # full minus full + believed
    assert row["gain_no_reward_believed"] == pytest.approx(-0.10)  # replaces the reward lags


def test_believed_table_needs_the_believed_weights():
    weights = cl.weights_table("m1", ("reward_lag1",), np.array([0.2]))
    comparison = comparison_of(
        {"full": 0.8, "full + believed": 0.78, "choice + tower + believed": 0.9}
    )
    with pytest.raises(ValueError, match="believed_value"):
        ext.believed_value_table(comparison, weights)


# --- gain summary ------------------------------------------------------------------------------


def test_gain_summary_with_enough_mice_gives_mean_se_and_tests():
    table = pd.DataFrame({"gain": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]})
    row = ext.gain_summary(table, ["gain"]).iloc[0]
    assert row["n"] == 6
    assert row["mean"] == pytest.approx(0.035)
    assert row["se"] == pytest.approx(table["gain"].std(ddof=1) / np.sqrt(6))
    assert row["n_positive"] == 6
    assert row["p_sign"] == pytest.approx(2 * 0.5**6)  # all six on one side
    assert row["p_wilcoxon"] == pytest.approx(2 / 64)


def test_five_mice_cannot_reach_significance_so_six_are_needed():
    assert ext.MIN_TEST_MICE == 6
    # the smallest two-sided p of five mice, all on one side
    assert 2 * 0.5**5 > 0.05
    assert 2 * 0.5**6 < 0.05


def test_gain_summary_with_too_few_mice_has_no_mean_error_or_test():
    table = pd.DataFrame({"gain": [0.01, 0.02, 0.03, 0.04, 0.05]})
    row = ext.gain_summary(table, ["gain"]).iloc[0]
    assert row["n"] == 5 and row["n_positive"] == 5
    for column in ("mean", "se", "p_sign", "p_wilcoxon"):
        assert np.isnan(row[column]), column


def test_gain_summary_with_a_single_mouse_has_no_mean():
    row = ext.gain_summary(pd.DataFrame({"gain": [0.01]}), ["gain"]).iloc[0]
    assert row["n"] == 1 and np.isnan(row["mean"])


def test_gain_summary_of_all_zero_gains_is_not_significant():
    table = pd.DataFrame({"gain": [0.0] * 6})
    row = ext.gain_summary(table, ["gain"]).iloc[0]
    assert row["p_wilcoxon"] == 1.0 and row["p_sign"] == 1.0


# --- 3. unique and shared variance -------------------------------------------------------------


def test_variance_partition_on_hand_computed_numbers():
    comparison = comparison_of(
        {
            "full": 0.70, "no reward": 0.80, "no choice": 0.85, "no tower": 0.75,
            "reward only": 0.95, "choice only": 0.90, "tower only": 1.00,
        }
    )  # fmt: skip
    row = ext.variance_partition(comparison).iloc[0]
    assert row["unique_reward"] == pytest.approx(0.10)  # no reward - full
    assert row["unique_choice"] == pytest.approx(0.15)
    assert row["unique_tower"] == pytest.approx(0.05)
    # shared = [gain(reward + tower) + gain(choice + tower)] - gain(full), gains over tower only
    assert row["shared_reward_choice"] == pytest.approx(
        (1.00 - 0.85) + (1.00 - 0.80) - (1.00 - 0.70)
    )
    assert row["gain_full_over_tower"] == pytest.approx(0.30)
    # the gain of the full model over the tower-only model splits exactly into the three terms
    assert row["gain_full_over_tower"] == pytest.approx(
        row["unique_reward"] + row["unique_choice"] + row["shared_reward_choice"]
    )
    assert row["only_choice"] == pytest.approx(np.log(3) - 0.90)  # gain over chance


def test_variance_partition_needs_all_its_models():
    with pytest.raises(ValueError, match="tower only"):
        ext.variance_partition(comparison_of({"full": 0.7, "no reward": 0.8}))


def test_a_noise_group_has_no_unique_variance_and_generating_groups_do(noise_tower_mouse):
    comparison = cv.model_comparison(noise_tower_mouse.scores, models=list(ext.VARIANTS))
    row = ext.variance_partition(comparison).iloc[0]
    assert abs(row["unique_tower"]) < 0.005  # no tower signal: dropping it costs nothing
    assert row["unique_reward"] > 0.02
    assert row["unique_choice"] > 0.02


# --- 4. calibration ----------------------------------------------------------------------------


def test_calibration_bins_on_a_hand_built_prediction_set():
    probabilities = np.array([[0.6, 0.3, 0.1], [0.1, 0.2, 0.7]])
    chosen = np.array([0, 2])
    table = ext.calibration_bins(probabilities, chosen, n_bins=2)
    # sorted predictions .1 .1 .2 | .3 .6 .7; chosen: only .6 and .7 were chosen
    assert table["n_pairs"].tolist() == [3, 3]
    assert table["mean_predicted"].tolist() == pytest.approx([0.4 / 3, 1.6 / 3])
    assert table["observed"].tolist() == pytest.approx([0.0, 2 / 3])


def test_calibration_bins_cover_every_trial_arm_pair_once():
    rng = np.random.default_rng(0)
    probabilities = rng.dirichlet([1, 1, 1], size=500)
    chosen = rng.integers(0, 3, size=500)
    table = ext.calibration_bins(probabilities, chosen, n_bins=10)
    assert table["n_pairs"].sum() == 1500
    assert (table["n_pairs"] == 150).all()  # equal-count (decile) bins
    assert (np.diff(table["mean_predicted"]) >= 0).all()


def test_perfectly_calibrated_predictions_lie_on_the_identity():
    rng = np.random.default_rng(1)
    probabilities = rng.dirichlet([0.7, 0.7, 0.7], size=40000)
    cumulative = probabilities.cumsum(axis=1)
    chosen = (rng.random(40000)[:, None] > cumulative).sum(axis=1)  # a draw from each row
    table = ext.calibration_bins(probabilities, chosen, n_bins=10)
    np.testing.assert_allclose(table["observed"], table["mean_predicted"], atol=0.02)


def test_miscalibrated_predictions_are_off_the_identity():
    rng = np.random.default_rng(2)
    probabilities = rng.dirichlet([0.7, 0.7, 0.7], size=20000)
    chosen = rng.integers(0, 3, size=20000)  # choices ignore the predictions
    table = ext.calibration_bins(probabilities, chosen, n_bins=10)
    assert (table["observed"] - table["mean_predicted"]).abs().max() > 0.1


def test_calibration_table_has_one_block_per_mouse_and_a_summary():
    rng = np.random.default_rng(3)
    probabilities = rng.dirichlet([1, 1, 1], size=400)
    chosen = rng.integers(0, 3, size=400)
    mice = np.array(["a"] * 200 + ["b"] * 200)
    table = ext.calibration_table(probabilities, chosen, mice, n_bins=5)
    assert list(table["mouse"].unique()) == ["a", "b"]
    assert all(bins == [0, 1, 2, 3, 4] for bins in table.groupby("mouse")["bin"].apply(list))
    summary = ext.calibration_summary(table)
    assert list(summary["mouse"]) == ["a", "b"]
    expected = (table["observed"] - table["mean_predicted"]).abs()[table["mouse"] == "a"].mean()
    assert summary.loc[0, "mean_abs_error"] == pytest.approx(expected)
    assert (summary["weighted_abs_error"] >= 0).all()


def test_calibration_needs_enough_predictions_and_matching_shapes():
    with pytest.raises(ValueError, match="bins"):
        ext.calibration_bins(np.full((1, 3), 1 / 3), np.array([0]), n_bins=10)
    with pytest.raises(ValueError, match="shape"):
        ext.calibration_bins(np.full((4, 2), 0.5), np.zeros(4, dtype=int), n_bins=2)
    with pytest.raises(ValueError, match="per trial"):
        ext.calibration_bins(np.full((4, 3), 1 / 3), np.zeros(3, dtype=int), n_bins=2)


# --- calibration noise floor -------------------------------------------------------------------


def calibrated_draw(n, seed):
    rng = np.random.default_rng(seed)
    probabilities = rng.dirichlet([0.7, 0.7, 0.7], size=n)
    chosen = (rng.random(n)[:, None] > probabilities.cumsum(axis=1)).sum(axis=1)
    return probabilities, chosen


def test_noise_floor_matches_the_error_of_a_perfectly_calibrated_sample():
    probabilities, chosen = calibrated_draw(30000, 11)
    mice = np.array(["a"] * 30000, dtype=object)
    floor = ext.calibration_noise_floor(probabilities, mice, n_bins=10)
    table = ext.calibration_table(probabilities, chosen, mice, n_bins=10)
    actual = ext.calibration_summary(table).loc[0, "mean_abs_error"]
    assert floor.loc[0, "noise_floor_mean_abs_error"] == pytest.approx(actual, abs=0.003)
    assert floor.loc[0, "noise_floor_mean_abs_error"] > 0


def test_noise_floor_is_reproducible_and_per_mouse():
    probabilities, _ = calibrated_draw(2000, 12)
    mice = np.array(["a"] * 1000 + ["b"] * 1000, dtype=object)
    first = ext.calibration_noise_floor(probabilities, mice, n_bins=5)
    again = ext.calibration_noise_floor(probabilities, mice, n_bins=5)
    pd.testing.assert_frame_equal(first, again)
    assert list(first["mouse"]) == ["a", "b"]
    assert list(first.columns) == [
        "mouse", "noise_floor_mean_abs_error", "noise_floor_weighted_abs_error",
    ]  # fmt: skip
    other = ext.calibration_noise_floor(probabilities, mice, n_bins=5, seed=1)
    assert not np.allclose(first["noise_floor_mean_abs_error"], other["noise_floor_mean_abs_error"])


def test_noise_floor_of_certain_predictions_is_zero():
    probabilities = np.tile(np.array([1.0, 0.0, 0.0]), (200, 1))
    floor = ext.calibration_noise_floor(probabilities, np.array(["a"] * 200, dtype=object), 4)
    assert floor.loc[0, "noise_floor_mean_abs_error"] == pytest.approx(0.0)
