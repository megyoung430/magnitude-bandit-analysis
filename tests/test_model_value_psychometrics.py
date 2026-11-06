"""Psychometrics on a continuous value: ranks, quantile bins, binned curves, stay/switch pairs."""

import numpy as np
import pandas as pd
import pytest

from src.behavior_analysis import model_value_psychometrics as mv
from src.behavior_analysis.get_believed_value import rank_weights
from src.behavior_analysis.stay_switch_psychometrics import stay_switch_curves


def test_rank_table_orders_values_and_gives_the_chosen_arms_rank_shares():
    values = np.array([[3.0, 1.0, 2.0], [0.5, 0.9, 0.1]])
    table = mv.rank_table(["m", "m"], values, np.array([2, 0]))
    assert table[["v_best", "v_second", "v_third"]].to_numpy().tolist() == [
        [3.0, 2.0, 1.0],
        [0.9, 0.5, 0.1],
    ]
    assert table["best_share"].tolist() == [0.0, 0.0]
    assert table["second_share"].tolist() == [1.0, 1.0]


def test_tied_values_share_their_rank_positions_evenly():
    values = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [2.0, 1.0, 1.0]])
    table = mv.rank_table(["m"] * 3, values, np.array([0, 1, 1]))
    assert table["best_share"].tolist() == pytest.approx([0.5, 1 / 3, 0.0])
    assert table["second_share"].tolist() == pytest.approx([0.5, 1 / 3, 0.5])


def test_rank_shares_agree_with_the_believed_value_rank_weights():
    rng = np.random.default_rng(0)
    values = rng.integers(0, 3, size=(200, 3)).astype(float)
    chosen = rng.integers(0, 3, size=200)
    table = mv.rank_table(["m"] * 200, values, chosen)
    expected = [rank_weights(v[c], v, n_ranks=2) for v, c in zip(values, chosen, strict=True)]
    np.testing.assert_allclose(table[["best_share", "second_share"]].to_numpy(), expected)


def test_quantile_bins_are_equally_filled_and_numbered_from_one():
    bins = mv.quantile_bins(np.arange(10.0), 5)
    assert bins.tolist() == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]


def test_quantile_bins_put_tied_values_in_one_bin():
    bins = mv.quantile_bins(np.array([0.0] * 8 + [1.0, 2.0]), 5)
    assert set(bins[:8]) == {1}
    assert bins[-1] == 5


def test_quantile_bins_reject_fewer_than_two_bins():
    with pytest.raises(ValueError, match="n_bins"):
        mv.quantile_bins(np.arange(4.0), 1)


def toy_rank_table():
    """Mouse a: P(best) 0 in the lower half of v_best, 1 in the upper; mouse b: always 1."""
    a = pd.DataFrame(
        dict(
            mouse="a",
            v_best=np.arange(10.0),
            v_second=np.arange(10.0) / 2,
            v_third=0.0,
            best_share=[0.0] * 5 + [1.0] * 5,
            second_share=[1.0] * 5 + [0.0] * 5,
        )
    )
    b = a.assign(mouse="b", best_share=1.0, second_share=0.0)
    return pd.concat([a, b], ignore_index=True)


def test_binned_curves_have_the_known_answer():
    curves = mv.binned_psychometric_curves(toy_rank_table(), n_bins=5)
    best = curves["best"]["best_val"]
    assert best.x.tolist() == [1, 2, 3, 4, 5]
    assert best.mice == ["a", "b"]
    np.testing.assert_allclose(best.per_mouse[0], [0, 0, 0.5, 1, 1])
    np.testing.assert_allclose(best.per_mouse[1], [1, 1, 1, 1, 1])
    np.testing.assert_allclose(best.mean, [0.5, 0.5, 0.75, 1, 1])
    np.testing.assert_allclose(best.se[0], 0.5)
    assert best.n_mice.tolist() == [2] * 5
    assert best.n_trials.tolist() == [4] * 5
    second = curves["second"]["best_val"]
    np.testing.assert_allclose(second.per_mouse[0], [1, 1, 0.5, 0, 0])


def test_binned_curves_cover_all_three_axes():
    curves = mv.binned_psychometric_curves(toy_rank_table(), n_bins=4)
    assert set(curves) == {"best", "second"}
    for choice in curves.values():
        assert set(choice) == {"best_val", "second_val", "diff"}
        assert all(len(c.x) == 4 for c in choice.values())


def test_a_mouse_is_binned_by_its_own_quantiles():
    low = toy_rank_table().query("mouse == 'a'")
    high = low.assign(mouse="b", v_best=low["v_best"] * 100 + 1000)
    curves = mv.binned_psychometric_curves(pd.concat([low, high]), n_bins=5)
    np.testing.assert_allclose(
        curves["best"]["best_val"].per_mouse[0], curves["best"]["best_val"].per_mouse[1]
    )


# ---------------------------------------------------------------------------
# Stay/switch pairs
# ---------------------------------------------------------------------------


def toy_trials():
    """Two sessions of one mouse's problem: trials 0-3, then 4-5."""
    return pd.DataFrame(
        dict(
            mouse="m",
            problem=1,
            trial=range(6),
            trial_in_session=[0, 1, 2, 3, 0, 1],
            after_gap=False,
            reward=[2, 0, 3, 1, 0, 4],
        )
    )


TOY_VALUES = np.array(
    [
        [0.0, 0.0, 0.0],
        [2.0, 5.0, 1.0],  # prev 0 (value 2); alternatives 5 (arm 1) and 1 (arm 2)
        [3.0, 1.0, 1.0],  # previous arm 0 (value 3); the two alternatives tie at 1
        [1.0, 0.0, 4.0],
        [9.0, 9.0, 9.0],  # session start: no pair
        [1.0, 2.0, 3.0],
    ]
)
TOY_CHOSEN = np.array([0, 0, 1, 2, 1, 2])


def test_pairs_follow_the_stay_switch_definition():
    pairs = mv.stay_switch_pairs_from_values(toy_trials(), TOY_VALUES, TOY_CHOSEN, n_bins=2)
    assert pairs["trial"].tolist() == [1, 2, 3, 5]
    assert pairs["raw_chosen"].tolist() == [2.0, 3.0, 0.0, 2.0]
    assert pairs["raw_alt"].tolist() == [5.0, 1.0, 4.0, 3.0]
    assert pairs["v_other"].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert pairs["stay"].tolist() == [1.0, 0.0, 0.0, 0.0]
    # trial 2: the two alternatives tie, so the switch is shared between them
    assert pairs["switch_best_alt"].tolist() == [0.0, 0.5, 1.0, 1.0]
    assert pairs["switch_other"].tolist() == [0.0, 0.5, 0.0, 0.0]
    assert pairs["rewarded"].tolist() == [True, False, True, False]


def test_pairs_never_span_a_dropped_trial():
    trials = toy_trials()
    trials.loc[2, "after_gap"] = True
    pairs = mv.stay_switch_pairs_from_values(trials, TOY_VALUES, TOY_CHOSEN, n_bins=2)
    assert pairs["trial"].tolist() == [1, 3, 5]


def test_pairs_do_not_cross_problems_or_mice():
    trials = pd.concat([toy_trials(), toy_trials().assign(mouse="n")], ignore_index=True)
    values = np.concatenate([TOY_VALUES, TOY_VALUES])
    chosen = np.concatenate([TOY_CHOSEN, TOY_CHOSEN])
    pairs = mv.stay_switch_pairs_from_values(trials, values, chosen, n_bins=2)
    assert pairs.groupby("mouse").size().to_dict() == {"m": 4, "n": 4}


def test_binned_pairs_give_stay_switch_curves_on_the_bin_axes():
    rng = np.random.default_rng(0)
    n = 600
    trials = pd.DataFrame(
        dict(
            mouse=np.repeat(["a", "b"], n // 2),
            problem=1,
            trial=np.tile(np.arange(n // 2), 2),
            trial_in_session=np.tile(np.arange(n // 2) % 50, 2),
            after_gap=False,
            reward=rng.integers(0, 5, n),
        )
    )
    values = rng.normal(size=(n, 3))
    chosen = rng.integers(0, 3, n)
    pairs = mv.stay_switch_pairs_from_values(trials, values, chosen, n_bins=4)
    curves = stay_switch_curves(pairs, pool_trials=True)
    assert curves["stay"]["chosen"].x.tolist() == [1, 2, 3, 4]
    assert curves["stay"]["diff"].x.tolist() == [1, 2, 3, 4]  # bins of the raw difference
    assert curves["stay"]["alt"].mice == ["a", "b"]


def test_split_by_previous_reward_is_a_partition():
    pairs = mv.stay_switch_pairs_from_values(toy_trials(), TOY_VALUES, TOY_CHOSEN, n_bins=2)
    rows = mv.split_by_previous_reward(pairs)
    assert list(rows) == ["rewarded", "unrewarded"]
    assert len(rows["rewarded"]) + len(rows["unrewarded"]) == len(pairs)
    assert rows["rewarded"]["rewarded"].all() and not rows["unrewarded"]["rewarded"].any()


def test_curves_by_reward_row_leave_out_an_empty_row():
    pairs = mv.stay_switch_pairs_from_values(toy_trials(), TOY_VALUES, TOY_CHOSEN, n_bins=2)
    only_rewarded = pairs[pairs["rewarded"]]
    curves = mv.stay_switch_curves_by_reward(only_rewarded, pool_trials=True)
    assert curves["unrewarded"] is None and curves["rewarded"] is not None
    targets = mv.switch_target_curves_by_reward(pairs, pool_trials=True)
    assert set(targets) == {"rewarded", "unrewarded"}


def test_stay_switch_axes_accept_a_precomputed_diff_column():
    pairs = pd.DataFrame(
        dict(
            mouse=["a", "a", "b", "b"],
            v_chosen=[1, 2, 1, 2],
            v_alt=[1, 1, 2, 2],
            diff=[7, 8, 7, 8],
            stay=[1.0, 0.0, 1.0, 1.0],
        )
    )
    curves = stay_switch_curves(pairs, pool_trials=True)
    assert curves["stay"]["diff"].x.tolist() == [7, 8]
    without = stay_switch_curves(pairs.drop(columns="diff"), pool_trials=True)
    assert without["stay"]["diff"].x.tolist() == [-1, 0, 1]
