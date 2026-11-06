"""Psychometrics on a continuous value: ranks, bins, curves with a baseline, stay/switch pairs."""

import numpy as np
import pandas as pd
import pytest

from src.behavior_analysis import model_value_psychometrics as mvp
from src.behavior_analysis.get_believed_value import rank_weights
from src.behavior_analysis.stay_switch_psychometrics import stay_switch_curves


def test_rank_table_orders_values_and_gives_the_chosen_arms_rank_shares():
    values = np.array([[3.0, 1.0, 2.0], [0.5, 0.9, 0.1]])
    table = mvp.rank_table(["m", "m"], values, np.array([2, 0]))
    assert table[["v_best", "v_second", "v_third"]].to_numpy().tolist() == [
        [3.0, 2.0, 1.0],
        [0.9, 0.5, 0.1],
    ]
    assert table["best_share"].tolist() == [0.0, 0.0]
    assert table["second_share"].tolist() == [1.0, 1.0]
    assert table["gap"].tolist() == pytest.approx([1.0, 0.4])


def test_tied_values_share_their_rank_positions_evenly():
    values = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [2.0, 1.0, 1.0]])
    table = mvp.rank_table(["m"] * 3, values, np.array([0, 1, 1]))
    assert table["best_share"].tolist() == pytest.approx([0.5, 1 / 3, 0.0])
    assert table["second_share"].tolist() == pytest.approx([0.5, 1 / 3, 0.5])


def test_rank_shares_agree_with_the_believed_value_rank_weights():
    rng = np.random.default_rng(0)
    values = rng.integers(0, 3, size=(200, 3)).astype(float)
    chosen = rng.integers(0, 3, size=200)
    table = mvp.rank_table(["m"] * 200, values, chosen)
    expected = [rank_weights(v[c], v, n_ranks=2) for v, c in zip(values, chosen, strict=True)]
    np.testing.assert_allclose(table[["best_share", "second_share"]].to_numpy(), expected)


def test_baseline_columns_weight_each_arm_by_its_probability_and_rank_share():
    values = np.array([[3.0, 1.0, 2.0], [1.0, 1.0, 0.0]])
    probabilities = np.array([[0.5, 0.2, 0.3], [0.2, 0.3, 0.5]])
    table = mvp.rank_table(["m", "m"], values, np.array([0, 0]), probabilities)
    # trial 1: arm 0 best, arm 2 second; trial 2: arms 0 and 1 tie for best and second
    assert table["best_baseline"].tolist() == pytest.approx([0.5, 0.25])
    assert table["second_baseline"].tolist() == pytest.approx([0.3, 0.25])


def test_quantile_bins_are_equally_filled_and_numbered_from_one():
    bins = mvp.quantile_bins(np.arange(10.0), 5)
    assert bins.tolist() == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]


def test_quantile_bins_are_never_empty_even_when_most_values_tie():
    x = np.array([0.0] * 80 + list(np.arange(1.0, 21.0)))
    bins = mvp.quantile_bins(x, 5)
    assert np.bincount(bins)[1:].tolist() == [20] * 5
    # the tied zeros are split over the lower bins, the larger values keep their order
    assert set(bins[:80]) == {1, 2, 3, 4}
    assert (np.diff(bins[80:]) >= 0).all()


def test_quantile_bins_are_deterministic_and_respect_value_order():
    x = np.random.default_rng(1).integers(0, 4, 200).astype(float)
    bins = mvp.quantile_bins(x, 4)
    np.testing.assert_array_equal(bins, mvp.quantile_bins(x, 4))
    assert bins[x == x.min()].max() <= bins[x == x.max()].min()


def test_quantile_bins_reject_bad_bin_counts():
    with pytest.raises(ValueError, match="n_bins"):
        mvp.quantile_bins(np.arange(4.0), 1)
    with pytest.raises(ValueError, match="cannot fill"):
        mvp.quantile_bins(np.arange(3.0), 5)


def toy_rank_table():
    """Mouse a: P(best) 0 in the lower half of v_best, 1 in the upper; mouse b: always 1."""
    v = np.arange(10.0)
    a = pd.DataFrame(
        dict(
            mouse="a",
            v_best=v,
            v_second=v / 2,
            v_third=0.0,
            gap=v / 2,
            best_share=[0.0] * 5 + [1.0] * 5,
            second_share=[1.0] * 5 + [0.0] * 5,
            best_baseline=0.5,
            second_baseline=0.25,
        )
    )
    b = a.assign(mouse="b", best_share=1.0, second_share=0.0)
    return pd.concat([a, b], ignore_index=True)


def test_binned_curves_have_the_known_answer_and_sit_at_bin_means():
    curves, baselines = mvp.psychometric_curves_from_table(toy_rank_table(), n_bins=5)
    best = curves["best"]["best_val"]
    np.testing.assert_allclose(best.x, [0.5, 2.5, 4.5, 6.5, 8.5])  # mean value of each bin
    assert best.mice == ["a", "b"]
    np.testing.assert_allclose(best.per_mouse[0], [0, 0, 0.5, 1, 1])
    np.testing.assert_allclose(best.per_mouse[1], [1, 1, 1, 1, 1])
    np.testing.assert_allclose(best.mean, [0.5, 0.5, 0.75, 1, 1])
    np.testing.assert_allclose(best.se[0], 0.5)
    assert best.n_trials.tolist() == [4] * 5
    np.testing.assert_allclose(baselines["best"]["best_val"].mean, [0.5] * 5)
    np.testing.assert_allclose(baselines["second"]["best_val"].mean, [0.25] * 5)
    assert baselines["best"]["best_val"].x.tolist() == best.x.tolist()


def test_curves_without_baseline_columns_have_no_baseline():
    table = toy_rank_table().drop(columns=["best_baseline", "second_baseline"])
    _, baselines = mvp.psychometric_curves_from_table(table, n_bins=4)
    assert baselines is None


def test_exact_values_are_used_without_bins():
    table = toy_rank_table().assign(v_best=lambda t: (t["v_best"] // 3).astype(float))
    curves, _ = mvp.psychometric_curves_from_table(table, n_bins=None)
    assert curves["best"]["best_val"].x.tolist() == [0.0, 1.0, 2.0, 3.0]


def test_a_mean_from_too_few_mice_is_not_drawn():
    table = toy_rank_table()
    only_a = table[(table["mouse"] == "a") | (table["v_best"] < 2)]
    curves, baselines = mvp.psychometric_curves_from_table(only_a, n_bins=None, min_mice=2)
    best = curves["best"]["best_val"]
    assert best.n_mice.tolist() == [2, 2, 1, 1, 1, 1, 1, 1, 1, 1]
    assert np.isfinite(best.mean[:2]).all() and np.isnan(best.mean[2:]).all()
    assert np.isnan(baselines["best"]["best_val"].mean[2:]).all()


def test_a_mouse_with_fewer_values_than_bins_is_left_out_of_binned_curves():
    table = toy_rank_table()
    short = pd.concat([table, table.iloc[:3].assign(mouse="c")], ignore_index=True)
    curves, _ = mvp.psychometric_curves_from_table(short, n_bins=5)
    assert curves["best"]["best_val"].mice == ["a", "b"]


def test_all_means_are_drawn_when_there_are_fewer_mice_than_the_minimum():
    single = toy_rank_table().query("mouse == 'a'")
    curves, _ = mvp.psychometric_curves_from_table(single, n_bins=5, min_mice=3)
    assert np.isfinite(curves["best"]["best_val"].mean).all()


def test_a_mouse_is_binned_by_its_own_values():
    low = toy_rank_table().query("mouse == 'a'")
    high = low.assign(mouse="b", v_best=low["v_best"] * 100 + 1000)
    curves, _ = mvp.psychometric_curves_from_table(pd.concat([low, high]), n_bins=5)
    best = curves["best"]["best_val"]
    np.testing.assert_allclose(best.per_mouse[0], best.per_mouse[1])


def test_value_effect_is_the_baseline_corrected_rise():
    table = toy_rank_table()
    curves, baselines = mvp.psychometric_curves_from_table(table, n_bins=5)
    # mouse a: observed 0 -> 1 against a flat baseline 0.5; mouse b: 1 -> 1
    assert mvp.value_effect(curves, baselines) == pytest.approx(0.5)
    flat = table.assign(best_share=0.5, best_baseline=0.5)
    flat_curves, flat_baselines = mvp.psychometric_curves_from_table(flat, n_bins=5)
    assert mvp.value_effect(flat_curves, flat_baselines) == pytest.approx(0.0)


def test_zero_fraction():
    assert mvp.zero_fraction(np.array([[0.0, 1.0, 0.0], [0.0, 0.5, 0.2]])) == pytest.approx(0.5)


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
TOY_PROBABILITIES = np.array(
    [
        [1 / 3, 1 / 3, 1 / 3],
        [0.6, 0.3, 0.1],
        [0.2, 0.4, 0.4],
        [0.1, 0.3, 0.6],
        [0.3, 0.3, 0.4],
        [0.2, 0.2, 0.6],
    ]
)


def test_pairs_follow_the_stay_switch_definition():
    pairs = mvp.stay_switch_pairs_from_values(toy_trials(), TOY_VALUES, TOY_CHOSEN)
    assert pairs["trial"].tolist() == [1, 2, 3, 5]
    assert pairs["raw_chosen"].tolist() == [2.0, 3.0, 0.0, 2.0]
    assert pairs["raw_alt"].tolist() == [5.0, 1.0, 4.0, 3.0]
    assert pairs["v_other"].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert pairs["gap"].tolist() == [-3.0, 2.0, -4.0, -1.0]
    assert pairs["stay"].tolist() == [1.0, 0.0, 0.0, 0.0]
    # trial 2: the two alternatives tie, so the switch is shared between them
    assert pairs["switch_best_alt"].tolist() == [0.0, 0.5, 1.0, 1.0]
    assert pairs["switch_other"].tolist() == [0.0, 0.5, 0.0, 0.0]
    assert pairs["rewarded"].tolist() == [True, False, True, False]
    assert "base_stay" not in pairs


def test_pair_baseline_is_the_models_probability_of_the_same_outcome():
    pairs = mvp.stay_switch_pairs_from_values(
        toy_trials(), TOY_VALUES, TOY_CHOSEN, TOY_PROBABILITIES
    )
    # trial 1: previous arm 0 -> P(stay) 0.6; alternatives arm 1 (0.3, best) and arm 2 (0.1)
    # trial 2: tied alternatives -> half each; trial 3: previous arm 1 (0.3), best alt arm 2
    assert pairs["base_stay"].tolist() == pytest.approx([0.6, 0.2, 0.3, 0.2])
    assert pairs["base_best_alt"].tolist() == pytest.approx([0.75, 0.5, 0.6 / 0.7, 0.6 / 0.8])
    np.testing.assert_allclose(pairs["base_best_alt"] + pairs["base_other"], 1.0)


def test_pairs_never_span_a_dropped_trial():
    trials = toy_trials()
    trials.loc[2, "after_gap"] = True
    pairs = mvp.stay_switch_pairs_from_values(trials, TOY_VALUES, TOY_CHOSEN)
    assert pairs["trial"].tolist() == [1, 3, 5]


def test_pairs_leave_out_trials_with_an_arm_that_has_no_value():
    values = TOY_VALUES.copy()
    values[2, 1] = np.nan
    pairs = mvp.stay_switch_pairs_from_values(toy_trials(), values, TOY_CHOSEN)
    assert pairs["trial"].tolist() == [1, 3, 5]


def test_pairs_do_not_cross_problems_or_mice():
    trials = pd.concat([toy_trials(), toy_trials().assign(mouse="n")], ignore_index=True)
    values = np.concatenate([TOY_VALUES, TOY_VALUES])
    chosen = np.concatenate([TOY_CHOSEN, TOY_CHOSEN])
    pairs = mvp.stay_switch_pairs_from_values(trials, values, chosen)
    assert pairs.groupby("mouse").size().to_dict() == {"m": 4, "n": 4}


def random_pairs(n_mice=4, n=800, seed=0, with_baseline=True):
    rng = np.random.default_rng(seed)
    total = n_mice * n
    meta = pd.DataFrame(
        dict(
            mouse=np.repeat([f"m{i}" for i in range(n_mice)], n),
            problem=1,
            trial=np.tile(np.arange(n), n_mice),
            trial_in_session=np.tile(np.arange(n) % 50, n_mice),
            after_gap=False,
            reward=rng.integers(0, 5, total),
        )
    )
    probabilities = rng.dirichlet([2, 2, 2], size=total) if with_baseline else None
    return mvp.stay_switch_pairs_from_values(
        meta, rng.normal(size=(total, 3)), rng.integers(0, 3, total), probabilities
    )


def test_split_by_previous_reward_is_a_partition():
    pairs = mvp.stay_switch_pairs_from_values(toy_trials(), TOY_VALUES, TOY_CHOSEN)
    rows = mvp.split_by_previous_reward(pairs)
    assert list(rows) == ["rewarded", "unrewarded"]
    assert len(rows["rewarded"]) + len(rows["unrewarded"]) == len(pairs)
    assert rows["rewarded"]["rewarded"].all() and not rows["unrewarded"]["rewarded"].any()


def test_bins_are_cut_within_each_reward_row():
    pairs = random_pairs()
    # a reward-dependent shift of the values would give the rows different bin edges
    pairs["raw_chosen"] += np.where(pairs["rewarded"], 10.0, 0.0)
    curves, _ = mvp.stay_switch_curves_by_reward(pairs, pool_trials=True, n_bins=4)
    for row in ("rewarded", "unrewarded"):
        chosen = curves[row]["stay"]["chosen"]
        assert len(chosen.x) == 4
        counts = chosen.n_trials
        assert counts.max() - counts.min() <= len(chosen.mice)  # equal bins, to within 1 per mouse
        assert (chosen.n_mice == len(chosen.mice)).all()  # every bin holds every mouse
    shift = (
        curves["rewarded"]["stay"]["chosen"].x.min()
        - curves["unrewarded"]["stay"]["chosen"].x.min()
    )
    assert shift > 5  # x is the mean value in the bin, per row


def test_curves_by_reward_give_a_baseline_on_the_same_axes():
    pairs = random_pairs()
    curves, baselines = mvp.stay_switch_curves_by_reward(pairs, pool_trials=True, n_bins=4)
    for row in ("rewarded", "unrewarded"):
        for outcome in ("stay", "switch"):
            for axis in ("diff", "chosen", "alt"):
                observed, base = curves[row][outcome][axis], baselines[row][outcome][axis]
                np.testing.assert_array_equal(observed.x, base.x)
        stay = baselines[row]["stay"]["chosen"].mean
        switch = baselines[row]["switch"]["chosen"].mean
        np.testing.assert_allclose(stay + switch, 1.0)


def test_baseline_of_a_value_blind_chooser_matches_the_observed_curves():
    pairs = random_pairs(n_mice=4, n=3000, seed=3)
    curves, baselines = mvp.stay_switch_curves_by_reward(pairs, pool_trials=True, n_bins=4)
    # random choices vs random probabilities: not equal, but both flat around the same level
    observed = curves["rewarded"]["stay"]["chosen"].mean
    assert np.ptp(observed) < 0.15
    assert np.ptp(baselines["rewarded"]["stay"]["chosen"].mean) < 0.15


def target_pairs():
    """Hand-made pairs: only ``alt_gap = raw_alt - v_other``, the outcome and its baseline matter.

    Rewarded row: mouse a switches at gaps 1, 1, 2, 2 (best alternative taken 0, 1, 1, 1; baseline
    0.5, 0.5, 0.8, 0.8), mouse b at gaps 1, 2 (taken 0, 1; baseline 0.4, 0.9). Each mouse also has
    a stay (not a switch) and a switch between tied alternatives (gap 0), which are left out.
    """
    rows = []

    def add(mouse, gap, stay, best, base, rewarded=True):
        v_other = 1.0
        rows.append(
            dict(
                mouse=mouse, raw_alt=v_other + gap, v_other=v_other, stay=stay,
                switch_best_alt=best, base_best_alt=base, rewarded=rewarded,
            )
        )  # fmt: skip

    for gap, best, base in [(1, 0, 0.5), (1, 1, 0.5), (2, 1, 0.8), (2, 1, 0.8), (0, 0.5, 0.5)]:
        add("a", gap, 0.0, best, base)
    add("a", 3, 1.0, 0.0, 0.5)  # a stay: not a switch
    for gap, best, base in [(1, 0, 0.4), (2, 1, 0.9), (0, 0.5, 0.5)]:
        add("b", gap, 0.0, best, base)
    return pd.DataFrame(rows)


def test_target_curve_is_the_best_alternative_share_of_untied_switches_by_gap():
    curves, baselines = mvp.switch_target_curves_by_reward(target_pairs(), n_bins=None)
    best = curves["rewarded"]
    assert best.x.tolist() == [1.0, 2.0]  # the gap between the unchosen arms, not a bin number
    # mouse a: gap 1 -> (0 + 1) / 2, gap 2 -> 1; mouse b: gap 1 -> 0, gap 2 -> 1
    np.testing.assert_allclose(best.per_mouse, [[0.5, 1.0], [0.0, 1.0]])
    np.testing.assert_allclose(best.mean, [0.25, 1.0])
    assert best.n_trials.tolist() == [3, 3]  # switches per gap, ties and stays not counted
    base = baselines["rewarded"]
    np.testing.assert_allclose(base.per_mouse, [[0.5, 0.8], [0.4, 0.9]])
    np.testing.assert_allclose(base.mean, [0.45, 0.85])
    np.testing.assert_array_equal(base.x, best.x)


def test_switches_between_tied_alternatives_are_counted_not_drawn():
    pairs = target_pairs()
    assert mvp.tied_switches_by_reward(pairs) == {"rewarded": 2, "unrewarded": 0}
    curves, _ = mvp.switch_target_curves_by_reward(pairs, n_bins=None)
    assert 0.0 not in curves["rewarded"].x
    assert curves["unrewarded"] is None  # no untied switch in the row


def test_target_gap_bins_are_cut_within_each_reward_row_and_sit_at_the_bin_mean():
    mouse = lambda gaps, best, rewarded: pd.DataFrame(  # noqa: E731
        dict(
            mouse="a", raw_alt=1.0 + np.array(gaps, dtype=float), v_other=1.0, stay=0.0,
            switch_best_alt=best, base_best_alt=0.5, rewarded=rewarded,
        )
    )  # fmt: skip
    pairs = pd.concat(
        [
            mouse([1, 2, 3, 4], [0.0, 0.0, 1.0, 1.0], True),
            mouse([10, 20, 30, 40], [0.0, 1.0, 1.0, 1.0], False),
        ],
        ignore_index=True,
    )
    curves, baselines = mvp.switch_target_curves_by_reward(pairs, n_bins=2)
    assert curves["rewarded"].x.tolist() == [1.5, 3.5]  # mean gap of {1, 2} and {3, 4}
    assert curves["rewarded"].mean.tolist() == [0.0, 1.0]
    assert curves["unrewarded"].x.tolist() == [15.0, 35.0]  # its own bins, not the other row's
    assert curves["unrewarded"].mean.tolist() == [0.5, 1.0]
    assert baselines["unrewarded"].mean.tolist() == [0.5, 0.5]


def test_a_target_mean_from_too_few_mice_is_not_drawn():
    pairs = target_pairs()
    pairs = pairs[~((pairs["mouse"] == "b") & (pairs["raw_alt"] == 3.0))]  # b has gap 1 only
    curves, baselines = mvp.switch_target_curves_by_reward(pairs, n_bins=None, min_mice=2)
    assert curves["rewarded"].n_mice.tolist() == [2, 1]
    assert np.isfinite(curves["rewarded"].mean[0]) and np.isnan(curves["rewarded"].mean[1])
    assert np.isnan(baselines["rewarded"].mean[1])


def test_target_curves_have_no_baseline_without_the_column():
    pairs = target_pairs().drop(columns="base_best_alt")
    curves, baselines = mvp.switch_target_curves_by_reward(pairs, n_bins=None)
    assert baselines is None and curves["rewarded"] is not None


def test_target_baseline_comes_from_the_pairs_of_real_values():
    pairs = mvp.stay_switch_pairs_from_values(
        toy_trials(), TOY_VALUES, TOY_CHOSEN, TOY_PROBABILITIES
    )
    # trial 3 (previous choice rewarded? reward on trial 2 was 3): switch to the best alternative
    # arm 2 at gap 4 - 1 = 3, baseline 0.6 / 0.7
    curves, baselines = mvp.switch_target_curves_by_reward(pairs, n_bins=None, min_mice=1)
    assert curves["rewarded"].x.tolist() == [3.0]
    assert curves["rewarded"].mean.tolist() == [1.0]
    assert baselines["rewarded"].mean.tolist() == pytest.approx([0.6 / 0.7])
    # trial 2 is a switch between tied alternatives, left out of the curve and counted
    assert mvp.tied_switches_by_reward(pairs)["unrewarded"] == 1


def test_curves_by_reward_leave_out_an_empty_row():
    pairs = random_pairs()
    curves, baselines = mvp.stay_switch_curves_by_reward(
        pairs[pairs["rewarded"]], pool_trials=True, n_bins=4
    )
    assert curves["unrewarded"] is None and baselines["unrewarded"] is None
    assert curves["rewarded"] is not None


def test_curves_without_baseline_columns_have_no_baseline_by_reward():
    pairs = random_pairs(with_baseline=False)
    curves, baselines = mvp.stay_switch_curves_by_reward(pairs, pool_trials=True, n_bins=4)
    assert baselines is None and curves["rewarded"] is not None


def test_exact_value_pairs_use_the_values_as_x():
    pairs = random_pairs().assign(
        raw_chosen=lambda p: (p["raw_chosen"] > 0).astype(float),
        raw_alt=lambda p: (p["raw_alt"] > 0).astype(float),
    )
    curves, _ = mvp.stay_switch_curves_by_reward(pairs, pool_trials=True, n_bins=None)
    assert curves["rewarded"]["stay"]["chosen"].x.tolist() == [0.0, 1.0]


def test_stay_switch_axes_accept_a_precomputed_diff_column():
    pairs = pd.DataFrame(
        dict(
            mouse=["a", "a", "b", "b"],
            v_chosen=[1, 2, 1, 2],
            v_alt=[1, 1, 2, 2],
            v_other=[0, 0, 0, 0],
            diff=[7, 8, 7, 8],
            stay=[1.0, 0.0, 1.0, 1.0],
        )
    )
    curves = stay_switch_curves(pairs, pool_trials=True)
    assert curves["stay"]["diff"].x.tolist() == [7, 8]
    without = stay_switch_curves(pairs.drop(columns="diff"), pool_trials=True)
    assert without["stay"]["diff"].x.tolist() == [-1, 0, 1]
