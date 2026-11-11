"""Second-best test: leaving the best, believed and true rankings, graded curves, minimums.

Every number below is worked out by hand from ``tests/second_best_fixtures.py``.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from src.behavior_analysis import second_best as sb
from src.behavior_analysis import switch_regression as sr
from tests.second_best_fixtures import hand_mouse
from tests.switch_regression_fixtures import simulate_switch_mouse


@pytest.fixture(scope="module")
def hand():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sb.analyze_mouse(hand_mouse())


def _row(table, measure, condition):
    rows = table[(table["measure"] == measure) & (table["condition"] == condition)]
    assert len(rows) == 1
    return rows.iloc[0]


# ---------------------------------------------------------------------------
# Leaving the best
# ---------------------------------------------------------------------------


def test_leaving_the_best_counts_chosen_second_bests_and_excluded_ties(hand):
    row = _row(hand.proportions, sb.SECOND_BEST, sr.AFTER_NONZERO)
    # r1 and r6 go to the second best, r9 and r13 to the third; r5 leaves a tied top (A = B = 3)
    # and r16 leaves the best with tied alternatives (A = C = 2): both excluded
    assert (row["n_used"], row["n_success"], row["n_tied"]) == (4, 2, 2)
    assert row["p"] == 0.5


def test_leaving_the_best_never_follows_a_zero_payout(hand):
    # an arm that paid 0 was the zero arm, which is never the best: the cell is empty
    row = _row(hand.proportions, sb.SECOND_BEST, sr.AFTER_ZERO)
    assert (row["n_used"], row["n_success"], row["n_tied"]) == (0, 0, 0)
    assert np.isnan(row["p"])


def test_pooled_row_adds_the_two_conditions(hand):
    table = hand.proportions
    for measure in sb.MEASURES:
        zero, nonzero, pooled = (
            _row(table, measure, c) for c in (sr.AFTER_ZERO, sr.AFTER_NONZERO, sb.POOLED)
        )
        for column in ("n_switches", "n_used", "n_success", "n_tied", "n_unvisited"):
            assert pooled[column] == zero[column] + nonzero[column]


def test_ranking_uses_the_magnitudes_of_the_previous_trial_not_the_switch_trial():
    # r6: B (3) is the strict best at r5 and the mouse leaves it for A; the table at r6 is the
    # same here, so shift the magnitudes of the switch trial only and the outcome must not move
    table = hand_mouse(magnitudes={6: (0, 1, 4)})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = sb.analyze_mouse(table)
    row = _row(result.proportions, sb.SECOND_BEST, sr.AFTER_NONZERO)
    assert (row["n_used"], row["n_success"]) == (4, 2)  # r6 still counts as a success


# ---------------------------------------------------------------------------
# Believed and true rankings of the alternatives
# ---------------------------------------------------------------------------


def test_believed_better_counts_visited_alternatives_that_differ(hand):
    nonzero = _row(hand.proportions, sb.BELIEVED_BETTER, sr.AFTER_NONZERO)
    zero = _row(hand.proportions, sb.BELIEVED_BETTER, sr.AFTER_ZERO)
    # r1, r2, r13 and r14 have an alternative that was never chosen in the problem
    assert (nonzero["n_used"], nonzero["n_success"], nonzero["n_unvisited"]) == (8, 5, 4)
    assert nonzero["p"] == 5 / 8
    assert (zero["n_used"], zero["n_success"], zero["n_unvisited"]) == (1, 1, 0)


def test_believed_values_carry_over_sessions_and_reset_at_a_new_problem(hand):
    # r11 (second session) ranks B above C from r8 and r9 of the first session: counted;
    # r13 and r14 (problem 2) find A, C and B, C unvisited although visited in problem 1
    nonzero = _row(hand.proportions, sb.BELIEVED_BETTER, sr.AFTER_NONZERO)
    assert nonzero["n_unvisited"] == 4


def test_true_better_counts_all_switches_where_the_alternatives_differ(hand):
    nonzero = _row(hand.proportions, sb.TRUE_BETTER, sr.AFTER_NONZERO)
    zero = _row(hand.proportions, sb.TRUE_BETTER, sr.AFTER_ZERO)
    assert (nonzero["n_used"], nonzero["n_success"], nonzero["n_tied"]) == (11, 6, 1)
    assert (zero["n_used"], zero["n_success"], zero["n_tied"]) == (0, 0, 1)  # r3: A = B = 3


def test_every_switch_is_counted_once_in_its_condition(hand):
    table = hand.proportions
    for measure in sb.MEASURES:
        zero, nonzero = (_row(table, measure, c) for c in (sr.AFTER_ZERO, sr.AFTER_NONZERO))
        assert (zero["n_switches"], nonzero["n_switches"]) == (1, 12)


def test_session_starts_and_stays_are_not_switches():
    # r0, r10 and r12 start sessions (r10 follows a different arm) and r4 stays: 13 switches only
    pooled = _row(sb.analyze_mouse(hand_mouse()).proportions, sb.TRUE_BETTER, sb.POOLED)
    assert pooled["n_switches"] == 13


# ---------------------------------------------------------------------------
# Graded curves
# ---------------------------------------------------------------------------


def _curve(graded, measure, condition):
    rows = graded[(graded["measure"] == measure) & (graded["condition"] == condition)]
    return rows.set_index("difference")[["n_used", "n_success"]]


def test_graded_curve_bins_by_the_magnitude_difference_of_the_alternatives(hand):
    true = _curve(hand.graded, sb.TRUE_BETTER, sr.AFTER_NONZERO)
    assert true.loc[1].tolist() == [4, 2]
    assert true.loc[2].tolist() == [6, 4]
    assert true.loc[3].tolist() == [1, 0]
    assert true.loc[4].tolist() == [0, 0]
    believed = _curve(hand.graded, sb.BELIEVED_BETTER, sr.AFTER_NONZERO)
    assert believed.loc[1].tolist() == [3, 1]
    assert believed.loc[2].tolist() == [2, 2]
    assert believed.loc[3].tolist() == [3, 2]
    assert believed.loc[4].tolist() == [0, 0]


def test_graded_curves_use_the_same_differences_levels_for_every_cell(hand):
    graded = hand.graded
    assert set(graded["difference"]) == set(sb.DIFFERENCES)
    assert set(graded["measure"]) == {sb.BELIEVED_BETTER, sb.TRUE_BETTER}
    assert set(graded["condition"]) == {sr.AFTER_ZERO, sr.AFTER_NONZERO, sb.POOLED}


def test_graded_proportions_sum_back_to_the_ungraded_counts(hand):
    for measure in (sb.BELIEVED_BETTER, sb.TRUE_BETTER):
        graded = _curve(hand.graded, measure, sb.POOLED)
        plain = _row(hand.proportions, measure, sb.POOLED)
        assert graded["n_used"].sum() == plain["n_used"]
        assert graded["n_success"].sum() == plain["n_success"]


# ---------------------------------------------------------------------------
# Minimums
# ---------------------------------------------------------------------------


def test_cells_below_the_minimum_are_dropped_and_named(hand):
    kept, dropped = sb.apply_minimum(hand.proportions, min_used=5)
    assert (kept["n_used"] >= 5).all()
    assert (dropped["n_used"] < 5).all()
    assert len(kept) + len(dropped) == len(hand.proportions)
    # the leave-the-best cell has 4 used switches: dropped at 5, kept at 4
    assert not ((kept["measure"] == sb.SECOND_BEST) & (kept["condition"] == sr.AFTER_NONZERO)).any()
    kept4, _ = sb.apply_minimum(hand.proportions, min_used=4)
    assert ((kept4["measure"] == sb.SECOND_BEST) & (kept4["condition"] == sr.AFTER_NONZERO)).any()


def test_the_default_minimum_is_thirty_used_switches():
    assert sb.MIN_USED == 30


def test_no_switches_at_all_raises():
    stays = hand_mouse().iloc[:1]
    with pytest.raises(sr.TooFewSwitchesError):
        sb.analyze_mouse(stays)


def test_ranking_outside_zero_to_four_pulses_raises():
    table = hand_mouse(magnitudes={8: (2, 13, 1)})
    with pytest.raises(ValueError, match="difference"):
        sb.analyze_mouse(table)


# ---------------------------------------------------------------------------
# Known rule: a mouse that follows the true value goes to the better alternative
# ---------------------------------------------------------------------------


def test_a_value_following_mouse_goes_to_the_truly_better_alternative():
    trials = simulate_switch_mouse("v", seed=1, value_weight=1.5, n_problems=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table = sb.analyze_mouse(trials).proportions
    p = _row(table, sb.TRUE_BETTER, sb.POOLED)["p"]
    blind = simulate_switch_mouse("b", seed=1, value_weight=0.0, n_problems=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p_blind = _row(sb.analyze_mouse(blind).proportions, sb.TRUE_BETTER, sb.POOLED)["p"]
    assert p > 0.75
    assert abs(p_blind - 0.5) < 0.06


# ---------------------------------------------------------------------------
# Calibration against a null
# ---------------------------------------------------------------------------


def _null_draws(values, mouse="m1", measure=sb.TRUE_BETTER, condition=sb.POOLED):
    return pd.DataFrame(
        dict(mouse=mouse, measure=measure, condition=condition, p=values,
             replicate=range(len(values)))
    )  # fmt: skip


def test_calibration_puts_the_observed_proportion_in_its_null():
    observed = pd.DataFrame(
        dict(mouse=["m1"], measure=[sb.TRUE_BETTER], condition=[sb.POOLED], p=[0.9])
    )
    null = _null_draws(np.linspace(0.4, 0.6, 21))
    out = sb.calibrate(observed, null)
    row = out.iloc[0]
    assert row["observed"] == 0.9
    assert row["n_null"] == 21
    assert row["null_mean"] == pytest.approx(0.5)
    assert row["percentile"] == 1.0
    assert row["p_empirical"] == pytest.approx(1 / 22)
    assert row["z"] == pytest.approx((0.9 - 0.5) / np.linspace(0.4, 0.6, 21).std(ddof=1))
    assert set(out.columns) >= {"mouse", "measure", "condition", "z", "percentile", "p_empirical"}


def test_calibration_without_a_null_raises():
    observed = pd.DataFrame(
        dict(mouse=["m1"], measure=[sb.TRUE_BETTER], condition=[sb.POOLED], p=[0.9])
    )
    with pytest.raises(ValueError, match="no null replicates"):
        sb.calibrate(observed, _null_draws([0.5, 0.4], mouse="other"))


def test_calibration_without_any_replicate_raises():
    observed = pd.DataFrame(
        dict(mouse=["m1"], measure=[sb.TRUE_BETTER], condition=[sb.POOLED], p=[0.9])
    )
    with pytest.raises(ValueError, match="no null replicates"):
        sb.calibrate(observed, pd.DataFrame())


def test_calibrated_summary_tests_z_across_mice_from_six_mice():
    rng = np.random.default_rng(0)
    rows = [
        dict(mouse=f"m{i}", measure=sb.TRUE_BETTER, condition=sb.POOLED, z=z)
        for i, z in enumerate(rng.normal(2.0, 0.5, 8))
    ]
    summary = sb.calibrated_summary(pd.DataFrame(rows))
    row = summary.iloc[0]
    assert row["n"] == 8 and np.isfinite(row["se"]) and row["p_wilcoxon"] < 0.05
    few = sb.calibrated_summary(pd.DataFrame(rows[:5]))
    assert np.isnan(few.iloc[0]["mean"]) and np.isnan(few.iloc[0]["p_wilcoxon"])
