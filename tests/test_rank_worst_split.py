"""Where the worst alternative's positive stay weight lives (stale beliefs, one zero arm)."""

import numpy as np
import pandas as pd
import pytest

from src.behavior_analysis.rank_tests import rank_decisions as rd
from src.behavior_analysis.rank_tests import rank_state as rs
from src.behavior_analysis.rank_tests import rank_worst_split as ws
from tests.switch_regression_fixtures import table_from_rows

NAN = np.nan
PORTS = ("A1", "B3", "C2")
MAGS = (4, 2, 0)


def visits(arms, mags=MAGS):
    """A one-problem mouse that visits the given arms in order (payouts fixed per arm)."""
    rows = [
        ((1, 1), 0, t, False, PORTS[a], mags[a], dict(zip(PORTS, mags, strict=True)))
        for t, a in enumerate(arms)
    ]
    return table_from_rows(rows)


def test_the_age_of_an_arm_is_the_trials_since_it_was_last_chosen():
    state = rs.mouse_state(visits([0, 1, 2, 0, 0, 1]))
    expected = np.array(
        [[0, NAN, NAN], [1, 0, NAN], [2, 1, 0], [0, 2, 1], [0, 3, 2], [1, 0, 3]], dtype=float
    )
    np.testing.assert_array_equal(ws.trials_since_visit(state), expected)


def test_the_ages_of_the_best_and_worst_alternative_follow_their_believed_values():
    state = rs.mouse_state(visits([0, 1, 2, 0, 0, 1]))
    table = ws.with_ages(state, rd.decision_table(state))
    by_row = table.set_index("t")
    # row 2: own C; A (4) is the best alternative, last visited 2 trials ago; B (2) 1 trial ago
    assert (by_row.loc[2, "age_best"], by_row.loc[2, "age_worst"]) == (2, 1)
    # row 3: own A; B (2) is the best alternative (age 2), C (0) the worst (age 1)
    assert (by_row.loc[3, "age_best"], by_row.loc[3, "age_worst"]) == (2, 1)
    assert by_row.loc[3, "worst_pos"] == 0 and by_row.loc[2, "worst_pos"] == 1
    np.testing.assert_allclose(by_row.loc[3, "log_age_best"], np.log1p(2))


def test_a_state_is_consistent_with_one_zero_arm_when_exactly_one_belief_is_zero():
    table = pd.DataFrame(
        dict(
            payout=[2, 2, 0, 0, 2],
            v_best=[3, 3, 3, 3, 0],
            v_worst=[0, 2, 2, 0, 0],
        )
    )
    # own pays, one alternative believed 0: ok; own pays, both alternatives pay: stale; own 0 and
    # both alternatives pay: ok; own 0 and an alternative 0: stale; two zeros: stale
    np.testing.assert_array_equal(
        ws.consistent_with_one_zero(table), [True, False, True, False, False]
    )


def test_variance_inflation_flags_collinear_regressors_and_not_independent_ones():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(2, 4000))
    independent = ws.variance_inflation(np.column_stack([a, b]))
    collinear = ws.variance_inflation(np.column_stack([a, a + 0.1 * b, b]))
    np.testing.assert_allclose(independent, 1.0, atol=0.05)
    assert collinear[0] > 50 and collinear[1] > 50


def test_perfectly_collinear_regressors_have_an_infinite_variance_inflation():
    a = np.random.default_rng(1).normal(size=100)
    assert np.isinf(ws.variance_inflation(np.column_stack([a, 2 * a]))).all()


def synthetic_decisions(flag_weight, worst_weight=0.0, n=6000, seed=0):
    """Decision-table columns with a known stay logit; ``worst_pos`` is a stale-belief flag."""
    rng = np.random.default_rng(seed)
    own_zero = (rng.random(n) < 0.3).astype(float)
    payout = (1 - own_zero) * rng.integers(1, 5, n)
    worst_pos = (rng.random(n) < 0.5).astype(float)
    v_worst = worst_pos * rng.integers(1, 5, n)
    v_best = np.maximum(v_worst, rng.integers(0, 5, n)).astype(float)
    table = pd.DataFrame(
        dict(
            problem=rng.integers(0, 6, n),
            payout=payout,
            own_zero=own_zero,
            v_best=v_best,
            v_worst=v_worst,
            worst_pos=worst_pos,
            log_age_best=np.log1p(rng.integers(1, 30, n)),
            log_age_worst=np.log1p(rng.integers(1, 30, n)),
            **{c: rng.normal(scale=0.3, size=n) for c in ("run_log", "trace")},
            **{c: np.zeros(n) for c in ("lag2", "lag3", "lag4", "plus_own")},
        )
    )
    logit = 0.3 + 0.4 * payout - 0.3 * v_best + worst_weight * v_worst + flag_weight * worst_pos
    table["leave"] = (rng.random(n) >= 1 / (1 + np.exp(-logit))).astype(float)
    return table


def test_a_stale_belief_flag_explains_a_positive_worst_weight_that_is_not_about_the_value():
    table = synthetic_decisions(flag_weight=1.0)
    out = ws.worst_weight_split(table).readouts
    assert out["ws_w_worst_ages"] > 0.15  # the value stands in for the flag
    assert abs(out["ws_w_worst_flag"]) < 0.15
    assert out["ws_w_flag"] == pytest.approx(1.0, abs=0.3)
    assert out["ws_w_best_ages"] == pytest.approx(-0.3, abs=0.1)


def test_without_a_flag_effect_the_worst_weight_is_the_value_weight():
    table = synthetic_decisions(flag_weight=0.0, worst_weight=0.4, seed=3)
    out = ws.worst_weight_split(table).readouts
    assert out["ws_w_worst_flag"] == pytest.approx(0.4, abs=0.12)
    assert abs(out["ws_w_flag"]) < 0.3


def test_the_split_by_own_payout_recovers_the_weight_within_each_cell():
    table = synthetic_decisions(flag_weight=1.0, seed=5)
    out = ws.worst_weight_split(table).readouts
    assert out["ws_w_flag_nz"] == pytest.approx(1.0, abs=0.35)
    assert out["ws_w_flag_z"] == pytest.approx(1.0, abs=0.5)
    assert out["ws_w_best_nz"] == pytest.approx(-0.3, abs=0.12)


def test_the_unpenalized_fit_agrees_with_the_ridge_fit_on_plenty_of_data():
    table = synthetic_decisions(flag_weight=1.0, n=20000, seed=7)
    out = ws.worst_weight_split(table).readouts
    assert out["ws_w_worst_noridge"] == pytest.approx(out["ws_w_worst_ages"], abs=0.03)
    assert out["ws_w_worst_flag_noridge"] == pytest.approx(out["ws_w_worst_flag"], abs=0.03)


def test_regressor_checks_are_reported():
    table = synthetic_decisions(flag_weight=1.0)
    out = ws.worst_weight_split(table).readouts
    # the value of the worst alternative is built from the flag: strongly correlated, not perfectly
    assert 0.6 < out["ws_corr_worst_flag"] < 0.99
    assert out["ws_vif_worst_full"] > out["ws_vif_worst_base"] > 1.0
    assert out["ws_vif_max_full"] >= out["ws_vif_worst_full"]
    assert abs(out["ws_corr_worst_payout"]) < 0.1


def test_stay_by_consistency_cell_and_the_share_of_stale_states():
    table = synthetic_decisions(flag_weight=1.0, seed=2)
    out = ws.worst_weight_split(table).readouts
    ok = ws.consistent_with_one_zero(table).to_numpy()
    paid = table["own_zero"].to_numpy() == 0
    stay = 1 - table["leave"].to_numpy()
    assert out["ws_stay_stale_nz"] == pytest.approx(stay[paid & ~ok].mean())
    assert out["ws_stay_ok_nz"] == pytest.approx(stay[paid & ok].mean())
    assert out["ws_share_stale_nz"] == pytest.approx((paid & ~ok).sum() / paid.sum())
    assert out["ws_stay_stale_nz"] > out["ws_stay_ok_nz"]  # stale beliefs: staying is higher here


def test_a_cell_that_cannot_identify_the_worst_weight_gives_nan_not_an_error():
    table = synthetic_decisions(flag_weight=1.0)
    table = table.assign(v_worst=0.0, worst_pos=0.0)  # the worst alternative is always believed 0
    out = ws.worst_weight_split(table).readouts
    assert np.isnan(out["ws_w_worst_ages"]) and np.isnan(out["ws_w_flag"])
    assert np.isfinite(out["ws_w_best_ages"])


def test_too_few_decisions_in_a_cell_give_nan():
    table = synthetic_decisions(flag_weight=1.0, n=3000)
    table = table.assign(own_zero=0.0).iloc[:5000]
    table.loc[table.index[:50], "own_zero"] = 1.0  # fewer than the cell minimum
    out = ws.worst_weight_split(table).readouts
    assert np.isnan(out["ws_w_worst_z"]) and np.isfinite(out["ws_w_worst_nz"])


def test_the_listed_readouts_are_exactly_the_ones_returned():
    out = ws.worst_weight_split(synthetic_decisions(flag_weight=1.0)).readouts
    assert set(out) == set(ws.READOUTS) and len(ws.READOUTS) == len(set(ws.READOUTS))
