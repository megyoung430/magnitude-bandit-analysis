"""WSLS (Habit): WSLS (Magnitude) whose stay logit also grows with the choice kernel."""

import numpy as np
import pytest
from scipy.special import expit

from src.agents import kernel_q_agent as kq
from src.agents import null_agents as na
from tests.test_null_agents import hand_table, simulated_table

HABIT = na.WslsHabitParams(a_win=0.4, a_lose=-0.5, b=0.3, w=0.8, w_h=1.5, alpha_k=0.25)
SAME_STAY_RULE = na.WslsMagnitudeParams(a_win=0.4, a_lose=-0.5, b=0.3, w=0.8)


def test_habit_is_a_registered_kind_after_the_others():
    assert na.KINDS[-1] == "wsls_habit"
    assert na.AGENT_PARAMS["wsls_habit"] is na.WslsHabitParams


def test_parameters_are_validated_and_frozen():
    with pytest.raises(ValueError, match="alpha_k"):
        na.WslsHabitParams(0.4, -0.5, 0.3, 0.8, 1.0, alpha_k=1.5)
    with pytest.raises(ValueError, match="alpha_k"):
        na.WslsHabitParams(0.4, -0.5, 0.3, 0.8, 1.0, alpha_k=0.0)
    with pytest.raises(ValueError, match="finite"):
        na.WslsHabitParams(0.4, -0.5, 0.3, 0.8, np.nan, 0.25)
    with pytest.raises(AttributeError):
        HABIT.w_h = 0.0


# ---------------------------------------------------------------------------
# The kernel of the current arm
# ---------------------------------------------------------------------------


def test_kernel_of_the_current_arm_grows_along_a_run_and_restarts_after_a_switch():
    table = hand_table([0, 0, 0, 1, 1, 0], [1] * 6)
    kernel = na.current_kernel(table, 0.5)
    # decision t is about the arm chosen at t-1; its kernel includes that choice
    assert np.isnan(kernel[0])
    assert kernel[1:] == pytest.approx([0.5, 0.75, 0.875, 0.5, 0.75])


def test_the_kernel_is_cleared_at_a_session_start_and_after_a_gap():
    table = hand_table([0, 0, 0, 0, 0, 0], [1] * 6, starts=(3,), gaps=(5,))
    kernel = na.current_kernel(table, 0.5)
    assert np.isnan(kernel[[0, 3, 5]]).all()
    assert kernel[[1, 2, 4]] == pytest.approx([0.5, 0.75, 0.5])


def test_the_kernel_is_the_one_of_the_kernel_q_learner():
    rng = np.random.default_rng(0)
    choices = rng.integers(0, 3, 40).tolist()
    kernel = na.current_kernel(hand_table(choices, [1] * 40), 0.3)
    k = np.zeros(3)
    for t, choice in enumerate(choices[:-1]):
        _, k = kq.update_state(np.zeros(3), k, choice, 0.0, (0.5, 0.5, 1.0, 0.3, 1.0, 0.0, 0.0))
        assert kernel[t + 1] == pytest.approx(k[choice])


# ---------------------------------------------------------------------------
# Choice rule and likelihood
# ---------------------------------------------------------------------------


def test_stay_probability_follows_the_rule_by_hand():
    # previous arm 0 paid 2 pulses; its kernel is 0.75
    kernel = np.array([0.75, 0.1, 0.0])
    p = na.choice_probabilities(HABIT, 0, 2.0, np.zeros(3), kernel=kernel)
    stay = expit(0.4 + 0.3 * (2 - 1) + 1.5 * 0.75)
    assert p[0] == pytest.approx(stay)
    assert p.sum() == pytest.approx(1.0)
    unrewarded = na.choice_probabilities(HABIT, 0, 0.0, np.zeros(3), kernel=kernel)
    assert unrewarded[0] == pytest.approx(expit(-0.5 + 1.5 * 0.75))


def test_without_a_habit_weight_it_is_wsls_magnitude():
    table = simulated_table(SAME_STAY_RULE, seed=3)
    flat = na.WslsHabitParams(0.4, -0.5, 0.3, 0.8, w_h=0.0, alpha_k=0.1)
    assert na.log_likelihood(table, flat) == pytest.approx(na.log_likelihood(table, SAME_STAY_RULE))
    table = simulated_table(HABIT, seed=3)
    assert na.log_likelihood(table, flat) == pytest.approx(na.log_likelihood(table, SAME_STAY_RULE))


def test_the_policy_and_the_likelihood_are_the_same_model():
    table = simulated_table(HABIT, seed=4)
    policy, total = na.NullPolicy(HABIT), 0.0
    ports = sorted(table["magnitudes"].iloc[0])
    for row in table.sort_values("trial").itertuples():
        if row.trial_in_session == 0:
            policy.start_session()
        else:
            total += np.log(policy.probabilities()[ports.index(row.choice)])
        policy.update(ports.index(row.choice), row.reward)
    n_start = na.decisions(table, na.DEFAULT_DECAY).start.sum()
    assert na.log_likelihood(table, HABIT) - n_start * np.log(1 / 3) == pytest.approx(total)


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------


def _long_habit_table(params, seeds=(1, 2, 3)):
    import pandas as pd

    tables = [simulated_table(params, seed=s).assign(problem=s) for s in seeds]
    return pd.concat(tables, ignore_index=True)


def test_fit_recovers_the_habit_weight_and_its_rate():
    table = _long_habit_table(HABIT)
    fitted = na.fit_null_agents(table, "wsls_habit")["m"]
    assert isinstance(fitted, na.WslsHabitParams)
    assert fitted.w_h == pytest.approx(HABIT.w_h, abs=0.5)
    assert fitted.alpha_k in na.HABIT_RATES
    assert 0.1 <= fitted.alpha_k <= 0.5  # truth 0.25 on a grid of rates
    assert (
        na.log_likelihood(table, fitted) >= na.log_likelihood(table, HABIT) - 20.0
    )  # the rate is on a grid


def test_fit_finds_no_habit_where_there_is_none():
    table = _long_habit_table(SAME_STAY_RULE)
    fitted = na.fit_null_agents(table, "wsls_habit")["m"]
    assert abs(fitted.w_h) < 0.6


def test_the_fit_is_never_worse_than_wsls_magnitude_on_its_own_data():
    table = _long_habit_table(HABIT)
    habit = na.fit_null_agents(table, "wsls_habit")["m"]
    plain = na.fit_null_agents(table, "wsls_magnitude")["m"]
    assert na.log_likelihood(table, habit) >= na.log_likelihood(table, plain) - 1e-6
