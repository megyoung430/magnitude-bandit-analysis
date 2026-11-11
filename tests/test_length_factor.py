"""Longer schedules for the comparison agents; the null calibration keeps the mouse's own length."""

import numpy as np
import pandas as pd
import pytest

from src.agents import drifting_simulation as sim
from src.agents import null_agents as na
from src.behavior_analysis import strategy_models as sm
from src.pipelines import fitted_q as fq
from src.pipelines import null_agents as nulls
from src.pipelines import second_best as second
from src.pipelines import switch_regression as switch_pipe
from tests.strategy_fixtures import RULES, simulate_mouse

pytestmark = pytest.mark.filterwarnings("ignore:the problems of")
SIZE = dict(n_problems=3, sessions=2, trials=200)
Q_PARAMS = (0.3, 2.0, 0.5, 0.3)


@pytest.fixture(scope="module")
def mice():
    return pd.concat(
        [simulate_mouse(sm.Q_LEARNER, seed=7 + i, mouse=f"m{i}", **SIZE) for i in (1, 2)],
        ignore_index=True,
    )


@pytest.fixture(scope="module")
def wsls(mice):
    return na.fit_null_agents(mice, "wsls_magnitude")


def _counts(trials):
    return trials.groupby("mouse").size()


# ---------------------------------------------------------------------------
# The factor
# ---------------------------------------------------------------------------


def test_lengthening_repeats_each_problems_sessions_cyclically():
    layout = sim.MouseLayout(["B1", "B2", "B3"], "B2", {"B1": 1, "B2": 2, "B3": 3}, [100, 150, 80])
    longer = nulls.lengthen_layouts({1: layout}, 2.0)[1]
    assert longer.session_lengths == [100, 150, 80, 100, 150, 80]
    assert (longer.ports, longer.initiation, longer.start_magnitudes) == layout[:3]
    assert nulls.lengthen_layouts({1: layout}, 1.0)[1] == layout
    assert nulls.lengthen_layouts({1: layout}, 0.4)[1].session_lengths == [100]  # at least one


@pytest.mark.parametrize("factor", [0, -1.0, float("nan")])
def test_a_nonpositive_factor_raises(factor):
    layout = sim.MouseLayout(["B1", "B2", "B3"], "B2", {"B1": 1, "B2": 2, "B3": 3}, [100])
    with pytest.raises(ValueError, match="length_factor"):
        nulls.lengthen_layouts({1: layout}, factor)


def test_the_null_agents_play_the_scaled_number_of_trials(mice, wsls):
    base = nulls.simulate_null_cohort(mice, wsls, "wsls_magnitude", 1, RULES)
    twice = nulls.simulate_null_cohort(mice, wsls, "wsls_magnitude", 1, RULES, length_factor=2.0)
    assert (_counts(twice.trials) == 2 * _counts(base.trials)).all()
    assert (_counts(base.trials) == _counts(mice)).all()
    assert set(twice.trials["problem"]) == set(mice["problem"])  # same problems, more sessions


def test_the_fitted_q_learners_play_the_scaled_number_of_trials(mice):
    fits = dict.fromkeys(("m1", "m2"), Q_PARAMS)
    base = fq.simulate_fitted_q_cohort(mice, fits, 1, RULES)
    twice = fq.simulate_fitted_q_cohort(mice, fits, 1, RULES, length_factor=2.0)
    assert (_counts(twice.trials) == 2 * _counts(base.trials)).all()


def test_factor_one_reproduces_the_simulation_exactly(mice, wsls):
    fits = dict.fromkeys(("m1", "m2"), Q_PARAMS)
    pd.testing.assert_frame_equal(
        nulls.simulate_null_cohort(mice, wsls, "wsls_magnitude", 3, RULES).trials,
        nulls.simulate_null_cohort(
            mice, wsls, "wsls_magnitude", 3, RULES, length_factor=1.0
        ).trials,
    )
    pd.testing.assert_frame_equal(
        fq.simulate_fitted_q_cohort(mice, fits, 3, RULES).trials,
        fq.simulate_fitted_q_cohort(mice, fits, 3, RULES, length_factor=1.0).trials,
    )


def test_a_longer_schedule_keeps_the_first_sessions_of_the_shorter_one(mice, wsls):
    base = nulls.simulate_null_cohort(mice, wsls, "wsls_magnitude", 3, RULES).trials
    twice = nulls.simulate_null_cohort(
        mice, wsls, "wsls_magnitude", 3, RULES, length_factor=2.0
    ).trials
    first = twice[twice["mouse"] == "m1"].head(len(base[base["mouse"] == "m1"]))
    one = base[base["mouse"] == "m1"]
    assert first["choice"].tolist()[:50] == one["choice"].tolist()[:50]


# ---------------------------------------------------------------------------
# The calibration and the power check keep the mouse's own length
# ---------------------------------------------------------------------------


@pytest.fixture
def factors(monkeypatch):
    """Every length factor that reaches the simulation, recorded."""
    seen = []
    real = nulls.lengthen_layouts

    def spy(layouts, length_factor):
        seen.append(length_factor)
        return real(layouts, length_factor)

    monkeypatch.setattr(nulls, "lengthen_layouts", spy)
    return seen


def test_the_switch_regression_calibration_ignores_the_factor(mice, wsls, factors):
    source = switch_pipe.analyze_switch_source(mice, trace_grid=(0.9,), min_switches=30)
    switch_pipe.calibrate_source(
        mice, source, {"wsls_magnitude": wsls}, agents=("wsls_magnitude",), replicates=2,
        rules=RULES,
    )  # fmt: skip
    assert factors and set(factors) == {1.0}


def test_the_second_best_calibration_ignores_the_factor(mice, wsls, factors):
    source = second.analyze_second_best_source(mice, min_used=5)
    second.calibrate_second_best(
        mice, source, {"wsls_magnitude": wsls}, agents=("wsls_magnitude",), replicates=2,
        rules=RULES,
    )  # fmt: skip
    assert factors and set(factors) == {1.0}


def test_the_power_check_ignores_the_factor(mice, factors):
    source = switch_pipe.analyze_switch_source(mice, trace_grid=(0.9,), min_switches=30)
    fq.power_check(mice, source, dict.fromkeys(("m1", "m2"), Q_PARAMS), replicates=2, rules=RULES)
    assert factors and set(factors) == {1.0}


def test_null_controls_pass_the_factor_to_every_agent(mice):
    controls = nulls.null_controls(
        mice, 1, n_lags=3, tail_decay=0.9, l2=1.0, rules=RULES, length_factor=2.0
    )
    for kind in na.KINDS:
        assert (_counts(controls.cohorts[kind].trials) == 2 * _counts(mice)).all()
    assert set(controls.fits["wsls_magnitude"]) == {"m1", "m2"}  # fitted on the mice's own data


def test_the_replicates_of_a_calibration_have_the_mouses_trial_count(mice, wsls):
    simulated = nulls.simulate_null_cohort(
        mice, {"m1": wsls["m1"]}, "wsls_magnitude", 1, RULES
    ).trials
    assert len(simulated) == int((mice["mouse"] == "m1").sum()) and np.isfinite(len(simulated))
