"""Per-mouse null calibration of the switch regression: structure, caching and validation.

The calibration of a mouse plays its own fitted win-stay/lose-shift agents on its own task and
puts its value weight in that null distribution. Validated both ways: a value learner's believed
value lies beyond its null, and an agent that ignores value ("a null mouse") gets calibrated p
values spread over 0-1 rather than piled up near 0.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from src.agents import drifting_simulation as sim
from src.agents import null_agents as na
from src.behavior_analysis import switch_regression as sr
from src.pipelines import choice_logit_values as logit_pipe
from src.pipelines import null_agents as null_pipe
from src.pipelines import switch_regression as pipe

RULES = sim.TaskRules(drift_rate=0.1, block_lengths=np.arange(30, 36))
GRID = (0.5, 0.9)
SIZE = dict(n_problems=6, sessions_per_problem=2, trials_per_session=250)
pytestmark = pytest.mark.filterwarnings("ignore:the problems of")


def _analyzed(trials):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pipe.analyze_switch_source(trials, None, trace_grid=GRID, min_switches=60)


@pytest.fixture(scope="module")
def q_learners():
    return logit_pipe.simulate_cohort(
        [(0.3, 1.2, 0.0, 0.0), (0.3, 1.0, 1.0, 0.5)], mice_per_set=2, seed=9, **SIZE
    ).trials


@pytest.fixture(scope="module")
def q_calibration(q_learners):
    source = _analyzed(q_learners)
    fits = {"wsls_magnitude": na.fit_null_agents(q_learners, "wsls_magnitude")}
    return pipe.calibrate_source(
        q_learners, source, fits, agents=("wsls_magnitude",), replicates=19, rules=RULES, seed=1
    )


def test_calibration_has_every_mouse_condition_value_and_term(q_calibration, q_learners):
    table = q_calibration.calibration
    assert set(table["mouse"]) == set(q_learners["mouse"])
    assert set(table["value"]) == set(pipe.CALIBRATED_VALUES)
    assert set(table["term"]) == {"beta", "gain"}
    assert len(table) == 4 * 2 * 2 * 2
    assert (table["agent"] == "wsls_magnitude").all()
    assert (table["n_null"] == 19).all()
    assert set(q_calibration.null["replicate"]) == set(range(19))
    assert (q_calibration.failed["n_failed"] == 0).all()


def test_a_value_learners_believed_value_lies_beyond_its_null(q_calibration):
    believed = q_calibration.calibration.query("value == 'believed' and term == 'beta'")
    assert np.allclose(believed["p_empirical"], 1 / 20)  # beyond all 19 nulls
    assert (believed["z"] > 3).all()


def test_agents_that_ignore_value_get_calibrated_p_values_spread_over_0_1(q_learners):
    mice = sorted(q_learners["mouse"].unique())
    # strong recency: the case whose raw value weight is least reliable
    params = {m: na.WslsMagnitudeParams(0.2, -0.5, 0.5, 1.5, 0.9) for m in mice}
    null_mice = null_pipe.simulate_null_cohort(q_learners, params, "wsls_magnitude", 2, RULES)
    trials = null_mice.trials
    source = _analyzed(trials)
    fits = {"wsls_magnitude": na.fit_null_agents(trials, "wsls_magnitude")}
    result = pipe.calibrate_source(
        trials, source, fits, agents=("wsls_magnitude",), replicates=19, rules=RULES, seed=3
    )
    p = result.calibration.query("term == 'beta'")["p_empirical"]
    assert len(p) == 16
    # uniform p on {1/20, ..., 1}: about 1.6 of 16 below 0.1; mean about 0.5
    assert (p <= 0.1).sum() <= 4
    assert 0.3 < p.mean() < 0.7


def test_the_calibration_is_reproducible_and_cached(q_learners, tmp_path):
    two = q_learners[q_learners["mouse"].isin(sorted(q_learners["mouse"].unique())[:1])]
    source = _analyzed(two)
    fits = {kind: na.fit_null_agents(two, kind) for kind in pipe.NULL_AGENTS}
    kwargs = dict(replicates=3, rules=RULES, seed=5, cache_dir=tmp_path)
    first = pipe.calibrate_source(two, source, fits, **kwargs)
    assert any(tmp_path.rglob("*.pkl"))
    again = pipe.calibrate_source(two, source, fits, n_jobs=2, **kwargs)
    pd.testing.assert_frame_equal(first.calibration, again.calibration)
    other = pipe.calibrate_source(two, source, fits, replicates=3, rules=RULES, seed=6)
    assert not np.allclose(first.null["beta"], other.null["beta"])
    assert list(first.calibration["agent"].unique()) == list(pipe.NULL_AGENTS)
    assert set(first.summary["agent"]) == set(pipe.NULL_AGENTS)


def test_the_calibration_validates_its_inputs(q_learners):
    source = _analyzed(q_learners)
    fits = {"wsls_magnitude": {}}
    with pytest.raises(ValueError, match="no fitted null agent"):
        pipe.calibrate_source(q_learners, source, fits, agents=("wsls_magnitude",), rules=RULES)
    with pytest.raises(ValueError, match="unknown null agents"):
        pipe.calibrate_source(q_learners, source, fits, agents=("q",), rules=RULES)
    with pytest.raises(ValueError, match="at least 2"):
        pipe.calibrate_source(q_learners, source, fits, replicates=1, rules=RULES)


def test_replicate_seeds_are_distinct():
    seeds = {pipe.replicate_seed(s, r) for s in range(3) for r in range(100)}
    assert len(seeds) == 300
    assert sr.CONDITIONS  # the module under calibration is importable alongside
