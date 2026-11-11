"""The second-best pipeline: sources, minimums, per-mouse null replicates and calibration."""

import inspect
import warnings

import numpy as np
import pandas as pd
import pytest

from src.agents import drifting_simulation as sim
from src.agents import null_agents as na
from src.behavior_analysis import second_best as sb
from src.pipelines import choice_logit_values as logit_pipe
from src.pipelines import null_agents as null_pipe
from src.pipelines import second_best as pipe
from src.pipelines import switch_regression as switch_pipe
from tests.second_best_fixtures import hand_mouse
from tests.switch_regression_fixtures import simulate_switch_mouse

RULES = sim.TaskRules(drift_rate=0.1, block_lengths=np.arange(30, 36))
SIZE = dict(n_problems=4, sessions_per_problem=2, trials_per_session=250)
pytestmark = [
    pytest.mark.filterwarnings("ignore:the problems of"),
    pytest.mark.filterwarnings("ignore:second_best, after_zero"),
]


def _cohort(n_mice, seed):
    return pd.concat(
        [
            simulate_switch_mouse(f"m{i}", seed + i, value_weight=0.5, trace_weight=0.5, **SIZE)
            for i in range(1, n_mice + 1)
        ],
        ignore_index=True,
    )


@pytest.fixture(scope="module")
def cohort():
    return _cohort(3, 50)


@pytest.fixture(scope="module")
def source(cohort):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pipe.analyze_second_best_source(cohort, min_used=20)


@pytest.fixture(scope="module")
def q_learners():
    return logit_pipe.simulate_cohort(
        [(0.3, 2.0, 0.0, 0.0), (0.3, 1.5, 1.0, 0.5)],
        mice_per_set=1,
        seed=9,
        n_problems=6,
        sessions_per_problem=2,
        trials_per_session=250,
    ).trials


# ---------------------------------------------------------------------------
# One source
# ---------------------------------------------------------------------------


def test_every_mouse_has_its_cells_with_enough_used_switches(source, cohort):
    assert set(source.proportions["mouse"]) == set(cohort["mouse"])
    assert (source.proportions["n_used"] >= 20).all()
    assert set(source.proportions["measure"]) == set(sb.MEASURES)
    assert set(source.graded["measure"]) == set(sb.GRADED_MEASURES)


def test_cells_below_the_minimum_are_dropped_named_and_warned_about(cohort):
    with pytest.warns(UserWarning, match=r"second_best, after_zero: 3 of 3 mice"):
        result = pipe.analyze_second_best_source(cohort, min_used=20)
    dropped = result.dropped
    zero = dropped[(dropped["measure"] == sb.SECOND_BEST) & (dropped["condition"] == "after_zero")]
    assert len(zero) == 3 and (zero["n_used"] == 0).all()  # leaving the best never follows a zero
    assert result.min_used == 20


def test_a_mouse_that_never_switches_is_skipped_with_a_warning(cohort):
    stays = hand_mouse("stay").iloc[:1]
    with pytest.warns(UserWarning, match="mouse stay skipped"):
        result = pipe.analyze_second_best_source(pd.concat([cohort, stays]), min_used=20)
    assert "never switches" in result.skipped["stay"]
    assert "stay" not in set(result.proportions["mouse"])


def test_no_cell_with_enough_switches_raises(cohort):
    with pytest.raises(ValueError, match="no cell has at least 100000"):
        pipe.analyze_second_best_source(cohort, min_used=100_000)


def test_sources_are_analyzed_in_order(cohort):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sources = pipe.analyze_second_best_sources({"A": cohort, "B": cohort[:2000]}, min_used=20)
    assert list(sources) == ["A", "B"]
    assert set(sources["B"].proportions["mouse"]) == {"m1"}


def test_source_summaries_and_per_mouse_table(source):
    summary = pipe.source_summaries({"Mice": source})
    assert summary["source"].eq("Mice").all()
    assert summary["mean"].isna().all()  # three mice: no mean, se or test
    table = pipe.per_mouse_table({"Mice": source})
    assert list(table.columns[:2]) == ["source", "mouse"]
    assert len(table) == len(source.proportions)


# ---------------------------------------------------------------------------
# Null replicates and calibration
# ---------------------------------------------------------------------------


def test_the_code_version_follows_the_module_sources(monkeypatch):
    before = pipe.code_version()
    assert before == pipe.code_version()
    real = inspect.getsource
    monkeypatch.setattr(inspect, "getsource", lambda module: real(module) + "# edit\n")
    assert pipe.code_version() != before


@pytest.fixture(scope="module")
def q_calibration(q_learners):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        source = pipe.analyze_second_best_source(q_learners, min_used=20)
        fits = {"wsls_magnitude": na.fit_null_agents(q_learners, "wsls_magnitude")}
        calibration = pipe.calibrate_second_best(
            q_learners, source, fits, agents=("wsls_magnitude",), replicates=19, rules=RULES, seed=1
        )
    return source, calibration


def test_calibration_has_every_kept_cell_of_every_mouse(q_calibration):
    source, calibration = q_calibration
    table = calibration.calibration
    assert len(table) == len(source.proportions)
    assert set(table["agent"]) == {"wsls_magnitude"}
    assert (table["n_null"] > 0).all() and (table["n_null"] <= 19).all()
    assert set(calibration.null["replicate"]) <= set(range(19))
    assert calibration.replicates == 19 and set(calibration.failed["agent"]) == {"wsls_magnitude"}


def test_a_value_learners_true_better_choices_lie_beyond_its_null(q_calibration):
    table = q_calibration[1].calibration
    pooled = table.query("measure == 'true_better' and condition == 'pooled'")
    assert len(pooled) == 2 and (pooled["z"] > 2).all()
    assert (pooled["percentile"] == 1.0).all()  # beyond all 19 null replicates
    assert (pooled["p_empirical"] <= 2 / 20).all()


def test_the_null_replicates_use_the_switch_regressions_seeds(q_learners):
    mouse = sorted(q_learners["mouse"].unique())[0]
    trials = q_learners[q_learners["mouse"] == mouse].reset_index(drop=True)
    params = na.fit_null_agents(trials, "wsls_magnitude")[mouse]
    settings = (trials, params, "wsls_magnitude", RULES, 2, 5, sb.cl.DEFAULT_LAGS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        direct = pipe.simulate_second_best_null(*settings, pipe.code_version())
        replay = null_pipe.simulate_null_cohort(
            trials, {mouse: params}, "wsls_magnitude", switch_pipe.replicate_seed(5, 1), RULES
        ).trials
        expected = sb.analyze_mouse(replay).proportions
    got = direct.values[direct.values["replicate"] == 1].drop(columns="replicate")
    pd.testing.assert_frame_equal(got.reset_index(drop=True), expected)


def test_the_calibration_is_reproducible_and_cached(q_learners, tmp_path):
    one = q_learners[q_learners["mouse"] == sorted(q_learners["mouse"].unique())[0]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        source = pipe.analyze_second_best_source(one, min_used=20)
        fits = {kind: na.fit_null_agents(one, kind) for kind in switch_pipe.NULL_AGENTS}
        kwargs = dict(replicates=3, rules=RULES, seed=5, cache_dir=tmp_path)
        first = pipe.calibrate_second_best(one, source, fits, **kwargs)
        assert any(tmp_path.rglob("*.pkl"))
        again = pipe.calibrate_second_best(one, source, fits, n_jobs=2, **kwargs)
        other = pipe.calibrate_second_best(one, source, fits, replicates=3, rules=RULES, seed=6)
    pd.testing.assert_frame_equal(first.calibration, again.calibration)
    assert not np.allclose(first.null["p"], other.null["p"])
    assert list(first.calibration["agent"].unique()) == list(switch_pipe.NULL_AGENTS)


def test_the_calibration_validates_its_inputs(q_learners, source):
    fits = {"wsls_magnitude": {}}
    with pytest.raises(ValueError, match="no fitted null agent"):
        pipe.calibrate_second_best(
            q_learners, source, fits, agents=("wsls_magnitude",), rules=RULES
        )
    with pytest.raises(ValueError, match="unknown null agents"):
        pipe.calibrate_second_best(q_learners, source, fits, agents=("q",), rules=RULES)
    with pytest.raises(ValueError, match="at least 2"):
        pipe.calibrate_second_best(q_learners, source, fits, replicates=1, rules=RULES)
