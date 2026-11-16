"""Mice, replays and calibration through the pipeline, on small simulated mice."""

import pandas as pd
import pytest

from src.behavior_analysis.rank_tests import rank_signatures as sg
from src.behavior_analysis.strategy_models import strategy_models as sm
from src.pipelines.rank_tests import rank_signatures as pipe
from tests.strategy_fixtures import PARAMETERS, RULES, simulate_mouse

SIZE = dict(n_problems=4, sessions=2, trials=250)
MODELS = (sm.WSLS_MAGNITUDE,)
pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture(scope="module")
def cohort():
    return pd.concat(
        [
            simulate_mouse(sm.WSLS_MAGNITUDE, seed=1, mouse="m1", **SIZE),
            simulate_mouse(sm.KERNEL_Q, seed=2, mouse="m2", **SIZE),
        ],
        ignore_index=True,
    )


@pytest.fixture(scope="module")
def fits():
    # the generating parameters stand in for fitted ones: the pipeline only plays them
    return {m: {model: PARAMETERS[model] for model in pipe.MODELS} for m in ("m1", "m2")}


@pytest.fixture(scope="module")
def mice(cohort, fits):
    return pipe.analyze_mice(cohort, {m: f[sm.KERNEL_Q] for m, f in fits.items()})


@pytest.fixture(scope="module")
def replays(cohort, fits):
    return pipe.replay_source(cohort, fits, models=MODELS, replicates=3, rules=RULES)


def test_every_mouse_has_every_readout_and_curves(mice):
    assert list(mice.readouts["mouse"]) == ["m1", "m2"] and not mice.skipped
    assert {"w_diff", "step_weight", "q_w_diff", "worst_slope", "slope_p100", "ws_w_flag"} <= set(
        mice.readouts.columns
    )
    assert set(mice.curves["mouse"]) == {"m1", "m2"}


def test_a_mouse_with_too_little_data_is_skipped_and_named(cohort):
    short = cohort[(cohort["mouse"] != "m2") | (cohort["trial"] < 5)]
    with pytest.warns(UserWarning, match="mouse m2 skipped"):
        source = pipe.analyze_mice(short)
    assert list(source.readouts["mouse"]) == ["m1"] and "m2" in source.skipped


def test_no_analyzable_mouse_raises(cohort):
    with pytest.raises(ValueError, match="no mouse can be analyzed"):
        pipe.analyze_mice(cohort[cohort["trial"] < 5])


def test_a_cached_analysis_gives_the_same_numbers(cohort, tmp_path):
    first = pipe.analyze_mice(cohort, cache_dir=tmp_path)
    again = pipe.analyze_mice(cohort, cache_dir=tmp_path)
    pd.testing.assert_frame_equal(first.readouts, again.readouts)


def test_replays_give_a_row_per_replicate_and_mouse_and_a_mean_curve(replays):
    table = replays.readouts[sm.WSLS_MAGNITUDE]
    assert sorted(set(table["mouse"])) == ["m1", "m2"]
    assert len(table) + replays.failed["n_failed"].sum() == 2 * 3
    assert {"step_weight", "w_diff"} <= set(table.columns)
    assert set(replays.curves[sm.WSLS_MAGNITUDE]["mouse"]) == {"m1", "m2"}


def test_replays_are_reproducible_from_the_seed(cohort, fits, replays):
    again = pipe.replay_source(cohort, fits, models=MODELS, replicates=3, rules=RULES)
    pd.testing.assert_frame_equal(
        replays.readouts[sm.WSLS_MAGNITUDE], again.readouts[sm.WSLS_MAGNITUDE]
    )
    other = pipe.replay_source(cohort, fits, models=MODELS, replicates=3, rules=RULES, seed=5)
    assert not other.readouts[sm.WSLS_MAGNITUDE]["step_weight"].equals(
        replays.readouts[sm.WSLS_MAGNITUDE]["step_weight"]
    )


def test_a_replay_with_too_little_data_is_counted_not_dropped_silently(cohort, fits, monkeypatch):
    real = sg.analyze_mouse
    calls = []

    def flaky(trials, kq_params=None):
        calls.append(1)
        if len(calls) % 2 == 0:
            raise sg.UnanalyzableError("too short")
        return real(trials, kq_params)

    monkeypatch.setattr(sg, "analyze_mouse", flaky)
    out = pipe.replay_source(cohort, fits, models=MODELS, replicates=4, rules=RULES)
    assert (out.failed["n_failed"] == 2).all()
    assert len(out.readouts[sm.WSLS_MAGNITUDE]) == 4  # two mice, two kept each


def test_calibration_gives_a_z_and_p_per_mouse_readout_and_agent(mice, replays):
    calibration = pipe.calibrate_source(mice.readouts, replays)
    table = calibration.table
    assert set(table["agent"]) == {sm.WSLS_MAGNITUDE} and set(table["mouse"]) == {"m1", "m2"}
    assert {"z", "p_empirical", "passes", "predicted"} <= set(table.columns)
    assert {"mean", "n_pass", "passing"} <= set(calibration.summary.columns)


def test_calibration_raises_when_every_replay_of_a_mouse_failed(mice, replays):
    failed = replays.failed.assign(n_failed=replays.replicates)
    with pytest.raises(ValueError, match="too little data"):
        pipe.calibrate_source(mice.readouts, replays._replace(failed=failed))


def test_fewer_than_two_replicates_raise(cohort, fits):
    with pytest.raises(ValueError, match="at least 2"):
        pipe.replay_source(cohort, fits, models=MODELS, replicates=1, rules=RULES)


def test_a_mouse_without_kernel_q_fits_raises(cohort, fits):
    broken = {"m1": {sm.WSLS_MAGNITUDE: PARAMETERS[sm.WSLS_MAGNITUDE]}, "m2": fits["m2"]}
    with pytest.raises(ValueError, match="missing fits"):
        pipe.replay_source(cohort, broken, models=MODELS, replicates=2, rules=RULES)


def test_the_cache_version_is_a_short_digest_of_the_analysis_code():
    assert len(pipe.code_version()) == 16
