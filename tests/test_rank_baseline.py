"""The held-out baseline comparison: a model subset, cached, with an optional shared cache."""

import warnings

import pandas as pd
import pytest

from src.behavior_analysis import strategy_models as sm
from src.pipelines import rank_baseline as rb
from src.pipelines import strategy_comparison as strategy
from tests.strategy_fixtures import simulate_mouse

MODELS = (sm.WSLS_MAGNITUDE, sm.WSLS_HABIT)
SIZE = dict(n_problems=3, sessions=2, trials=150)
COMPARISONS = ((sm.WSLS_HABIT, sm.WSLS_MAGNITUDE),)


@pytest.fixture(scope="module")
def cohort():
    return pd.concat(
        [
            simulate_mouse(sm.WSLS_MAGNITUDE, seed=1, mouse="m1", **SIZE),
            simulate_mouse(sm.WSLS_HABIT, seed=2, mouse="m2", **SIZE),
        ],
        ignore_index=True,
    )


@pytest.fixture(scope="module")
def result(cohort):
    return rb.heldout_baseline(cohort, MODELS, comparisons=COMPARISONS, n_starts=2)


def test_every_mouse_and_model_has_held_out_nats_per_trial(result):
    assert set(zip(result.nats["mouse"], result.nats["model"], strict=True)) == {
        (m, model) for m in ("m1", "m2") for model in MODELS
    }
    assert (result.nats["nats_per_choice"] > 0).all()


def test_the_fits_of_every_model_are_kept_for_replays(result):
    assert set(result.fits) == {"m1", "m2"}
    assert set(result.fits["m1"]) == set(MODELS)


def test_the_paired_difference_is_the_first_model_minus_the_second(result):
    wide = result.nats.pivot(index="mouse", columns="model", values="nats_per_choice")
    expected = wide[sm.WSLS_HABIT] - wide[sm.WSLS_MAGNITUDE]
    got = result.differences.set_index("mouse")[sm.comparison_name(*COMPARISONS[0])]
    assert got.to_dict() == pytest.approx(expected.to_dict())


def test_a_cached_run_gives_the_same_numbers(cohort, tmp_path):
    first = rb.heldout_baseline(cohort, MODELS, n_starts=2, cache_dir=tmp_path)
    again = rb.heldout_baseline(cohort, MODELS, n_starts=2, cache_dir=tmp_path)
    pd.testing.assert_frame_equal(first.nats, again.nats)


def test_a_mouse_in_the_shared_cache_is_not_refitted(cohort, tmp_path, monkeypatch):
    shared = tmp_path / "shared"
    mine = cohort[cohort["mouse"] == "m1"].reset_index(drop=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # a full comparison of one mouse in a strategy cache, as a notebook run leaves it
        strategy.mouse_comparer(shared)(mine, 2, 0, strategy.code_version())
    fitted = []
    real = sm.cross_validate

    def spy(rows, *args, **kwargs):
        fitted.append(rows["mouse"].iloc[0])
        return real(rows, *args, **kwargs)

    monkeypatch.setattr(sm, "cross_validate", spy)
    rb.heldout_baseline(
        cohort, MODELS, n_starts=2, cache_dir=tmp_path / "mine", shared_cache_dir=shared
    )
    assert fitted == ["m2"]


def test_no_trials_raises():
    with pytest.raises(ValueError, match="no trials"):
        rb.heldout_baseline(pd.DataFrame(columns=["mouse", "problem"]), MODELS)


def test_a_comparison_outside_the_models_raises(cohort):
    with pytest.raises(ValueError, match="outside"):
        rb.heldout_baseline(cohort, MODELS, comparisons=((sm.KERNEL_Q, sm.WSLS_HABIT),))
