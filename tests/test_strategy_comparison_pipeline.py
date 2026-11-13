"""The strategy-comparison pipeline: sources, skipping, caching, and model recovery."""

import inspect
import warnings

import numpy as np
import pandas as pd
import pytest

from src.agents import drifting_simulation as sim
from src.agents import kernel_q_agent as kq
from src.behavior_analysis.strategy_models import strategy_models as sm
from src.pipelines.choice_logit import choice_logit_values as logit_pipe
from src.pipelines.strategy_models import strategy_comparison as pipe
from tests.strategy_fixtures import RULES, simulate_mouse

N_STARTS = 3
SIZE = dict(n_problems=3, sessions=2, trials=150)
pytestmark = pytest.mark.filterwarnings("ignore:the problems of")


def _cohort(models, seed=10):
    return pd.concat(
        [
            simulate_mouse(model, seed=seed + i, mouse=f"m{i}", **SIZE)
            for i, model in enumerate(models, start=1)
        ],
        ignore_index=True,
    )


@pytest.fixture(scope="module")
def cohort():
    return _cohort([sm.HEURISTIC_PLUS, sm.Q_LEARNER, sm.WSLS_MAGNITUDE])


@pytest.fixture(scope="module")
def analysis(cohort):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return logit_pipe.analyse_trial_table(cohort, n_lags=3, tail_decay=0.9, min_trials=100)


@pytest.fixture(scope="module")
def source(cohort, analysis):
    return pipe.analyze_comparison_source(
        cohort, analysis, n_starts=N_STARTS, min_trials=100, n_jobs=1
    )


# ---------------------------------------------------------------------------
# One source
# ---------------------------------------------------------------------------


def test_every_mouse_is_scored_with_every_model_in_every_fold(source, cohort):
    assert set(source.scores["mouse"]) == set(cohort["mouse"])
    per = source.scores.groupby(["mouse", "model"])["problem"].nunique()
    assert (per == 3).all() and len(per) == 3 * len(sm.COMPARED)


def test_the_tables_are_those_of_the_analysis_layer(source):
    pd.testing.assert_frame_equal(source.nats, sm.nats_table(source.scores))
    pd.testing.assert_frame_equal(source.differences, sm.difference_table(source.nats))
    assert set(source.winners["mouse"]) == set(source.nats["mouse"])


def test_the_choice_logit_is_listed_as_a_descriptive_reference(source, analysis):
    logit = source.logit.set_index("mouse")["nats_per_choice"]
    full = analysis.comparison.query("model == 'full'").set_index("mouse")["nats_per_choice"]
    assert np.allclose(logit.loc[full.index], full)


def test_fits_are_kept_for_simulating_the_models(source):
    assert set(source.fits) == set(source.nats["mouse"])
    assert set(source.fits["m1"]) == set(sm.COMPARED)


def test_a_mouse_that_cannot_be_cross_validated_is_skipped_with_its_reason(cohort):
    single = simulate_mouse(sm.WSLS_MAGNITUDE, mouse="solo", n_problems=1)
    with pytest.warns(UserWarning, match="mouse solo skipped"):
        result = pipe.analyze_comparison_source(
            pd.concat([cohort, single], ignore_index=True), None, n_starts=N_STARTS, min_trials=100
        )
    assert "problem" in result.skipped["solo"]
    assert "solo" not in set(result.scores["mouse"])
    assert result.logit is None


def test_mice_the_logit_skipped_are_not_analyzed(cohort, analysis):
    extra = simulate_mouse(sm.WSLS_MAGNITUDE, mouse="late", **SIZE)
    with pytest.warns(UserWarning, match="mouse late skipped"):
        result = pipe.analyze_comparison_source(
            pd.concat([cohort, extra], ignore_index=True), analysis, n_starts=N_STARTS,
            min_trials=100,
        )  # fmt: skip
    assert "not in the logit analysis" in result.skipped["late"]


def test_no_analyzable_mouse_raises():
    single = simulate_mouse(sm.WSLS_MAGNITUDE, n_problems=1)
    with pytest.raises(ValueError, match="no mouse can be compared"):
        with pytest.warns(UserWarning):
            pipe.analyze_comparison_source(single, None, n_starts=N_STARTS, min_trials=100)


def test_per_mouse_table_and_summaries(source):
    table = pipe.per_mouse_table({"Mice": source})
    assert list(table.columns[:2]) == ["source", "mouse"]
    parameters = {"v", "u", "q_learning_rate", "q_beta", "graded_w_m", "p_recent"}
    parameters |= {f"kq_{name}" for name in kq.PARAMETER_NAMES}
    assert {"winner", "choice_logit"} | parameters <= set(table.columns)
    assert set(sm.COMPARED) <= set(table.columns)  # nats per choice, ablations included
    assert len(table) == 3
    summary = pipe.source_summaries({"Mice": source})
    assert summary["source"].eq("Mice").all() and summary["mean"].isna().all()  # three mice
    wins = pipe.win_table({"Mice": source})
    assert wins.loc[wins["source"] == "Mice", "n_mice"].sum() == 3


def test_ablations_are_scored_on_the_same_folds_and_never_win(source):
    assert set(sm.ABLATIONS) <= set(source.nats["model"])
    assert set(source.winners["winner"]) <= set(sm.MODELS)
    table = pipe.ablation_table(source)
    assert list(table.columns) == [
        "mouse",
        *[sm.comparison_name(a, sm.KERNEL_Q) for a in sm.ABLATIONS],
    ]
    wide = source.nats.pivot(index="mouse", columns="model", values="nats_per_choice")
    name = sm.comparison_name(sm.ABLATIONS[0], sm.KERNEL_Q)
    expected = wide[sm.ABLATIONS[0]] - wide[sm.KERNEL_Q]
    assert np.allclose(table.set_index("mouse")[name], expected.loc[table["mouse"]])
    summary = pipe.ablation_summaries({"Mice": source})
    assert summary["source"].eq("Mice").all() and len(summary) == len(sm.ABLATIONS)


def test_ablation_table_names_missing_scores(source):
    without = source._replace(nats=source.nats[~source.nats["model"].isin(sm.ABLATIONS)])
    with pytest.raises(ValueError, match="ablations"):
        pipe.ablation_table(without)


def test_the_kernel_parameters_table_lists_the_fits_and_the_decay(source):
    table = pipe.kernel_parameters_table(source).set_index("mouse")
    assert list(table.columns) == [*kq.PARAMETER_NAMES, "decay", "at_bounds"]
    fit = source.fits["m1"][sm.KERNEL_Q]
    assert table.loc["m1", "alpha_k"] == pytest.approx(fit[kq.INDEX["alpha_k"]])
    assert table.loc["m1", "decay"] == pytest.approx(1 - fit[kq.INDEX["alpha_k"]])


def test_decay_profile_table_covers_the_mice_and_is_cached(cohort, tmp_path):
    kwargs = dict(decays=(0.5, 0.9), cache_dir=tmp_path)
    table = pipe.decay_profile_table(cohort, ["m1", "m3"], **kwargs)
    assert set(table["mouse"]) == {"m1", "m3"}
    assert sorted(table["decay"].unique()) == [sm.NO_TRACE, 0.5, 0.9]
    assert any(tmp_path.rglob("*.pkl"))
    pd.testing.assert_frame_equal(table, pipe.decay_profile_table(cohort, ["m1", "m3"], **kwargs))
    with pytest.raises(ValueError, match="no trials for mice"):
        pipe.decay_profile_table(cohort, ["nobody"])


# ---------------------------------------------------------------------------
# Caching and parallel workers
# ---------------------------------------------------------------------------


def test_the_cache_returns_the_same_result_and_is_written(cohort, source, analysis, tmp_path):
    kwargs = dict(n_starts=N_STARTS, min_trials=100, cache_dir=tmp_path)
    first = pipe.analyze_comparison_source(cohort, analysis, **kwargs)
    assert any(tmp_path.rglob("*.pkl"))
    again = pipe.analyze_comparison_source(cohort, analysis, n_jobs=2, **kwargs)
    pd.testing.assert_frame_equal(first.scores, again.scores)
    pd.testing.assert_frame_equal(first.scores, source.scores)


def test_changing_a_setting_misses_the_cache(cohort, tmp_path):
    cached = pipe.mouse_comparer(tmp_path)
    trials = cohort[cohort["mouse"] == "m1"].reset_index(drop=True)
    arguments = (trials, N_STARTS, 0, pipe.code_version())
    cached(*arguments)
    assert cached.check_call_in_cache(*arguments)
    assert not cached.check_call_in_cache(trials, N_STARTS + 1, 0, pipe.code_version())
    assert not cached.check_call_in_cache(trials, N_STARTS, 1, pipe.code_version())
    assert not cached.check_call_in_cache(trials, N_STARTS, 0, "another version")


def test_the_code_version_follows_the_module_sources(monkeypatch):
    before = pipe.code_version()
    assert before == pipe.code_version()
    real = inspect.getsource
    monkeypatch.setattr(inspect, "getsource", lambda module: real(module) + "# edit\n")
    assert pipe.code_version() != before


# ---------------------------------------------------------------------------
# Model recovery
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def recovery(source, cohort):
    two = cohort[cohort["mouse"].isin(["m1", "m2"])]
    restricted = source._replace(fits={m: source.fits[m] for m in ("m1", "m2")})
    return pipe.recover_models(two, restricted, rules=RULES, replicates=1, seed=3,
                               n_starts=N_STARTS)  # fmt: skip


def test_every_mouse_is_simulated_with_every_model_and_compared_again(recovery):
    won = recovery.won
    assert len(won) == 2 * len(sm.MODELS)
    assert set(won["generator"]) == set(sm.MODELS) and set(won["mouse"]) == {"m1", "m2"}
    assert set(won["winner"]) <= set(sm.MODELS)
    assert recovery.counts.to_numpy().sum() == len(won)
    assert recovery.failed.empty


def test_heuristic_parameters_are_recovered_with_their_generating_values(recovery):
    table = recovery.parameters
    assert set(table["generator"]) == set(sm.MODELS)
    heuristic = table[table["generator"] == sm.HEURISTIC_PLUS]
    assert {"v", "u", "w"} <= set(heuristic["parameter"])
    assert set(recovery.summary["generator"]) == set(sm.MODELS)


def test_the_simulated_mice_play_the_layouts_of_the_real_ones(cohort, source):
    trials = cohort[cohort["mouse"] == "m1"].reset_index(drop=True)
    simulated = pipe.simulate_model_trials(
        trials, source.fits["m1"][sm.Q_LEARNER], sm.Q_LEARNER, RULES, seed=1
    )
    real = sim.mouse_layouts(trials)
    played = sim.mouse_layouts(simulated)
    assert list(real) == list(played)
    for problem, layout in real.items():
        assert played[problem].ports == layout.ports
        assert played[problem].session_lengths == layout.session_lengths
    assert set(simulated["mouse"]) == {"m1"}


def test_the_simulation_is_reproducible_and_seeded(cohort, source):
    trials = cohort[cohort["mouse"] == "m1"].reset_index(drop=True)
    params = source.fits["m1"][sm.HEURISTIC_PLUS]
    first = pipe.simulate_model_trials(trials, params, sm.HEURISTIC_PLUS, RULES, seed=1)
    again = pipe.simulate_model_trials(trials, params, sm.HEURISTIC_PLUS, RULES, seed=1)
    other = pipe.simulate_model_trials(trials, params, sm.HEURISTIC_PLUS, RULES, seed=2)
    pd.testing.assert_frame_equal(first, again)
    assert not first["choice"].equals(other["choice"])


def test_recovery_is_cached_per_mouse(cohort, source, tmp_path):
    one = cohort[cohort["mouse"] == "m1"]
    restricted = source._replace(fits={"m1": source.fits["m1"]})
    kwargs = dict(rules=RULES, replicates=1, seed=3, n_starts=N_STARTS, cache_dir=tmp_path)
    first = pipe.recover_models(one, restricted, **kwargs)
    assert any(tmp_path.rglob("*.pkl"))
    again = pipe.recover_models(one, restricted, n_jobs=2, **kwargs)
    pd.testing.assert_frame_equal(first.won, again.won)


def test_recovery_validates_its_inputs(cohort, source):
    with pytest.raises(ValueError, match="at least 1"):
        pipe.recover_models(cohort, source, rules=RULES, replicates=0)
