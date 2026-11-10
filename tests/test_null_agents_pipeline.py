"""Null agents fitted to a cohort and simulated on each mouse's own task."""

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from src.agents import drifting_simulation as sim
from src.agents import null_agents as na
from src.behavior_analysis.drifting_trial_table import drifting_trial_table
from src.pipelines import choice_logit_values as logit_pipe
from src.pipelines import null_agents as pipe

RULES = sim.TaskRules(drift_rate=0.1, block_lengths=np.arange(30, 36))
RECENCY = na.WslsRecencyParams(a_win=1.0, a_lose=-0.6, w=1.0)


@pytest.fixture(scope="module")
def real_like():
    """Four Q-learning mice standing in for real ones: six problems, uneven session lengths."""
    cohort = logit_pipe.simulate_cohort(
        [(0.3, 1.2, 0.0, 0.0), (0.3, 1.2, 0.8, 0.6)],
        mice_per_set=2,
        n_problems=3,
        sessions_per_problem=3,
        trials_per_session=300,
        seed=1,
    )
    return cohort.trials


@pytest.fixture(scope="module")
def fits(real_like):
    return {kind: na.fit_null_agents(real_like, kind) for kind in na.KINDS}


@pytest.fixture(scope="module")
def cohort(real_like, fits):
    return pipe.simulate_null_cohort(real_like, fits["wsls_recency"], "wsls_recency", 3, RULES)


def test_each_mouse_is_simulated_on_its_own_layouts(real_like, cohort):
    assert sorted(cohort.trials["mouse"].unique()) == sorted(real_like["mouse"].unique())
    for mouse, rows in real_like.groupby("mouse"):
        mine = cohort.trials[cohort.trials["mouse"] == mouse]
        assert len(mine) == len(rows)
        real_layouts = sim.mouse_layouts(rows)
        for number, layout in enumerate(real_layouts.values(), start=1):
            simulated = mine[mine["problem"] == number]
            assert simulated["magnitudes"].iloc[0] == layout.start_magnitudes
            assert simulated.groupby("session_index").size().tolist() == layout.session_lengths
            assert set(simulated["choice"]) <= set(layout.ports)
            assert simulated["initiation"].iloc[0] == layout.initiation


def test_cohort_has_the_synthetic_cohort_interface(cohort, fits):
    assert isinstance(cohort, logit_pipe.SyntheticCohort)
    assert list(cohort.parameters.columns) == ["mouse", "a_win", "a_lose", "w", "decay"]
    mouse = cohort.parameters["mouse"].iloc[0]
    assert cohort.parameters.set_index("mouse").loc[mouse, "w"] == fits["wsls_recency"][mouse].w
    assert sorted(next(iter(cohort.problems.values()))) == sorted(cohort.trials["mouse"].unique())


def test_simulation_is_reproducible_and_independent_of_the_other_mice(real_like, fits, cohort):
    again = pipe.simulate_null_cohort(real_like, fits["wsls_recency"], "wsls_recency", 3, RULES)
    pd.testing.assert_frame_equal(cohort.trials, again.trials)
    one = sorted(fits["wsls_recency"])[1]
    alone = pipe.simulate_null_cohort(
        real_like, {one: fits["wsls_recency"][one]}, "wsls_recency", 3, RULES
    )
    kept = cohort.trials[cohort.trials["mouse"] == one].reset_index(drop=True)
    pd.testing.assert_frame_equal(alone.trials.reset_index(drop=True), kept)


def test_another_agent_or_seed_gives_another_stream(real_like, fits, cohort):
    other_seed = pipe.simulate_null_cohort(
        real_like, fits["wsls_recency"], "wsls_recency", 4, RULES
    )
    assert not other_seed.trials["choice"].equals(cohort.trials["choice"])


@pytest.mark.filterwarnings("ignore:the problems of:UserWarning")  # three disjoint tower triples
def test_simulated_cohort_goes_through_the_logit_pipeline(cohort):
    analysis = logit_pipe.analyse_trial_table(cohort.trials, n_lags=3, tail_decay=0.9)
    assert sorted(analysis.designs) == sorted(cohort.trials["mouse"].unique())


def test_fits_to_the_simulated_cohort_recover_the_agents(real_like, fits, cohort):
    refit = na.fit_wsls_recency(cohort.trials)
    for mouse, truth in fits["wsls_recency"].items():
        assert refit[mouse].a_win == pytest.approx(truth.a_win, abs=0.5)
        assert refit[mouse].a_lose == pytest.approx(truth.a_lose, abs=0.5)


def test_rules_are_measured_from_the_trial_table_when_not_given():
    layout = sim.MouseLayout(["A1", "B3", "C2"], "B2", {"A1": 4, "B3": 2, "C2": 0}, [250] * 8)
    sessions = sim.play_problem(na.NullPolicy(RECENCY), layout, RULES, np.random.default_rng(0))
    table = drifting_trial_table({1: {"m": sim.simulated_sessions("m", 1, layout, sessions)}})
    cohort = pipe.simulate_null_cohort(table, {"m": RECENCY}, "wsls_recency", seed=0)
    assert len(cohort.trials) == len(table)


def test_parameters_must_match_the_agent_and_the_mice(real_like, fits):
    with pytest.raises(ValueError, match="wsls_magnitude"):
        pipe.simulate_null_cohort(real_like, fits["wsls"], "wsls_magnitude", 0, RULES)
    with pytest.raises(ValueError, match="unknown agent"):
        pipe.simulate_null_cohort(real_like, fits["wsls"], "other", 0, RULES)
    with pytest.raises(ValueError, match="not in the trial table"):
        pipe.simulate_null_cohort(real_like, {"ghost": fits["wsls"]["sim01"]}, "wsls", 0, RULES)


def test_parameters_table_has_one_row_per_mouse(fits):
    table = pipe.parameters_table(fits["wsls_magnitude"])
    assert list(table.columns) == ["mouse", "a_win", "a_lose", "b", "w", "decay"]
    assert table["mouse"].tolist() == sorted(fits["wsls_magnitude"])


def test_scores_table_is_long_with_nats_per_choice(real_like, fits):
    table = pipe.scores_table(real_like, fits)
    assert set(table["model"]) == set(na.KINDS)
    assert len(table) == len(na.KINDS) * real_like["mouse"].nunique()
    row = table[(table["model"] == "wsls") & (table["mouse"] == "sim01")].iloc[0]
    mine = real_like[real_like["mouse"] == "sim01"]
    assert row["n_choices"] == len(mine)
    assert row["nats_per_choice"] == pytest.approx(na.nats_per_choice(mine, fits["wsls"]["sim01"]))
    assert row["log_likelihood"] == pytest.approx(-row["nats_per_choice"] * len(mine))


def test_nested_agents_never_fit_worse_than_plain_wsls(real_like, fits):
    table = pipe.scores_table(real_like, fits).pivot(
        index="mouse", columns="model", values="nats_per_choice"
    )
    # same stay rule, plus a fitted switch weight / slope: maximum likelihood cannot lose
    assert (table["wsls_recency"] <= table["wsls"] + 1e-9).all()
    assert (table["wsls_magnitude"] <= table["wsls_recency"] + 1e-9).all()


def test_observed_stay_by_previous_magnitude(real_like):
    table = pipe.stay_by_magnitude(real_like[real_like["mouse"] == "sim01"])
    assert list(table.columns) == ["mouse", "magnitude", "n_pairs", "p_stay"]
    assert sorted(table["magnitude"]) == sorted(set(table["magnitude"]))
    assert table["n_pairs"].sum() == (
        len(real_like[real_like["mouse"] == "sim01"])
        - real_like[real_like["mouse"] == "sim01"]["trial_in_session"].eq(0).sum()
    )
    assert table["p_stay"].between(0, 1).all()


def test_implied_stay_curves_follow_the_fitted_stay_logit(fits):
    table = pipe.implied_stay_curves(fits)
    row = table[(table["model"] == "wsls_magnitude") & (table["mouse"] == "sim01")]
    params = fits["wsls_magnitude"]["sim01"]
    assert row["magnitude"].tolist() == [0, 1, 2, 3, 4]
    assert row["p_stay"].to_numpy() == pytest.approx(expit(params.stay_logit(np.arange(5))))
