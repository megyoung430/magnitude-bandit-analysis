"""Held-out (leave-one-problem-out) scores of the null agents, on the folds of the choice logit."""

import warnings

import numpy as np
import pandas as pd
import pytest

from src.agents import null_agents as na
from src.behavior_analysis.strategy_models import strategy_models as sm
from src.pipelines.choice_logit import choice_logit_values as logit_pipe
from src.pipelines.strategy_models import null_agents as pipe
from tests.test_null_agents import RULES, hand_table

pytestmark = pytest.mark.filterwarnings("ignore:the problems of")


@pytest.fixture(scope="module")
def mice():
    return logit_pipe.simulate_cohort(
        [(0.3, 1.2, 0.0, 0.0), (0.3, 1.2, 0.8, 0.6)],
        mice_per_set=1,
        n_problems=3,
        sessions_per_problem=3,
        trials_per_session=300,
        seed=1,
    ).trials


@pytest.fixture(scope="module")
def heldout(mice):
    return pipe.heldout_scores(mice)


@pytest.fixture(scope="module")
def analysis(mice):
    return logit_pipe.analyse_trial_table(mice, n_lags=3, tail_decay=0.9)


def test_every_null_agent_is_scored_on_every_held_out_problem(heldout, mice):
    folds = heldout.folds
    assert set(folds["model"]) == set(sm.NULL_MODELS)
    per = folds.groupby(["mouse", "model"])["problem"].nunique()
    assert (per == 3).all() and len(per) == mice["mouse"].nunique() * len(sm.NULL_MODELS)


def test_the_folds_are_those_of_the_choice_logit(heldout, analysis):
    for mouse, cv in analysis.cv.items():
        logit = cv.scores.query("model == 'full'").set_index("problem")["n_trials"]
        mine = heldout.folds[(heldout.folds["mouse"] == mouse) & (heldout.folds["model"] == "wsls")]
        agents = mine.set_index("problem")
        assert list(agents.index) == list(logit.index)  # the same problems, in the same order
        # every trial of the held-out problem is scored: the choices and the segment starts
        assert (agents["n_choices"] + agents["n_segment_starts"]).tolist() == logit.tolist()


def test_the_table_is_nats_per_trial_with_starts_at_log_one_third(heldout):
    table = heldout.table
    assert set(table.columns) >= {
        "mouse",
        "model",
        "n_choices",
        "log_likelihood",
        "nats_per_choice",
    }
    folds = heldout.folds[heldout.folds["model"] == "wsls"]
    row = table[(table["model"] == "wsls") & (table["mouse"] == "sim01")].iloc[0]
    by_hand = folds[folds["mouse"] == "sim01"]
    n = (by_hand["n_choices"] + by_hand["n_segment_starts"]).sum()
    ll = by_hand["held_out_log_likelihood"].sum() + by_hand["n_segment_starts"].sum() * np.log(
        1 / 3
    )
    assert row["n_choices"] == n and row["nats_per_choice"] == pytest.approx(-ll / n)


def test_held_out_nats_are_close_to_but_not_below_chance_for_a_value_blind_fit(heldout, mice):
    in_sample = pipe.scores_table(
        mice, {k: na.fit_null_agents(mice, k) for k in sm.NULL_MODELS}
    ).set_index(["mouse", "model"])["nats_per_choice"]
    held = heldout.table.set_index(["mouse", "model"])["nats_per_choice"]
    assert (held < np.log(3)).all()
    assert (held - in_sample).abs().max() < 0.05  # 2 to 4 parameters: little optimism
    assert (held - in_sample).mean() > -0.005  # held out is not systematically better


def test_a_mouse_with_one_problem_is_skipped_with_a_warning(mice):
    solo = hand_table([0, 1, 2] * 100, [1, 0, 1] * 100, mouse="solo")
    table = pd.concat([mice, solo], ignore_index=True)
    with pytest.warns(UserWarning, match="mouse solo skipped"):
        result = pipe.heldout_scores(table)
    assert "problem" in result.skipped["solo"]
    assert "solo" not in set(result.table["mouse"])


def test_nothing_to_score_raises(mice):
    solo = hand_table([0, 1, 2] * 100, [1, 0, 1] * 100, mouse="solo")
    with pytest.raises(ValueError, match="no mouse can be cross-validated"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.heldout_scores(solo)


def test_the_comparison_scores_the_agents_held_out_and_keeps_the_in_sample_column(mice, analysis):
    comparison = pipe.compare_with_controls(
        mice, analysis, "Mice", seed=1, n_bins=4, min_mice=1,
        rules=RULES,
    )  # fmt: skip
    held = comparison.scores.query("model != 'choice_logit'")
    assert set(held["model"]) == set(sm.NULL_MODELS)
    assert comparison.in_sample is not None
    assert set(comparison.in_sample["model"]) == set(sm.NULL_MODELS)
    both = held.merge(comparison.in_sample, on=["mouse", "model"], suffixes=("_held", "_in"))
    assert not np.allclose(both["nats_per_choice_held"], both["nats_per_choice_in"])
