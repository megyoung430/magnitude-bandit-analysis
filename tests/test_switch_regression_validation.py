"""Validation of the raw switch regression on agents whose switch rule is known.

The win-stay/lose-shift agents choose a switch target from their choice history alone, so their
value weight should be about 0 and adding a value should not improve the held-out fit; the
Q-learner switches towards the arm it values more. The stress agents have a strong recency
weight (w = 1.5) with the slow trace (decay 0.9) that puts the old choice-logit baseline about
0.3 below the observed switch targets (``tests/test_null_agents_pipeline.py``).

Every check is per condition and per value variant, at switch counts like a mouse's: about
300-700 per condition (``REALISTIC``) and, for the near-minimum case, 100-250 (``NEAR_MINIMUM``).
What these tests can and cannot say: with 12 mice a bias of one per-mouse standard error gives a
pooled z of about 3.5, detected with about 67% power at the Bonferroni limit below; smaller
biases pass unnoticed, and the null spread of a strong recency agent at low counts is wide
(``test_strong_recency_spreads_the_null_weights_near_the_minimum``). That is why the inferential
output for a mouse is its null-calibrated weight (``tests/test_switch_null_calibration.py``).
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binomtest

from src.agents import drifting_simulation as sim
from src.agents import null_agents as na
from src.behavior_analysis import switch_regression as sr
from src.pipelines import choice_logit_values as logit_pipe
from src.pipelines import null_agents as null_pipe

RULES = sim.TaskRules(drift_rate=0.1, block_lengths=np.arange(30, 36))
GRID = (0.5, 0.9, 0.95)
N_MICE = 12
REALISTIC = dict(n_problems=6, sessions_per_problem=2, trials_per_session=250)
NEAR_MINIMUM = dict(n_problems=6, sessions_per_problem=1, trials_per_session=220)
NULL_SOURCES = ("wsls", "wsls_recency", "wsls_magnitude", "wsls_recency stress",
                "wsls_magnitude stress")  # fmt: skip
FITTED = NULL_SOURCES[:3]
# 5 sources x 2 conditions x 2 values = 20 checks per size; Bonferroni for a family-wise 5%
CHECKS = 20
Z_LIMIT = 3.02  # two-sided p = 0.05 / 20
SIGN_ALPHA = 0.05 / CHECKS  # with 12 mice only 12/12 or 0/12 reaches it


def _value_tables(trials: pd.DataFrame) -> pd.DataFrame:
    tables = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _, rows in trials.groupby("mouse"):
            result = sr.analyze_mouse(rows, trace_grid=GRID, min_switches=80)
            tables.append(
                sr.value_table(result.scores, result.weights).merge(
                    result.counts, on=["mouse", "condition"]
                )
            )
    return pd.concat(tables, ignore_index=True)


def _sources(size: dict, seed: int) -> dict:
    q = logit_pipe.simulate_cohort(
        [(0.3, 1.2, 0.0, 0.0), (0.3, 1.0, 1.0, 0.5)], mice_per_set=N_MICE // 2, seed=seed, **size
    ).trials
    mice = sorted(q["mouse"].unique())
    sources = {"Q-Learner": q}
    for kind in na.KINDS:
        fitted = na.fit_null_agents(q, kind)
        sources[kind] = null_pipe.simulate_null_cohort(q, fitted, kind, seed + 1, RULES).trials
    stress = {
        "wsls_recency": {m: na.WslsRecencyParams(1.0, -0.5, 1.5, 0.9) for m in mice},
        "wsls_magnitude": {m: na.WslsMagnitudeParams(0.2, -0.5, 0.5, 1.5, 0.9) for m in mice},
    }
    for kind, params in stress.items():
        sources[f"{kind} stress"] = null_pipe.simulate_null_cohort(
            q, params, kind, seed + 2, RULES
        ).trials
    return {label: _value_tables(trials) for label, trials in sources.items()}


@pytest.fixture(scope="module")
def realistic():
    return _sources(REALISTIC, 4)


@pytest.fixture(scope="module")
def near_minimum():
    return _sources(NEAR_MINIMUM, 14)


def _check_null(table: pd.DataFrame, label: str) -> None:
    for (condition, value), cell in table.groupby(["condition", "value"]):
        if value == sr.MODEL:
            continue
        z = cell["beta"] / cell["se"]
        pooled = z.sum() / np.sqrt(len(z))
        n_positive = int((cell["beta"] > 0).sum())
        where = (label, condition, value, round(pooled, 2), n_positive)
        assert abs(pooled) < Z_LIMIT, where
        assert binomtest(n_positive, len(cell), 0.5).pvalue > SIGN_ALPHA, where
        assert cell["gain"].mean() < 0.003, where


def test_switch_counts_are_like_a_mouses(realistic, near_minimum):
    realistic_counts = pd.concat([t["n_switches"] for t in realistic.values()])
    low_counts = pd.concat([t["n_switches"] for t in near_minimum.values()])
    assert 250 < realistic_counts.median() < 800
    assert 80 <= low_counts.min() and low_counts.median() < 260


@pytest.mark.parametrize("source", NULL_SOURCES)
def test_null_agents_have_no_value_weight_at_realistic_counts(realistic, source):
    assert realistic[source]["mouse"].nunique() == N_MICE
    _check_null(realistic[source], source)


@pytest.mark.parametrize("source", FITTED)
def test_fitted_null_agents_have_no_value_weight_near_the_minimum(near_minimum, source):
    _check_null(near_minimum[source], source)


def test_strong_recency_spreads_the_null_weights_near_the_minimum(near_minimum):
    # a value-blind agent can show a true-value weight larger than the Q-learner's after zero
    for source in ("wsls_recency stress", "wsls_magnitude stress"):
        true = near_minimum[source].query("value == 'true'")
        assert true["beta"].std() > 0.1, source
        assert true["beta"].abs().max() > 0.2, source


def test_q_learners_switch_towards_value(realistic):
    q = sr.group_summary(realistic["Q-Learner"]).set_index(["condition", "value", "term"])
    for condition in sr.CONDITIONS:
        believed = q.loc[(condition, "believed", "beta")]
        assert believed["n_positive"] == N_MICE
        assert believed["mean"] > 0.2 and believed["p_wilcoxon"] < 0.01
        assert q.loc[(condition, "believed", "gain")]["mean"] > 0.01
    true_after_reward = q.loc[("after_nonzero", "true", "beta")]
    assert true_after_reward["n_positive"] >= N_MICE - 1
    assert q.loc[("after_nonzero", "true", "gain")]["mean"] > 0.005


def test_the_stress_cases_pick_the_agents_own_trace_decay(realistic):
    # one mouse is enough to show the decay of the generating trace wins the comparison
    q = logit_pipe.simulate_cohort([(0.3, 1.2, 0.0, 0.0)], mice_per_set=1, seed=4, **REALISTIC)
    params = {"sim01": na.WslsRecencyParams(1.0, -0.5, 1.5, 0.9)}
    trials = null_pipe.simulate_null_cohort(q.trials, params, "wsls_recency", 6, RULES).trials
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert sr.analyze_mouse(trials, trace_grid=GRID).trace_decay == 0.9
