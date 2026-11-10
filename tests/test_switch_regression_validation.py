"""Validation of the switch regression on agents whose switch rule is known.

The win-stay/lose-shift agents choose a switch target from their choice history alone, so the
value weight must be about 0 and adding a value must not improve the held-out fit; the
Q-learner switches towards the arm it values more, so its value weight must be positive. The
stress cases have a strong recency weight (w = 1.5) with the slow trace (decay 0.9) that puts the
old choice-logit baseline about 0.3 below the observed switch targets
(``tests/test_null_agents_pipeline.py``).
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

RULES = sim.TaskRules(drift_rate=0.1, block_lengths=np.arange(30, 36))
GRID = (0.5, 0.9, 0.95)
N_MICE = 6  # the fewest mice with a mean and a test


@pytest.fixture(scope="module")
def q_learners():
    return logit_pipe.simulate_cohort(
        [(0.3, 1.2, 0.0, 0.0), (0.3, 1.0, 1.0, 0.5)],
        mice_per_set=N_MICE // 2,
        n_problems=6,
        sessions_per_problem=3,
        trials_per_session=300,
        seed=4,
    ).trials


def _value_tables(trials: pd.DataFrame) -> pd.DataFrame:
    """:func:`switch_regression.value_table` of every mouse of *trials*."""
    tables = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _, rows in trials.groupby("mouse"):
            result = sr.analyse_mouse(rows, trace_grid=GRID)
            tables.append(sr.value_table(result.scores, result.weights))
    return pd.concat(tables, ignore_index=True)


@pytest.fixture(scope="module")
def tables(q_learners):
    mice = sorted(q_learners["mouse"].unique())
    fitted = {kind: na.fit_null_agents(q_learners, kind) for kind in na.KINDS}
    stress = {
        "wsls_recency": {m: na.WslsRecencyParams(1.0, -0.5, 1.5, 0.9) for m in mice},
        "wsls_magnitude": {m: na.WslsMagnitudeParams(0.2, -0.5, 0.5, 1.5, 0.9) for m in mice},
    }
    sources = {"Q-Learner": q_learners}
    for kind in na.KINDS:
        sources[kind] = null_pipe.simulate_null_cohort(q_learners, fitted[kind], kind, 5, RULES)
    for kind, params in stress.items():
        sources[f"{kind} stress"] = null_pipe.simulate_null_cohort(
            q_learners, params, kind, 6, RULES
        )
    return {
        label: _value_tables(source if isinstance(source, pd.DataFrame) else source.trials)
        for label, source in sources.items()
    }


NULL_SOURCES = ("wsls", "wsls_recency", "wsls_magnitude", "wsls_recency stress",
                "wsls_magnitude stress")  # fmt: skip
# Two-sided threshold of the pooled z of each null check: Bonferroni over the 10 checks (5
# sources, 2 values) for a family-wise false-alarm rate of 1%. A bias of about one standard
# error per mouse (0.05-0.15 logit per pulse) still exceeds it, a fraction of the Q-learner's
# 0.47.
Z_LIMIT = 3.5


@pytest.mark.parametrize("source", NULL_SOURCES)
@pytest.mark.parametrize("value", ["true", "believed"])
def test_null_agents_have_no_value_weight_and_no_value_gain(tables, source, value):
    table = tables[source].query("value == @value")
    assert len(table) == 2 * N_MICE
    # Stouffer's z over the mice and both conditions: about N(0, 1) when the weight is 0
    z = table["beta"] / table["se"]
    assert abs(z.sum() / np.sqrt(len(z))) < Z_LIMIT, (source, value, table)
    assert 0 < (table["beta"] > 0).sum() < len(table)  # no consistent sign
    summary = sr.group_summary(table).set_index(["condition", "term"])
    for condition in sr.CONDITIONS:
        assert summary.loc[(condition, "beta"), "n"] == N_MICE
        assert abs(summary.loc[(condition, "beta"), "mean"]) < 0.2
        assert summary.loc[(condition, "gain"), "mean"] < 0.003, (source, condition)


def test_the_stress_cases_pick_the_agents_own_trace_decay(q_learners):
    mice = sorted(q_learners["mouse"].unique())[:2]
    params = {m: na.WslsRecencyParams(1.0, -0.5, 1.5, 0.9) for m in mice}
    trials = null_pipe.simulate_null_cohort(q_learners, params, "wsls_recency", 6, RULES).trials
    for _, rows in trials.groupby("mouse"):
        assert sr.analyse_mouse(rows, trace_grid=GRID).trace_decay == 0.9


def test_q_learners_switch_towards_value(tables):
    q = sr.group_summary(tables["Q-Learner"]).set_index(["condition", "value", "term"])
    for condition in sr.CONDITIONS:
        believed = q.loc[(condition, "believed", "beta")]
        assert believed["n_positive"] == N_MICE
        assert believed["mean"] > 0.2 and believed["p_wilcoxon"] < 0.05
        assert q.loc[(condition, "believed", "gain")]["mean"] > 0.01
    true_after_reward = q.loc[("after_nonzero", "true", "beta")]
    assert true_after_reward["n_positive"] >= N_MICE - 1
    assert q.loc[("after_nonzero", "true", "gain")]["mean"] > 0.005
