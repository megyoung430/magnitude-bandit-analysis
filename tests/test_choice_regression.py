"""The conditional-logit choice regression: its features, and recovery of known weights."""

import numpy as np
import pandas as pd
from scipy.special import softmax

from src.behavior_analysis.choice_regression import (
    FEATURES,
    choice_design,
    fit_by_mouse,
    fit_conditional_logit,
    group_summary,
)

# From initiation B2: B1 is a plus port, A1 and C1 are X ports.
MAGS = {"A1": 3, "B1": 1, "C1": 0}


def _table(choices, trial_in_session, mouse="m", magnitudes=MAGS):
    n = len(choices)
    return pd.DataFrame(
        dict(
            mouse=mouse,
            problem=1,
            trial=range(n),
            trial_in_session=trial_in_session,
            choice=choices,
            magnitudes=[magnitudes] * n,
            initiation="B2",
            after_gap=False,
        )
    )


def _features(design, trial, port):
    return dict(zip(FEATURES, design.X[trial, sorted(MAGS).index(port)], strict=True))


def test_features_follow_what_the_mouse_last_saw():
    design = choice_design(_table(["A1", "C1", "B1", "A1"], [0, 1, 0, 1]))
    assert design.chosen.tolist() == [0, 2, 1, 0]
    first = _features(design, 0, "A1")
    assert first == dict(last_value=0, last_zero=0, unseen=1, stay=0, plus_port=0)
    # trial 1: A1 last paid 3 and was just chosen
    assert _features(design, 1, "A1") == dict(
        last_value=3, last_zero=0, unseen=0, stay=1, plus_port=0
    )
    # trial 2 is a new session: C1 last paid 0, stay resets, beliefs carry over
    assert _features(design, 2, "C1") == dict(
        last_value=0, last_zero=1, unseen=0, stay=0, plus_port=0
    )
    assert _features(design, 2, "B1")["plus_port"] == 1
    assert _features(design, 2, "B1")["unseen"] == 1


def test_beliefs_reset_at_a_new_problem():
    first, second = _table(["A1"], [0]), _table(["B1"], [0]).assign(problem=2)
    design = choice_design(pd.concat([first, second], ignore_index=True))
    assert _features(design, 1, "A1")["unseen"] == 1


def test_fit_recovers_known_weights():
    rng = np.random.default_rng(1)
    true = np.array([0.8, -1.0, 0.5])
    X = rng.normal(size=(20000, 3, 3))
    chosen = np.array([rng.choice(3, p=softmax(x @ true)) for x in X])
    fit = fit_conditional_logit(X, chosen, ridge=0.0)
    assert np.allclose(fit.weights, true, atol=4 * fit.standard_errors.max())
    assert np.all(fit.standard_errors < 0.05)


def test_fit_by_mouse_skips_small_mice_and_summary_tests_against_zero():
    rng = np.random.default_rng(2)
    ports = sorted(MAGS)
    big = [_table(list(rng.choice(ports, 300)), list(range(300)), mouse=f"m{i}") for i in range(3)]
    small = _table(["A1"] * 10, list(range(10)), mouse="tiny")
    weights = fit_by_mouse(choice_design(pd.concat([*big, small], ignore_index=True)))
    assert sorted(weights["mouse"].unique()) == ["m0", "m1", "m2"]
    summary = group_summary(weights)
    assert summary["feature"].tolist() == list(FEATURES)
    assert (summary["n_mice"] == 3).all()
    one = group_summary(weights[weights["mouse"] == "m0"])
    assert one["t"].isna().all()
