"""The nuisance-only baseline of the leave-one-problem-out validation."""

import numpy as np
from scipy.special import logsumexp

from src.behavior_analysis import choice_logit as cl
from src.behavior_analysis.choice_logit_cv import cross_validate_mouse
from tests.test_choice_logit import three_problem_design


def test_baseline_is_the_held_out_nuisance_model_without_reward_terms():
    design = three_problem_design()
    result = cross_validate_mouse(design, l2=1.0)
    keep = [i for i, n in enumerate(design.feature_names) if n.startswith(("choice", "tower"))]
    names = tuple(design.feature_names[i] for i in keep)
    X = design.X[:, :, keep]
    for problem in (1, 2, 3):
        train = (design.meta["problem"] != problem).to_numpy()
        fit = cl.fit_logit(X[train], design.chosen[train], names, l2=1.0)
        utility = X[~train] @ fit.beta
        expected = np.exp(utility - logsumexp(utility, axis=1, keepdims=True))
        np.testing.assert_allclose(result.baseline[~train], expected, atol=1e-6)
    np.testing.assert_allclose(result.baseline.sum(axis=1), 1.0)
