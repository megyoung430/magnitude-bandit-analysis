"""Leave-one-problem-out validation of the choice logit, and the held-out model values.

For each problem of a mouse the logit is fitted on the mouse's other problems and scored on the
problem it did not see, so a model is judged on a new problem (new towers, new value drift)
rather than on trials it was fitted to. Three nested models are compared:

- ``reward``: reward lags only;
- ``reward+choice``: plus choice lags;
- ``full``: plus the tower biases.

The held-out *model values* come from the ``full`` model's fold fit that excluded the problem
(its reward weights, see :func:`src.behavior_analysis.choice_logit.model_values`), so a value
never comes from weights that were fitted to the trials it describes.

Everything here is pure: designs in, arrays and tables out.
"""

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from src.behavior_analysis import choice_logit as cl

MODELS = {
    "reward": (cl.REWARD,),
    "reward+choice": (cl.REWARD, cl.CHOICE),
    "full": (cl.REWARD, cl.CHOICE, cl.TOWER),
}
VALUE_MODEL = "full"
# Choice history and tower bias without any reward term: what perseveration and port preference
# alone predict. Its held-out probabilities are the baseline of every psychometric on a value.
NUISANCE_GROUPS = (cl.CHOICE, cl.TOWER)


class MouseCV(NamedTuple):
    """The validation of one mouse.

    Attributes:
        scores (pd.DataFrame): One row per held-out problem and model: ``mouse, problem, model,
            n_trials, held_out_log_likelihood`` (summed over the problem's trials).
        values (np.ndarray): ``(n_trials, 3)`` held-out model values, rows as in the design.
        baseline (np.ndarray): ``(n_trials, 3)`` held-out choice probabilities of the
            nuisance-only model (:data:`NUISANCE_GROUPS`), fitted on the other problems.
    """

    scores: pd.DataFrame
    values: np.ndarray
    baseline: np.ndarray


def cross_validate_mouse(design: cl.LogitDesign, l2: float = cl.DEFAULT_L2) -> MouseCV:
    """Fit every model on all but one problem and score it on that problem, for each problem.

    Args:
        design (cl.LogitDesign): One mouse's design (:func:`choice_logit.design_matrix`).
        l2 (float): Penalty of the fits. Default ``choice_logit.DEFAULT_L2``.

    Returns:
        MouseCV: Held-out scores of every model, the held-out values of the ``full`` model and
        the nuisance-only baseline probabilities.

    Raises:
        ValueError: If the mouse has fewer than two problems.
        RuntimeError: If a fit does not converge.
    """
    codes, problems = pd.factorize(design.meta["problem"])
    if len(problems) < 2:
        raise ValueError("leave-one-problem-out validation needs at least two problems")
    mouse = design.meta["mouse"].iloc[0]
    reduced = {name: cl.select_features(design, groups) for name, groups in MODELS.items()}
    nuisance = cl.select_features(design, NUISANCE_GROUPS)
    values = np.full(design.X.shape[:2], np.nan)
    baseline = np.full(design.X.shape[:2], np.nan)
    rows = []
    for code, problem in enumerate(problems):
        test = codes == code
        base_fit = cl.fit_logit(
            nuisance.X[~test], nuisance.chosen[~test], nuisance.feature_names, l2
        )
        utility = nuisance.X[test] @ base_fit.beta
        baseline[test] = np.exp(utility - logsumexp(utility, axis=1, keepdims=True))
        for name, model in reduced.items():
            fit = cl.fit_logit(model.X[~test], model.chosen[~test], model.feature_names, l2)
            rows.append(
                dict(
                    mouse=mouse,
                    problem=problem,
                    model=name,
                    n_trials=int(test.sum()),
                    held_out_log_likelihood=cl.log_likelihood(
                        model.X[test], model.chosen[test], fit.beta
                    ),
                )
            )
            if name == VALUE_MODEL:
                values[test] = cl.model_values(model.X[test], fit)
    return MouseCV(pd.DataFrame(rows), values, baseline)


def model_comparison(scores: pd.DataFrame, models: Sequence[str] | None = None) -> pd.DataFrame:
    """Held-out log likelihood per choice of each model and mouse.

    Args:
        scores (pd.DataFrame): ``MouseCV.scores`` of one or more mice, concatenated.
        models (Sequence[str] | None): Model names in the order to list them, for scores of
            models other than ``MODELS``. Default ``None``: the order of ``MODELS``.

    Returns:
        pd.DataFrame: One row per mouse and model (models in the order of *models*) with
        ``mouse, model, n_trials, nats_per_choice`` (negative mean log likelihood over all the
        mouse's held-out trials, lower is better) and ``gain_over_chance``
        (``log(3) - nats_per_choice``; positive: better than choosing at random).
    """
    total = scores.groupby(["mouse", "model"], sort=False).agg(
        n_trials=("n_trials", "sum"), log_likelihood=("held_out_log_likelihood", "sum")
    )
    total["nats_per_choice"] = -total["log_likelihood"] / total["n_trials"]
    total["gain_over_chance"] = np.log(cl.N_ARMS) - total["nats_per_choice"]
    table = total.drop(columns="log_likelihood").reset_index()
    order = {name: i for i, name in enumerate(MODELS if models is None else models)}
    return table.sort_values(
        ["mouse", "model"], key=lambda s: s.map(order) if s.name == "model" else s, kind="stable"
    ).reset_index(drop=True)
