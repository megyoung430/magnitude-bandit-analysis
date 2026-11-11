"""A heuristic or a value learner? Three models scored on held-out problems.

Per mouse, each model is fitted on all but one problem and scored on the held-out one, for every
problem in turn (the leave-one-problem-out folds of the choice logit,
:mod:`src.behavior_analysis.choice_logit_cv`):

- ``wsls_magnitude``: the win-stay/lose-shift agent whose staying grows with the last payout and
  whose switches follow the choice trace (:mod:`src.agents.null_agents`, its fit and likelihood);
- ``heuristic_plus``: the same stay rule, with switches that also follow what the alternatives
  last paid and whether they were ever chosen (:mod:`src.agents.heuristic_agent`);
- ``q_learner``: the value learner (:mod:`src.agents.value_agent`) with a learning rate, an
  inverse temperature, stickiness and a bias to the plus-shaped ports, fitted by multi-start
  maximum likelihood.

**What is scored.** Every choice that has a previous choice in its segment (so not a session's
first trial or the first trial after a dropped trial). The first choice of a segment is uniform
for both heuristics, while the Q-learner predicts it from its values and its port bias; scoring
it would credit the Q-learner for a rule the others do not have. The Q-learner still learns from
the segment starts. The score is the summed log likelihood of those choices on the held-out
problem; nats per choice pools the folds of a mouse.

Everything here is pure: trial tables in, tables out. Fitting and scoring are small functions
registered in ``SPECS`` so that the folds, the simulation behind model recovery and the
parameter tables treat the models alike.
"""

from collections.abc import Callable, Sequence
from dataclasses import fields
from typing import NamedTuple

import numpy as np
import pandas as pd

from src.agents import drifting_simulation as sim
from src.agents import heuristic_agent as ha
from src.agents import null_agents as na
from src.agents import value_agent as va
from src.behavior_analysis import choice_logit_extensions as ext

WSLS, WSLS_RECENCY = "wsls", "wsls_recency"
WSLS_MAGNITUDE, WSLS_GRADED = "wsls_magnitude", "wsls_graded"
HEURISTIC_PLUS, Q_LEARNER = "heuristic_plus", "q_learner"
# The models of the strategy comparison and its recovery, in table order.
MODELS = (WSLS_MAGNITUDE, HEURISTIC_PLUS, Q_LEARNER)
# The null agents, scored held out like the comparison's models (``cross_validate(models=...)``).
NULL_MODELS = (WSLS, WSLS_RECENCY, WSLS_MAGNITUDE, WSLS_GRADED)
# Every model that can be fitted and scored, in table order.
ALL_MODELS = (WSLS, WSLS_RECENCY, WSLS_MAGNITUDE, HEURISTIC_PLUS, Q_LEARNER, WSLS_GRADED)
# The pairs whose held-out nats are differenced, first minus second: negative means the first
# model predicts held-out choices better.
COMPARISONS = (
    (HEURISTIC_PLUS, Q_LEARNER),
    (WSLS_MAGNITUDE, Q_LEARNER),
    (HEURISTIC_PLUS, WSLS_MAGNITUDE),
)
# The Q-learner's likelihood has a slow and a fast mode for some mice, so every fit starts from
# several points (see value_agent).
Q_STARTS = 5
Q_KIND = "magnitude"  # the Q-learner learns from the pulses
# As the choice logit's minimum (choice_logit_values.MIN_TRIALS).
MIN_TRIALS = 300


class ModelSpec(NamedTuple):
    """What the comparison needs to know about one model.

    Attributes:
        fit (Callable): ``(train trials, n_starts, seed) -> params`` of one mouse.
        score (Callable): ``(test trials, params) -> held-out log likelihood`` of the choices
            that have a previous choice.
        policy (Callable): ``(params, layout) -> Policy`` that plays a problem.
        describe (Callable): ``params -> {parameter name: value}``.
        at_bounds (Callable): ``params -> names of the parameters at a fitting bound``.
    """

    fit: Callable
    score: Callable
    policy: Callable
    describe: Callable
    at_bounds: Callable


# ---------------------------------------------------------------------------
# The three models
# ---------------------------------------------------------------------------


def _n_starts(trials: pd.DataFrame) -> int:
    return int(na.decisions(trials, na.DEFAULT_DECAY).start.sum())


def _fit_null(kind: str):
    def fit(train: pd.DataFrame, n_starts: int, seed: int):
        return na.fit_null_agents(train, kind)[train["mouse"].iloc[0]]

    return fit


def _score_wsls(test: pd.DataFrame, params) -> float:
    # the null agents' likelihood counts every segment start as log(1/3)
    return na.log_likelihood(test, params) - _n_starts(test) * np.log(1 / na.N_ARMS)


def _fit_heuristic(train: pd.DataFrame, n_starts: int, seed: int) -> ha.HeuristicPlusParams:
    return ha.fit(train)


def _score_heuristic(test: pd.DataFrame, params: ha.HeuristicPlusParams) -> float:
    return ha.log_likelihood(test, params, include_starts=False)


def _fit_q(train: pd.DataFrame, n_starts: int, seed: int) -> tuple[float, ...]:
    fit = va.fit(sim.problem_sequences(train, Q_KIND), n_starts, seed)
    return tuple(float(p) for p in fit.parameters)


def _score_q(test: pd.DataFrame, params: Sequence[float]) -> float:
    seqs = sim.problem_sequences(test, Q_KIND)
    per_trial = va.trial_log_probabilities(params, seqs)
    return float(per_trial[seqs.mask & ~seqs.session_start].sum())


def _fields(params) -> dict:
    return {f.name: float(getattr(params, f.name)) for f in fields(params)}


def _null_spec(kind: str) -> ModelSpec:
    return ModelSpec(
        fit=_fit_null(kind),
        score=_score_wsls,
        policy=lambda params, layout: na.NullPolicy(params),
        describe=_fields,
        at_bounds=lambda params: (),
    )


SPECS = {
    WSLS: _null_spec(WSLS),
    WSLS_RECENCY: _null_spec(WSLS_RECENCY),
    WSLS_MAGNITUDE: _null_spec(WSLS_MAGNITUDE),
    WSLS_GRADED: _null_spec(WSLS_GRADED),
    HEURISTIC_PLUS: ModelSpec(
        fit=_fit_heuristic,
        score=_score_heuristic,
        policy=lambda params, layout: ha.HeuristicPolicy(params),
        describe=_fields,
        at_bounds=lambda params: (),
    ),
    Q_LEARNER: ModelSpec(
        fit=_fit_q,
        score=_score_q,
        policy=lambda params, layout: sim.ValuePolicy(params, Q_KIND, layout),
        describe=lambda params: dict(zip(va.PARAMETER_NAMES, params, strict=True)),
        at_bounds=va.parameters_at_bounds,
    ),
}


# ---------------------------------------------------------------------------
# Folds
# ---------------------------------------------------------------------------


def _unfittable(train: pd.DataFrame) -> str | None:
    """Why the null agents' stay and switch rules cannot be fitted to *train*, or ``None``."""
    d = na.decisions(train, na.DEFAULT_DECAY)
    pairs = ~d.start
    rewarded = pairs & (np.nan_to_num(d.prev_magnitude) > 0)
    unrewarded = pairs & ~rewarded
    for name, mask in (("rewarded", rewarded), ("unrewarded", unrewarded)):
        if mask.sum() < na.MIN_PAIRS:
            return f"{mask.sum()} {name} pairs; at least {na.MIN_PAIRS} are needed"
    if len(np.unique(d.prev_magnitude[rewarded])) < 2:
        return "fewer than two distinct rewarded magnitudes; a magnitude slope needs two"
    informative = int((d.delta_h[d.switched] != 0).sum())
    if informative < na.MIN_SWITCHES:
        return f"{informative} switches with a history difference; at least {na.MIN_SWITCHES}"
    recent = int((d.recent[d.switched] != 0).sum())
    if recent < na.MIN_SWITCHES:
        return f"{recent} switches with a more recent alternative; at least {na.MIN_SWITCHES}"
    return None


def skip_reason(trials: pd.DataFrame, min_trials: int = MIN_TRIALS) -> str | None:
    """Why one mouse cannot be cross-validated, or ``None`` if it can.

    A mouse needs at least two problems, *min_trials* trials, and in every fold enough rewarded
    and unrewarded pairs and switches in the training problems to fit the stay and switch rules.

    Args:
        trials (pd.DataFrame): One mouse's rows of the drifting trial table.
        min_trials (int): Trials needed. Default ``MIN_TRIALS``.

    Returns:
        str | None: The reason, naming the fold when a training set is the problem.
    """
    codes, problems = pd.factorize(trials["problem"])
    if len(problems) < 2:
        return f"{len(problems)} problem(s); leave-one-problem-out needs at least two problems"
    if len(trials) < min_trials:
        return f"{len(trials)} trials; at least {min_trials} are needed"
    for fold, problem in enumerate(problems):
        reason = _unfittable(trials[codes != fold])
        if reason:
            return f"training on the problems other than {problem!r}: {reason}"
    return None


class MouseComparison(NamedTuple):
    """The held-out comparison of one mouse.

    Attributes:
        scores (pd.DataFrame): ``mouse, problem, model, n_choices, held_out_log_likelihood``,
            one row per held-out problem and model.
        parameters (pd.DataFrame): ``mouse, model, parameter, value, at_bound`` of every model
            fitted to all of the mouse's trials.
        fits (dict): ``{model: params}`` of those fits, to simulate the models.
    """

    scores: pd.DataFrame
    parameters: pd.DataFrame
    fits: dict


def fit_all(trials: pd.DataFrame, n_starts: int, seed: int, models: Sequence[str] = MODELS) -> dict:
    """``{model: params}`` of each of *models* fitted to *trials*."""
    return {name: SPECS[name].fit(trials, n_starts, seed) for name in models}


def score_fold(
    mouse: str,
    problem,
    train: pd.DataFrame,
    test: pd.DataFrame,
    n_starts: int,
    seed: int,
    models: Sequence[str] = MODELS,
) -> list[dict]:
    """The held-out score of each of *models*: fitted on *train*, scored on *test*.

    ``n_choices`` counts the choices with a previous choice (the ones scored) and
    ``n_segment_starts`` the segment starts of *test*, which no model is scored on.
    """
    n_start = _n_starts(test)
    n_choices = len(test) - n_start
    fits = fit_all(train, n_starts, seed, models)
    return [
        dict(mouse=mouse, problem=problem, model=name, n_choices=n_choices,
             n_segment_starts=n_start,
             held_out_log_likelihood=SPECS[name].score(test, params))
        for name, params in fits.items()
    ]  # fmt: skip


def parameters_table(mouse: str, fits: dict) -> pd.DataFrame:
    """``mouse, model, parameter, value, at_bound`` for fitted parameters ``{model: params}``."""
    rows = []
    for model, params in fits.items():
        at_bounds = SPECS[model].at_bounds(params)
        rows += [
            dict(mouse=mouse, model=model, parameter=name, value=value, at_bound=name in at_bounds)
            for name, value in SPECS[model].describe(params).items()
        ]
    return pd.DataFrame(rows)


def cross_validate(
    trials: pd.DataFrame, n_starts: int = Q_STARTS, seed: int = 0, models: Sequence[str] = MODELS
) -> MouseComparison:
    """Leave-one-problem-out held-out scores of *models*, and their full-data parameters.

    The folds are those of the choice logit (``choice_logit_cv``): one per problem, in order of
    first appearance.

    Args:
        trials (pd.DataFrame): One mouse's rows of the drifting trial table.
        n_starts (int): Random starts of each Q-learner fit. Default ``Q_STARTS``.
        seed (int): Seed of the starting points; each fold has its own stream. Default 0.
        models (Sequence[str]): The models to fit and score. Default ``MODELS``;
            ``NULL_MODELS`` for the null agents.

    Returns:
        MouseComparison: Scores per fold and model, and the parameters of the fits to all trials.

    Raises:
        ValueError: If *trials* hold other than one mouse, fewer than two problems, or a training
            set that a stay or switch rule cannot be fitted to (the message names the mouse).
        RuntimeError: If a fit does not converge or runs to a bound.
    """
    if trials["mouse"].nunique() != 1:
        raise ValueError(f"cross_validate takes one mouse, got {trials['mouse'].nunique()}")
    mouse = trials["mouse"].iloc[0]
    codes, problems = pd.factorize(trials["problem"])
    if len(problems) < 2:
        raise ValueError(f"mouse {mouse}: {len(problems)} problem(s); at least two problems needed")
    rows = []
    for fold, problem in enumerate(problems):
        test = codes == fold
        rows += score_fold(
            mouse, problem, trials[~test], trials[test], n_starts, 1000 * seed + fold, models
        )
    fits = fit_all(trials, n_starts, seed, models)
    return MouseComparison(pd.DataFrame(rows), parameters_table(mouse, fits), fits)


# ---------------------------------------------------------------------------
# Tables across mice
# ---------------------------------------------------------------------------


def nats_table(scores: pd.DataFrame) -> pd.DataFrame:
    """Held-out nats per choice of each model and mouse.

    Args:
        scores (pd.DataFrame): ``MouseComparison.scores`` of one or more mice, concatenated.

    Returns:
        pd.DataFrame: ``mouse, model, n_choices, nats_per_choice`` (negative mean log likelihood
        over the mouse's held-out choices, lower is better), models in ``MODELS`` order.
    """
    total = scores.groupby(["mouse", "model"], sort=False).agg(
        n_choices=("n_choices", "sum"), log_likelihood=("held_out_log_likelihood", "sum")
    )
    total["nats_per_choice"] = -total["log_likelihood"] / total["n_choices"]
    order = {name: i for i, name in enumerate(ALL_MODELS)}
    return (
        total.drop(columns="log_likelihood")
        .reset_index()
        .sort_values(["mouse", "model"], key=lambda s: s.map(order) if s.name == "model" else s,
                     kind="stable")
        .reset_index(drop=True)
    )  # fmt: skip


def heldout_nats_per_trial(scores: pd.DataFrame) -> pd.DataFrame:
    """Held-out nats per *trial* of each model and mouse, segment starts at ``log(1/3)``.

    The scores leave out the segment starts (no model but the Q-learner predicts them). To
    compare with a model scored on every trial, such as the choice logit, each start is credited
    with the uniform ``log(1/3)`` the null agents give it.

    Args:
        scores (pd.DataFrame): ``MouseComparison.scores`` of one or more mice, concatenated.

    Returns:
        pd.DataFrame: ``mouse, model, n_choices`` (all trials of the held-out problems),
        ``log_likelihood`` and ``nats_per_choice`` (negative mean log likelihood, lower is
        better), models in ``ALL_MODELS`` order.
    """
    total = scores.groupby(["mouse", "model"], sort=False).agg(
        n_choices=("n_choices", "sum"), n_segment_starts=("n_segment_starts", "sum"),
        scored=("held_out_log_likelihood", "sum"),
    )  # fmt: skip
    total["log_likelihood"] = total["scored"] + total["n_segment_starts"] * np.log(1 / na.N_ARMS)
    total["n_choices"] = total["n_choices"] + total["n_segment_starts"]
    total["nats_per_choice"] = -total["log_likelihood"] / total["n_choices"]
    order = {name: i for i, name in enumerate(ALL_MODELS)}
    return (
        total[["n_choices", "log_likelihood", "nats_per_choice"]]
        .reset_index()
        .sort_values(["mouse", "model"], key=lambda c: c.map(order) if c.name == "model" else c,
                     kind="stable")
        .reset_index(drop=True)
    )  # fmt: skip


def comparison_name(first: str, second: str) -> str:
    """The column of the held-out nats of *first* minus those of *second*."""
    return f"{first}_minus_{second}"


def difference_table(nats: pd.DataFrame) -> pd.DataFrame:
    """Per mouse, the paired differences of held-out nats per choice of ``COMPARISONS``.

    Args:
        nats (pd.DataFrame): :func:`nats_table` of the mice.

    Returns:
        pd.DataFrame: ``mouse`` and one column per comparison, first model minus second:
        negative means the first predicts held-out choices better.
    """
    wide = nats.pivot(index="mouse", columns="model", values="nats_per_choice")
    table = pd.DataFrame({comparison_name(a, b): wide[a] - wide[b] for a, b in COMPARISONS})
    return table.reset_index()


def difference_summary(table: pd.DataFrame, min_mice: int = ext.MIN_TEST_MICE) -> pd.DataFrame:
    """Across mice: mean ± se and the Wilcoxon and sign tests of each paired difference.

    Args:
        table (pd.DataFrame): :func:`difference_table` of one source.
        min_mice (int): Fewer mice give NaN mean, se and tests. Default ``MIN_TEST_MICE``.

    Returns:
        pd.DataFrame: ``comparison, n, mean, se, n_positive`` (mice where the first model fits
        worse), ``p_wilcoxon, p_sign``.
    """
    columns = [comparison_name(a, b) for a, b in COMPARISONS]
    return ext.gain_summary(table, columns, min_mice).rename(columns={"gain": "comparison"})


def winners(nats: pd.DataFrame) -> pd.DataFrame:
    """``mouse, winner``: the model with the lowest held-out nats per choice of each mouse."""
    best = nats.loc[nats.groupby("mouse", sort=False)["nats_per_choice"].idxmin()]
    return best.rename(columns={"model": "winner"})[["mouse", "winner"]].reset_index(drop=True)


def win_counts(won: pd.DataFrame) -> pd.Series:
    """Mice won by each model of ``MODELS`` (0 for a model that never wins)."""
    return won["winner"].value_counts().reindex(list(MODELS), fill_value=0)


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------


def recovery_matrix(won: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The confusion matrix of model recovery.

    Args:
        won (pd.DataFrame): ``generator, winner``: for each simulated mouse the model that
            generated its choices and the model with the best held-out score.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Counts and fractions of each row (generating model)
        won by each column (winning model); a generator without simulated mice has NaN fractions.
    """
    counts = (
        pd.crosstab(won["generator"], won["winner"])
        .reindex(index=list(MODELS), columns=list(MODELS), fill_value=0)
    )  # fmt: skip
    fractions = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)
    return counts, fractions


def parameter_recovery_table(true: pd.DataFrame, fitted: pd.DataFrame) -> pd.DataFrame:
    """Generating against recovered parameters.

    Args:
        true (pd.DataFrame): ``mouse, parameter, value`` of the generating parameters.
        fitted (pd.DataFrame): The same columns for the parameters fitted to the simulated mice.

    Returns:
        pd.DataFrame: ``mouse, parameter, true, estimate, error`` (estimate minus true).

    Raises:
        ValueError: If a generating parameter has no fitted counterpart.
    """
    keys = ["mouse", "parameter"]
    merged = true[[*keys, "value"]].merge(
        fitted[[*keys, "value"]], on=keys, how="left", suffixes=("_true", "_fitted")
    )
    if merged["value_fitted"].isna().any():
        missing = merged.loc[merged["value_fitted"].isna(), keys].to_dict("records")
        raise ValueError(f"no fitted parameter for {missing}")
    table = merged.rename(columns={"value_true": "true", "value_fitted": "estimate"})
    return table.assign(error=table["estimate"] - table["true"])


def parameter_recovery_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Per parameter: bias, root-mean-square error and correlation of estimate with truth.

    Args:
        table (pd.DataFrame): :func:`parameter_recovery_table`.

    Returns:
        pd.DataFrame: ``parameter, n, bias, rmse, correlation`` (NaN below three simulated mice
        or without spread in the truth or the estimate).
    """
    rows = []
    for parameter, group in table.groupby("parameter", sort=False):
        true, estimate = group["true"].to_numpy(), group["estimate"].to_numpy()
        spread = len(group) >= 3 and true.std() > 0 and estimate.std() > 0
        rows.append(
            dict(parameter=parameter, n=len(group), bias=float(group["error"].mean()),
                 rmse=float(np.sqrt((group["error"] ** 2).mean())),
                 correlation=float(np.corrcoef(true, estimate)[0, 1]) if spread else np.nan)
        )  # fmt: skip
    return pd.DataFrame(rows)
