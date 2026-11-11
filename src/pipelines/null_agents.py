"""Null agents fitted to a cohort and played on each mouse's own task.

The negative controls of the choice-logit analyses (:mod:`src.agents.null_agents`) in four
steps, each a plain function:

1. **Fit** every mouse of a drifting trial table with :func:`src.agents.null_agents.
   fit_null_agents`, and summarize the fits (:func:`parameters_table`, :func:`scores_table`,
   :func:`stay_by_magnitude`, :func:`implied_stay_curves`).
2. **Simulate** (:func:`simulate_null_cohort`) one agent per fitted mouse on that mouse's own
   problems: the same offered towers, initiation port, starting magnitudes and number of
   sessions and trials per session. The magnitudes then follow the task's rules (drift and block
   changes, measured per cohort from the data, :func:`task_rules_by_cohort`) rather than
   replaying the mouse's own schedule: the block change zeroes the arm the *agent* chose most,
   so a mouse's recorded schedule is only the right schedule for the mouse's own choices.
3. **Analyze** the simulated cohort like the mice (:func:`null_controls`). The cohort is a
   :class:`~src.pipelines.choice_logit_values.SyntheticCohort` tagged with the agent kind, so
   it goes through :func:`~src.pipelines.choice_logit_values.analyse_trial_table` and the
   summaries of that pipeline, except those that need the value learner's generating rule.
4. **Compare** the agents with a reference analysis (the mice, or simulated Q-learners) in
   one set of tables (:func:`compare_with_controls`); drawing is in
   :mod:`src.behavior_visualization.plot_null_agents`.

**What the switch-target comparison can and cannot say.** The choice-history baseline of the
logit sees five choice lags and no choice tail. An agent whose switches follow a slowly
decaying trace of past choices (the default decay 0.9 keeps most of the trace beyond five
lags) therefore shows above-baseline switch targets with no value guidance at all
(:func:`switch_target_excess`); a fast trace that fits in five lags does not.
"""

import zlib
from collections.abc import Mapping
from dataclasses import fields
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.special import expit

from src.agents import drifting_simulation as sim
from src.agents import null_agents as na
from src.behavior_analysis import model_value_psychometrics as mvp
from src.behavior_analysis.drifting_trial_table import drifting_trial_table
from src.pipelines.agent_baselines import measure_task_rules
from src.pipelines.choice_logit_values import (
    LogitAnalysis,
    SyntheticCohort,
    analyse_trial_table,
    value_curves,
    value_pairs,
)

MAGNITUDES = (0, 1, 2, 3, 4)
AGENT_LABELS = {
    "wsls": "WSLS",
    "wsls_recency": "WSLS (Recency)",
    "wsls_magnitude": "WSLS (Magnitude)",
}


# ---------------------------------------------------------------------------
# Summaries of fits
# ---------------------------------------------------------------------------


def parameters_table(params_by_mouse: dict) -> pd.DataFrame:
    """The fitted parameters of one agent, one row per mouse (``mouse`` and one column each).

    Raises:
        ValueError: If *params_by_mouse* is empty.
    """
    if not params_by_mouse:
        raise ValueError("no fitted parameters to tabulate")
    rows = [
        {"mouse": mouse, **{f.name: getattr(p, f.name) for f in fields(p)}}
        for mouse, p in sorted(params_by_mouse.items())
    ]
    return pd.DataFrame(rows)


def scores_table(trial_table: pd.DataFrame, fits_by_kind: dict) -> pd.DataFrame:
    """In-sample negative log likelihood of each fitted agent on each mouse's choices.

    Two to four fitted parameters against thousands of choices, so the score is not held out.

    Args:
        trial_table (pd.DataFrame): The trial table the agents were fitted to.
        fits_by_kind (dict): ``{kind: {mouse: params}}``.

    Returns:
        pd.DataFrame: ``mouse, model, n_choices, log_likelihood, nats_per_choice``, one row per
        agent and mouse.
    """
    by_mouse = {m: rows for m, rows in trial_table.groupby("mouse")}
    rows = []
    for kind, fits in fits_by_kind.items():
        for mouse, params in sorted(fits.items()):
            mine = by_mouse[mouse]
            ll = na.log_likelihood(mine, params)
            rows.append((mouse, kind, len(mine), ll, -ll / len(mine)))
    columns = ["mouse", "model", "n_choices", "log_likelihood", "nats_per_choice"]
    return pd.DataFrame(rows, columns=columns)


def stay_by_magnitude(trial_table: pd.DataFrame, decay: float = na.DEFAULT_DECAY) -> pd.DataFrame:
    """Observed P(stay) after a choice that paid each magnitude, per mouse.

    Args:
        trial_table (pd.DataFrame): A drifting trial table.
        decay (float): Decay of the history trace, which the decisions are built with (it does
            not change the stays). Default ``DEFAULT_DECAY``.

    Returns:
        pd.DataFrame: ``mouse, magnitude, n_pairs, p_stay`` for the magnitudes that occurred.

    Raises:
        ValueError: If *trial_table* is empty.
    """
    if trial_table.empty:
        raise ValueError("the trial table is empty")
    tables = []
    for mouse, rows in sorted(trial_table.groupby("mouse"), key=lambda item: item[0]):
        d = na.decisions(rows, decay)
        pairs = pd.DataFrame(
            {"magnitude": d.prev_magnitude[~d.start], "stayed": d.stayed[~d.start]}
        )
        summary = pairs.groupby("magnitude")["stayed"].agg(["size", "mean"]).reset_index()
        summary.columns = ["magnitude", "n_pairs", "p_stay"]
        tables.append(summary.assign(mouse=mouse))
    return pd.concat(tables, ignore_index=True)[["mouse", "magnitude", "n_pairs", "p_stay"]]


def implied_stay_curves(fits_by_kind: dict) -> pd.DataFrame:
    """P(stay) each fitted agent implies after 0 to 4 pulses.

    Returns:
        pd.DataFrame: ``mouse, model, magnitude, p_stay``.
    """
    rows = [
        (mouse, kind, m, float(expit(params.stay_logit(m))))
        for kind, fits in fits_by_kind.items()
        for mouse, params in sorted(fits.items())
        for m in MAGNITUDES
    ]
    return pd.DataFrame(rows, columns=["mouse", "model", "magnitude", "p_stay"])


# ---------------------------------------------------------------------------
# Task rules, per cohort
# ---------------------------------------------------------------------------


def cohort_of(problem) -> int | None:
    """The cohort of a problem key: ``cohort`` for ``(cohort, k)`` keys, ``None`` for others."""
    return problem[0] if isinstance(problem, tuple) else None


def _cohort_name(cohort) -> str:
    return "the table" if cohort is None else f"cohort {cohort}"


def _mouse_cohorts(trial_table: pd.DataFrame) -> dict:
    """``{mouse: cohort}``; raises if a mouse has problems of more than one cohort."""
    cohorts = trial_table.groupby("mouse")["problem"].agg(lambda p: {cohort_of(k) for k in p})
    crossing = sorted(m for m, c in cohorts.items() if len(c) > 1)
    if crossing:
        raise ValueError(f"mice {crossing} have problems of more than one cohort")
    return {m: next(iter(c)) for m, c in cohorts.items()}


def task_rules_by_cohort(trial_table: pd.DataFrame, seed: int = 0) -> dict:
    """The drifting task's rules (drift rate, block lengths), measured within each cohort.

    Cohorts may run with different task settings, so nothing is pooled across them. Problems
    keyed ``(cohort, k)`` are grouped by ``cohort``; any other key gives a single group (key
    ``None``).

    Args:
        trial_table (pd.DataFrame): A drifting trial table.
        seed (int): Seed of the drift-rate calibration. Default 0.

    Returns:
        dict: ``{cohort: sim.TaskRules}``.

    Raises:
        ValueError: If a mouse has problems of more than one cohort, or a cohort's rules
            cannot be estimated (no complete block, or the data do not follow the simulated
            block-change rule); the message names the cohort.
    """
    if trial_table.empty:
        raise ValueError("the trial table is empty")
    mouse_cohort = _mouse_cohorts(trial_table)
    rules = {}
    for cohort in sorted(set(mouse_cohort.values()), key=lambda c: (c is None, c)):
        members = [m for m, c in mouse_cohort.items() if c == cohort]
        rows = trial_table[trial_table["mouse"].isin(members)]
        try:
            rules[cohort] = measure_task_rules(rows, seed).rules
        except ValueError as error:
            raise ValueError(
                f"cannot measure the task rules of {_cohort_name(cohort)}: {error}"
            ) from error
    return rules


def _rules_for_mice(trial_table: pd.DataFrame, mice, rules) -> dict:
    """``{mouse: TaskRules}`` from one rule set, a ``{cohort: rules}`` dict, or ``None``."""
    mouse_cohort = _mouse_cohorts(trial_table)
    if rules is None:
        rules = task_rules_by_cohort(trial_table)
    if isinstance(rules, sim.TaskRules):
        return dict.fromkeys(mice, rules)
    missing = sorted({_cohort_name(mouse_cohort[m]) for m in mice if mouse_cohort[m] not in rules})
    if missing:
        raise ValueError(f"no task rules for {missing}")
    return {m: rules[mouse_cohort[m]] for m in mice}


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def _check_parameters(trial_table: pd.DataFrame, params_by_mouse: dict, agent: str) -> None:
    if agent not in na.AGENT_PARAMS:
        raise ValueError(f"unknown agent {agent!r}; expected one of {na.KINDS}")
    if not params_by_mouse:
        raise ValueError("no mice to simulate: params_by_mouse is empty")
    expected = na.AGENT_PARAMS[agent]
    wrong = sorted(m for m, p in params_by_mouse.items() if type(p) is not expected)
    if wrong:
        raise ValueError(f"mice {wrong} do not have {expected.__name__} parameters for {agent!r}")
    missing = sorted(set(params_by_mouse) - set(trial_table["mouse"]))
    if missing:
        raise ValueError(f"mice {missing} are not in the trial table")


def simulate_null_cohort(
    trial_table: pd.DataFrame,
    params_by_mouse: dict,
    agent: str,
    seed: int,
    rules: sim.TaskRules | dict | None = None,
) -> SyntheticCohort:
    """One null agent per mouse, each playing that mouse's own problems.

    Every agent has its own random stream, seeded from *seed*, a checksum of the mouse id and
    the agent, so adding or removing a mouse changes no other mouse's simulation. Simulated
    animals keep their mouse's id.

    Args:
        trial_table (pd.DataFrame): The mice's drifting trial table (the layouts to play).
        params_by_mouse (dict): ``{mouse: params}`` of *agent*, e.g. from
            :func:`~src.agents.null_agents.fit_null_agents`.
        agent (str): One of ``null_agents.KINDS``.
        seed (int): Base seed.
        rules (sim.TaskRules | dict | None): The simulated task: one set for every mouse, or
            ``{cohort: TaskRules}``. Default ``None``: measured within each cohort of
            *trial_table* (:func:`task_rules_by_cohort`).

    Returns:
        SyntheticCohort: The simulated trials, the agents' parameters
        (:func:`parameters_table`), the sessions, and ``kind=agent``.

    Raises:
        ValueError: For an unknown *agent*, no mice, parameters of another agent, a mouse that
            is not in *trial_table*, a mouse in several cohorts, a cohort without rules or whose
            rules cannot be measured.
    """
    _check_parameters(trial_table, params_by_mouse, agent)
    mice = sorted(params_by_mouse)
    rules_by_mouse = _rules_for_mice(trial_table, mice, rules)
    problems: dict = {}
    for mouse in mice:
        params = params_by_mouse[mouse]
        rng = np.random.default_rng([seed, zlib.crc32(str(mouse).encode()), na.KINDS.index(agent)])
        layouts = sim.mouse_layouts(trial_table[trial_table["mouse"] == mouse])
        played = sim.simulate_policy_mouse(
            lambda layout, params=params: na.NullPolicy(params),
            layouts,
            rules_by_mouse[mouse],
            rng,
            mouse,
        )
        for problem, animals in played.items():
            problems.setdefault(problem, {}).update(animals)
    return SyntheticCohort(
        drifting_trial_table(problems), parameters_table(params_by_mouse), problems, agent
    )


# ---------------------------------------------------------------------------
# All agents, end to end
# ---------------------------------------------------------------------------


class NullControls(NamedTuple):
    """Every null agent fitted to a trial table, played on it, and analyzed like the mice.

    Attributes:
        fits (dict): ``{kind: {mouse: params}}``.
        cohorts (dict): ``{kind: SyntheticCohort}`` of the simulated agents.
        analyses (dict): ``{kind: LogitAnalysis}`` of each simulated cohort.
    """

    fits: dict
    cohorts: dict
    analyses: dict


def null_controls(
    trial_table: pd.DataFrame,
    seed: int,
    *,
    n_lags: int,
    tail_decay: float | None,
    l2: float,
    decay: float = na.DEFAULT_DECAY,
    rules: sim.TaskRules | dict | None = None,
) -> NullControls:
    """Fit all of ``null_agents.KINDS`` to *trial_table*, simulate them and run the choice logit.

    The logit settings should be those of the analysis of *trial_table* itself, so the agents
    and the mice go through the same pipeline.

    Args:
        trial_table (pd.DataFrame): The mice's drifting trial table; every mouse must be fittable
            (see :func:`~src.agents.null_agents.fit_null_agents`) and analyzable.
        seed (int): Base seed of the simulations.
        n_lags (int): Lags of the logit. Keyword-only.
        tail_decay (float | None): Reward-tail decay of the logit. Keyword-only.
        l2 (float): Penalty of the logit. Keyword-only.
        decay (float): Decay of the agents' history trace. Default ``DEFAULT_DECAY``.
        rules (sim.TaskRules | dict | None): The simulated task, see
            :func:`simulate_null_cohort`; measured once per cohort if ``None``.

    Returns:
        NullControls: Fits, simulated cohorts and their logit analyses, keyed by agent kind.
    """
    rules = task_rules_by_cohort(trial_table) if rules is None else rules
    fits = {kind: na.fit_null_agents(trial_table, kind, decay) for kind in na.KINDS}
    cohorts = {
        kind: simulate_null_cohort(trial_table, fits[kind], kind, seed, rules) for kind in na.KINDS
    }
    analyses: dict[str, LogitAnalysis] = {
        kind: analyse_trial_table(cohort.trials, n_lags=n_lags, tail_decay=tail_decay, l2=l2)
        for kind, cohort in cohorts.items()
    }
    return NullControls(fits, cohorts, analyses)


# ---------------------------------------------------------------------------
# Comparison with a reference analysis
# ---------------------------------------------------------------------------


def switch_target_excess(
    analysis: LogitAnalysis, n_bins: int | None = mvp.DEFAULT_BINS, values: np.ndarray | None = None
) -> pd.DataFrame:
    """How far P(best alternative | switch) lies above the choice baseline, per reward row.

    The quantity of the switch-target figure: observed minus baseline in each bin of the gap
    between the two unchosen arms' values, averaged over bins (and mice).

    Args:
        analysis (LogitAnalysis): The analysis.
        n_bins (int | None): Bins per mouse and row. Default ``model_value_psychometrics.
            DEFAULT_BINS``.
        values (np.ndarray | None): ``(n_trials, 3)`` values to use instead of the held-out
            model values.

    Returns:
        pd.DataFrame: ``row, observed_top, baseline_top, mean_excess``: the observed and baseline
        probability in the bin with the largest gap, and the mean over bins of observed minus
        baseline (NaN for a row without untied switches).
    """
    pairs = value_pairs(analysis, values)
    curves, baselines = mvp.switch_target_curves_by_reward(pairs, n_bins=n_bins, min_mice=1)
    rows = []
    for row in mvp.REWARD_ROWS:
        observed, baseline = curves[row], baselines[row]
        if observed is None:
            rows.append((row, np.nan, np.nan, np.nan))
            continue
        gap = observed.mean - baseline.mean
        rows.append((row, observed.mean[-1], baseline.mean[-1], float(np.nanmean(gap))))
    return pd.DataFrame(rows, columns=["row", "observed_top", "baseline_top", "mean_excess"])


class NullComparison(NamedTuple):
    """The null agents next to a reference analysis (the mice, or simulated Q-learners).

    Attributes:
        controls (NullControls): The fitted, simulated and analyzed agents.
        analyses (dict): ``{label: LogitAnalysis}``, the reference first, then the agents (labels
            ``AGENT_LABELS``).
        scores (pd.DataFrame): ``mouse, model, nats_per_choice``: each agent in-sample, and the
            reference logit's held-out ``choice_logit`` rows.
        observed_stay (pd.DataFrame): :func:`stay_by_magnitude` of the reference's mice.
        implied_stay (pd.DataFrame): :func:`implied_stay_curves` of the fits.
        psychometrics (dict): ``{label: (curves, baselines)}`` of the model value.
        effects (pd.DataFrame): ``source, raw_rise, rise_beyond_baseline`` of P(best).
        stay_switch (dict): ``{label: (curves, baselines)}`` of ``stay_switch_curves_by_reward``.
        switch_targets (dict): ``{label: (curves, baselines, tied)}`` of
            ``switch_target_curves_by_reward`` and the tied-switch counts.
        switch_excess (pd.DataFrame): :func:`switch_target_excess` of every source, with a
            ``source`` column.
        n_bins (int): Bins used. min_mice (int): Minimum mice per mean. pool_trials (bool): Pairs
            pooled within a mouse.
        extra_cohorts (dict | None): ``{label: SyntheticCohort}`` of further simulated sources
            analyzed like the agents and listed after them (e.g. the fitted Q-learner, a positive
            control), or ``None``.
    """

    controls: NullControls
    analyses: dict
    scores: pd.DataFrame
    observed_stay: pd.DataFrame
    implied_stay: pd.DataFrame
    psychometrics: dict
    effects: pd.DataFrame
    stay_switch: dict
    switch_targets: dict
    switch_excess: pd.DataFrame
    n_bins: int
    min_mice: int
    pool_trials: bool
    extra_cohorts: dict | None = None


def compare_with_controls(
    trial_table: pd.DataFrame,
    reference: LogitAnalysis,
    reference_label: str,
    seed: int,
    *,
    n_bins: int = mvp.DEFAULT_BINS,
    min_mice: int = mvp.MIN_MICE,
    pool_trials: bool = True,
    decay: float = na.DEFAULT_DECAY,
    rules: sim.TaskRules | dict | None = None,
    extra_cohorts: Mapping[str, SyntheticCohort] | None = None,
) -> NullComparison:
    """Fit, simulate and analyze the null agents of the mice behind *reference*, and tabulate.

    The logit settings (lags, tail decay, penalty) are those of *reference*. Only the mice the
    reference could analyze are fitted.

    Args:
        trial_table (pd.DataFrame): The trial table *reference* was made from.
        reference (LogitAnalysis): The analysis of *trial_table* to compare against.
        reference_label (str): Its name in the tables, e.g. ``"Mice"``.
        seed (int): Base seed of the simulations.
        n_bins (int): Quantile bins of the model value per mouse. Default
            ``model_value_psychometrics.DEFAULT_BINS``. Keyword-only.
        min_mice (int): Means from fewer mice are not drawn. Keyword-only.
        pool_trials (bool): Pool a bin's pairs within a mouse in the stay/switch curves.
            Keyword-only.
        decay (float): Decay of the agents' history trace. Keyword-only.
        rules (sim.TaskRules | dict | None): The simulated task, see
            :func:`simulate_null_cohort`. Keyword-only.
        extra_cohorts (Mapping[str, SyntheticCohort] | None): Further simulated sources, e.g.
            ``{"Q-Learner (Fitted)": fitted_q.simulate_fitted_q_cohort(...)}``, run through the
            reference's logit and every table and figure after the agents. Keyword-only.

    Returns:
        NullComparison: All tables of the comparison.

    Raises:
        ValueError: If an extra label is also the label of the reference or an agent.
    """
    extra_cohorts = dict(extra_cohorts or {})
    taken = {reference_label, *AGENT_LABELS.values()}
    if clash := sorted(taken & set(extra_cohorts)):
        raise ValueError(f"extra cohort labels {clash} are already sources of the comparison")
    table = trial_table[trial_table["mouse"].isin(sorted(reference.designs))]
    controls = null_controls(
        table,
        seed,
        n_lags=reference.n_lags,
        tail_decay=reference.tail_decay,
        l2=reference.l2,
        decay=decay,
        rules=rules,
    )
    analyses = (
        {reference_label: reference}
        | {AGENT_LABELS[kind]: controls.analyses[kind] for kind in na.KINDS}
        | {
            label: analyse_trial_table(
                cohort.trials, n_lags=reference.n_lags, tail_decay=reference.tail_decay,
                l2=reference.l2,
            )
            for label, cohort in extra_cohorts.items()
        }
    )  # fmt: skip
    logit = reference.comparison.query("model == 'full'").assign(model="choice_logit")
    scores = pd.concat(
        [scores_table(table, controls.fits), logit[["mouse", "model", "nats_per_choice"]]],
        ignore_index=True,
    )
    psychometrics, effects, stay_switch, switch_targets, excess = {}, [], {}, {}, []
    for label, analysis in analyses.items():
        curves, baselines = value_curves(analysis, n_bins, None, min_mice)
        psychometrics[label] = (curves, baselines)
        best = curves["best"]["best_val"].mean
        effects.append(
            dict(
                source=label,
                raw_rise=best[-1] - best[0],
                rise_beyond_baseline=mvp.value_effect(curves, baselines),
            )
        )
        pairs = value_pairs(analysis)
        stay_switch[label] = mvp.stay_switch_curves_by_reward(
            pairs, pool_trials=pool_trials, n_bins=n_bins, min_mice=min_mice
        )
        targets, target_baselines = mvp.switch_target_curves_by_reward(
            pairs, n_bins=n_bins, min_mice=min_mice
        )
        switch_targets[label] = (targets, target_baselines, mvp.tied_switches_by_reward(pairs))
        excess.append(switch_target_excess(analysis, n_bins).assign(source=label))
    return NullComparison(
        controls=controls,
        analyses=analyses,
        scores=scores,
        observed_stay=stay_by_magnitude(table, decay),
        implied_stay=implied_stay_curves(controls.fits),
        psychometrics=psychometrics,
        effects=pd.DataFrame(effects),
        stay_switch=stay_switch,
        switch_targets=switch_targets,
        switch_excess=pd.concat(excess, ignore_index=True)[
            ["source", "row", "observed_top", "baseline_top", "mean_excess"]
        ],
        n_bins=n_bins,
        min_mice=min_mice,
        pool_trials=pool_trials,
        extra_cohorts=extra_cohorts or None,
    )
