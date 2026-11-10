"""Null agents fitted to a cohort and played on each mouse's own task.

The negative controls of the choice-logit analyses (:mod:`src.agents.null_agents`) in three
steps, each a plain function:

1. **Fit** every mouse of a drifting trial table with :func:`src.agents.null_agents.
   fit_null_agents`, and summarise the fits (:func:`parameters_table`, :func:`scores_table`,
   :func:`stay_by_magnitude`, :func:`implied_stay_curves`).
2. **Simulate** (:func:`simulate_null_cohort`) one agent per fitted mouse on that mouse's own
   problems: the same offered towers, initiation port, starting magnitudes and number of
   sessions and trials per session. The magnitudes then follow the task's rules (drift and block
   changes, measured from the data) rather than replaying the mouse's own schedule: the block
   change zeroes the arm the *agent* chose most, so a mouse's recorded schedule is only the
   right schedule for the mouse's own choices.
3. The result is a :class:`~src.pipelines.choice_logit_values.SyntheticCohort`, so it goes
   through :func:`~src.pipelines.choice_logit_values.analyse_trial_table` and every summary of
   that pipeline like the simulated Q-learners. Its ``parameters`` table holds the null agents'
   parameters instead of the value learner's.
"""

import zlib
from dataclasses import fields
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.special import expit

from src.agents import drifting_simulation as sim
from src.agents import null_agents as na
from src.behavior_analysis.drifting_trial_table import drifting_trial_table
from src.pipelines.agent_baselines import measure_task_rules
from src.pipelines.choice_logit_values import LogitAnalysis, SyntheticCohort, analyse_trial_table

MAGNITUDES = (0, 1, 2, 3, 4)


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
    """
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
# Simulation
# ---------------------------------------------------------------------------


def _check_parameters(trial_table: pd.DataFrame, params_by_mouse: dict, agent: str) -> None:
    if agent not in na.AGENT_PARAMS:
        raise ValueError(f"unknown agent {agent!r}; expected one of {na.KINDS}")
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
    rules: sim.TaskRules | None = None,
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
        rules (sim.TaskRules | None): The simulated task. Default ``None``: measured from
            *trial_table* (drift rate and block lengths pooled over its mice; the data must
            follow the block-change rule, see ``agent_baselines.measure_task_rules``).

    Returns:
        SyntheticCohort: The simulated trials, the agents' parameters
        (:func:`parameters_table`) and the sessions.

    Raises:
        ValueError: For an unknown *agent*, parameters of another agent, or a mouse that is not
            in *trial_table*.
    """
    _check_parameters(trial_table, params_by_mouse, agent)
    rules = measure_task_rules(trial_table).rules if rules is None else rules
    problems: dict = {}
    for mouse, params in sorted(params_by_mouse.items()):
        rng = np.random.default_rng([seed, zlib.crc32(str(mouse).encode()), na.KINDS.index(agent)])
        layouts = sim.mouse_layouts(trial_table[trial_table["mouse"] == mouse])
        played = sim.simulate_policy_mouse(
            lambda layout, params=params: na.NullPolicy(params), layouts, rules, rng, mouse
        )
        for problem, animals in played.items():
            problems.setdefault(problem, {}).update(animals)
    return SyntheticCohort(
        drifting_trial_table(problems), parameters_table(params_by_mouse), problems
    )


# ---------------------------------------------------------------------------
# All agents, end to end
# ---------------------------------------------------------------------------


class NullControls(NamedTuple):
    """Every null agent fitted to a trial table, played on it, and analysed like the mice.

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
    rules: sim.TaskRules | None = None,
) -> NullControls:
    """Fit all of ``null_agents.KINDS`` to *trial_table*, simulate them and run the choice logit.

    The logit settings should be those of the analysis of *trial_table* itself, so the agents
    and the mice go through the same pipeline.

    Args:
        trial_table (pd.DataFrame): The mice's drifting trial table; every mouse must be fittable
            (see :func:`~src.agents.null_agents.fit_null_agents`) and analysable.
        seed (int): Base seed of the simulations.
        n_lags (int): Lags of the logit. Keyword-only.
        tail_decay (float | None): Reward-tail decay of the logit. Keyword-only.
        l2 (float): Penalty of the logit. Keyword-only.
        decay (float): Decay of the agents' history trace. Default ``DEFAULT_DECAY``.
        rules (sim.TaskRules | None): The simulated task; ``None`` measures it from
            *trial_table* once.

    Returns:
        NullControls: Fits, simulated cohorts and their logit analyses, keyed by agent kind.
    """
    rules = measure_task_rules(trial_table).rules if rules is None else rules
    fits = {kind: na.fit_null_agents(trial_table, kind, decay) for kind in na.KINDS}
    cohorts = {
        kind: simulate_null_cohort(trial_table, fits[kind], kind, seed, rules) for kind in na.KINDS
    }
    analyses: dict[str, LogitAnalysis] = {
        kind: analyse_trial_table(cohort.trials, n_lags=n_lags, tail_decay=tail_decay, l2=l2)
        for kind, cohort in cohorts.items()
    }
    return NullControls(fits, cohorts, analyses)
