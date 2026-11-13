"""The switch regression for one source of choices: the mice, or a simulated cohort.

Steps, each a plain function:

1. **Analyze** (:func:`analyze_switch_source`): per mouse, the switch regression of
   :mod:`src.behavior_analysis.switching.switch_regression` (trace decay chosen by held-out
   likelihood,
   every value variant in both conditions, leave-one-problem-out gains). The ``model`` value is
   the held-out model value of the choice-logit analysis of the same trials, and the regression
   uses that analysis' lags and penalty, so a source goes through the logit and the switch
   regression with one set of settings. Mice are independent: the work is parallel over mice
   and each mouse's result is cached, keyed on its trials, the settings and the source of the
   modules (:func:`code_version`). Warnings raised in a worker travel with its result and are
   re-emitted in the parent, also for a cached result.
2. **Summarize** (:func:`source_summaries`, :func:`per_mouse_table`): mean ± se and tests across
   mice from ``choice_logit_extensions.MIN_TEST_MICE`` mice, and every mouse's numbers.
3. **Calibrate** (:func:`calibrate_source`): each mouse's value weights and gains against its
   own null distribution. Its fitted win-stay/lose-shift agents (PR #115's fits) are played on
   the mouse's own task, matched in problems, sessions and trials, ``replicates`` times, and
   every replicate goes through the identical analysis (decay selection included). The raw
   value weight of an agent that ignores the alternatives' values is not reliably 0 at a mouse's
   switch counts: a nearly deterministic recency rule, and block changes that zero the arm chosen
   most, correlate the true value with the trace. For a strong recency agent (w = 1.5) the null
   weights spread with an SD of 0.1-0.3 logit per pulse, and near the 100-switch minimum they are
   biased (about -0.1 for the true value after a reward). The calibrated z and the empirical p
   carry the spread and the bias in the null, and are the inferential output for the mice.

The same sources as the regression extensions go through it (the mice, the simulated Q-learner
and the win-stay/lose-shift agents,
:func:`src.pipelines.choice_logit.regression_extensions.comparison_inputs`).
Drawing is in :mod:`src.behavior_visualization.switching.plot_switch_regression`.
"""

import hashlib
import inspect
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

import pandas as pd
from joblib import Memory, Parallel, delayed

from src.agents import drifting_simulation as sim
from src.agents import null_agents as na
from src.behavior_analysis.choice_logit import choice_logit as cl
from src.behavior_analysis.switching import switch_regression as sr
from src.pipelines.strategy_models import null_agents as null_pipe

# Part of the cache key next to the source of the modules; bump for a change outside them.
CACHE_VERSION = 1


def code_version() -> str:
    """A cache version that changes with ``CACHE_VERSION`` and the source of the modules used.

    joblib hashes the cached worker's arguments but not the code it calls; this string is an
    argument, so editing the design matrix, the null agents' switch definition or the switch
    regression invalidates every cached mouse.
    """
    modules = (cl, na, sr)
    digest = hashlib.sha256(
        "\n".join([str(CACHE_VERSION), *(inspect.getsource(m) for m in modules)]).encode()
    )
    return digest.hexdigest()[:16]


class SourceSwitchRegression(NamedTuple):
    """The switch regression of every analyzable mouse of one source.

    Attributes:
        values (pd.DataFrame): ``switch_regression.value_table`` of all mice (``mouse, condition,
            value, beta, se, nats_history, nats_value, gain``).
        history (pd.DataFrame): ``switch_regression.history_weight_table`` of all mice.
        counts (pd.DataFrame): ``mouse, condition, n_switches, n_problems``.
        decays (pd.DataFrame): ``mouse, trace_decay``: the decay chosen for each mouse.
        decay_table (pd.DataFrame): ``mouse, trace_decay, nats_per_switch`` of every candidate.
        skipped (dict[str, str]): Mice not analyzed and why.
        n_lags (int): Last choice lag K.
        l2 (float): Penalty of the fits.
        trace_grid (tuple): Candidate trace decays.
        recency_cap (int): Cap of the trials since chosen.
        min_switches (int): Switches each mouse needs per condition.
        firth (bool): Firth's bias reduction in the fits. Default ``switch_regression.FIRTH``.
    """

    values: pd.DataFrame
    history: pd.DataFrame
    counts: pd.DataFrame
    decays: pd.DataFrame
    decay_table: pd.DataFrame
    skipped: dict
    n_lags: int
    l2: float
    trace_grid: tuple
    recency_cap: int
    min_switches: int
    firth: bool = sr.FIRTH


class _MouseResult(NamedTuple):
    result: sr.MouseSwitchRegression
    warnings: tuple


def analyze_mouse_switches(
    trials: pd.DataFrame,
    n_lags: int,
    trace_grid: tuple,
    l2: float,
    recency_cap: int,
    min_switches: int,
    model_values: sr.ModelValues | None,
    firth: bool,
    version: str,
) -> _MouseResult:
    """:func:`switch_regression.analyze_mouse` with the warnings it raised.

    Args:
        trials (pd.DataFrame): One mouse's drifting trial table.
        n_lags (int): Last choice lag K.
        trace_grid (tuple): Candidate trace decays.
        l2 (float): Penalty.
        recency_cap (int): Cap of the trials since chosen.
        min_switches (int): Switches needed per condition.
        model_values (sr.ModelValues | None): Values of the ``model`` variant, or ``None``.
        firth (bool): Firth's bias reduction in the fits.
        version (str): ``code_version()``; only part of the cache key.

    Returns:
        _MouseResult: The mouse's result and the messages of its warnings.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = sr.analyze_mouse(
            trials, n_lags, trace_grid, l2, recency_cap, min_switches, model_values, firth
        )
    return _MouseResult(result, tuple(dict.fromkeys(str(w.message) for w in caught)))


def mouse_analyzer(cache_dir: Path | None):
    """:func:`analyze_mouse_switches`, cached under *cache_dir* if one is given."""
    if cache_dir is None:
        return analyze_mouse_switches
    return Memory(cache_dir, verbose=0).cache(analyze_mouse_switches)


def _skip_reason(trials: pd.DataFrame, n_lags: int, min_switches: int) -> str | None:
    """Why one mouse has too few switches, or ``None`` if it has enough.

    Only a lack of switches skips a mouse; any other problem with its trials (e.g. a problem
    that does not offer three arms) raises, as it would in the analysis.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the worker re-raises the design's warnings
        try:
            data = sr.switch_data(trials, n_lags, trace_decay=0.5)
        except sr.TooFewSwitchesError as error:
            return str(error)
    counts = sr.switch_counts(data)
    count_text = (
        f"{counts[sr.AFTER_ZERO]} switches after zero, {counts[sr.AFTER_NONZERO]} after non-zero"
    )
    if min(counts.values()) < min_switches:
        return f"{count_text}; at least {min_switches} per condition are needed"
    for condition in sr.CONDITIONS:
        if len(set(sr.select_condition(data, condition).problem)) < 2:
            label = condition.replace("_", " ")
            return f"{count_text}; the switches {label} come from fewer than two problems"
    return None


def _model_values(analysis, mouse: str) -> sr.ModelValues:
    meta = analysis.designs[mouse].meta
    return sr.ModelValues(
        meta["problem"].to_numpy(), meta["trial"].to_numpy(), analysis.cv[mouse].values
    )


def analyze_switch_source(
    trial_table: pd.DataFrame,
    analysis=None,
    *,
    trace_grid: Sequence[float] = sr.TRACE_GRID,
    recency_cap: int = sr.RECENCY_CAP,
    min_switches: int = sr.MIN_SWITCHES,
    firth: bool = sr.FIRTH,
    n_jobs: int = 1,
    cache_dir: Path | None = None,
) -> SourceSwitchRegression:
    """The switch regression of every mouse of *trial_table*.

    Args:
        trial_table (pd.DataFrame): A drifting trial table (the mice, or a simulated cohort).
        analysis (LogitAnalysis | None): The choice-logit analysis of *trial_table*
            (``choice_logit_values.analyse_trial_table``): its mice are analyzed, with its lags,
            penalty and held-out model values. ``None``: every mouse, the default lags and
            penalty, and no ``model`` variant.
        trace_grid (Sequence[float]): Candidate trace decays. Default
            ``switch_regression.TRACE_GRID``.
        recency_cap (int): Cap of the trials since chosen. Default ``RECENCY_CAP``.
        min_switches (int): Mice with fewer switches in a condition, or switches of a condition
            in fewer than two problems, are skipped with a warning. Default ``MIN_SWITCHES``.
        firth (bool): Firth's bias reduction in the fits. Default ``switch_regression.FIRTH``.
        n_jobs (int): Parallel workers over mice (joblib). Default 1.
        cache_dir (Path | None): Cache each mouse's result here. Default ``None``: no cache.

    Returns:
        SourceSwitchRegression: Results for the analyzable mice, sorted by id.

    Raises:
        ValueError: If no mouse has enough switches.
        RuntimeError: If a fit does not converge.

    Warns:
        UserWarning: For each skipped mouse, and whatever a mouse's analysis warned about, also
            when served from the cache.
    """
    n_lags = cl.DEFAULT_LAGS if analysis is None else analysis.n_lags
    l2 = cl.DEFAULT_L2 if analysis is None else analysis.l2
    by_mouse = dict(sorted(trial_table.groupby("mouse"), key=lambda item: item[0]))
    skipped = {}
    if analysis is not None:
        skipped = {m: "not in the logit analysis" for m in by_mouse if m not in analysis.designs}
    for mouse, rows in by_mouse.items():
        if mouse not in skipped and (reason := _skip_reason(rows, n_lags, min_switches)):
            skipped[mouse] = reason
    for mouse, reason in skipped.items():
        warnings.warn(f"mouse {mouse} skipped: {reason}", stacklevel=2)
    kept = [m for m in by_mouse if m not in skipped]
    if not kept:
        raise ValueError(
            "no mouse has enough switches: " + "; ".join(f"{m}: {r}" for m, r in skipped.items())
        )
    analyze = mouse_analyzer(cache_dir)
    version = code_version()
    results = Parallel(n_jobs=n_jobs)(
        delayed(analyze)(
            by_mouse[m].reset_index(drop=True),
            n_lags,
            tuple(trace_grid),
            l2,
            recency_cap,
            min_switches,
            None if analysis is None else _model_values(analysis, m),
            firth,
            version,
        )
        for m in kept
    )
    for message in dict.fromkeys(m for r in results for m in r.warnings):
        warnings.warn(message, stacklevel=2)
    mice = [r.result for r in results]
    return SourceSwitchRegression(
        values=pd.concat([sr.value_table(r.scores, r.weights) for r in mice], ignore_index=True),
        history=pd.concat([sr.history_weight_table(r.weights) for r in mice], ignore_index=True),
        counts=pd.concat([r.counts for r in mice], ignore_index=True),
        decays=pd.DataFrame({"mouse": kept, "trace_decay": [r.trace_decay for r in mice]}),
        decay_table=pd.concat([r.decay_table for r in mice], ignore_index=True),
        skipped=skipped,
        n_lags=n_lags,
        l2=l2,
        trace_grid=tuple(trace_grid),
        recency_cap=recency_cap,
        min_switches=min_switches,
        firth=firth,
    )


def analyze_switch_sources(
    inputs: Mapping[str, tuple], **kwargs
) -> dict[str, SourceSwitchRegression]:
    """:func:`analyze_switch_source` of every source, each with its own logit analysis.

    Args:
        inputs (Mapping[str, tuple]): ``{label: (trial table, LogitAnalysis or None)}``, e.g.
            ``regression_extensions.comparison_inputs``.
        **kwargs: Keyword arguments of :func:`analyze_switch_source` (grid, cap, minimum,
            workers, cache).

    Returns:
        dict[str, SourceSwitchRegression]: One result per label, in order.
    """
    return {
        label: analyze_switch_source(table, analysis, **kwargs)
        for label, (table, analysis) in inputs.items()
    }


def source_summaries(sources: Mapping[str, SourceSwitchRegression]) -> pd.DataFrame:
    """``switch_regression.group_summary`` of every source, with a ``source`` column first."""
    tables = [sr.group_summary(s.values).assign(source=label) for label, s in sources.items()]
    table = pd.concat(tables, ignore_index=True)
    return table[["source", *[c for c in table.columns if c != "source"]]]


def per_mouse_table(sources: Mapping[str, SourceSwitchRegression]) -> pd.DataFrame:
    """Every mouse's value weights and gains with its switch count and trace decay.

    Returns:
        pd.DataFrame: ``source, mouse, condition, value, beta, se, nats_history, nats_value,
        gain, n_switches, n_problems, trace_decay``.
    """
    frames = [
        s.values.merge(s.counts, on=["mouse", "condition"])
        .merge(s.decays, on="mouse")
        .assign(source=label)
        for label, s in sources.items()
    ]
    table = pd.concat(frames, ignore_index=True)
    return table[["source", *[c for c in table.columns if c != "source"]]]


# ---------------------------------------------------------------------------
# Null calibration
# ---------------------------------------------------------------------------

# Null agents each mouse is calibrated against: the realistic one (magnitude-graded staying,
# recency switches) first, whose band is the one drawn; then plain recency switches; then the
# graded agent (linear stay logit, switches to the most recent alternative); then the habit agent
# (WSLS (Magnitude) whose stay logit grows with a slowly built choice kernel): it revisits its usual
# arms, which may also be the arms that paid more, so habit alone could produce a value weight.
NULL_AGENTS = ("wsls_magnitude", "wsls_recency", "wsls_graded", "wsls_habit")
DEFAULT_REPLICATES = 100
CALIBRATION_SEED = 0
# The calibrated variants. The model value is left out: each replicate would need its own
# leave-one-problem-out choice logit, and the model value is a consistency check only.
CALIBRATED_VALUES = (sr.TRUE, sr.BELIEVED)


def replicate_seed(seed: int, replicate: int) -> int:
    """The simulation seed of one replicate: distinct for every (seed, replicate < 100000)."""
    return seed * 100_000 + replicate


class _MouseNull(NamedTuple):
    values: pd.DataFrame
    n_failed: int
    warnings: tuple


def simulate_mouse_null(
    trials: pd.DataFrame,
    params,
    kind: str,
    rules: sim.TaskRules,
    replicates: int,
    seed: int,
    settings: tuple,
    version: str,
) -> _MouseNull:
    """The value tables of *replicates* simulations of one mouse's null agent.

    Args:
        trials (pd.DataFrame): The mouse's drifting trial table (the task to play).
        params: The mouse's fitted parameters of the null agent *kind*.
        kind (str): One of ``null_agents.KINDS``.
        rules (sim.TaskRules): Drift and block lengths of the simulated task.
        replicates (int): Number of simulations.
        seed (int): Base seed (:func:`replicate_seed`).
        settings (tuple): ``(n_lags, trace_grid, l2, recency_cap, min_switches, firth)`` of the
            analysis of the mouse.
        version (str): ``code_version()``; only part of the cache key.

    Returns:
        _MouseNull: ``value_table`` rows of the true and believed values with a ``replicate``
        column, the number of replicates with too few switches to analyze (left out), and the
        warnings raised.
    """
    n_lags, trace_grid, l2, recency_cap, min_switches, firth = settings
    mouse = trials["mouse"].iloc[0]
    frames, failed = [], 0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for replicate in range(replicates):
            simulated = null_pipe.simulate_null_cohort(
                trials, {mouse: params}, kind, replicate_seed(seed, replicate), rules
            ).trials
            try:
                result = sr.analyze_mouse(
                    simulated, n_lags, trace_grid, l2, recency_cap, min_switches, None, firth
                )
            except sr.TooFewSwitchesError:
                failed += 1
                continue
            table = sr.value_table(result.scores, result.weights)
            frames.append(table.assign(replicate=replicate))
    values = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return _MouseNull(values, failed, tuple(dict.fromkeys(str(w.message) for w in caught)))


def rules_by_mouse(trial_table: pd.DataFrame, mice: Sequence[str], rules) -> dict:
    """``{mouse: TaskRules}`` from one rule set, ``{cohort: rules}`` or ``None`` (measured)."""
    if rules is None:
        rules = null_pipe.task_rules_by_cohort(trial_table)
    if isinstance(rules, sim.TaskRules):
        return dict.fromkeys(mice, rules)
    cohort = trial_table.groupby("mouse")["problem"].first().map(null_pipe.cohort_of)
    missing = sorted({str(cohort[m]) for m in mice if cohort[m] not in rules})
    if missing:
        raise ValueError(f"no task rules for cohorts {missing}")
    return {m: rules[cohort[m]] for m in mice}


def null_simulator(cache_dir: Path | None):
    """:func:`simulate_mouse_null`, cached under *cache_dir* if one is given."""
    if cache_dir is None:
        return simulate_mouse_null
    return Memory(cache_dir, verbose=0).cache(simulate_mouse_null)


def _agent_first(table: pd.DataFrame) -> pd.DataFrame:
    return table[["agent", *[c for c in table.columns if c != "agent"]]]


class NullCalibration(NamedTuple):
    """Every mouse of a source against the null of each of its fitted null agents.

    Attributes:
        calibration (pd.DataFrame): ``agent`` and ``switch_regression.null_calibration`` of the
            true and believed values of every mouse.
        null (pd.DataFrame): ``agent`` and every replicate's ``value_table`` row.
        summary (pd.DataFrame): ``agent`` and ``switch_regression.calibrated_summary``.
        failed (pd.DataFrame): ``mouse, agent, n_failed``: replicates left out for too few
            switches.
        replicates (int): Replicates simulated per mouse and agent.
        seed (int): Base seed.
    """

    calibration: pd.DataFrame
    null: pd.DataFrame
    summary: pd.DataFrame
    failed: pd.DataFrame
    replicates: int
    seed: int


def calibrate_source(
    trial_table: pd.DataFrame,
    source: SourceSwitchRegression,
    fits: Mapping[str, Mapping],
    *,
    agents: Sequence[str] = NULL_AGENTS,
    replicates: int = DEFAULT_REPLICATES,
    rules: sim.TaskRules | Mapping | None = None,
    seed: int = CALIBRATION_SEED,
    n_jobs: int = 1,
    cache_dir: Path | None = None,
) -> NullCalibration:
    """Calibrate every analyzed mouse of *source* against its own null agents.

    Args:
        trial_table (pd.DataFrame): The trial table *source* was analyzed from.
        source (SourceSwitchRegression): From :func:`analyze_switch_source`; its settings are
            used for every replicate.
        fits (Mapping[str, Mapping]): ``{kind: {mouse: params}}`` of the null agents fitted to
            the mice, e.g. ``NullComparison.controls.fits``.
        agents (Sequence[str]): The null agents to calibrate against. Default ``NULL_AGENTS``.
        replicates (int): Simulations per mouse and agent. Default ``DEFAULT_REPLICATES``.
        rules (sim.TaskRules | Mapping | None): The simulated task, one rule set or ``{cohort:
            rules}``. Default ``None``: measured within each cohort of *trial_table*, as the
            null agents of ``null_agents.compare_with_controls`` are.
        seed (int): Base seed; each mouse and agent also has its own stream. Default
            ``CALIBRATION_SEED``.
        n_jobs (int): Parallel workers over mice and agents. Default 1.
        cache_dir (Path | None): Cache each mouse's null here. Default ``None``.

    Returns:
        NullCalibration: The calibrated numbers, the null draws and the summaries.

    Raises:
        ValueError: If fewer than two replicates are asked for, an agent is unknown, or a mouse
            of *source* has no fitted parameters for an agent.

    Warns:
        UserWarning: For a mouse whose replicates were partly too short to analyze, and whatever
            the replicates' analyses warned about.
    """
    if replicates < 2:
        raise ValueError(f"replicates must be at least 2, got {replicates}")
    unknown = [a for a in agents if a not in na.KINDS]
    if unknown:
        raise ValueError(f"unknown null agents {unknown}; expected some of {na.KINDS}")
    mice = list(source.decays["mouse"])
    missing = [(a, m) for a in agents for m in mice if m not in fits.get(a, {})]
    if missing:
        raise ValueError(f"no fitted null agent for (agent, mouse) {missing}")
    rules_of = rules_by_mouse(trial_table, mice, rules)
    settings = (source.n_lags, source.trace_grid, source.l2, source.recency_cap,
                source.min_switches, source.firth)  # fmt: skip
    by_mouse = {m: rows.reset_index(drop=True) for m, rows in trial_table.groupby("mouse")}
    simulate = null_simulator(cache_dir)
    jobs = [(a, m) for a in agents for m in mice]
    results = Parallel(n_jobs=n_jobs)(
        delayed(simulate)(by_mouse[m], fits[a][m], a, rules_of[m], replicates, seed, settings,
                          code_version())
        for a, m in jobs
    )  # fmt: skip
    for message in dict.fromkeys(w for r in results for w in r.warnings):
        warnings.warn(message, stacklevel=2)
    observed = source.values[source.values["value"].isin(CALIBRATED_VALUES)]
    calibrations, nulls, failed = [], [], []
    for (agent, mouse), result in zip(jobs, results, strict=True):
        failed.append((mouse, agent, result.n_failed))
        if result.n_failed:
            warnings.warn(
                f"mouse {mouse}, {agent}: {result.n_failed} of {replicates} null replicates had "
                "too few switches and are left out",
                stacklevel=2,
            )
        null = result.values[result.values["value"].isin(CALIBRATED_VALUES)]
        nulls.append(null.assign(agent=agent))
        mine = observed[observed["mouse"] == mouse]
        calibrations.append(sr.null_calibration(mine, null).assign(agent=agent))
    calibration = pd.concat(calibrations, ignore_index=True)
    summary = pd.concat(
        [sr.calibrated_summary(group).assign(agent=agent)
         for agent, group in calibration.groupby("agent", sort=False)],
        ignore_index=True,
    )  # fmt: skip
    return NullCalibration(
        calibration=_agent_first(calibration),
        null=_agent_first(pd.concat(nulls, ignore_index=True)),
        summary=_agent_first(summary),
        failed=pd.DataFrame(failed, columns=["mouse", "agent", "n_failed"]),
        replicates=replicates,
        seed=seed,
    )
