"""The switch regression for one source of choices: the mice, or a simulated cohort.

Steps, each a plain function:

1. **Analyze** (:func:`analyse_switch_source`): per mouse, the switch regression of
   :mod:`src.behavior_analysis.switch_regression` (trace decay chosen by held-out likelihood,
   every value variant in both conditions, leave-one-problem-out gains). The ``model`` value is
   the held-out model value of the choice-logit analysis of the same trials, and the regression
   uses that analysis' lags and penalty, so a source goes through the logit and the switch
   regression with one set of settings. Mice are independent: the work is parallel over mice
   and each mouse's result is cached, keyed on its trials, the settings and the source of the
   modules (:func:`code_version`). Warnings raised in a worker travel with its result and are
   re-emitted in the parent, also for a cached result.
2. **Summarize** (:func:`source_summaries`, :func:`per_mouse_table`): mean ± se and tests across
   mice from ``choice_logit_extensions.MIN_TEST_MICE`` mice, and every mouse's numbers.

The same sources as the regression extensions go through it (the mice, the simulated Q-learner
and the win-stay/lose-shift agents, :func:`src.pipelines.regression_extensions.comparison_inputs`).
Drawing is in :mod:`src.behavior_visualization.plot_switch_regression`.
"""

import hashlib
import inspect
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

import pandas as pd
from joblib import Memory, Parallel, delayed

from src.agents import null_agents as na
from src.behavior_analysis import choice_logit as cl
from src.behavior_analysis import switch_regression as sr

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


class _MouseResult(NamedTuple):
    result: sr.MouseSwitchRegression
    warnings: tuple


def analyse_mouse_switches(
    trials: pd.DataFrame,
    n_lags: int,
    trace_grid: tuple,
    l2: float,
    recency_cap: int,
    min_switches: int,
    model_values: sr.ModelValues | None,
    version: str,
) -> _MouseResult:
    """:func:`switch_regression.analyse_mouse` with the warnings it raised.

    Args:
        trials (pd.DataFrame): One mouse's drifting trial table.
        n_lags (int): Last choice lag K.
        trace_grid (tuple): Candidate trace decays.
        l2 (float): Penalty.
        recency_cap (int): Cap of the trials since chosen.
        min_switches (int): Switches needed per condition.
        model_values (sr.ModelValues | None): Values of the ``model`` variant, or ``None``.
        version (str): ``code_version()``; only part of the cache key.

    Returns:
        _MouseResult: The mouse's result and the messages of its warnings.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = sr.analyse_mouse(
            trials, n_lags, trace_grid, l2, recency_cap, min_switches, model_values
        )
    return _MouseResult(result, tuple(dict.fromkeys(str(w.message) for w in caught)))


def mouse_analyser(cache_dir: Path | None):
    """:func:`analyse_mouse_switches`, cached under *cache_dir* if one is given."""
    if cache_dir is None:
        return analyse_mouse_switches
    return Memory(cache_dir, verbose=0).cache(analyse_mouse_switches)


def _skip_reason(trials: pd.DataFrame, n_lags: int, min_switches: int) -> str | None:
    """Why one mouse has too few switches, or ``None`` if it has enough."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the worker re-raises the design's warnings
        try:
            data = sr.switch_data(trials, n_lags, trace_decay=0.5)
        except ValueError as error:
            return str(error)
    for condition in sr.CONDITIONS:
        mine = sr.select_condition(data, condition)
        label = condition.replace("_", " ")
        if len(mine.chosen) < min_switches:
            return f"{len(mine.chosen)} switches {label}; at least {min_switches} are needed"
        if len(set(mine.problem)) < 2:
            return f"switches {label} in fewer than two problems"
    return None


def _model_values(analysis, mouse: str) -> sr.ModelValues:
    meta = analysis.designs[mouse].meta
    return sr.ModelValues(
        meta["problem"].to_numpy(), meta["trial"].to_numpy(), analysis.cv[mouse].values
    )


def analyse_switch_source(
    trial_table: pd.DataFrame,
    analysis=None,
    *,
    trace_grid: Sequence[float] = sr.TRACE_GRID,
    recency_cap: int = sr.RECENCY_CAP,
    min_switches: int = sr.MIN_SWITCHES,
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
    analyse = mouse_analyser(cache_dir)
    version = code_version()
    results = Parallel(n_jobs=n_jobs)(
        delayed(analyse)(
            by_mouse[m].reset_index(drop=True),
            n_lags,
            tuple(trace_grid),
            l2,
            recency_cap,
            min_switches,
            None if analysis is None else _model_values(analysis, m),
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
    )


def analyse_switch_sources(
    inputs: Mapping[str, tuple], **kwargs
) -> dict[str, SourceSwitchRegression]:
    """:func:`analyse_switch_source` of every source, each with its own logit analysis.

    Args:
        inputs (Mapping[str, tuple]): ``{label: (trial table, LogitAnalysis or None)}``, e.g.
            ``regression_extensions.comparison_inputs``.
        **kwargs: Keyword arguments of :func:`analyse_switch_source` (grid, cap, minimum,
            workers, cache).

    Returns:
        dict[str, SourceSwitchRegression]: One result per label, in order.
    """
    return {
        label: analyse_switch_source(table, analysis, **kwargs)
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
