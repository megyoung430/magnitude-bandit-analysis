"""The second-best test for one source of choices: the mice, or a simulated cohort.

Steps, each a plain function:

1. **Analyze** (:func:`analyze_second_best_source`): per mouse, the three measures and the
   graded curves of :mod:`src.behavior_analysis.switching.second_best`. A cell (a measure after
   zero, after
   non-zero or pooled) with fewer than ``MIN_USED`` used switches is dropped, named in a warning
   and kept in ``dropped``; a mouse that never switches is skipped.
2. **Summarize** (:func:`source_summaries`, :func:`per_mouse_table`): mean ± se across mice from
   ``choice_logit_extensions.MIN_TEST_MICE`` mice, and every mouse's numbers.
3. **Calibrate** (:func:`calibrate_second_best`): 0.5 is not the reference of any measure, because
   the arm chosen most recently tends to be the better one. Each mouse's proportions are read
   against the same proportions of its own fitted win-stay/lose-shift agents played on its own
   task: the replicates are those of the switch regression's calibration
   (:func:`src.pipelines.switching.switch_regression.calibrate_source`: the same agents, replicate
   count and
   seeds, so replicate ``r`` is the same simulated mouse), cached per mouse and agent.

Drawing is in :mod:`src.behavior_visualization.switching.plot_second_best`.
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
from src.behavior_analysis.switching import second_best as sb
from src.behavior_analysis.switching import switch_regression as sr
from src.pipelines.strategy_models import null_agents as null_pipe
from src.pipelines.switching import switch_regression as switch_pipe

# Part of the cache key next to the source of the modules; bump for a change outside them.
CACHE_VERSION = 1


def code_version() -> str:
    """A cache version that changes with ``CACHE_VERSION`` and the source of the modules used.

    joblib hashes the cached worker's arguments but not the code it calls; this string is an
    argument, so editing the design matrix, the null agents, the switch definition, the ranking
    or the simulation invalidates every cached replicate.
    """
    modules = (cl, na, sr, sb, sim)
    digest = hashlib.sha256(
        "\n".join([str(CACHE_VERSION), *(inspect.getsource(m) for m in modules)]).encode()
    )
    return digest.hexdigest()[:16]


class SourceSecondBest(NamedTuple):
    """The second-best test of every analyzable mouse of one source.

    Attributes:
        proportions (pd.DataFrame): ``second_best.PROPORTION_COLUMNS`` of the cells with at
            least *min_used* used switches.
        graded (pd.DataFrame): ``second_best.GRADED_COLUMNS`` of the mice analyzed.
        dropped (pd.DataFrame): The proportions cells below the minimum, as *proportions*.
        skipped (dict[str, str]): Mice not analyzed and why.
        min_used (int): Used switches a cell needs.
        n_lags (int): Lags of the switch regression's design.
    """

    proportions: pd.DataFrame
    graded: pd.DataFrame
    dropped: pd.DataFrame
    skipped: dict
    min_used: int
    n_lags: int


def _warn_dropped(dropped: pd.DataFrame, n_mice: int, min_used: int) -> None:
    """One warning per measure and condition, naming the mice below the minimum."""
    for (measure, condition), cells in dropped.groupby(["measure", "condition"], sort=False):
        mice = ", ".join(f"{r.mouse} ({r.n_used})" for r in cells.itertuples())
        warnings.warn(
            f"{measure}, {condition}: {len(cells)} of {n_mice} mice below {min_used} used "
            f"switches and left out: {mice}",
            stacklevel=3,
        )


def analyze_second_best_source(
    trial_table: pd.DataFrame, *, n_lags: int = cl.DEFAULT_LAGS, min_used: int = sb.MIN_USED
) -> SourceSecondBest:
    """The second-best test of every mouse of *trial_table*.

    Args:
        trial_table (pd.DataFrame): A drifting trial table (the mice, or a simulated cohort).
        n_lags (int): Lags of the switch regression's design. Default
            ``choice_logit.DEFAULT_LAGS``.
        min_used (int): Used switches a cell needs. Default ``second_best.MIN_USED``.

    Returns:
        SourceSecondBest: The cells that reach the minimum, those that do not, and the mice
        skipped, sorted by mouse id.

    Raises:
        ValueError: If no cell has *min_used* used switches.

    Warns:
        UserWarning: For each measure and condition with mice below the minimum, for each mouse
            skipped, and whatever the design raised (tower groups that share no tower).
    """
    results, skipped = {}, {}
    for mouse, rows in sorted(trial_table.groupby("mouse"), key=lambda item: item[0]):
        try:
            results[mouse] = sb.analyze_mouse(rows.reset_index(drop=True), n_lags)
        except sr.TooFewSwitchesError as error:
            skipped[mouse] = str(error)
            warnings.warn(f"mouse {mouse} skipped: {error}", stacklevel=2)
    proportions = pd.concat([r.proportions for r in results.values()], ignore_index=True)
    kept, dropped = sb.apply_minimum(proportions, min_used)
    if kept.empty:
        raise ValueError(f"no cell has at least {min_used} used switches")
    _warn_dropped(dropped, len(results), min_used)
    return SourceSecondBest(
        proportions=kept,
        graded=pd.concat([r.graded for r in results.values()], ignore_index=True),
        dropped=dropped,
        skipped=skipped,
        min_used=min_used,
        n_lags=n_lags,
    )


def analyze_second_best_sources(
    tables: Mapping[str, pd.DataFrame], **kwargs
) -> dict[str, SourceSecondBest]:
    """:func:`analyze_second_best_source` of every source, in order.

    Args:
        tables (Mapping[str, pd.DataFrame]): ``{label: trial table}``.
        **kwargs: Keyword arguments of :func:`analyze_second_best_source`.
    """
    return {label: analyze_second_best_source(t, **kwargs) for label, t in tables.items()}


def _with_source_first(table: pd.DataFrame) -> pd.DataFrame:
    return table[["source", *[c for c in table.columns if c != "source"]]]


def source_summaries(sources: Mapping[str, SourceSecondBest]) -> pd.DataFrame:
    """``second_best.group_summary`` of every source, with a ``source`` column first."""
    tables = [sb.group_summary(s.proportions).assign(source=label) for label, s in sources.items()]
    return _with_source_first(pd.concat(tables, ignore_index=True))


def per_mouse_table(sources: Mapping[str, SourceSecondBest]) -> pd.DataFrame:
    """Every mouse's proportions with their counts and exclusions, with a ``source`` column."""
    frames = [s.proportions.assign(source=label) for label, s in sources.items()]
    return _with_source_first(pd.concat(frames, ignore_index=True))


# ---------------------------------------------------------------------------
# Null calibration
# ---------------------------------------------------------------------------


class _MouseNull(NamedTuple):
    values: pd.DataFrame
    n_failed: int
    warnings: tuple


def simulate_second_best_null(
    trials: pd.DataFrame,
    params,
    kind: str,
    rules: sim.TaskRules,
    replicates: int,
    seed: int,
    n_lags: int,
    version: str,
) -> _MouseNull:
    """The proportions of *replicates* simulations of one mouse's null agent.

    Replicate ``r`` is the simulated mouse of the switch regression's null with the same
    arguments (:func:`~src.pipelines.switching.switch_regression.replicate_seed`).

    Args:
        trials (pd.DataFrame): The mouse's drifting trial table (the task to play).
        params: The mouse's fitted parameters of the null agent *kind*.
        kind (str): One of ``null_agents.KINDS``.
        rules (sim.TaskRules): Drift and block lengths of the simulated task.
        replicates (int): Number of simulations.
        seed (int): Base seed.
        n_lags (int): Lags of the switch regression's design.
        version (str): ``code_version()``; only part of the cache key.

    Returns:
        _MouseNull: ``second_best.analyze_mouse`` proportions of every replicate with a
        ``replicate`` column, the number of replicates without any switch (left out), and the
        warnings raised.
    """
    mouse = trials["mouse"].iloc[0]
    frames, failed = [], 0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for replicate in range(replicates):
            simulated = null_pipe.simulate_null_cohort(
                trials, {mouse: params}, kind, switch_pipe.replicate_seed(seed, replicate), rules
            ).trials
            try:
                table = sb.analyze_mouse(simulated, n_lags).proportions
            except sr.TooFewSwitchesError:
                failed += 1
                continue
            frames.append(table.assign(replicate=replicate))
    values = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return _MouseNull(values, failed, tuple(dict.fromkeys(str(w.message) for w in caught)))


def null_simulator(cache_dir: Path | None):
    """:func:`simulate_second_best_null`, cached under *cache_dir* if one is given."""
    if cache_dir is None:
        return simulate_second_best_null
    return Memory(cache_dir, verbose=0).cache(simulate_second_best_null)


class SecondBestCalibration(NamedTuple):
    """Every mouse of a source against the null of each of its fitted null agents.

    Attributes:
        calibration (pd.DataFrame): ``agent`` and ``second_best.calibrate`` of every kept cell.
        null (pd.DataFrame): ``agent`` and every replicate's proportions.
        summary (pd.DataFrame): ``agent`` and ``second_best.calibrated_summary``.
        failed (pd.DataFrame): ``mouse, agent, n_failed``: replicates left out for no switch.
        replicates (int): Replicates simulated per mouse and agent.
        seed (int): Base seed.
    """

    calibration: pd.DataFrame
    null: pd.DataFrame
    summary: pd.DataFrame
    failed: pd.DataFrame
    replicates: int
    seed: int


def calibrate_second_best(
    trial_table: pd.DataFrame,
    source: SourceSecondBest,
    fits: Mapping[str, Mapping],
    *,
    agents: Sequence[str] = switch_pipe.NULL_AGENTS,
    replicates: int = switch_pipe.DEFAULT_REPLICATES,
    rules: sim.TaskRules | Mapping | None = None,
    seed: int = switch_pipe.CALIBRATION_SEED,
    n_jobs: int = 1,
    cache_dir: Path | None = None,
) -> SecondBestCalibration:
    """Calibrate every kept cell of *source* against its mouse's own null agents.

    Args:
        trial_table (pd.DataFrame): The trial table *source* was analyzed from.
        source (SourceSecondBest): From :func:`analyze_second_best_source`.
        fits (Mapping[str, Mapping]): ``{kind: {mouse: params}}`` of the null agents fitted to
            the mice, e.g. ``NullComparison.controls.fits``.
        agents (Sequence[str]): The null agents to calibrate against. Default
            ``switch_regression.NULL_AGENTS``.
        replicates (int): Simulations per mouse and agent. Default
            ``switch_regression.DEFAULT_REPLICATES``.
        rules (sim.TaskRules | Mapping | None): The simulated task, one rule set or
            ``{cohort: rules}``. Default ``None``: measured within each cohort of *trial_table*.
        seed (int): Base seed. Default ``switch_regression.CALIBRATION_SEED``.
        n_jobs (int): Parallel workers over mice and agents. Default 1.
        cache_dir (Path | None): Cache each mouse's null here. Default ``None``.

    Returns:
        SecondBestCalibration: The calibrated numbers, the null draws and the summaries.

    Raises:
        ValueError: If fewer than two replicates are asked for, an agent is unknown, a mouse of
            *source* has no fitted parameters for an agent, or a kept cell has no null replicate.

    Warns:
        UserWarning: For a mouse whose replicates were partly too short to analyze, and whatever
            the replicates' analyses warned about.
    """
    if replicates < 2:
        raise ValueError(f"replicates must be at least 2, got {replicates}")
    unknown = [a for a in agents if a not in na.KINDS]
    if unknown:
        raise ValueError(f"unknown null agents {unknown}; expected some of {na.KINDS}")
    mice = sorted(set(source.proportions["mouse"]))
    missing = [(a, m) for a in agents for m in mice if m not in fits.get(a, {})]
    if missing:
        raise ValueError(f"no fitted null agent for (agent, mouse) {missing}")
    rules_of = switch_pipe.rules_by_mouse(trial_table, mice, rules)
    by_mouse = {m: rows.reset_index(drop=True) for m, rows in trial_table.groupby("mouse")}
    simulate = null_simulator(cache_dir)
    version = code_version()
    jobs = [(a, m) for a in agents for m in mice]
    results = Parallel(n_jobs=n_jobs)(
        delayed(simulate)(by_mouse[m], fits[a][m], a, rules_of[m], replicates, seed,
                          source.n_lags, version)
        for a, m in jobs
    )  # fmt: skip
    for message in dict.fromkeys(w for r in results for w in r.warnings):
        warnings.warn(message, stacklevel=2)
    calibrations, nulls, failed = [], [], []
    for (agent, mouse), result in zip(jobs, results, strict=True):
        failed.append((mouse, agent, result.n_failed))
        if result.n_failed:
            warnings.warn(
                f"mouse {mouse}, {agent}: {result.n_failed} of {replicates} null replicates had "
                "no switch and are left out",
                stacklevel=2,
            )
        mine = source.proportions[source.proportions["mouse"] == mouse]
        nulls.append(result.values.assign(agent=agent))
        calibrations.append(sb.calibrate(mine, result.values).assign(agent=agent))
    calibration = pd.concat(calibrations, ignore_index=True)
    summary = pd.concat(
        [sb.calibrated_summary(group).assign(agent=agent)
         for agent, group in calibration.groupby("agent", sort=False)],
        ignore_index=True,
    )  # fmt: skip
    return SecondBestCalibration(
        calibration=_agent_first(calibration),
        null=_agent_first(pd.concat(nulls, ignore_index=True)),
        summary=_agent_first(summary),
        failed=pd.DataFrame(failed, columns=["mouse", "agent", "n_failed"]),
        replicates=replicates,
        seed=seed,
    )


def _agent_first(table: pd.DataFrame) -> pd.DataFrame:
    return table[["agent", *[c for c in table.columns if c != "agent"]]]
