"""The rank tests on the mice, and on replays of agents fitted to each mouse.

Steps, each a plain function:

1. **Analyze** (:func:`analyze_mice`): :func:`src.behavior_analysis.rank_signatures.analyze_mouse`
   on every mouse of a trial table, in parallel over mice and cached per mouse (keyed on its
   trials, its kernel Q parameters and the source of the analysis modules, :func:`code_version`). A
   mouse with too little data is skipped with a warning and listed.
2. **Replay** (:func:`replay_source`): each mouse's fitted WSLS (Magnitude), WSLS (Habit) and
   Q-Learner (Kernel + Forgetting) play that mouse's own problems (its towers, initiation port,
   starting magnitudes, sessions and trials, in a task whose drift and block lengths are measured
   per cohort), ``replicates`` times, and every replay goes through the identical analysis. A
   replay with too little data is left out and counted. The work is parallel over (mouse, agent)
   and cached per pair.
3. **Calibrate** (:func:`calibrate_source`): every mouse readout against each agent's replays
   (:mod:`src.behavior_analysis.rank_calibration`). The Q-learner is the key comparator: a rank
   signature means the mice differ from it in the predicted direction.

Fitting the agents is not done here: pass the fits of the held-out comparison
(:func:`src.pipelines.rank_baseline.heldout_baseline`). Drawing is in
:mod:`src.behavior_visualization.plot_rank_signatures`.
"""

import hashlib
import inspect
import warnings
import zlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from joblib import Memory, Parallel, delayed

from src.agents import drifting_simulation as sim
from src.behavior_analysis import rank_calibration as rcal
from src.behavior_analysis import (
    rank_crossings,
    rank_decisions,
    rank_logit,
    rank_stay,
    rank_switch,
    rank_third,
    rank_transfer,
    rank_worst_split,
)
from src.behavior_analysis import rank_signatures as sg
from src.behavior_analysis import rank_state as rs
from src.behavior_analysis import strategy_models as sm
from src.behavior_analysis.drifting_trial_table import drifting_trial_table
from src.pipelines import switch_regression as switch_pipe

# Part of the cache key next to the source of the modules; bump for a change outside them.
CACHE_VERSION = 1
MODELS = (sm.WSLS_MAGNITUDE, sm.WSLS_HABIT, sm.KERNEL_Q)
AGENT_LABELS = {
    sm.WSLS_MAGNITUDE: "WSLS (Magnitude)",
    sm.WSLS_HABIT: "WSLS (Habit)",
    sm.KERNEL_Q: "Q-Learner (Kernel + Forgetting, Fitted)",
}
KEY_AGENT = sm.KERNEL_Q  # a rank signature means the mice differ from this comparator
DEFAULT_REPLICATES = 100
REPLAY_SEED = 0


def code_version() -> str:
    """A cache version that changes with ``CACHE_VERSION`` and the source of the analysis."""
    modules = (rs, sg, rank_crossings, rank_decisions, rank_logit, rank_stay, rank_switch,
               rank_third, rank_transfer, rank_worst_split, sm, sim)  # fmt: skip
    digest = hashlib.sha256(
        "\n".join([str(CACHE_VERSION), *(inspect.getsource(m) for m in modules)]).encode()
    )
    return digest.hexdigest()[:16]


class MouseAnalysis(NamedTuple):
    """One mouse's readouts and curves, or why it could not be analyzed.

    Attributes:
        readouts (dict | None): ``{readout: value}``; ``None`` if the mouse is unanalyzable.
        curves (pd.DataFrame | None): ``rank_result.CURVE_COLUMNS`` rows; ``None`` likewise.
        reason (str | None): Why the mouse is unanalyzable.
        warnings (tuple): Messages of the warnings the analysis raised.
    """

    readouts: dict | None
    curves: pd.DataFrame | None
    reason: str | None
    warnings: tuple


def analyze_one(trials: pd.DataFrame, kq_params, version: str) -> MouseAnalysis:
    """:func:`rank_signatures.analyze_mouse` with its warnings; *version* only keys the cache."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = sg.analyze_mouse(trials, kq_params)
        except sg.UnanalyzableError as error:
            return MouseAnalysis(None, None, str(error), ())
    messages = tuple(dict.fromkeys(str(w.message) for w in caught))
    return MouseAnalysis(result.readouts, result.curves, None, messages)


def _memoized(function, cache_dir: Path | None):
    return function if cache_dir is None else Memory(cache_dir, verbose=0).cache(function)


class SourceAnalysis(NamedTuple):
    """The rank tests on every analyzable mouse of a trial table.

    Attributes:
        readouts (pd.DataFrame): ``mouse`` and one column per readout.
        curves (pd.DataFrame): ``mouse`` and ``rank_result.CURVE_COLUMNS``.
        skipped (dict[str, str]): Mice not analyzed and why.
    """

    readouts: pd.DataFrame
    curves: pd.DataFrame
    skipped: dict


def analyze_mice(
    trial_table: pd.DataFrame,
    kq_fits: Mapping[str, Sequence[float]] | None = None,
    *,
    n_jobs: int = 1,
    cache_dir: Path | None = None,
) -> SourceAnalysis:
    """The rank tests on every mouse of *trial_table*.

    Args:
        trial_table (pd.DataFrame): A drifting trial table.
        kq_fits (Mapping[str, Sequence[float]] | None): ``{mouse: kernel Q parameters}``; a mouse
            with fits also gets the Q-value readouts of test 1. Default ``None``.
        n_jobs (int): Parallel workers over mice. Default 1.
        cache_dir (Path | None): Cache each mouse here. Default ``None``: no cache.

    Returns:
        SourceAnalysis: Readouts and curves of the analyzable mice (sorted by id) and the rest.

    Raises:
        ValueError: If no mouse can be analyzed.

    Warns:
        UserWarning: For each skipped mouse, and whatever a mouse's analysis warned about.
    """
    by_mouse = dict(sorted(trial_table.groupby("mouse"), key=lambda item: item[0]))
    analyze = _memoized(analyze_one, cache_dir)
    version = code_version()
    results = Parallel(n_jobs=n_jobs)(
        delayed(analyze)(rows.reset_index(drop=True), (kq_fits or {}).get(mouse), version)
        for mouse, rows in by_mouse.items()
    )
    skipped = {m: r.reason for m, r in zip(by_mouse, results, strict=True) if r.reason}
    for mouse, reason in skipped.items():
        warnings.warn(f"mouse {mouse} skipped: {reason}", stacklevel=2)
    for message in dict.fromkeys(m for r in results for m in r.warnings):
        warnings.warn(message, stacklevel=2)
    kept = [(m, r) for m, r in zip(by_mouse, results, strict=True) if not r.reason]
    if not kept:
        raise ValueError("no mouse can be analyzed: " + "; ".join(skipped.values()))
    readouts = pd.DataFrame([{"mouse": m, **r.readouts} for m, r in kept])
    curves = pd.concat([r.curves.assign(mouse=m) for m, r in kept], ignore_index=True)
    return SourceAnalysis(readouts, curves, skipped)


# ---------------------------------------------------------------------------
# Replays
# ---------------------------------------------------------------------------


class MouseReplays(NamedTuple):
    """The analyses of one agent's replays of one mouse.

    Attributes:
        readouts (pd.DataFrame): ``replicate`` and one column per readout, one row per analyzable
            replay.
        curves (pd.DataFrame): ``curve, series, x, y, n`` averaged over the replays.
        n_failed (int): Replays left out for too little data.
        warnings (tuple): Messages of the warnings the analyses raised.
    """

    readouts: pd.DataFrame
    curves: pd.DataFrame
    n_failed: int
    warnings: tuple


def play_mouse(
    trials: pd.DataFrame, model: str, params, rules: sim.TaskRules, seed: int
) -> pd.DataFrame:
    """The trial table of one fitted agent playing one mouse's problems once.

    Args:
        trials (pd.DataFrame): The mouse's drifting trial table (the layouts to play).
        model (str): One of ``MODELS``.
        params: The mouse's fitted parameters of *model*.
        rules (sim.TaskRules): Drift and block lengths of the simulated task.
        seed (int): Replay seed (:func:`src.pipelines.switch_regression.replicate_seed`).

    Returns:
        pd.DataFrame: The simulated trials, under the mouse's id.
    """
    mouse = trials["mouse"].iloc[0]
    rng = np.random.default_rng([seed, zlib.crc32(str(mouse).encode()), MODELS.index(model)])
    played = sim.simulate_policy_mouse(
        lambda layout: sm.SPECS[model].policy(params, layout),
        sim.mouse_layouts(trials),
        rules,
        rng,
        mouse,
    )
    return drifting_trial_table(played)


def replay_one(
    trials: pd.DataFrame,
    model: str,
    params,
    kq_params,
    rules: sim.TaskRules,
    replicates: int,
    seed: int,
    version: str,
) -> MouseReplays:
    """*replicates* replays of one mouse by one fitted agent, each analyzed like the mouse.

    Args:
        trials (pd.DataFrame): The mouse's drifting trial table.
        model (str): One of ``MODELS``.
        params: The mouse's fitted parameters of *model*.
        kq_params: The mouse's fitted kernel Q parameters (for the Q-value readouts), or ``None``.
        rules (sim.TaskRules): Drift and block lengths of the simulated task.
        replicates (int): Number of replays.
        seed (int): Base seed.
        version (str): ``code_version()``; only part of the cache key.

    Returns:
        MouseReplays: Readouts per analyzable replay, mean curves and the failure count.
    """
    rows, curves, failed = [], [], 0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for replicate in range(replicates):
            played = play_mouse(
                trials, model, params, rules, switch_pipe.replicate_seed(seed, replicate)
            )
            try:
                result = sg.analyze_mouse(played, kq_params)
            except sg.UnanalyzableError:
                failed += 1
                continue
            rows.append({"replicate": replicate, **result.readouts})
            curves.append(result.curves)
    mean_curves = (
        pd.concat(curves)
        .groupby(["curve", "series", "x"], sort=False)[["y", "n"]]
        .mean()
        .reset_index()
        if curves
        else pd.DataFrame(columns=["curve", "series", "x", "y", "n"])
    )
    messages = tuple(dict.fromkeys(str(w.message) for w in caught))
    return MouseReplays(pd.DataFrame(rows), mean_curves, failed, messages)


class Replays(NamedTuple):
    """Every agent's replays of every mouse.

    Attributes:
        readouts (dict[str, pd.DataFrame]): ``{model: mouse, replicate, <readouts>}``.
        curves (dict[str, pd.DataFrame]): ``{model: mouse + mean curves}``.
        failed (pd.DataFrame): ``mouse, agent, n_failed``.
        replicates (int): Replays asked for per mouse and agent.
        seed (int): Base seed.
    """

    readouts: dict
    curves: dict
    failed: pd.DataFrame
    replicates: int
    seed: int


def replay_source(
    trial_table: pd.DataFrame,
    fits: Mapping[str, Mapping],
    *,
    models: Sequence[str] = MODELS,
    replicates: int = DEFAULT_REPLICATES,
    rules=None,
    seed: int = REPLAY_SEED,
    n_jobs: int = 1,
    cache_dir: Path | None = None,
) -> Replays:
    """Replay every mouse of *fits* with each of its fitted agents.

    Args:
        trial_table (pd.DataFrame): The mice's drifting trial table.
        fits (Mapping[str, Mapping]): ``{mouse: {model: params}}`` of the fitted agents, e.g.
            ``rank_baseline.Baseline.fits``; must include the kernel Q-learner of every mouse.
        models (Sequence[str]): The agents to replay. Default ``MODELS``.
        replicates (int): Replays per mouse and agent, at least 2. Default ``DEFAULT_REPLICATES``.
        rules (sim.TaskRules | Mapping | None): The simulated task: one rule set or ``{cohort:
            rules}``. Default ``None``: measured within each cohort of *trial_table*.
        seed (int): Base seed. Default ``REPLAY_SEED``.
        n_jobs (int): Parallel workers over (mouse, agent). Default 1.
        cache_dir (Path | None): Cache each (mouse, agent) here. Default ``None``.

    Returns:
        Replays: Readouts of every analyzable replay, mean curves and failure counts.

    Raises:
        ValueError: For fewer than 2 replicates, an unknown model, or a mouse without fits.
    """
    if replicates < 2:
        raise ValueError(f"replicates must be at least 2, got {replicates}")
    unknown = [m for m in models if m not in MODELS]
    missing = [
        (m, mouse) for mouse, f in fits.items() for m in (*models, sm.KERNEL_Q) if m not in f
    ]
    if unknown or missing:
        raise ValueError(f"unknown models {unknown}; missing fits (model, mouse) {missing}")
    mice = sorted(fits)
    rules_of = switch_pipe.rules_by_mouse(trial_table, mice, rules)
    by_mouse = {m: rows.reset_index(drop=True) for m, rows in trial_table.groupby("mouse")}
    replay = _memoized(replay_one, cache_dir)
    jobs = [(model, mouse) for model in models for mouse in mice]
    version = code_version()
    results = Parallel(n_jobs=n_jobs)(
        delayed(replay)(by_mouse[mouse], model, fits[mouse][model], fits[mouse][sm.KERNEL_Q],
                        rules_of[mouse], replicates, seed, version)
        for model, mouse in jobs
    )  # fmt: skip
    for message in dict.fromkeys(w for r in results for w in r.warnings):
        warnings.warn(message, stacklevel=2)
    readouts: dict = {model: [] for model in models}
    curves: dict = {model: [] for model in models}
    failed = []
    for (model, mouse), result in zip(jobs, results, strict=True):
        readouts[model].append(result.readouts.assign(mouse=mouse))
        curves[model].append(result.curves.assign(mouse=mouse))
        failed.append((mouse, model, result.n_failed))
    return Replays(
        readouts={m: pd.concat(frames, ignore_index=True) for m, frames in readouts.items()},
        curves={m: pd.concat(frames, ignore_index=True) for m, frames in curves.items()},
        failed=pd.DataFrame(failed, columns=["mouse", "agent", "n_failed"]),
        replicates=replicates,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def _agent_first(table: pd.DataFrame) -> pd.DataFrame:
    return table[["agent", *[c for c in table.columns if c != "agent"]]]


class Calibration(NamedTuple):
    """Every mouse readout against every agent's replays.

    Attributes:
        table (pd.DataFrame): ``agent`` and ``rank_calibration.calibrate`` with ``predicted`` and
            ``passes``.
        summary (pd.DataFrame): ``agent`` and ``rank_calibration.summarize``.
    """

    table: pd.DataFrame
    summary: pd.DataFrame


def calibrate_source(observed: pd.DataFrame, replays: Replays) -> Calibration:
    """Calibrate each mouse of *observed* against each agent's replays of it.

    Args:
        observed (pd.DataFrame): ``SourceAnalysis.readouts``.
        replays (Replays): :func:`replay_source`.

    Returns:
        Calibration: Per-mouse z and p, and the across-mice summary, per agent.

    Raises:
        ValueError: If a mouse of *observed* has no replays, or every replay of a (mouse, agent)
            had too little data.
    """
    empty = replays.failed[replays.failed["n_failed"] >= replays.replicates]
    if len(empty):
        raise ValueError(
            "every replay had too little data for (mouse, agent) "
            + ", ".join(f"({r.mouse}, {r.agent})" for r in empty.itertuples())
        )
    tables, summaries = [], []
    for model, null in replays.readouts.items():
        table = rcal.calibrate(observed, null)
        tables.append(rcal.passes(table, sg.PREDICTED).assign(agent=model))
        summaries.append(rcal.summarize(table, sg.PREDICTED).assign(agent=model))
    return Calibration(
        _agent_first(pd.concat(tables, ignore_index=True)),
        _agent_first(pd.concat(summaries, ignore_index=True)),
    )
