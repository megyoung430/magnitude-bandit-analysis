"""A Q-learner fitted to each mouse, played on that mouse's own task: a positive control.

The win-stay/lose-shift agents of :mod:`src.pipelines.null_agents` are the negative controls of
the value analyses: they never look at the alternatives' values. This is the matching positive
control, made the same way. A value learner (:mod:`src.agents.value_agent`: learning rate, beta,
stickiness and a bias to the plus-shaped ports) is fitted to every mouse, and then plays that
mouse's own problems (its towers, initiation port, starting magnitudes and sessions) in a task
whose drift and block lengths are measured per cohort. The simulated mice keep their mouse's id,
so they take that mouse's color in figures, as the null agents do.

It answers what a synthetic Q-learner with round parameters cannot: would a value learner with
*this mouse's* noise level (a low beta, strong stickiness) have shown the value effect that the
analysis looks for? :func:`power_check` gives that as the share of fitted-Q replicates whose
switch-regression value weight lies above the mouse's own.

The fits are those of the held-out comparison (:mod:`src.pipelines.strategy_comparison`), reused
through :func:`fits_of`; :func:`fit_q_learners` fits them from a trial table. Nothing here is
calibrated against: the fitted Q-learner is a positive control, the mice are calibrated against
their WSLS agents (:func:`src.pipelines.switch_regression.calibrate_source`).
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
from src.agents import null_agents as na
from src.agents import value_agent as va
from src.behavior_analysis import choice_logit as cl
from src.behavior_analysis import strategy_models as sm
from src.behavior_analysis import switch_regression as sr
from src.behavior_analysis.drifting_trial_table import drifting_trial_table
from src.pipelines import switch_regression as switch_pipe
from src.pipelines.choice_logit_values import SyntheticCohort

KIND = "q_fitted"
LABEL = "Q-Learner (Fitted)"
# Part of the cache key next to the source of the modules; bump for a change outside them.
CACHE_VERSION = 1
# The stream index of this agent: after the null agents' (``null_agents.KINDS``).
_STREAM = len(na.KINDS)


def code_version() -> str:
    """A cache version that changes with ``CACHE_VERSION`` and the source of the modules used."""
    modules = (cl, na, sr, va, sm, sim)
    digest = hashlib.sha256(
        "\n".join([str(CACHE_VERSION), *(inspect.getsource(m) for m in modules)]).encode()
    )
    return digest.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Fits
# ---------------------------------------------------------------------------


def fits_of(source) -> dict[str, tuple[float, ...]]:
    """``{mouse: Q-learner parameters}`` of a strategy comparison's full-data fits.

    Args:
        source (SourceComparison): :func:`src.pipelines.strategy_comparison.
            analyze_comparison_source`.

    Returns:
        dict[str, tuple[float, ...]]: ``(learning_rate, beta, kappa, plus_bias)`` per mouse.
    """
    return {mouse: fits[sm.Q_LEARNER] for mouse, fits in source.fits.items()}


def fit_q_learners(
    trial_table: pd.DataFrame, *, n_starts: int = sm.Q_STARTS, seed: int = 0
) -> dict[str, tuple[float, ...]]:
    """Fit the Q-learner to every mouse on all of its problems, from several starts.

    Args:
        trial_table (pd.DataFrame): A drifting trial table.
        n_starts (int): Random starts of each fit. Default ``strategy_models.Q_STARTS``.
        seed (int): Seed of the starts. Default 0.

    Returns:
        dict[str, tuple[float, ...]]: ``(learning_rate, beta, kappa, plus_bias)`` per mouse.

    Raises:
        ValueError: If the table has no trials.
        RuntimeError: If a fit does not converge (from :func:`src.agents.value_agent.fit`).
    """
    if trial_table.empty:
        raise ValueError("no trials to fit")
    fit = sm.SPECS[sm.Q_LEARNER].fit
    return {
        mouse: fit(rows.reset_index(drop=True), n_starts, seed)
        for mouse, rows in trial_table.groupby("mouse", sort=True)
    }


def parameters_table(params_by_mouse: Mapping[str, Sequence[float]]) -> pd.DataFrame:
    """One row per mouse: the four parameters and whether any lies at a bound of the fit.

    Args:
        params_by_mouse (Mapping[str, Sequence[float]]): As :func:`fit_q_learners`.

    Returns:
        pd.DataFrame: ``mouse``, the ``value_agent`` parameter names, and ``at_bound`` (a
        parameter at a bound is poorly identified; ``bounded`` names them).
    """
    rows = []
    for mouse, params in sorted(params_by_mouse.items()):
        bounded = va.parameters_at_bounds(params)
        rows.append(
            {"mouse": mouse, **dict(zip(va.PARAMETER_NAMES, params, strict=True)),
             "at_bound": bool(bounded), "bounded": ", ".join(bounded)}
        )  # fmt: skip
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Simulation on each mouse's own task
# ---------------------------------------------------------------------------


def _check(trial_table: pd.DataFrame, params_by_mouse: Mapping) -> None:
    if not params_by_mouse:
        raise ValueError("no mice to simulate: params_by_mouse is empty")
    missing = sorted(set(params_by_mouse) - set(trial_table["mouse"]))
    if missing:
        raise ValueError(f"mice {missing} are not in the trial table")


def _simulate_mouse(trials: pd.DataFrame, params, rules: sim.TaskRules, seed: int) -> dict:
    """``{problem: {mouse: sessions}}`` of one fitted Q-learner playing one mouse's problems."""
    mouse = trials["mouse"].iloc[0]
    rng = np.random.default_rng([seed, zlib.crc32(str(mouse).encode()), _STREAM])
    return sim.simulate_policy_mouse(
        lambda layout: sm.SPECS[sm.Q_LEARNER].policy(params, layout),
        sim.mouse_layouts(trials),
        rules,
        rng,
        mouse,
    )


def simulate_fitted_q_cohort(
    trial_table: pd.DataFrame,
    params_by_mouse: Mapping[str, Sequence[float]],
    seed: int,
    rules: sim.TaskRules | Mapping | None = None,
) -> SyntheticCohort:
    """One fitted Q-learner per mouse, each playing that mouse's own problems.

    Built as :func:`src.pipelines.null_agents.simulate_null_cohort` is: the same layouts, the same
    per-cohort task rules, a random stream per mouse (so adding a mouse changes no other), and the
    mouse's id on its simulated animal.

    Args:
        trial_table (pd.DataFrame): The mice's drifting trial table (the layouts to play).
        params_by_mouse (Mapping[str, Sequence[float]]): ``{mouse: (learning_rate, beta, kappa,
            plus_bias)}``, e.g. :func:`fits_of`.
        seed (int): Base seed.
        rules (sim.TaskRules | Mapping | None): The simulated task: one set for every mouse, or
            ``{cohort: TaskRules}``. Default ``None``: measured within each cohort of
            *trial_table*.

    Returns:
        SyntheticCohort: The simulated trials, the parameters, the sessions and
        ``kind="q_fitted"``.

    Raises:
        ValueError: For no mice, a mouse that is not in *trial_table*, or a cohort without rules.
    """
    _check(trial_table, params_by_mouse)
    mice = sorted(params_by_mouse)
    rules_of = switch_pipe.rules_by_mouse(trial_table, mice, rules)
    problems: dict = {}
    for mouse in mice:
        mine = trial_table[trial_table["mouse"] == mouse]
        played = _simulate_mouse(mine, params_by_mouse[mouse], rules_of[mouse], seed)
        for problem, animals in played.items():
            problems.setdefault(problem, {}).update(animals)
    parameters = pd.DataFrame(
        [{"mouse": m, **dict(zip(va.PARAMETER_NAMES, params_by_mouse[m], strict=True))}
         for m in mice]
    )  # fmt: skip
    return SyntheticCohort(drifting_trial_table(problems), parameters, problems, KIND)


# ---------------------------------------------------------------------------
# Power check
# ---------------------------------------------------------------------------


class _MouseReplicates(NamedTuple):
    values: pd.DataFrame
    n_failed: int
    warnings: tuple


def simulate_mouse_replicates(
    trials: pd.DataFrame,
    params,
    rules: sim.TaskRules,
    replicates: int,
    seed: int,
    settings: tuple,
    version: str,
) -> _MouseReplicates:
    """The value tables of *replicates* simulations of one mouse's fitted Q-learner.

    Args:
        trials (pd.DataFrame): The mouse's drifting trial table (the task to play).
        params (Sequence[float]): The mouse's fitted Q-learner parameters.
        rules (sim.TaskRules): Drift and block lengths of the simulated task.
        replicates (int): Number of simulations.
        seed (int): Base seed (:func:`src.pipelines.switch_regression.replicate_seed`).
        settings (tuple): ``(n_lags, trace_grid, l2, recency_cap, min_switches, firth)`` of the
            mouse's switch regression.
        version (str): ``code_version()``; only part of the cache key.

    Returns:
        _MouseReplicates: ``value_table`` rows of every analyzable replicate (with a
        ``replicate`` column), the number of replicates with too few switches, and the warnings
        raised.
    """
    n_lags, trace_grid, l2, recency_cap, min_switches, firth = settings
    frames, failed = [], 0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for replicate in range(replicates):
            problems = _simulate_mouse(
                trials, params, rules, switch_pipe.replicate_seed(seed, replicate)
            )
            simulated = drifting_trial_table(problems)
            try:
                result = sr.analyze_mouse(
                    simulated, n_lags, trace_grid, l2, recency_cap, min_switches, None, firth
                )
            except sr.TooFewSwitchesError:
                failed += 1
                continue
            frames.append(sr.value_table(result.scores, result.weights).assign(replicate=replicate))
    values = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return _MouseReplicates(values, failed, tuple(dict.fromkeys(str(w.message) for w in caught)))


def replicate_simulator(cache_dir: Path | None):
    """:func:`simulate_mouse_replicates`, cached under *cache_dir* if one is given."""
    if cache_dir is None:
        return simulate_mouse_replicates
    return Memory(cache_dir, verbose=0).cache(simulate_mouse_replicates)


POWER_COLUMNS = ["mouse", "condition", "value", "observed", "n_replicates", "null_mean",
                 "fraction_above"]  # fmt: skip


def power_check(
    trial_table: pd.DataFrame,
    source,
    params_by_mouse: Mapping[str, Sequence[float]],
    *,
    replicates: int = 20,
    rules: sim.TaskRules | Mapping | None = None,
    seed: int = 0,
    n_jobs: int = 1,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Would a Q-learner with each mouse's fitted parameters have shown a value effect?

    Each analyzed mouse's fitted Q-learner is replayed on the mouse's task *replicates* times
    and goes through the identical switch regression. The share of replicates whose value
    weight lies above the mouse's own says how detectable the effect is at the mouse's noise
    level: near 1, a value learner like this would show more value-guided switching than the
    mouse does; near 0.5, the mouse looks like such a learner.

    Args:
        trial_table (pd.DataFrame): The trial table *source* was analyzed from.
        source (SourceSwitchRegression): The mice's switch regression; its settings are used for
            every replicate.
        params_by_mouse (Mapping[str, Sequence[float]]): ``{mouse: Q-learner parameters}``.
        replicates (int): Simulations per mouse. Default 20.
        rules (sim.TaskRules | Mapping | None): As :func:`simulate_fitted_q_cohort`.
        seed (int): Base seed. Default 0.
        n_jobs (int): Parallel workers over mice. Default 1.
        cache_dir (Path | None): Cache each mouse's replicates here. Default ``None``.

    Returns:
        pd.DataFrame: ``mouse, condition, value`` (true and believed), the mouse's ``observed``
        weight, ``n_replicates`` analyzed, the replicates' ``null_mean`` weight and
        ``fraction_above``: the fraction of replicates with a larger weight (ties count half).

    Raises:
        ValueError: If fewer than one replicate is asked for or a mouse has no fitted Q-learner.
    """
    if replicates < 1:
        raise ValueError(f"replicates must be at least 1, got {replicates}")
    mice = list(source.decays["mouse"])
    missing = [m for m in mice if m not in params_by_mouse]
    if missing:
        raise ValueError(f"no fitted Q-learner for mice {missing}")
    rules_of = switch_pipe.rules_by_mouse(trial_table, mice, rules)
    settings = (source.n_lags, source.trace_grid, source.l2, source.recency_cap,
                source.min_switches, source.firth)  # fmt: skip
    by_mouse = {m: rows.reset_index(drop=True) for m, rows in trial_table.groupby("mouse")}
    simulate, version = replicate_simulator(cache_dir), code_version()
    results = Parallel(n_jobs=n_jobs)(
        delayed(simulate)(by_mouse[m], params_by_mouse[m], rules_of[m], replicates, seed, settings,
                          version)
        for m in mice
    )  # fmt: skip
    for message in dict.fromkeys(w for r in results for w in r.warnings):
        warnings.warn(message, stacklevel=2)
    observed = source.values[source.values["value"].isin(switch_pipe.CALIBRATED_VALUES)]
    rows = []
    for mouse, result in zip(mice, results, strict=True):
        if result.n_failed:
            warnings.warn(
                f"mouse {mouse}: {result.n_failed} of {replicates} fitted-Q replicates had too "
                "few switches and are left out",
                stacklevel=2,
            )
        for row in observed[observed["mouse"] == mouse].itertuples():
            beta = _replicate_betas(result.values, row.condition, row.value)
            rows.append(
                dict(
                    mouse=mouse,
                    condition=row.condition,
                    value=row.value,
                    observed=row.beta,
                    n_replicates=len(beta),
                    null_mean=float(beta.mean()) if len(beta) else np.nan,
                    fraction_above=_fraction_above(beta, row.beta),
                )  # fmt: skip
            )
    return pd.DataFrame(rows, columns=POWER_COLUMNS)


def _replicate_betas(values: pd.DataFrame, condition: str, value: str) -> np.ndarray:
    """The replicates' value weights of one condition and value variant."""
    if values.empty:
        return np.array([])
    rows = values[(values["condition"] == condition) & (values["value"] == value)]
    return rows["beta"].to_numpy(dtype=float)


def _fraction_above(beta: np.ndarray, observed: float) -> float:
    """Fraction of *beta* above *observed*, ties counted half; NaN without replicates."""
    if not len(beta):
        return np.nan
    return float(((beta > observed).sum() + 0.5 * (beta == observed).sum()) / len(beta))
