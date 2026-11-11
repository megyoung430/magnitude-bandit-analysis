"""A Q-learner fitted to each mouse, played on that mouse's own task: a positive control.

Two value learners can be the control: the Q-learner of :mod:`src.agents.value_agent` (``"Q-Learner
(Fitted)"``) and the richer Q-learner of :mod:`src.agents.kernel_q_agent`, with asymmetric
learning rates, forgetting and a multi-trial choice kernel (``"Q-Learner (Kernel + Forgetting,
Fitted)"``). Every function takes ``model`` (``strategy_models.Q_LEARNER``, the default, or
``strategy_models.KERNEL_Q``); ``CONTROLS`` holds the kind, label and random stream of each.

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
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from joblib import Memory, Parallel, delayed

from src.agents import drifting_simulation as sim
from src.agents import kernel_q_agent as kq
from src.agents import null_agents as na
from src.agents import value_agent as va
from src.behavior_analysis import choice_logit as cl
from src.behavior_analysis import strategy_models as sm
from src.behavior_analysis import switch_regression as sr
from src.behavior_analysis.drifting_trial_table import drifting_trial_table
from src.pipelines import null_agents as null_pipe
from src.pipelines import switch_regression as switch_pipe
from src.pipelines.choice_logit_values import SyntheticCohort

KIND = "q_fitted"
LABEL = "Q-Learner (Fitted)"
# Part of the cache key next to the source of the modules; bump for a change outside them.
CACHE_VERSION = 1


class Control(NamedTuple):
    """One fitted value learner used as a positive control.

    Attributes:
        kind (str): ``SyntheticCohort.kind`` of its simulated cohort.
        label (str): Name of the control in figures and tables.
        stream (int): Index of its random stream: after the null agents' (``null_agents.KINDS``)
            and distinct from the other control's, so no two agents draw the same numbers.
        parameter_names (tuple[str, ...]): Names of its parameters.
        at_bounds (Callable): ``params -> names of the parameters at a fitting bound``.
    """

    kind: str
    label: str
    stream: int
    parameter_names: tuple
    at_bounds: Callable


CONTROLS = {
    sm.Q_LEARNER: Control(KIND, LABEL, len(na.KINDS), va.PARAMETER_NAMES, va.parameters_at_bounds),
    sm.KERNEL_Q: Control(
        "kernel_q_fitted",
        "Q-Learner (Kernel + Forgetting, Fitted)",
        len(na.KINDS) + 1,
        kq.PARAMETER_NAMES,
        kq.parameters_at_bounds,
    ),
}


def _control(model: str) -> Control:
    if model not in CONTROLS:
        raise ValueError(f"no fitted control for {model!r}; expected one of {list(CONTROLS)}")
    return CONTROLS[model]


def code_version() -> str:
    """A cache version that changes with ``CACHE_VERSION`` and the source of the modules used."""
    modules = (cl, na, sr, va, kq, sm, sim)
    digest = hashlib.sha256(
        "\n".join([str(CACHE_VERSION), *(inspect.getsource(m) for m in modules)]).encode()
    )
    return digest.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Fits
# ---------------------------------------------------------------------------


def fits_of(source, model: str = sm.Q_LEARNER) -> dict[str, tuple[float, ...]]:
    """``{mouse: Q-learner parameters}`` of a strategy comparison's full-data fits.

    Args:
        source (SourceComparison): :func:`src.pipelines.strategy_comparison.
            analyze_comparison_source`.
        model (str): ``strategy_models.Q_LEARNER`` (default) or ``strategy_models.KERNEL_Q``.

    Returns:
        dict[str, tuple[float, ...]]: The parameters in the model's ``PARAMETER_NAMES`` order,
        per mouse.
    """
    _control(model)
    return {mouse: fits[model] for mouse, fits in source.fits.items()}


def fit_q_learners(
    trial_table: pd.DataFrame,
    *,
    n_starts: int = sm.Q_STARTS,
    seed: int = 0,
    model: str = sm.Q_LEARNER,
) -> dict[str, tuple[float, ...]]:
    """Fit the Q-learner to every mouse on all of its problems, from several starts.

    Args:
        trial_table (pd.DataFrame): A drifting trial table.
        n_starts (int): Random starts of each fit. Default ``strategy_models.Q_STARTS``.
        seed (int): Seed of the starts. Default 0.
        model (str): ``strategy_models.Q_LEARNER`` (default) or ``strategy_models.KERNEL_Q``.

    Returns:
        dict[str, tuple[float, ...]]: The parameters in the model's ``PARAMETER_NAMES`` order,
        per mouse.

    Raises:
        ValueError: If the table has no trials.
        RuntimeError: If a fit does not converge.
    """
    _control(model)
    if trial_table.empty:
        raise ValueError("no trials to fit")
    fit = sm.SPECS[model].fit
    return {
        mouse: fit(rows.reset_index(drop=True), n_starts, seed)
        for mouse, rows in trial_table.groupby("mouse", sort=True)
    }


def parameters_table(
    params_by_mouse: Mapping[str, Sequence[float]], model: str = sm.Q_LEARNER
) -> pd.DataFrame:
    """One row per mouse: the parameters and whether any lies at a bound of the fit.

    Args:
        params_by_mouse (Mapping[str, Sequence[float]]): As :func:`fit_q_learners`.
        model (str): ``strategy_models.Q_LEARNER`` (default) or ``strategy_models.KERNEL_Q``.

    Returns:
        pd.DataFrame: ``mouse``, the model's parameter names, and ``at_bound`` (a parameter at a
        bound is poorly identified; ``bounded`` names them).
    """
    control = _control(model)
    rows = []
    for mouse, params in sorted(params_by_mouse.items()):
        bounded = control.at_bounds(params)
        rows.append(
            {"mouse": mouse, **dict(zip(control.parameter_names, params, strict=True)),
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


def _simulate_mouse(
    trials: pd.DataFrame,
    params,
    rules: sim.TaskRules,
    seed: int,
    length_factor: float = 1.0,
    model: str = sm.Q_LEARNER,
) -> dict:
    """``{problem: {mouse: sessions}}`` of one fitted Q-learner playing one mouse's problems."""
    mouse = trials["mouse"].iloc[0]
    rng = np.random.default_rng([seed, zlib.crc32(str(mouse).encode()), CONTROLS[model].stream])
    return sim.simulate_policy_mouse(
        lambda layout: sm.SPECS[model].policy(params, layout),
        null_pipe.lengthen_layouts(sim.mouse_layouts(trials), length_factor),
        rules,
        rng,
        mouse,
    )


def simulate_fitted_q_cohort(
    trial_table: pd.DataFrame,
    params_by_mouse: Mapping[str, Sequence[float]],
    seed: int,
    rules: sim.TaskRules | Mapping | None = None,
    length_factor: float = 1.0,
    model: str = sm.Q_LEARNER,
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
        length_factor (float): Schedule length relative to the mouse's own, as
            :func:`src.pipelines.null_agents.lengthen_layouts`. Default 1.0; above 1 for comparison
            columns only, never for the power check or a calibration.
        model (str): ``strategy_models.Q_LEARNER`` (default) or ``strategy_models.KERNEL_Q``.

    Returns:
        SyntheticCohort: The simulated trials, the parameters, the sessions and the control's
        ``kind`` (``"q_fitted"`` or ``"kernel_q_fitted"``).

    Raises:
        ValueError: For no mice, a mouse that is not in *trial_table*, a cohort without rules, or
            a *model* without a fitted control.
    """
    control = _control(model)
    _check(trial_table, params_by_mouse)
    mice = sorted(params_by_mouse)
    rules_of = switch_pipe.rules_by_mouse(trial_table, mice, rules)
    problems: dict = {}
    for mouse in mice:
        mine = trial_table[trial_table["mouse"] == mouse]
        played = _simulate_mouse(
            mine, params_by_mouse[mouse], rules_of[mouse], seed, length_factor, model
        )
        for problem, animals in played.items():
            problems.setdefault(problem, {}).update(animals)
    parameters = pd.DataFrame(
        [{"mouse": m, **dict(zip(control.parameter_names, params_by_mouse[m], strict=True))}
         for m in mice]
    )  # fmt: skip
    return SyntheticCohort(drifting_trial_table(problems), parameters, problems, control.kind)


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
    model: str = sm.Q_LEARNER,
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
        model (str): ``strategy_models.Q_LEARNER`` (default) or ``strategy_models.KERNEL_Q``.

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
                trials, params, rules, switch_pipe.replicate_seed(seed, replicate), model=model
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
    model: str = sm.Q_LEARNER,
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
        model (str): ``strategy_models.Q_LEARNER`` (default) or ``strategy_models.KERNEL_Q``.

    Returns:
        pd.DataFrame: ``mouse, condition, value`` (true and believed), the mouse's ``observed``
        weight, ``n_replicates`` analyzed, the replicates' ``null_mean`` weight and
        ``fraction_above``: the fraction of replicates with a larger weight (ties count half).

    Raises:
        ValueError: If fewer than one replicate is asked for or no mouse has a fitted Q-learner.

    Warns:
        UserWarning: For each mouse without a fitted Q-learner (it was not compared), which is left
            out.
    """
    _control(model)
    if replicates < 1:
        raise ValueError(f"replicates must be at least 1, got {replicates}")
    mice = [m for m in source.decays["mouse"] if m in params_by_mouse]
    for mouse in source.decays["mouse"]:
        if mouse not in params_by_mouse:
            warnings.warn(f"mouse {mouse} left out: no fitted Q-learner", stacklevel=2)
    if not mice:
        raise ValueError("no fitted Q-learner for any mouse of the switch regression")
    rules_of = switch_pipe.rules_by_mouse(trial_table, mice, rules)
    settings = (source.n_lags, source.trace_grid, source.l2, source.recency_cap,
                source.min_switches, source.firth)  # fmt: skip
    by_mouse = {m: rows.reset_index(drop=True) for m, rows in trial_table.groupby("mouse")}
    simulate, version = replicate_simulator(cache_dir), code_version()
    results = Parallel(n_jobs=n_jobs)(
        delayed(simulate)(by_mouse[m], params_by_mouse[m], rules_of[m], replicates, seed, settings,
                          version, model)
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
