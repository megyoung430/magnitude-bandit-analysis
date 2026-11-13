"""The choice kernel to a long lag, with and without a choice tail, and a known-kernel recovery.

In the choice logit with five lags the mice's choice-history weights are flat (about 0.2-0.3 at
every lag) while every agent we fitted has a sharp kernel. This module asks whether that is real:

1. **Long lags** (:func:`analyze_long_kernel`): per mouse, the logit with many choice lags
   (default 15) and, as a second model, one more regressor for the choice history older than
   the lags (``choice_logit.CHOICE_TAIL``). Both are scored held out (leave one problem out) so
   the tail's worth is a number. The same function is run on every agent's simulated table, so
   the agents get the identical design.
2. **Recovery** (:func:`recover_choice_kernel`): does the L2 penalty, which shares weight between
   collinear lags, flatten a sharp kernel? Agents whose choices follow *only* a known choice
   kernel (rate ``alpha_k``, weight ``beta_k``) play the mice's own schedules and go through the
   same analysis; the fitted lag weights are set against the weights the kernel implies
   (:func:`src.behavior_analysis.choice_logit.choice_kernel_timescale.true_kernel`).

Mice are independent: the work is parallel over mice and each mouse's result is cached, keyed on
its trials, the settings and the source of the modules (:func:`code_version`).
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
from src.agents import kernel_q_agent as kq
from src.behavior_analysis.choice_logit import choice_kernel_timescale as ts
from src.behavior_analysis.choice_logit import choice_logit as cl
from src.behavior_analysis.choice_logit import choice_logit_cv as cv
from src.behavior_analysis.choice_logit import choice_logit_extensions as ext
from src.behavior_analysis.data.drifting_trial_table import drifting_trial_table
from src.behavior_analysis.data.stats import nan_standard_error
from src.pipelines.choice_logit import choice_logit_values as logit_pipe
from src.pipelines.switching import switch_regression as switch_pipe

# Part of the cache key next to the source of the modules; bump for a change outside them.
CACHE_VERSION = 1
LONG_LAGS = 15
# Decay of the choice tail: a half-life of about 7 trials puts its weight on the 6-30 trials
# before a choice, beyond the lags. The reward tail's default (0.9) is used so that both tails
# have the same reach; ``analyze_long_kernel`` takes it as a parameter.
CHOICE_TAIL_DECAY = 0.9
MIN_TRIALS = logit_pipe.MIN_TRIALS


def code_version() -> str:
    """A cache version that changes with ``CACHE_VERSION`` and the source of the modules used."""
    modules = (cl, cv, ext, ts)
    digest = hashlib.sha256(
        "\n".join([str(CACHE_VERSION), *(inspect.getsource(m) for m in modules)]).encode()
    )
    return digest.hexdigest()[:16]


class LongKernel(NamedTuple):
    """The long-lag choice logit of every analyzable mouse of one source.

    Attributes:
        weights (pd.DataFrame): ``choice_logit.weights_table`` of the model with the lags and no
            tail, fitted to all of each mouse's trials.
        weights_tail (pd.DataFrame): The same for the model with the choice tail.
        comparison (pd.DataFrame): ``mouse, model, n_trials, nats_per_choice`` of the two models
            (``"full"``: no tail, ``"full + choice tail"``), held out by problem.
        skipped (dict[str, str]): Mice not analyzed and why.
        n_lags (int): Reward and choice lags.
        tail_decay (float): Decay of the reward tail (both models have it).
        choice_tail_decay (float): Decay of the choice tail.
        l2 (float): Penalty.
    """

    weights: pd.DataFrame
    weights_tail: pd.DataFrame
    comparison: pd.DataFrame
    skipped: dict
    n_lags: int
    tail_decay: float
    choice_tail_decay: float
    l2: float


def analyze_mouse_long(
    trials: pd.DataFrame,
    n_lags: int,
    tail_decay: float,
    choice_tail_decay: float,
    l2: float,
    version: str,
):
    """``(weights, weights with tail, held-out scores)`` of one mouse (:func:`analyze_long_kernel`).

    *version* (``code_version()``) is only part of the cache key.
    """
    mouse = trials["mouse"].iloc[0]
    design = cl.design_matrix(
        trials, n_lags, tail_decay, extras=(cl.CHOICE_TAIL,), choice_tail_decay=choice_tail_decay
    )
    weights = {}
    for name, groups in ext.TAIL_VARIANTS.items():
        model = cl.select_features(design, groups)
        fit = cl.fit_logit(model.X, model.chosen, model.feature_names, l2)
        weights[name] = cl.weights_table(mouse, fit.feature_names, fit.beta)
    scores = ext.cross_validate_variants(design, ext.TAIL_VARIANTS, l2).scores
    return weights["full"], weights["full + choice tail"], scores


def mouse_analyzer(cache_dir: Path | None):
    """:func:`analyze_mouse_long`, cached under *cache_dir* if one is given."""
    if cache_dir is None:
        return analyze_mouse_long
    return Memory(cache_dir, verbose=0).cache(analyze_mouse_long)


def analyze_long_kernel(
    trial_table: pd.DataFrame,
    *,
    n_lags: int = LONG_LAGS,
    tail_decay: float = cl.DEFAULT_TAIL_DECAY,
    choice_tail_decay: float = CHOICE_TAIL_DECAY,
    l2: float = cl.DEFAULT_L2,
    min_trials: int = MIN_TRIALS,
    n_jobs: int = 1,
    cache_dir: Path | None = None,
) -> LongKernel:
    """Fit the long-lag choice logit, with and without the choice tail, to every mouse.

    Args:
        trial_table (pd.DataFrame): A drifting trial table (the mice, or a simulated cohort).
        n_lags (int): Reward and choice lags. Default ``LONG_LAGS``.
        tail_decay (float): Decay of the reward tail. Default ``choice_logit.DEFAULT_TAIL_DECAY``.
        choice_tail_decay (float): Decay of the choice tail. Default ``CHOICE_TAIL_DECAY``.
        l2 (float): Penalty. Default ``choice_logit.DEFAULT_L2``.
        min_trials (int): Mice with fewer trials, or fewer than two problems, are skipped.
            Default ``MIN_TRIALS``.
        n_jobs (int): Parallel workers over mice (joblib). Default 1.
        cache_dir (Path | None): Cache each mouse's result here. Default ``None``.

    Returns:
        LongKernel: Results for the analyzable mice; the others are in ``skipped``.

    Raises:
        ValueError: If no mouse can be analyzed.

    Warns:
        UserWarning: For each skipped mouse.
    """
    by_mouse = dict(sorted(trial_table.groupby("mouse"), key=lambda item: item[0]))
    skipped = {
        m: r for m, rows in by_mouse.items() if (r := logit_pipe.skip_reason(rows, min_trials))
    }
    for mouse, reason in skipped.items():
        warnings.warn(f"mouse {mouse} skipped: {reason}", stacklevel=2)
    kept = [m for m in by_mouse if m not in skipped]
    if not kept:
        raise ValueError(
            "no mouse can be analyzed: " + "; ".join(f"{m}: {r}" for m, r in skipped.items())
        )
    analyze, version = mouse_analyzer(cache_dir), code_version()
    results = Parallel(n_jobs=n_jobs)(
        delayed(analyze)(
            by_mouse[m].reset_index(drop=True), n_lags, tail_decay, choice_tail_decay, l2, version
        )
        for m in kept
    )
    scores = pd.concat([r[2] for r in results], ignore_index=True)
    return LongKernel(
        weights=pd.concat([r[0] for r in results], ignore_index=True),
        weights_tail=pd.concat([r[1] for r in results], ignore_index=True),
        comparison=cv.model_comparison(scores, models=list(ext.TAIL_VARIANTS)),
        skipped=skipped,
        n_lags=n_lags,
        tail_decay=tail_decay,
        choice_tail_decay=choice_tail_decay,
        l2=l2,
    )


# ---------------------------------------------------------------------------
# Recovery of a known kernel
# ---------------------------------------------------------------------------


def _beta_of(beta_k: float | Mapping[str, float], mouse: str) -> float:
    return float(beta_k[mouse] if isinstance(beta_k, Mapping) else beta_k)


def simulate_kernel_only(
    trial_table: pd.DataFrame,
    alpha_k: float,
    beta_k: float | Mapping[str, float],
    rules: sim.TaskRules | Mapping | None,
    seed: int,
) -> pd.DataFrame:
    """Agents that choose by a choice kernel alone, each playing one mouse's own problems.

    The agent is the kernel Q-learner with no value (inverse temperature near 0), no bias and a
    kernel of rate *alpha_k* and weight *beta_k*; it keeps its mouse's id and schedule.

    Args:
        trial_table (pd.DataFrame): The mice's drifting trial table (the schedules to play).
        alpha_k (float): Kernel rate, in (0, 1].
        beta_k (float | Mapping[str, float]): Kernel weight, one for all or per mouse.
        rules (sim.TaskRules | Mapping | None): The simulated task, as
            :func:`src.pipelines.switching.switch_regression.rules_by_mouse`. ``None``: measured per
            cohort.
        seed (int): Base seed; every mouse has its own stream.

    Returns:
        pd.DataFrame: The simulated drifting trial table.
    """
    mice = sorted(trial_table["mouse"].unique())
    rules_of = switch_pipe.rules_by_mouse(trial_table, mice, rules)
    problems: dict = {}
    for mouse in mice:
        rows = trial_table[trial_table["mouse"] == mouse]
        params = (
            0.5,
            0.5,
            1.0,
            alpha_k,
            kq.BOUNDS_BY_NAME["beta"][0],
            _beta_of(beta_k, mouse),
            0.0,
        )
        rng = np.random.default_rng([seed, zlib.crc32(str(mouse).encode()), 7])
        played = sim.simulate_policy_mouse(
            lambda layout, params=params: sim.KernelQPolicy(params, "magnitude", layout),
            sim.mouse_layouts(rows),
            rules_of[mouse],
            rng,
            mouse,
        )
        for problem, animals in played.items():
            problems.setdefault(problem, {}).update(animals)
    return drifting_trial_table(problems)


def recover_choice_kernel(
    trial_table: pd.DataFrame,
    *,
    alpha_ks: Sequence[float],
    beta_k: float | Mapping[str, float],
    rules: sim.TaskRules | Mapping | None = None,
    n_lags: int = LONG_LAGS,
    seed: int = 0,
    min_trials: int = MIN_TRIALS,
    n_jobs: int = 1,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Fitted against generating choice-lag weights of kernel-only agents on the mice's schedules.

    Args:
        trial_table (pd.DataFrame): The mice's drifting trial table.
        alpha_ks (Sequence[float]): Generating kernel rates, one simulated cohort each.
        beta_k (float | Mapping[str, float]): Generating kernel weight, one for all or per mouse
            (e.g. the fitted ``beta_k``, to give mouse-like run lengths).
        rules (sim.TaskRules | Mapping | None): As :func:`simulate_kernel_only`.
        n_lags (int): Choice lags of the fit. Default ``LONG_LAGS``.
        seed (int): Base seed. Default 0.
        min_trials (int): As :func:`analyze_long_kernel`.
        n_jobs (int): Parallel workers over mice. Default 1.
        cache_dir (Path | None): Cache each simulated mouse's analysis. Default ``None``.

    Returns:
        pd.DataFrame: ``alpha_k, lag, true, fitted_mean, fitted_se, n_mice``: the weight of the
        kernel at each lag (for the mice's mean weight ``beta_k`` when it varies), and the fitted
        mean and standard error across the simulated mice.
    """
    rows = []
    for alpha_k in alpha_ks:
        simulated = simulate_kernel_only(trial_table, alpha_k, beta_k, rules, seed)
        analysis = analyze_long_kernel(
            simulated, n_lags=n_lags, min_trials=min_trials, n_jobs=n_jobs, cache_dir=cache_dir
        )
        choice = analysis.weights[analysis.weights["group"] == cl.CHOICE]
        by_lag = choice.pivot(index="mouse", columns="lag", values="weight")
        kept = list(by_lag.index)
        mean_beta = float(np.mean([_beta_of(beta_k, m) for m in kept]))
        true = ts.true_kernel(alpha_k, mean_beta, by_lag.columns.to_numpy())
        se = nan_standard_error(by_lag.to_numpy())
        for k, lag in enumerate(by_lag.columns):
            rows.append(
                dict(alpha_k=alpha_k, lag=int(lag), true=float(true[k]),
                     fitted_mean=float(by_lag[lag].mean()), fitted_se=float(se[k]),
                     n_mice=len(kept))
            )  # fmt: skip
    return pd.DataFrame(rows)
