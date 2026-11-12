"""The held-out strategy comparison on a model subset: the baseline of the rank-signature tests.

A selection of problems (:mod:`src.behavior_analysis.problem_selection`) changes the data of
some mice, so their fits must be redone. The full strategy comparison of
:mod:`src.pipelines.strategy_comparison` fits eleven models per fold; the baseline needs four
(WSLS (Magnitude), WSLS (Habit), Q-Learner, Kernel Q-Learner), so this module fits only the ones
asked for, with the same folds, seeds and scoring (``strategy_models.cross_validate``), so a
number is identical to the full comparison's.

Each mouse is cached on its trials, the models and the code. A mouse whose full comparison is
already in a *shared* cache (a notebook's strategy cache, same arguments) is read from it instead
of refitted: the cache key contains the mouse's trials, so a mouse whose data the selection
changed is never served a stale result.
"""

import hashlib
import inspect
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import pandas as pd
from joblib import Memory, Parallel, delayed

from src.behavior_analysis import strategy_models as sm
from src.pipelines import strategy_comparison as strategy

# Part of the cache key next to the source of the modules; bump for a change outside them.
CACHE_VERSION = 1
DEFAULT_COMPARISONS = (
    (sm.KERNEL_Q, sm.WSLS_MAGNITUDE),
    (sm.KERNEL_Q, sm.Q_LEARNER),
    (sm.WSLS_HABIT, sm.WSLS_MAGNITUDE),
    (sm.WSLS_HABIT, sm.KERNEL_Q),
)


class Baseline(NamedTuple):
    """The held-out comparison of a model subset.

    Attributes:
        nats (pd.DataFrame): ``mouse, model, n_choices, log_likelihood, nats_per_choice``: held-out
            nats per trial (``strategy_models.heldout_nats_per_trial``, segment starts included).
        differences (pd.DataFrame): ``strategy_models.difference_table`` of *comparisons*: first
            model minus second, negative means the first predicts better.
        fits (dict): ``{mouse: {model: params}}`` fitted to all of the mouse's trials.
    """

    nats: pd.DataFrame
    differences: pd.DataFrame
    fits: dict


def code_version() -> str:
    """A cache version that changes with ``CACHE_VERSION`` and the source of ``strategy_models``."""
    digest = hashlib.sha256(
        "\n".join([str(CACHE_VERSION), strategy.code_version(), inspect.getsource(sm)]).encode()
    )
    return digest.hexdigest()[:16]


def compare_models(
    trials: pd.DataFrame, n_starts: int, seed: int, models: tuple, version: str
) -> sm.MouseComparison:
    """:func:`strategy_models.cross_validate` of one mouse; *version* only keys the cache."""
    return sm.cross_validate(trials, n_starts, seed, models)


def _from_shared(shared_cache_dir: Path | None, rows, n_starts, seed, models):
    """The mouse's full comparison from the shared cache restricted to *models*, or ``None``."""
    if shared_cache_dir is None:
        return None
    cached = Memory(shared_cache_dir, verbose=0).cache(strategy.compare_mouse)
    args = (rows, n_starts, seed, strategy.code_version())
    if not cached.check_call_in_cache(*args):
        return None
    full = cached(*args)
    return sm.MouseComparison(
        scores=full.scores[full.scores["model"].isin(models)].reset_index(drop=True),
        parameters=full.parameters[full.parameters["model"].isin(models)],
        fits={m: full.fits[m] for m in models},
    )


def _one_mouse(rows, models, n_starts, seed, cache_dir, shared_cache_dir):
    shared = _from_shared(shared_cache_dir, rows, n_starts, seed, models)
    if shared is not None:
        return shared
    run = compare_models
    if cache_dir is not None:
        run = Memory(cache_dir, verbose=0).cache(compare_models)
    return run(rows, n_starts, seed, tuple(models), code_version())


def heldout_baseline(
    trial_table: pd.DataFrame,
    models: Sequence[str],
    *,
    comparisons: Sequence[tuple[str, str]] = (),
    n_starts: int = sm.Q_STARTS,
    seed: int = 0,
    n_jobs: int = 1,
    cache_dir: Path | None = None,
    shared_cache_dir: Path | None = None,
) -> Baseline:
    """Leave-one-problem-out nats of *models* for every mouse of *trial_table*.

    Args:
        trial_table (pd.DataFrame): A drifting trial table.
        models (Sequence[str]): Models of ``strategy_models.SPECS`` to fit and score.
        comparisons (Sequence[tuple[str, str]]): ``(first, second)`` pairs for the paired
            differences, both in *models*. Default none.
        n_starts (int): Random starts of each Q-learner fit. Default ``strategy_models.Q_STARTS``.
        seed (int): Seed of the starts. Default 0.
        n_jobs (int): Parallel workers over mice. Default 1.
        cache_dir (Path | None): Cache each mouse here. Default ``None``: no cache.
        shared_cache_dir (Path | None): A strategy-comparison cache to read finished mice from.
            Default ``None``.

    Returns:
        Baseline: Held-out nats per trial, paired differences and the full-data fits.

    Raises:
        ValueError: If there are no trials, a model is unknown, or a comparison names a model
            outside *models*.
        RuntimeError: If a fit does not converge.
    """
    if trial_table.empty:
        raise ValueError("no trials to compare")
    unknown = [m for m in models if m not in sm.SPECS]
    outside = [p for p in comparisons if not set(p) <= set(models)]
    if unknown or outside:
        raise ValueError(f"unknown models {unknown}; comparisons outside the models: {outside}")
    by_mouse = dict(sorted(trial_table.groupby("mouse"), key=lambda item: item[0]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the design's tower-set warnings, as in every analysis
        results = Parallel(n_jobs=n_jobs)(
            delayed(_one_mouse)(
                rows.reset_index(drop=True), models, n_starts, seed, cache_dir, shared_cache_dir
            )
            for rows in by_mouse.values()
        )
    scores = pd.concat([r.scores for r in results], ignore_index=True)
    nats = sm.heldout_nats_per_trial(scores)
    return Baseline(
        nats=nats,
        differences=sm.difference_table(nats, comparisons) if comparisons else pd.DataFrame(),
        fits={m: r.fits for m, r in zip(by_mouse, results, strict=True)},
    )
