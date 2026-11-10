"""The choice-logit extensions for one source of choices: the mice, or a simulated cohort.

Steps, each a plain function:

1. **Analyze** (:func:`analyse_extensions`): per mouse, every model variant of
   :data:`~src.behavior_analysis.choice_logit_extensions.VARIANTS` validated leave-one-problem-out
   on one design that holds all optional columns, plus the weights of the believed-value model
   fitted to all of the mouse's trials. Mice are independent, so the work is parallel over mice
   and each mouse's result can be cached.
2. **Tabulate** (:func:`extension_tables`): the believed-value, binary-against-pulse,
   unique/shared-variance and calibration tables of
   :mod:`src.behavior_analysis.choice_logit_extensions`.

A source is analyzed with the logit settings of the main analysis of the same trials (lags, tail
decay, penalty), and the same sources go through the same functions: the mice, a simulated
Q-learner as the positive control and the win-stay/lose-shift agents as negative controls.
Drawing is in :mod:`src.behavior_visualization.plot_regression_extensions`.
"""

import warnings
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from joblib import Memory, Parallel, delayed

from src.behavior_analysis import choice_logit as cl
from src.behavior_analysis import choice_logit_cv as cv
from src.behavior_analysis import choice_logit_extensions as ext
from src.pipelines.choice_logit_values import MIN_TRIALS, skip_reason

# Bump when the design or the fits change in a way the cache key (the mouse's trials and the
# settings) does not see.
CACHE_VERSION = 1
# The variant whose full-data fit gives the believed-value weights.
BELIEVED_MODEL = "full + believed"
# The variants whose held-out probabilities are kept for the calibration figure: the full model
# and the model with choice history only.
CALIBRATION_MODELS = ("full", "choice only")


class MouseExtensions(NamedTuple):
    """The extension analyses of one mouse.

    Attributes:
        scores (pd.DataFrame): Held-out scores of every variant (``VariantCV.scores``).
        weights (pd.DataFrame): ``choice_logit.weights_table`` of the believed-value model fitted
            to all of the mouse's trials.
        chosen (np.ndarray): ``(n_trials,)`` chosen arm index.
        probabilities (dict): ``{model: (n_trials, 3)}`` held-out probabilities of
            ``CALIBRATION_MODELS``.
    """

    scores: pd.DataFrame
    weights: pd.DataFrame
    chosen: np.ndarray
    probabilities: dict


class SourceExtensions(NamedTuple):
    """The extension analyses of every analysable mouse of one source.

    Attributes:
        comparison (pd.DataFrame): ``choice_logit_cv.model_comparison`` of all variants, all mice.
        weights (pd.DataFrame): Believed-value model weights, all mice.
        mice (np.ndarray): ``(n_trials,)`` mouse of each trial.
        chosen (np.ndarray): ``(n_trials,)`` chosen arm index.
        probabilities (dict): ``{model: (n_trials, 3)}`` held-out probabilities, rows as *mice*.
        skipped (dict[str, str]): Mice not analyzed and why.
        n_lags (int): Lags of the design.
        tail_decay (float | None): Decay of the reward tail, ``None`` without one.
        l2 (float): Penalty of the fits.
    """

    comparison: pd.DataFrame
    weights: pd.DataFrame
    mice: np.ndarray
    chosen: np.ndarray
    probabilities: dict
    skipped: dict
    n_lags: int
    tail_decay: float | None
    l2: float


def analyse_mouse_extensions(
    trials: pd.DataFrame, n_lags: int, tail_decay: float | None, l2: float, version: int
) -> MouseExtensions:
    """Validate every variant on one mouse and fit the believed-value model to all its trials.

    Args:
        trials (pd.DataFrame): One mouse's drifting trial table.
        n_lags (int): Reward and choice lags.
        tail_decay (float | None): Decay of the reward tail; ``None`` for none.
        l2 (float): Penalty of the fits.
        version (int): ``CACHE_VERSION``; only part of the cache key.

    Returns:
        MouseExtensions: Scores, believed-model weights, choices and held-out probabilities.
    """
    design = cl.design_matrix(trials, n_lags, tail_decay, extras=cl.EXTRAS)
    validation = ext.cross_validate_variants(design, ext.VARIANTS, l2)
    believed = cl.select_features(design, ext.VARIANTS[BELIEVED_MODEL])
    fit = cl.fit_logit(believed.X, believed.chosen, believed.feature_names, l2)
    return MouseExtensions(
        validation.scores,
        cl.weights_table(design.meta["mouse"].iloc[0], fit.feature_names, fit.beta),
        design.chosen,
        {name: validation.probabilities[name] for name in CALIBRATION_MODELS},
    )


def analyse_extensions(
    trial_table: pd.DataFrame,
    n_lags: int,
    tail_decay: float | None,
    l2: float,
    min_trials: int = MIN_TRIALS,
    n_jobs: int = 1,
    cache_dir: Path | None = None,
) -> SourceExtensions:
    """The extension analyses of every mouse of *trial_table*.

    Args:
        trial_table (pd.DataFrame): A drifting trial table (the mice, or a simulated cohort).
        n_lags (int): Reward and choice lags; those of the main analysis of the same trials.
        tail_decay (float | None): Reward-tail decay of the main analysis.
        l2 (float): Penalty of the main analysis.
        min_trials (int): Mice with fewer trials, or fewer than two problems, are skipped with a
            warning, as in the main analysis. Default ``MIN_TRIALS``.
        n_jobs (int): Parallel workers over mice (joblib). Default 1.
        cache_dir (Path | None): Cache each mouse's result here, keyed on the mouse's trials, the
            settings and ``CACHE_VERSION``. Default ``None``: no cache.

    Returns:
        SourceExtensions: Results for the analysable mice, sorted by id.

    Raises:
        ValueError: If no mouse can be analyzed.
        RuntimeError: If a fit does not converge.
    """
    by_mouse = dict(sorted(trial_table.groupby("mouse"), key=lambda item: item[0]))
    skipped = {m: r for m, rows in by_mouse.items() if (r := skip_reason(rows, min_trials))}
    for mouse, reason in skipped.items():
        warnings.warn(f"mouse {mouse} skipped: {reason}", stacklevel=2)
    kept = [m for m in by_mouse if m not in skipped]
    if not kept:
        raise ValueError(
            "no mouse can be analysed: " + "; ".join(f"{m}: {r}" for m, r in skipped.items())
        )
    analyse = analyse_mouse_extensions
    if cache_dir is not None:
        analyse = Memory(cache_dir, verbose=0).cache(analyse_mouse_extensions)
    results = Parallel(n_jobs=n_jobs)(
        delayed(analyse)(by_mouse[m].reset_index(drop=True), n_lags, tail_decay, l2, CACHE_VERSION)
        for m in kept
    )
    return SourceExtensions(
        comparison=cv.model_comparison(
            pd.concat([r.scores for r in results], ignore_index=True), models=list(ext.VARIANTS)
        ),
        weights=pd.concat([r.weights for r in results], ignore_index=True),
        mice=np.concatenate(
            [np.full(len(r.chosen), m, dtype=object) for m, r in zip(kept, results, strict=True)]
        ),
        chosen=np.concatenate([r.chosen for r in results]),
        probabilities={
            name: np.concatenate([r.probabilities[name] for r in results])
            for name in CALIBRATION_MODELS
        },
        skipped=skipped,
        n_lags=n_lags,
        tail_decay=tail_decay,
        l2=l2,
    )


class ExtensionTables(NamedTuple):
    """The tables of the four analyses for one source.

    Attributes:
        believed (pd.DataFrame): :func:`~src.behavior_analysis.choice_logit_extensions.
            believed_value_table`.
        binary_pulse (pd.DataFrame): ``binary_pulse_table``.
        partition (pd.DataFrame): ``variance_partition``.
        calibration (dict): ``{model: calibration_table}`` for ``CALIBRATION_MODELS``.
        calibration_summary (dict): ``{model: calibration_summary}``.
    """

    believed: pd.DataFrame
    binary_pulse: pd.DataFrame
    partition: pd.DataFrame
    calibration: dict
    calibration_summary: dict


def extension_tables(
    source: SourceExtensions, n_bins: int = ext.DEFAULT_CALIBRATION_BINS
) -> ExtensionTables:
    """Tabulate the four analyses of *source*.

    Args:
        source (SourceExtensions): From :func:`analyse_extensions`.
        n_bins (int): Quantile bins of the calibration. Default
            ``choice_logit_extensions.DEFAULT_CALIBRATION_BINS``.

    Returns:
        ExtensionTables: One table (or one per calibrated model) per analysis.
    """
    calibration = {
        name: ext.calibration_table(source.probabilities[name], source.chosen, source.mice, n_bins)
        for name in CALIBRATION_MODELS
    }
    return ExtensionTables(
        believed=ext.believed_value_table(source.comparison, source.weights),
        binary_pulse=ext.binary_pulse_table(source.comparison),
        partition=ext.variance_partition(source.comparison),
        calibration=calibration,
        calibration_summary={name: ext.calibration_summary(t) for name, t in calibration.items()},
    )
