"""The choice-logit extensions for one source of choices: the mice, or a simulated cohort.

Steps, each a plain function:

1. **Analyze** (:func:`analyse_extensions`): per mouse, every model variant of
   :data:`~src.behavior_analysis.choice_logit_extensions.VARIANTS` validated leave-one-problem-out,
   plus the weights of the believed-value model fitted to all of the mouse's trials. The binary
   reward coding has its own tail decay, chosen from a grid by the binary model's pooled held-out
   likelihood (:func:`select_binary_decay`), as the pulse model's is. Mice are independent, so the
   work is parallel over mice and each mouse's result can be cached; warnings raised in a worker
   travel with its result and are re-emitted in the parent, also for a cached result.
2. **Tabulate** (:func:`extension_tables`): the believed-value, binary-against-pulse,
   unique/shared-variance and calibration tables of
   :mod:`src.behavior_analysis.choice_logit_extensions`.

A source is analyzed with the logit settings of the main analysis of the same trials (lags, tail
decay, penalty), and the same sources go through the same functions: the mice, a simulated
Q-learner as the positive control and the win-stay/lose-shift agents as negative controls.
Drawing is in :mod:`src.behavior_visualization.plot_regression_extensions`.
"""

import hashlib
import inspect
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from joblib import Memory, Parallel, delayed

from src.agents import null_agents as na
from src.behavior_analysis import choice_logit as cl
from src.behavior_analysis import choice_logit_cv as cv
from src.behavior_analysis import choice_logit_extensions as ext
from src.pipelines.choice_logit_values import MIN_TRIALS, TAIL_GRID, skip_reason
from src.pipelines.null_agents import AGENT_LABELS, NullComparison

# Part of the cache key next to the source of the logit modules; bump for a change outside them.
CACHE_VERSION = 2
# The variant whose full-data fit gives the believed-value weights.
BELIEVED_MODEL = "full + believed"
# The variants whose held-out probabilities are kept for the calibration figure: the full model
# and the model with choice history only.
CALIBRATION_MODELS = ("full", "choice only")
# The variant whose held-out likelihood chooses the binary tail decay.
BINARY_MODEL = "binary"
NOISE_FLOOR_SEED = 0


def code_version() -> str:
    """A cache version that changes with ``CACHE_VERSION`` and the source of the logit modules.

    The cached worker's arguments (trials, settings, variants) are hashed by joblib, but the
    code the worker calls is not: this string is passed as an argument so that editing the design
    matrix, the validation or the analyses invalidates every cached mouse.
    """
    modules = (cl, cv, ext)
    digest = hashlib.sha256(
        "\n".join([str(CACHE_VERSION), *(inspect.getsource(m) for m in modules)]).encode()
    )
    return digest.hexdigest()[:16]


class MouseExtensions(NamedTuple):
    """The extension analyses of one mouse.

    Attributes:
        scores (pd.DataFrame): Held-out scores of the variants without binary columns
            (``VariantCV.scores``).
        binary_scores (dict): ``{binary tail decay: scores}`` of the variants with binary
            columns, one entry per candidate decay.
        weights (pd.DataFrame): ``choice_logit.weights_table`` of the believed-value model fitted
            to all of the mouse's trials.
        chosen (np.ndarray): ``(n_trials,)`` chosen arm index.
        probabilities (dict): ``{model: (n_trials, 3)}`` held-out probabilities of
            ``CALIBRATION_MODELS``.
        warnings (tuple[str, ...]): Messages of the warnings raised while analyzing the mouse.
    """

    scores: pd.DataFrame
    binary_scores: dict
    weights: pd.DataFrame
    chosen: np.ndarray
    probabilities: dict
    warnings: tuple


class SourceExtensions(NamedTuple):
    """The extension analyses of every analyzable mouse of one source.

    Attributes:
        comparison (pd.DataFrame): ``choice_logit_cv.model_comparison`` of all variants, all mice
            (the binary variants with the chosen binary tail decay).
        weights (pd.DataFrame): Believed-value model weights, all mice.
        mice (np.ndarray): ``(n_trials,)`` mouse of each trial.
        chosen (np.ndarray): ``(n_trials,)`` chosen arm index.
        probabilities (dict): ``{model: (n_trials, 3)}`` held-out probabilities, rows as *mice*.
        skipped (dict[str, str]): Mice not analyzed and why.
        n_lags (int): Lags of the design.
        tail_decay (float | None): Decay of the reward tail, ``None`` without one.
        l2 (float): Penalty of the fits.
        binary_tail_decay (float | None): Tail decay of the binary coding, ``None`` without a tail.
        binary_decay_table (pd.DataFrame): ``binary_tail_decay, nats_per_choice`` of the binary
            model for each candidate decay, pooled over mice.
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
    binary_tail_decay: float | None
    binary_decay_table: pd.DataFrame


def _needs_binary(groups: tuple[str, ...]) -> bool:
    return cl.BINARY in groups


def analyse_mouse_extensions(
    trials: pd.DataFrame,
    n_lags: int,
    tail_decay: float | None,
    l2: float,
    variants: Mapping[str, tuple[str, ...]],
    extras: tuple[str, ...],
    binary_decays: tuple,
    version: str,
) -> MouseExtensions:
    """Validate every variant on one mouse and fit the believed-value model to all its trials.

    The variants without binary columns are validated on one design; those with binary columns
    once for each candidate decay of the binary tail.

    Args:
        trials (pd.DataFrame): One mouse's drifting trial table.
        n_lags (int): Reward and choice lags.
        tail_decay (float | None): Decay of the reward tail; ``None`` for none.
        l2 (float): Penalty of the fits.
        variants (Mapping[str, tuple[str, ...]]): ``{name: feature groups}``; must contain
            ``BELIEVED_MODEL`` and ``CALIBRATION_MODELS``.
        extras (tuple[str, ...]): Optional column groups the variants may use.
        binary_decays (tuple): Candidate tail decays of the binary coding; ignored (one run
            with no tail) when *tail_decay* is ``None``.
        version (str): ``code_version()``; only part of the cache key.

    Returns:
        MouseExtensions: Scores, believed-model weights, choices, held-out probabilities and the
        warnings raised on the way.

    Raises:
        ValueError: If *variants* lacks a model this analysis needs.
    """
    needed = (BELIEVED_MODEL, BINARY_MODEL, *CALIBRATION_MODELS)
    missing = [m for m in needed if m not in variants]
    if missing:
        raise ValueError(f"variants lack the models {missing}")
    decays = tuple(binary_decays) if tail_decay is not None else (None,)
    plain = {n: g for n, g in variants.items() if not _needs_binary(g)}
    binary = {n: g for n, g in variants.items() if _needs_binary(g)}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        design = cl.design_matrix(
            trials, n_lags, tail_decay, extras=tuple(e for e in extras if e != cl.BINARY)
        )
        validation = ext.cross_validate_variants(design, plain, l2)
        believed = cl.select_features(design, variants[BELIEVED_MODEL])
        fit = cl.fit_logit(believed.X, believed.chosen, believed.feature_names, l2)
        binary_scores = {
            decay: ext.cross_validate_variants(
                cl.design_matrix(
                    trials, n_lags, tail_decay, extras=(cl.BINARY,), binary_tail_decay=decay
                ),
                binary,
                l2,
            ).scores
            for decay in decays
        }
    messages = tuple(dict.fromkeys(str(w.message) for w in caught))
    return MouseExtensions(
        validation.scores,
        binary_scores,
        cl.weights_table(design.meta["mouse"].iloc[0], fit.feature_names, fit.beta),
        design.chosen,
        {name: validation.probabilities[name] for name in CALIBRATION_MODELS},
        messages,
    )


def mouse_analyser(cache_dir: Path | None):
    """:func:`analyse_mouse_extensions`, cached under *cache_dir* if one is given.

    The cache key is every argument of the function, including the variant definitions, the
    extras and ``code_version()``.
    """
    if cache_dir is None:
        return analyse_mouse_extensions
    return Memory(cache_dir, verbose=0).cache(analyse_mouse_extensions)


def select_binary_decay(binary_scores: Sequence[Mapping]) -> tuple[float | None, pd.DataFrame]:
    """The binary tail decay with the fewest held-out nats per choice, pooled over mice.

    Args:
        binary_scores (Sequence[Mapping]): ``MouseExtensions.binary_scores`` of each mouse.

    Returns:
        tuple[float | None, pd.DataFrame]: The chosen decay (``None`` when the only candidate has
        no tail; the first of equals wins) and a table ``binary_tail_decay, nats_per_choice``
        (NaN decay for no tail) of the ``BINARY_MODEL`` for each candidate.
    """
    decays = list(binary_scores[0])
    nats = {}
    for decay in decays:
        own = [m[decay].query("model == @BINARY_MODEL") for m in binary_scores]
        total = pd.concat(own)
        nats[decay] = -total["held_out_log_likelihood"].sum() / total["n_trials"].sum()
    table = pd.DataFrame(
        {
            "binary_tail_decay": [np.nan if d is None else d for d in decays],
            "nats_per_choice": [nats[d] for d in decays],
        }
    )
    return min(decays, key=lambda d: nats[d]), table


def analyse_extensions(
    trial_table: pd.DataFrame,
    n_lags: int,
    tail_decay: float | None,
    l2: float,
    min_trials: int = MIN_TRIALS,
    n_jobs: int = 1,
    cache_dir: Path | None = None,
    binary_tail_grid: Sequence[float] = TAIL_GRID,
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
            settings, the variants and the source of the logit modules. Default ``None``: no
            cache.
        binary_tail_grid (Sequence[float]): Candidate tail decays of the binary coding, from which
            the one with the best pooled held-out likelihood of the binary model is used. Default
            ``choice_logit_values.TAIL_GRID``.

    Returns:
        SourceExtensions: Results for the analyzable mice, sorted by id.

    Raises:
        ValueError: If no mouse can be analyzed or the grid is empty with a reward tail.
        RuntimeError: If a fit does not converge.

    Warns:
        UserWarning: For each skipped mouse, and whatever a mouse's analysis warned about (for
            example towers that never repeat across problems), also when served from the cache.
    """
    if tail_decay is not None and len(binary_tail_grid) == 0:
        raise ValueError("the binary tail grid is empty")
    by_mouse = dict(sorted(trial_table.groupby("mouse"), key=lambda item: item[0]))
    skipped = {m: r for m, rows in by_mouse.items() if (r := skip_reason(rows, min_trials))}
    for mouse, reason in skipped.items():
        warnings.warn(f"mouse {mouse} skipped: {reason}", stacklevel=2)
    kept = [m for m in by_mouse if m not in skipped]
    if not kept:
        raise ValueError(
            "no mouse can be analyzed: " + "; ".join(f"{m}: {r}" for m, r in skipped.items())
        )
    analyse = mouse_analyser(cache_dir)
    version = code_version()
    results = Parallel(n_jobs=n_jobs)(
        delayed(analyse)(
            by_mouse[m].reset_index(drop=True),
            n_lags,
            tail_decay,
            l2,
            ext.VARIANTS,
            cl.EXTRAS,
            tuple(binary_tail_grid),
            version,
        )
        for m in kept
    )
    for message in dict.fromkeys(m for result in results for m in result.warnings):
        warnings.warn(message, stacklevel=2)
    decay, decay_table = select_binary_decay([r.binary_scores for r in results])
    scores = pd.concat(
        [pd.concat([r.scores, r.binary_scores[decay]]) for r in results], ignore_index=True
    )
    return SourceExtensions(
        comparison=cv.model_comparison(scores, models=list(ext.VARIANTS)),
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
        binary_tail_decay=decay,
        binary_decay_table=decay_table,
    )


class ExtensionTables(NamedTuple):
    """The tables of the four analyses for one source.

    Attributes:
        believed (pd.DataFrame): :func:`~src.behavior_analysis.choice_logit_extensions.
            believed_value_table`.
        binary_pulse (pd.DataFrame): ``binary_pulse_table``.
        partition (pd.DataFrame): ``variance_partition``.
        calibration (dict): ``{model: calibration_table}`` for ``CALIBRATION_MODELS``.
        calibration_summary (dict): ``{model: summary}``: ``calibration_summary`` with the
            ``calibration_noise_floor`` columns and the model's held-out ``nats_per_choice``.
        binary_tail_decay (float | None): Tail decay of the binary coding. Default ``None``.
    """

    believed: pd.DataFrame
    binary_pulse: pd.DataFrame
    partition: pd.DataFrame
    calibration: dict
    calibration_summary: dict
    binary_tail_decay: float | None = None


def extension_tables(
    source: SourceExtensions,
    n_bins: int = ext.DEFAULT_CALIBRATION_BINS,
    floor_seed: int = NOISE_FLOOR_SEED,
) -> ExtensionTables:
    """Tabulate the four analyses of *source*.

    Args:
        source (SourceExtensions): From :func:`analyse_extensions`.
        n_bins (int): Quantile bins of the calibration. Default
            ``choice_logit_extensions.DEFAULT_CALIBRATION_BINS``.
        floor_seed (int): Seed of the calibration noise floor. Default ``NOISE_FLOOR_SEED``.

    Returns:
        ExtensionTables: One table (or one per calibrated model) per analysis.
    """
    calibration, summaries = {}, {}
    for name in CALIBRATION_MODELS:
        table = ext.calibration_table(
            source.probabilities[name], source.chosen, source.mice, n_bins
        )
        floor = ext.calibration_noise_floor(
            source.probabilities[name], source.mice, n_bins, seed=floor_seed
        )
        nats = source.comparison.query("model == @name")[["mouse", "nats_per_choice"]]
        calibration[name] = table
        summaries[name] = (
            ext.calibration_summary(table).merge(floor, on="mouse").merge(nats, on="mouse")
        )
    return ExtensionTables(
        believed=ext.believed_value_table(source.comparison, source.weights),
        binary_pulse=ext.binary_pulse_table(source.comparison),
        partition=ext.variance_partition(source.comparison),
        calibration=calibration,
        calibration_summary=summaries,
        binary_tail_decay=source.binary_tail_decay,
    )


# ---------------------------------------------------------------------------
# Several sources side by side
# ---------------------------------------------------------------------------


def comparison_inputs(
    reference_label: str, trial_table: pd.DataFrame, comparison: NullComparison
) -> dict:
    """The reference and each null agent as ``{label: (trial table, LogitAnalysis)}``.

    Args:
        reference_label (str): The reference's name in *comparison*, e.g. ``"Mice"``.
        trial_table (pd.DataFrame): The trial table the reference was analyzed from.
        comparison (NullComparison): From ``null_agents.compare_with_controls``; its simulated
            cohorts are reused, not simulated again.

    Returns:
        dict: The reference first, then the agents in ``null_agents.KINDS`` order.
    """
    inputs = {reference_label: (trial_table, comparison.analyses[reference_label])}
    for kind in na.KINDS:
        label = AGENT_LABELS[kind]
        inputs[label] = (comparison.controls.cohorts[kind].trials, comparison.analyses[label])
    return inputs


def analyse_sources(
    inputs: Mapping[str, tuple], n_jobs: int = 1, cache_dir: Path | None = None
) -> dict[str, SourceExtensions]:
    """:func:`analyse_extensions` of every source with the logit settings of its own analysis.

    Args:
        inputs (Mapping[str, tuple]): ``{label: (trial table, LogitAnalysis)}``, e.g. from
            :func:`comparison_inputs`; the analysis supplies lags, tail decay and penalty.
        n_jobs (int): Parallel workers over mice. Default 1.
        cache_dir (Path | None): Per-mouse cache folder. Default ``None``: no cache.

    Returns:
        dict[str, SourceExtensions]: One result per label, in order.
    """
    return {
        label: analyse_extensions(
            table, a.n_lags, a.tail_decay, a.l2, n_jobs=n_jobs, cache_dir=cache_dir
        )
        for label, (table, a) in inputs.items()
    }


class SourceSummaries(NamedTuple):
    """Means across mice of the four analyses, one block of rows per source.

    A source with fewer than ``choice_logit_extensions.MIN_TEST_MICE`` mice has NaN for the mean,
    standard error and tests: its mice are in :func:`per_mouse_summary`.

    Attributes:
        gains (pd.DataFrame): ``source, term, n, mean, se, n_positive, p_wilcoxon, p_sign`` of the
            held-out gains ``gain_believed``, ``gain_no_reward_believed``, ``gain_binary`` and
            ``gain_binary_pulse`` (see :func:`~src.behavior_analysis.choice_logit_extensions.
            gain_summary`).
        weights (pd.DataFrame): ``source, term, n, mean, se`` of the believed-value and
            never-visited weights.
        partition (pd.DataFrame): The same for the unique and shared variance terms.
        calibration (pd.DataFrame): ``source, model, term, n, mean, se`` of the calibration
            errors, their noise floors and the held-out nats per choice of each calibrated model.
    """

    gains: pd.DataFrame
    weights: pd.DataFrame
    partition: pd.DataFrame
    calibration: pd.DataFrame


BELIEVED_GAIN_TERMS = ("gain_believed", "gain_no_reward_believed")
BINARY_GAIN_TERMS = ("gain_binary", "gain_binary_pulse")
WEIGHT_TERMS = ("believed_weight", "unvisited_weight")
PARTITION_TERMS = ("unique_tower", "unique_reward", "unique_choice", "shared_reward_choice")
CALIBRATION_TERMS = (
    "mean_abs_error",
    "weighted_abs_error",
    "noise_floor_mean_abs_error",
    "noise_floor_weighted_abs_error",
    "nats_per_choice",
)
MEAN_SE = ["term", "n", "mean", "se"]
GAIN_COLUMNS = [*MEAN_SE, "n_positive", "p_wilcoxon", "p_sign"]


def _summary(table: pd.DataFrame, terms: Sequence[str]) -> pd.DataFrame:
    """``term, n, mean, se, ...`` of the columns *terms* across the mice of *table*."""
    return ext.gain_summary(table, terms).rename(columns={"gain": "term"})


def source_summaries(tables: Mapping[str, ExtensionTables]) -> SourceSummaries:
    """Mean ± se across mice of every analysis, for each source.

    Args:
        tables (Mapping[str, ExtensionTables]): ``{source label: tables}``.

    Returns:
        SourceSummaries: Long tables with a ``source`` column, sources in order.
    """
    gains, weights, partition, calibration = [], [], [], []
    for label, t in tables.items():
        gains += [
            _summary(t.believed, BELIEVED_GAIN_TERMS).assign(source=label),
            _summary(t.binary_pulse, BINARY_GAIN_TERMS).assign(source=label),
        ]
        weights.append(_summary(t.believed, WEIGHT_TERMS)[MEAN_SE].assign(source=label))
        partition.append(_summary(t.partition, PARTITION_TERMS)[MEAN_SE].assign(source=label))
        calibration += [
            _summary(summary, CALIBRATION_TERMS)[MEAN_SE].assign(source=label, model=model)
            for model, summary in t.calibration_summary.items()
        ]
    return SourceSummaries(
        gains=pd.concat(gains, ignore_index=True)[["source", *GAIN_COLUMNS]],
        weights=pd.concat(weights, ignore_index=True)[["source", *MEAN_SE]],
        partition=pd.concat(partition, ignore_index=True)[["source", *MEAN_SE]],
        calibration=pd.concat(calibration, ignore_index=True)[["source", "model", *MEAN_SE]],
    )


def per_mouse_summary(tables: Mapping[str, ExtensionTables]) -> pd.DataFrame:
    """One row per source and mouse with every per-mouse number of the four analyses.

    For reading the sources that have too few mice for a mean (see :class:`SourceSummaries`).

    Args:
        tables (Mapping[str, ExtensionTables]): ``{source label: tables}``.

    Returns:
        pd.DataFrame: ``source, mouse``, the believed-value weights and gains, the binary gains
        and preferred coding, the variance terms, and for each calibrated model the calibration
        errors, noise floors and held-out nats per choice (column suffix ``_<model>`` with
        spaces as underscores).
    """
    frames = []
    for label, t in tables.items():
        frame = t.believed[["mouse", *WEIGHT_TERMS, *BELIEVED_GAIN_TERMS]].merge(
            t.binary_pulse[["mouse", *BINARY_GAIN_TERMS, "preferred"]]
        )
        frame = frame.merge(t.partition[["mouse", *PARTITION_TERMS]])
        for model, summary in t.calibration_summary.items():
            suffix = model.replace(" ", "_")
            frame = frame.merge(
                summary[["mouse", *CALIBRATION_TERMS]].rename(
                    columns={c: f"{c}_{suffix}" for c in CALIBRATION_TERMS}
                )
            )
        frames.append(frame.assign(source=label))
    table = pd.concat(frames, ignore_index=True)
    return table[["source", "mouse", *[c for c in table.columns if c not in ("source", "mouse")]]]
