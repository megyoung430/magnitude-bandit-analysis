"""Four questions asked of the choice logit by comparing held-out model variants.

Every variant is fitted on all but one of a mouse's problems and scored on the problem it did
not see (:func:`cross_validate_variants`, the validation of
:mod:`src.behavior_analysis.choice_logit_cv` for any set of feature groups). Lower held-out
negative log likelihood per choice ("nats per choice") is better; chance is ``log(3)``.

1. **Believed value** (:func:`believed_value_table`): does the magnitude each arm last paid
   (:data:`~src.behavior_analysis.choice_logit.BELIEVED`), with an indicator for arms not yet
   visited, add anything to the reward lags, or replace them?
2. **Binary against pulse** (:func:`binary_pulse_table`): do choices follow how many pulses were
   received, or only whether any were?
3. **Unique and shared variance** (:func:`variance_partition`): what each group of regressors
   (tower bias, reward history, choice history) adds that the others do not, and what reward and
   choice history explain jointly.
4. **Calibration** (:func:`calibration_table`): does a held-out predicted probability match the
   fraction of times that arm was chosen?

Everything here is pure: designs and tables in, tables out.
"""

import zlib
from collections.abc import Mapping, Sequence
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import binomtest, wilcoxon

from src.behavior_analysis import choice_logit as cl
from src.behavior_analysis.stats import nan_mean, nan_standard_error

R, C, T, B, BIN = cl.REWARD, cl.CHOICE, cl.TOWER, cl.BELIEVED, cl.BINARY

# Variant name -> the feature groups of that model. "full" is the model of the other analyses.
VARIANTS = {
    "full": (R, C, T),
    "full + believed": (R, C, T, B),
    "choice + tower + believed": (C, T, B),
    "binary": (BIN, C, T),
    "binary + pulse": (BIN, R, C, T),
    "no reward": (C, T),
    "no choice": (R, T),
    "no tower": (R, C),
    "reward only": (R,),
    "choice only": (C,),
    "tower only": (T,),
}
# Fewer mice than this are shown as single mice, without a mean, error or test. Five mice can
# never reach p < 0.05 in a two-sided exact test (the smallest p is 2 / 2**5 = 0.0625); six can.
MIN_TEST_MICE = 6
DEFAULT_CALIBRATION_BINS = 10


class VariantCV(NamedTuple):
    """Leave-one-problem-out results of several model variants for one mouse.

    Attributes:
        scores (pd.DataFrame): One row per held-out problem and variant: ``mouse, problem,
            model, n_trials, held_out_log_likelihood`` (the layout of ``MouseCV.scores``).
        probabilities (dict[str, np.ndarray]): ``{variant: (n_trials, 3)}`` held-out choice
            probabilities, rows as in the design.
    """

    scores: pd.DataFrame
    probabilities: dict


# ---------------------------------------------------------------------------
# Held-out scores of variants
# ---------------------------------------------------------------------------


def _require_groups(design: cl.LogitDesign, groups: tuple[str, ...]) -> cl.LogitDesign:
    """The columns of *groups*; raises if the design has no column of one of them."""
    missing = [
        g for g in groups if not any(n.startswith(cl.GROUP_PREFIX[g]) for n in design.feature_names)
    ]
    if missing:
        raise ValueError(
            f"the design has no {missing} columns; build it with "
            f"design_matrix(..., extras={tuple(cl.EXTRAS)})"
        )
    return cl.select_features(design, groups)


def _held_out_log_probabilities(model: cl.LogitDesign, codes: np.ndarray, l2: float) -> np.ndarray:
    """``(n_trials, 3)`` log choice probabilities of each problem under a fit to the others."""
    log_probabilities = np.full(model.X.shape[:2], np.nan)
    for code in np.unique(codes):
        test = codes == code
        fit = cl.fit_logit(model.X[~test], model.chosen[~test], model.feature_names, l2)
        utility = model.X[test] @ fit.beta
        log_probabilities[test] = utility - logsumexp(utility, axis=1, keepdims=True)
    return log_probabilities


def cross_validate_variants(
    design: cl.LogitDesign, variants: Mapping[str, tuple[str, ...]], l2: float = cl.DEFAULT_L2
) -> VariantCV:
    """Fit each variant on all but one problem and score it on that problem, for each problem.

    Args:
        design (cl.LogitDesign): One mouse's design with the column groups every variant needs
            (:func:`choice_logit.design_matrix`, with ``extras`` for binary or believed groups).
        variants (Mapping[str, tuple[str, ...]]): ``{name: feature groups}``, e.g. ``VARIANTS``.
        l2 (float): Penalty of the fits. Default ``choice_logit.DEFAULT_L2``.

    Returns:
        VariantCV: Scores per held-out problem and held-out probabilities of every variant.

    Raises:
        ValueError: If the mouse has fewer than two problems or a variant needs a group the
            design lacks.
        RuntimeError: If a fit does not converge.
    """
    codes, problems = pd.factorize(design.meta["problem"])
    if len(problems) < 2:
        raise ValueError("leave-one-problem-out validation needs at least two problems")
    mouse = design.meta["mouse"].iloc[0]
    rows, probabilities = [], {}
    for name, groups in variants.items():
        model = _require_groups(design, groups)
        log_probabilities = _held_out_log_probabilities(model, codes, l2)
        probabilities[name] = np.exp(log_probabilities)
        chosen_log = log_probabilities[np.arange(len(model.chosen)), model.chosen]
        for code, problem in enumerate(problems):
            test = codes == code
            rows.append(
                dict(
                    mouse=mouse,
                    problem=problem,
                    model=name,
                    n_trials=int(test.sum()),
                    held_out_log_likelihood=float(chosen_log[test].sum()),
                )
            )
    return VariantCV(pd.DataFrame(rows), probabilities)


def _nats_by_mouse(comparison: pd.DataFrame, needed: Sequence[str]) -> pd.DataFrame:
    """Nats per choice, mice by the *needed* models (one column each, in that order)."""
    missing = [m for m in needed if m not in set(comparison["model"])]
    if missing:
        raise ValueError(f"the comparison lacks the models {missing}")
    wide = comparison.pivot(index="mouse", columns="model", values="nats_per_choice")
    return wide[list(needed)]


# ---------------------------------------------------------------------------
# 1. Believed value
# ---------------------------------------------------------------------------


def believed_value_table(comparison: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    """Per mouse: the believed-value weights and what the believed value adds to the model.

    Args:
        comparison (pd.DataFrame): ``choice_logit_cv.model_comparison`` with the ``full``,
            ``full + believed`` and ``choice + tower + believed`` variants.
        weights (pd.DataFrame): ``choice_logit.weights_table`` of the ``full + believed`` fit to
            all of each mouse's trials.

    Returns:
        pd.DataFrame: ``mouse``, ``believed_weight`` (logit per pulse of the believed value),
        ``unvisited_weight`` (logit of an arm not yet visited in the problem, against a visited
        arm that paid 0), ``nats_full``, ``nats_full_believed``, ``nats_no_reward_believed``,
        ``gain_believed`` (``nats_full - nats_full_believed``: held-out gain of adding the
        believed value to the full model, positive is better) and ``gain_no_reward_believed``
        (the same for the model with the believed value *instead of* the reward lags; negative
        if the believed value cannot replace them).

    Raises:
        ValueError: If a model or a believed-value weight is missing.
    """
    names = ["full", "full + believed", "choice + tower + believed"]
    nats = _nats_by_mouse(comparison, names)
    absent = [f for f in cl.BELIEVED_FEATURES if f not in set(weights["feature"])]
    if absent:
        raise ValueError(f"the weights lack {absent}")
    wide = weights.pivot(index="mouse", columns="feature", values="weight")
    table = pd.DataFrame(
        {
            "believed_weight": wide["believed_value"],
            "unvisited_weight": wide["believed_unvisited"],
            "nats_full": nats["full"],
            "nats_full_believed": nats["full + believed"],
            "nats_no_reward_believed": nats["choice + tower + believed"],
        }
    )
    table["gain_believed"] = table["nats_full"] - table["nats_full_believed"]
    table["gain_no_reward_believed"] = table["nats_full"] - table["nats_no_reward_believed"]
    return table.rename_axis("mouse").reset_index()


# ---------------------------------------------------------------------------
# 2. Binary against pulse
# ---------------------------------------------------------------------------


def binary_pulse_table(comparison: pd.DataFrame) -> pd.DataFrame:
    """Per mouse: held-out nats of the pulse, binary and binary + pulse reward codings.

    Args:
        comparison (pd.DataFrame): ``choice_logit_cv.model_comparison`` with the ``full`` (pulse),
            ``binary`` and ``binary + pulse`` variants.

    Returns:
        pd.DataFrame: ``mouse``, ``nats_pulse``, ``nats_binary``, ``nats_binary_pulse``,
        ``gain_binary`` (``nats_pulse - nats_binary``: positive when the binary coding predicts
        better), ``gain_binary_pulse`` (``nats_pulse - nats_binary_pulse``: positive when adding
        the binary lags to the pulse lags helps) and ``preferred`` (``"binary"`` or ``"pulse"``,
        whichever of the two single codings has fewer nats).

    Raises:
        ValueError: If a model is missing.
    """
    nats = _nats_by_mouse(comparison, ["full", "binary", "binary + pulse"])
    table = pd.DataFrame(
        {
            "nats_pulse": nats["full"],
            "nats_binary": nats["binary"],
            "nats_binary_pulse": nats["binary + pulse"],
        }
    )
    table["gain_binary"] = table["nats_pulse"] - table["nats_binary"]
    table["gain_binary_pulse"] = table["nats_pulse"] - table["nats_binary_pulse"]
    table["preferred"] = np.where(table["gain_binary"] > 0, "binary", "pulse")
    return table.rename_axis("mouse").reset_index()


def _two_sided_tests(values: np.ndarray) -> tuple[float, float]:
    """``(Wilcoxon p, sign-test p)`` of paired differences against 0; both 1 if all are 0."""
    nonzero = values[values != 0]
    if len(nonzero) == 0:  # no difference at all: no evidence either way
        return 1.0, 1.0
    p_sign = binomtest(int((nonzero > 0).sum()), len(nonzero), 0.5).pvalue
    return float(wilcoxon(values).pvalue), float(p_sign)


def gain_summary(
    table: pd.DataFrame, columns: Sequence[str], min_mice: int = MIN_TEST_MICE
) -> pd.DataFrame:
    """Mean, standard error and paired tests across mice of per-mouse gains.

    Args:
        table (pd.DataFrame): One row per mouse, with a column of gains for each of *columns*.
        columns (Sequence[str]): The gain columns to summarize.
        min_mice (int): With fewer mice the mean, standard error and tests are NaN: five mice
            cannot reach p < 0.05 in a two-sided exact test (the smallest p is 0.0625), and the
            figures show such mice individually. Default ``MIN_TEST_MICE``.

    Returns:
        pd.DataFrame: One row per column: ``gain, n, mean, se, n_positive, p_wilcoxon,
        p_sign``; NaN in ``mean`` to ``p_sign`` below *min_mice* mice.
    """
    rows = []
    for column in columns:
        values = table[column].dropna().to_numpy(dtype=float)
        enough = len(values) >= min_mice
        p_wilcoxon, p_sign = _two_sided_tests(values) if enough else (np.nan, np.nan)
        rows.append(
            dict(
                gain=column,
                n=len(values),
                mean=float(nan_mean(values[:, None])[0]) if enough else np.nan,
                se=float(nan_standard_error(values[:, None])[0]) if enough else np.nan,
                n_positive=int((values > 0).sum()),
                p_wilcoxon=p_wilcoxon,
                p_sign=p_sign,
            )
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Unique and shared variance
# ---------------------------------------------------------------------------

PARTITION_MODELS = (
    "full", "no reward", "no choice", "no tower", "reward only", "choice only", "tower only",
)  # fmt: skip


def variance_partition(comparison: pd.DataFrame) -> pd.DataFrame:
    """Per mouse: what each regressor group adds uniquely, and what reward and choice share.

    All terms are held-out nats per choice, ``loss`` below, and positive terms are explained
    variance.

    - ``unique_<group>`` = ``loss(full minus the group) - loss(full)``: what only that group
      explains, given the other two.
    - ``shared_reward_choice`` = ``[gain(reward + tower) + gain(choice + tower)] - gain(full)``
      with ``gain(model) = loss(tower only) - loss(model)``: the sum of what each history
      explains beyond the tower bias, minus what both explain together. It is positive when the
      two histories overlap, so the sum counts the same choices twice (a rewarded choice is also a
      repeated one), and negative when they complement each other. The tower bias stays in
      every model of this term so that it is measured against one baseline; the models without
      it would score below the tower-only model for a mouse with a strong port preference.
    - Together ``gain_full_over_tower = unique_reward + unique_choice + shared_reward_choice``
      exactly.
    - ``only_<group>`` is the gain over chance (``log(3) - loss``) of the model with that group
      alone.

    Args:
        comparison (pd.DataFrame): ``choice_logit_cv.model_comparison`` with the models of
            ``PARTITION_MODELS``.

    Returns:
        pd.DataFrame: ``mouse, nats_full, unique_tower, unique_reward, unique_choice,
        shared_reward_choice, gain_full_over_tower, only_tower, only_reward, only_choice``.

    Raises:
        ValueError: If a model is missing.
    """
    nats = _nats_by_mouse(comparison, PARTITION_MODELS)
    chance = np.log(cl.N_ARMS)
    table = pd.DataFrame(
        {
            "nats_full": nats["full"],
            "unique_tower": nats["no tower"] - nats["full"],
            "unique_reward": nats["no reward"] - nats["full"],
            "unique_choice": nats["no choice"] - nats["full"],
            # gain(reward + tower) = loss(tower only) - loss(no choice), and so on
            "shared_reward_choice": (nats["tower only"] - nats["no choice"])
            + (nats["tower only"] - nats["no reward"])
            - (nats["tower only"] - nats["full"]),
            "gain_full_over_tower": nats["tower only"] - nats["full"],
            "only_tower": chance - nats["tower only"],
            "only_reward": chance - nats["reward only"],
            "only_choice": chance - nats["choice only"],
        }
    )
    return table.rename_axis("mouse").reset_index()


# ---------------------------------------------------------------------------
# 4. Calibration
# ---------------------------------------------------------------------------


def calibration_bins(
    probabilities: np.ndarray, chosen: np.ndarray, n_bins: int = DEFAULT_CALIBRATION_BINS
) -> pd.DataFrame:
    """Observed choice frequency against predicted probability, in equal-count bins.

    Every (trial, arm) pair is one prediction, the arm's probability, with outcome 1 if that arm
    was chosen. The predictions are sorted and cut into *n_bins* groups of (nearly) equal size
    (deciles by default), so every bin is estimated from as many pairs as any other. Tied
    predictions can straddle a bin edge: equal values are split between the two bins in sorted
    order, so a model with few distinct predictions (e.g. choice history only) has adjacent bins
    with the same mean predicted probability.

    Args:
        probabilities (np.ndarray): ``(n_trials, 3)`` predicted choice probabilities.
        chosen (np.ndarray): ``(n_trials,)`` chosen arm index.
        n_bins (int): Number of quantile bins. Default ``DEFAULT_CALIBRATION_BINS``.

    Returns:
        pd.DataFrame: One row per bin, lowest predictions first: ``bin, n_pairs, mean_predicted,
        observed`` (fraction of the bin's arms that were chosen).

    Raises:
        ValueError: If the shapes are wrong or there are fewer predictions than bins.
    """
    if probabilities.ndim != 2 or probabilities.shape[1] != cl.N_ARMS:
        raise ValueError(f"probabilities must have shape (n_trials, 3), got {probabilities.shape}")
    if chosen.shape != (len(probabilities),):
        raise ValueError("chosen must hold one arm index per trial")
    predicted = probabilities.ravel()
    if predicted.size < n_bins:
        raise ValueError(f"{predicted.size} predictions cannot fill {n_bins} bins")
    outcome = (np.arange(cl.N_ARMS)[None, :] == chosen[:, None]).ravel().astype(float)
    order = np.argsort(predicted, kind="stable")
    bin_of = np.empty(predicted.size, dtype=int)
    bin_of[order] = np.arange(predicted.size) * n_bins // predicted.size
    frame = pd.DataFrame({"bin": bin_of, "predicted": predicted, "outcome": outcome})
    return (
        frame.groupby("bin")
        .agg(
            n_pairs=("predicted", "size"),
            mean_predicted=("predicted", "mean"),
            observed=("outcome", "mean"),
        )
        .reset_index()
    )


def calibration_table(
    probabilities: np.ndarray,
    chosen: np.ndarray,
    mice: np.ndarray,
    n_bins: int = DEFAULT_CALIBRATION_BINS,
) -> pd.DataFrame:
    """:func:`calibration_bins` of each mouse's trials.

    Args:
        probabilities (np.ndarray): ``(n_trials, 3)`` predicted probabilities of all mice.
        chosen (np.ndarray): ``(n_trials,)`` chosen arm index.
        mice (np.ndarray): ``(n_trials,)`` mouse of each trial.
        n_bins (int): Quantile bins per mouse. Default ``DEFAULT_CALIBRATION_BINS``.

    Returns:
        pd.DataFrame: ``mouse, bin, n_pairs, mean_predicted, observed``, mice in order of first
        appearance.

    Raises:
        ValueError: As :func:`calibration_bins`, or if *mice* is not one per trial.
    """
    if mice.shape != chosen.shape:
        raise ValueError("mice must hold one mouse id per trial")
    tables = [
        calibration_bins(probabilities[mice == mouse], chosen[mice == mouse], n_bins).assign(
            mouse=mouse
        )
        for mouse in pd.unique(mice)
    ]
    return pd.concat(tables, ignore_index=True)[
        ["mouse", "bin", "n_pairs", "mean_predicted", "observed"]
    ]


def calibration_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Per mouse: how far the bins lie from the identity line.

    Args:
        table (pd.DataFrame): :func:`calibration_table` output.

    Returns:
        pd.DataFrame: ``mouse``, ``mean_abs_error`` (mean over bins of ``|observed -
        predicted|``) and ``weighted_abs_error`` (the same weighted by each bin's pairs); both 0
        for a perfectly calibrated model.
    """
    deviation = (table["observed"] - table["mean_predicted"]).abs()
    by_mouse = table.assign(deviation=deviation, weighted=deviation * table["n_pairs"]).groupby(
        "mouse", sort=False
    )
    return pd.DataFrame(
        {
            "mean_abs_error": by_mouse["deviation"].mean(),
            "weighted_abs_error": by_mouse["weighted"].sum() / by_mouse["n_pairs"].sum(),
        }
    ).reset_index()


def calibration_noise_floor(
    probabilities: np.ndarray,
    mice: np.ndarray,
    n_bins: int = DEFAULT_CALIBRATION_BINS,
    seed: int = 0,
    n_draws: int = 20,
) -> pd.DataFrame:
    """The calibration error a perfectly calibrated model would show by sampling noise alone.

    For each mouse the outcomes are drawn from the mouse's own held-out predictions (each trial's
    choice from its three probabilities), and the calibration error is computed with the same
    bins as for the real choices; the floor is the mean over *n_draws* draws. A model whose
    error is near its floor is as calibrated as its trial count lets one tell.

    Args:
        probabilities (np.ndarray): ``(n_trials, 3)`` held-out predictions of all mice.
        mice (np.ndarray): ``(n_trials,)`` mouse of each trial.
        n_bins (int): Quantile bins per mouse. Default ``DEFAULT_CALIBRATION_BINS``.
        seed (int): Seed of the draws; each mouse has its own stream derived from it and a
            checksum of the mouse id, so adding a mouse changes no other mouse's floor. Default 0.
        n_draws (int): Simulated outcome sets averaged per mouse. Default 20.

    Returns:
        pd.DataFrame: ``mouse, noise_floor_mean_abs_error, noise_floor_weighted_abs_error`` (the
        two metrics of :func:`calibration_summary`), mice in order of first appearance.

    Raises:
        ValueError: If *mice* is not one per trial, or as :func:`calibration_bins`.
    """
    if mice.shape != (len(probabilities),):
        raise ValueError("mice must hold one mouse id per trial")
    rows = []
    for mouse in pd.unique(mice):
        own = probabilities[mice == mouse]
        rng = np.random.default_rng([seed, zlib.crc32(str(mouse).encode())])
        cumulative = own.cumsum(axis=1)
        errors = []
        for _ in range(n_draws):
            # a draw from each row: the first arm whose cumulative probability exceeds u
            simulated = np.minimum(
                (rng.random(len(own))[:, None] >= cumulative).sum(axis=1), cl.N_ARMS - 1
            )
            table = calibration_bins(own, simulated, n_bins).assign(mouse=mouse)
            errors.append(calibration_summary(table).iloc[0])
        rows.append(
            dict(
                mouse=mouse,
                noise_floor_mean_abs_error=float(np.mean([e["mean_abs_error"] for e in errors])),
                noise_floor_weighted_abs_error=float(
                    np.mean([e["weighted_abs_error"] for e in errors])
                ),
            )
        )
    return pd.DataFrame(rows)
