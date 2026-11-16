"""The choice logit as a source of value estimates, for the mice and for simulated agents.

Three steps, each a plain function so a notebook can call them one at a time:

1. **Load** (:func:`load_drifting_trial_table`) the drifting problems of the chosen cohorts as one
   trial table, from the newest import-cache snapshot (nothing is read from or written to the
   data share), or :func:`simulate_cohort` a cohort of value-learning agents whose generating
   values and choice rule are known.
2. **Analyse** (:func:`analyse_trial_table`): per mouse, the design matrix
   (:mod:`src.behavior_analysis.choice_logit.choice_logit`), the fit to all of the mouse's trials
   (for the
   kernels), and leave-one-problem-out validation
   (:mod:`src.behavior_analysis.choice_logit.choice_logit_cv`), which also gives the held-out model
   values
   and the held-out nuisance baseline. :func:`select_tail_decay` chooses the decay of the
   reward tail by held-out likelihood.
3. **Summarise** the held-out values as psychometric curves and stay/switch pairs against
   quantile bins of the value, each with the nuisance baseline (:func:`value_curves`,
   :func:`value_pairs`), and compare them with the task magnitudes, the last reward each arm paid
   (:func:`task_values`, :func:`believed_values`) and, for simulated agents, the generating
   values.

Drawing is in :mod:`src.behavior_visualization.choice_logit.plot_choice_logit`.
"""

import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.agents import drifting_simulation as sim
from src.agents.choice_logit_truth import generating_weights, q_kernel_bounds, true_utilities
from src.behavior_analysis.choice_logit import choice_logit as cl
from src.behavior_analysis.choice_logit import choice_logit_cv as cv
from src.behavior_analysis.data.drifting_trial_table import drifting_trial_table
from src.behavior_analysis.value import model_value_psychometrics as mvp
from src.pipelines.common.all_cohorts_drifting import (
    check_cohort_ids,
    drifting_problems,
    pool_cohorts,
)
from src.pipelines.common.cohort_analysis import load_cohort

# A mouse with fewer trials than this is not analysed: the weights would mostly be the prior.
MIN_TRIALS = 300
# Towers of the simulated problems; each triple avoids the initiation port B2 and mixes
# plus-shaped ports (same row or column as B2) with the others. Over the six triples every tower
# is offered in at least two problems, so each tower's bias is identified by held-out fits.
SIMULATED_PORTS = (
    ("A1", "B3", "C2"),
    ("A3", "B1", "C3"),
    ("A2", "B3", "C1"),
    ("A1", "B1", "C3"),
    ("A3", "B3", "C1"),
    ("A2", "B1", "C2"),
)
SIMULATED_INITIATION = "B2"
# Magnitudes (pulses) of the three ports on a simulated problem's first trial, dealt to the
# ports in a random order for each agent and problem so no port starts out best systematically.
START_MAGNITUDES = (4, 2, 0)
# Decays of the reward tail tried by select_tail_decay.
TAIL_GRID = (0.5, 0.7, 0.85, 0.95)


# ---------------------------------------------------------------------------
# Loading and simulating
# ---------------------------------------------------------------------------


def load_drifting_problems(
    cohort_ids: Sequence[int],
    data_root: Path | None = None,
    trials_by_cohort: dict | None = None,
) -> dict:
    """The drifting problems of *cohort_ids* pooled, ``{(cohort id, k): {mouse: sessions}}``.

    Each cohort is loaded cache-first: the newest snapshot of the import cache is used without
    reading the data share (sessions added since are missing; a ``StaleCacheWarning`` says
    so). A cohort with no cache at all is imported from the share and cached, which writes
    to the cache directory only.

    Args:
        cohort_ids (Sequence[int]): Cohorts to pool, e.g. ``[1, 3]``.
        data_root (Path | None): Folder holding ``3x3_field_magnitude_bandit``. Default ``None``:
            the ``BANDIT_DATA_ROOT`` setting.
        trials_by_cohort (dict | None): Already-loaded ``{cohort id: {overall problem: {mouse:
            sessions}}}``; skips loading. Default ``None``.

    Returns:
        dict: ``k`` is the drifting problem number within the cohort.

    Raises:
        ValueError: If a cohort id is invalid or repeated, a cohort has no drifting problem, or a
            mouse id occurs in two cohorts.
    """
    ids = check_cohort_ids(cohort_ids)
    if trials_by_cohort is None:
        trials_by_cohort = {
            c: load_cohort(c, data_root, use_cache=True, from_cache=True).trials_by_problem
            for c in ids
        }
    return pool_cohorts({c: drifting_problems(trials_by_cohort[c]) for c in ids}).problems


def load_drifting_trial_table(
    cohort_ids: Sequence[int],
    data_root: Path | None = None,
    trials_by_cohort: dict | None = None,
) -> pd.DataFrame:
    """The trial table of :func:`load_drifting_problems` (same arguments).

    Returns:
        pd.DataFrame: The table of
        :func:`~src.behavior_analysis.data.drifting_trial_table.drifting_trial_table`.
    """
    return drifting_trial_table(load_drifting_problems(cohort_ids, data_root, trials_by_cohort))


class SyntheticCohort(NamedTuple):
    """Simulated value-learning agents on the drifting task.

    Attributes:
        trials (pd.DataFrame): Their trial table.
        parameters (pd.DataFrame): ``mouse`` and the generating ``value_agent`` parameters.
        problems (dict): ``{problem: {mouse: sessions}}``, for the analyses that take sessions.
        kind (str): The agent that played: ``"value"`` for the value learner of
            :mod:`src.agents.value_agent`, the agent kind for null agents
            (:mod:`src.agents.null_agents`). Only ``"value"`` cohorts have the generating
            values and choice rule of :func:`generating_values` and :func:`reference_weights`.
            Default ``"value"``.
    """

    trials: pd.DataFrame
    parameters: pd.DataFrame
    problems: dict
    kind: str = "value"


def _layouts(
    n_problems: int, n_sessions: int, trials_per_session: int, rng: np.random.Generator
) -> dict:
    """The problems one simulated agent plays: other towers in each, B2 to initiate.

    The start magnitudes are dealt to the ports at random (*rng*), so which tower starts out
    best differs between problems.
    """
    ports = [sorted(SIMULATED_PORTS[k % len(SIMULATED_PORTS)]) for k in range(n_problems)]
    return {
        k: sim.MouseLayout(
            p,
            SIMULATED_INITIATION,
            dict(zip(p, (int(m) for m in rng.permutation(START_MAGNITUDES)), strict=True)),
            [trials_per_session] * n_sessions,
        )
        for k, p in enumerate(ports, start=1)
    }


def simulate_cohort(
    parameter_sets: Sequence[Sequence[float]],
    *,
    mice_per_set: int,
    n_problems: int,
    sessions_per_problem: int,
    trials_per_session: int,
    seed: int,
    drift_rate: float = 0.1,
    block_lengths: Sequence[int] = tuple(range(30, 36)),
) -> SyntheticCohort:
    """Agents of the value learner of :mod:`src.agents.value_agent` playing the drifting task.

    Every parameter set is played by ``mice_per_set`` agents (``sim01``, ``sim02``, ...); each
    agent has its own random streams (task and start magnitudes), so adding agents does not
    change the others.

    Args:
        parameter_sets (Sequence[Sequence[float]]): ``(learning_rate, beta, kappa, plus_bias)``
            of each setting. ``kappa`` is the stickiness and ``plus_bias`` a known bias towards
            the ports in the row or column of the initiation port.
        mice_per_set (int): Agents per setting. Keyword-only.
        n_problems (int): Problems each agent plays; the ``len(SIMULATED_PORTS)`` tower triples
            repeat beyond that. Keyword-only.
        sessions_per_problem (int): Sessions per problem. Keyword-only.
        trials_per_session (int): Trials per session. Keyword-only.
        seed (int): Base seed. Keyword-only.
        drift_rate (float): Per-trial drift attempt probability of the task. Default 0.1.
        block_lengths (Sequence[int]): Block lengths, sampled with replacement. Default 30-35.

    Returns:
        SyntheticCohort: The simulated trials, the generating parameters and the sessions.
    """
    rules = sim.TaskRules(drift_rate, np.asarray(block_lengths))
    settings = [p for p in parameter_sets for _ in range(mice_per_set)]
    problems: dict = {}
    for i, parameters in enumerate(settings, start=1):
        mouse = f"sim{i:02d}"
        fit = sim.AgentFit(mouse, "magnitude", tuple(parameters), 0.0, 0)
        layouts = _layouts(
            n_problems,
            sessions_per_problem,
            trials_per_session,
            np.random.default_rng([seed, i, 1]),
        )
        played = sim.simulate_mouse(fit, layouts, rules, np.random.default_rng([seed, i]), mouse)
        for problem, animals in played.items():
            problems.setdefault(problem, {}).update(animals)
    parameters_table = pd.DataFrame(
        [(f"sim{i:02d}", *p) for i, p in enumerate(settings, start=1)],
        columns=["mouse", *("learning_rate", "beta", "kappa", "plus_bias")],
    )
    return SyntheticCohort(drifting_trial_table(problems), parameters_table, problems)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


class LogitAnalysis(NamedTuple):
    """The choice logit of every analysable mouse.

    Attributes:
        designs (dict[str, cl.LogitDesign]): Design of each mouse.
        fits (dict[str, cl.LogitFit]): Full-model fit to all of each mouse's trials.
        cv (dict[str, cv.MouseCV]): Leave-one-problem-out scores, held-out values and baseline.
        weights (pd.DataFrame): ``choice_logit.weights_table`` of the full fits, all mice.
        comparison (pd.DataFrame): ``choice_logit_cv.model_comparison`` of all mice.
        skipped (dict[str, str]): Mice not analysed and why.
        meta (pd.DataFrame): Trial metadata of all analysed mice, rows aligned with ``chosen``,
            ``values`` and ``baseline``.
        chosen (np.ndarray): ``(n_trials,)`` chosen arm index.
        values (np.ndarray): ``(n_trials, 3)`` held-out model values.
        baseline (np.ndarray): ``(n_trials, 3)`` held-out choice probabilities of the model with
            choice history and tower bias only.
        n_lags (int): Lags of the design.
        tail_decay (float | None): Decay of the reward tail, ``None`` without one.
        l2 (float): Penalty of the fits.
    """

    designs: dict
    fits: dict
    cv: dict
    weights: pd.DataFrame
    comparison: pd.DataFrame
    skipped: dict
    meta: pd.DataFrame
    chosen: np.ndarray
    values: np.ndarray
    baseline: np.ndarray
    n_lags: int
    tail_decay: float | None
    l2: float


def skip_reason(trials: pd.DataFrame, min_trials: int) -> str | None:
    """Why one mouse's *trials* cannot be validated, or ``None`` if they can."""
    n_problems = trials["problem"].nunique()
    if n_problems < 2:
        return f"{n_problems} problem; leave-one-problem-out needs at least two"
    if len(trials) < min_trials:
        return f"{len(trials)} trials; at least {min_trials} are needed"
    return None


def analyse_mouse(trials: pd.DataFrame, n_lags: int, tail_decay: float | None, l2: float):
    """``(design, full-data fit, validation)`` of one mouse."""
    design = cl.design_matrix(trials, n_lags, tail_decay)
    fit = cl.fit_logit(design.X, design.chosen, design.feature_names, l2)
    return design, fit, cv.cross_validate_mouse(design, l2)


def analyse_trial_table(
    trial_table: pd.DataFrame,
    n_lags: int = cl.DEFAULT_LAGS,
    tail_decay: float | None = cl.DEFAULT_TAIL_DECAY,
    l2: float = cl.DEFAULT_L2,
    min_trials: int = MIN_TRIALS,
) -> LogitAnalysis:
    """Fit and validate the choice logit for every mouse of *trial_table*.

    Args:
        trial_table (pd.DataFrame): A drifting trial table.
        n_lags (int): Reward lags and choice lags. Default ``choice_logit.DEFAULT_LAGS``.
        tail_decay (float | None): Decay of the reward tail (rewards older than the lags, one
            summed regressor); ``None`` for no tail. Default ``choice_logit.DEFAULT_TAIL_DECAY``.
        l2 (float): Penalty. Default ``choice_logit.DEFAULT_L2``.
        min_trials (int): Mice with fewer trials are skipped. Default ``MIN_TRIALS``.

    Returns:
        LogitAnalysis: Results for the mice that can be validated; the others are in
        ``skipped`` and named in a warning.

    Raises:
        ValueError: If no mouse can be analysed.
        RuntimeError: If a fit does not converge.
    """
    by_mouse = {m: rows for m, rows in trial_table.groupby("mouse", sort=True)}
    skipped = {m: r for m, rows in by_mouse.items() if (r := skip_reason(rows, min_trials))}
    for mouse, reason in skipped.items():
        warnings.warn(f"mouse {mouse} skipped: {reason}", stacklevel=2)
    results = {
        m: analyse_mouse(rows, n_lags, tail_decay, l2)
        for m, rows in by_mouse.items()
        if m not in skipped
    }
    if not results:
        raise ValueError(
            "no mouse can be analysed: " + "; ".join(f"{m}: {r}" for m, r in skipped.items())
        )
    designs = {m: r[0] for m, r in results.items()}
    fits = {m: r[1] for m, r in results.items()}
    validations = {m: r[2] for m, r in results.items()}
    return LogitAnalysis(
        designs=designs,
        fits=fits,
        cv=validations,
        weights=pd.concat(
            [cl.weights_table(m, f.feature_names, f.beta) for m, f in fits.items()],
            ignore_index=True,
        ),
        comparison=cv.model_comparison(
            pd.concat([v.scores for v in validations.values()], ignore_index=True)
        ),
        skipped=skipped,
        meta=pd.concat([d.meta for d in designs.values()], ignore_index=True),
        chosen=np.concatenate([d.chosen for d in designs.values()]),
        values=np.concatenate([v.values for v in validations.values()]),
        baseline=np.concatenate([v.baseline for v in validations.values()]),
        n_lags=n_lags,
        tail_decay=tail_decay,
        l2=l2,
    )


class TailSelection(NamedTuple):
    """The reward-tail decay chosen by held-out likelihood.

    Attributes:
        decay (float): The decay of the grid with the lowest held-out nats per choice.
        table (pd.DataFrame): ``tail_decay`` (NaN: no tail) and ``nats_per_choice`` of the full
            model, trial-weighted over mice.
        analysis (LogitAnalysis): The analysis with the chosen decay.
    """

    decay: float
    table: pd.DataFrame
    analysis: LogitAnalysis


def _full_model_nats(analysis: LogitAnalysis) -> float:
    """Held-out nats per choice of the ``full`` model, pooled over the analysed mice."""
    full = analysis.comparison.query("model == 'full'")
    return float(np.average(full["nats_per_choice"], weights=full["n_trials"]))


def select_tail_decay(
    trial_table: pd.DataFrame,
    grid: Sequence[float] = TAIL_GRID,
    n_lags: int = cl.DEFAULT_LAGS,
    l2: float = cl.DEFAULT_L2,
    min_trials: int = MIN_TRIALS,
) -> TailSelection:
    """Choose the decay of the reward tail from *grid* by leave-one-problem-out likelihood.

    The tail is always part of the model (it is what gives long-unvisited arms a value), so the
    choice is among the decays of *grid*; the no-tail model is scored too, for comparison.

    Args:
        trial_table (pd.DataFrame): A drifting trial table.
        grid (Sequence[float]): Candidate decays in (0, 1).
        n_lags (int): Lags of the design. Default ``choice_logit.DEFAULT_LAGS``.
        l2 (float): Penalty. Default ``choice_logit.DEFAULT_L2``.
        min_trials (int): Mice with fewer trials are skipped. Default ``MIN_TRIALS``.

    Returns:
        TailSelection: The best decay, the scores, and the analysis with that decay.

    Raises:
        ValueError: If *grid* is empty or no mouse can be analysed.
    """
    if len(grid) == 0:
        raise ValueError("the decay grid is empty")
    analyses = {
        decay: analyse_trial_table(trial_table, n_lags, decay, l2, min_trials)
        for decay in (None, *grid)
    }
    table = pd.DataFrame(
        {
            "tail_decay": [np.nan if d is None else d for d in analyses],
            "nats_per_choice": [_full_model_nats(a) for a in analyses.values()],
        }
    )
    best = min(grid, key=lambda d: _full_model_nats(analyses[d]))
    return TailSelection(best, table, analyses[best])


# ---------------------------------------------------------------------------
# Value matrices
# ---------------------------------------------------------------------------


def task_values(analysis: LogitAnalysis) -> np.ndarray:
    """``(n_trials, 3)`` magnitude each port offers on each trial, rows as ``analysis.meta``."""
    ports = np.concatenate([d.ports for d in analysis.designs.values()])
    magnitudes = analysis.meta["magnitudes"].to_numpy()
    return np.array(
        [[m[p] for p in row] for m, row in zip(magnitudes, ports, strict=True)], dtype=float
    )


def believed_values(analysis: LogitAnalysis) -> np.ndarray:
    """``(n_trials, 3)`` last reward each port paid when chosen earlier in the problem.

    A port not yet chosen in the problem has no believed value (NaN). The reward history
    carries across the sessions of a problem, as the model's does.
    """
    meta = analysis.meta
    group = meta.groupby(["mouse", "problem"], sort=False).ngroup().to_numpy()
    reward = meta["reward"].to_numpy(dtype=float)
    believed = np.full((len(meta), cl.N_ARMS), np.nan)
    for arm in range(cl.N_ARMS):
        paid = pd.Series(np.where(analysis.chosen == arm, reward, np.nan)).groupby(group)
        believed[:, arm] = paid.ffill().groupby(group).shift(1).to_numpy()
    return believed


def generating_values(analysis: LogitAnalysis, synthetic: SyntheticCohort) -> np.ndarray:
    """The agents' own Q-values before each choice, rows aligned with ``analysis.meta``.

    Raises:
        ValueError: If *synthetic* is not a cohort of value learners.
    """
    _require_value_agents(synthetic)
    return np.concatenate(
        [
            true_utilities(
                synthetic.trials[synthetic.trials["mouse"] == m], _agent_parameters(synthetic, m)
            )[1]
            for m in analysis.designs
        ]
    )


# ---------------------------------------------------------------------------
# Value summaries
# ---------------------------------------------------------------------------


def _values_to_use(analysis: LogitAnalysis, values: np.ndarray | None) -> np.ndarray:
    return analysis.values if values is None else values


def value_rank_table(analysis: LogitAnalysis, values: np.ndarray | None = None) -> pd.DataFrame:
    """:func:`~src.behavior_analysis.value.model_value_psychometrics.rank_table` with the baseline.

    Trials on which an arm has no value (NaN, e.g. a believed value before the arm was chosen)
    are left out.

    Args:
        analysis (LogitAnalysis): The analysis.
        values (np.ndarray | None): ``(n_trials, 3)`` values to rank instead of the held-out
            model values, rows aligned with ``analysis.meta``.
    """
    chosen_values = _values_to_use(analysis, values)
    keep = np.isfinite(chosen_values).all(axis=1)
    return mvp.rank_table(
        analysis.meta["mouse"].to_numpy()[keep],
        chosen_values[keep],
        analysis.chosen[keep],
        analysis.baseline[keep],
    )


def value_curves(
    analysis: LogitAnalysis,
    n_bins: int | None = mvp.DEFAULT_BINS,
    values: np.ndarray | None = None,
    min_mice: int = mvp.MIN_MICE,
) -> tuple[dict, dict]:
    """P(best) and P(second) against the value, and the nuisance baseline's, see
    :func:`~src.behavior_analysis.value.model_value_psychometrics.psychometric_curves_from_table`."""
    return mvp.psychometric_curves_from_table(value_rank_table(analysis, values), n_bins, min_mice)


def value_pairs(analysis: LogitAnalysis, values: np.ndarray | None = None) -> pd.DataFrame:
    """Stay/switch trial pairs on the value, with the baseline, see
    :func:`~src.behavior_analysis.value.model_value_psychometrics.stay_switch_pairs_from_values`."""
    return mvp.stay_switch_pairs_from_values(
        analysis.meta, _values_to_use(analysis, values), analysis.chosen, analysis.baseline
    )


# ---------------------------------------------------------------------------
# Simulated agents: what the generating rule implies
# ---------------------------------------------------------------------------


def _require_value_agents(synthetic: SyntheticCohort) -> None:
    """Raise unless *synthetic* was played by the value learner (the only agent with a Q)."""
    if synthetic.kind != "value":
        raise ValueError(
            f"a cohort of {synthetic.kind!r} agents has no value learner's generating values or "
            "choice rule; this summary is for cohorts from simulate_cohort"
        )


def _agent_parameters(synthetic: SyntheticCohort, mouse: str) -> tuple:
    row = synthetic.parameters.set_index("mouse").loc[mouse]
    return tuple(float(x) for x in row)


def centred_spearman(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Spearman correlation of two ``(n_trials, 3)`` value arrays after removing each trial's mean.

    Only the differences between a trial's arms matter to the choice, so a per-trial offset is
    removed first.
    """
    centred = [v - v.mean(axis=1, keepdims=True) for v in (reference, estimate)]
    return float(spearmanr(centred[0].ravel(), centred[1].ravel()).statistic)


def value_agreement(analysis: LogitAnalysis, synthetic: SyntheticCohort) -> pd.DataFrame:
    """Per mouse, the centred Spearman of the generating Q with the model and believed values.

    The believed value (last reward each arm paid) is the baseline the model value has to beat;
    both are compared on the trials where every arm has a believed value.

    Returns:
        pd.DataFrame: ``mouse, model_value, believed_value``.
    """
    q = generating_values(analysis, synthetic)
    believed = believed_values(analysis)
    complete = np.isfinite(believed).all(axis=1)
    rows = []
    for mouse in analysis.designs:
        mine = (analysis.meta["mouse"] == mouse).to_numpy() & complete
        rows.append(
            dict(
                mouse=mouse,
                model_value=centred_spearman(q[mine], analysis.values[mine]),
                believed_value=centred_spearman(q[mine], believed[mine]),
            )
        )
    return pd.DataFrame(rows)


def reference_weights(analysis: LogitAnalysis, synthetic: SyntheticCohort) -> pd.DataFrame:
    """The logit weights closest to each agent's generating choice probabilities.

    Same layout as ``analysis.weights`` (see
    :func:`~src.agents.choice_logit_truth.generating_weights` for what these weights are).

    Raises:
        ValueError: If *synthetic* is not a cohort of value learners.
    """
    _require_value_agents(synthetic)
    tables = []
    for mouse, design in analysis.designs.items():
        trials = synthetic.trials[synthetic.trials["mouse"] == mouse]
        utilities, _ = true_utilities(trials, _agent_parameters(synthetic, mouse))
        beta = generating_weights(utilities, design.X, design.feature_names)
        tables.append(cl.weights_table(mouse, design.feature_names, beta))
    return pd.concat(tables, ignore_index=True)


def kernel_comparison(analysis: LogitAnalysis, synthetic: SyntheticCohort) -> pd.DataFrame:
    """Per mouse and reward lag: the Q-learner's weight bounds, the fitted and reference weights.

    Returns:
        pd.DataFrame: ``mouse, lag, lower, upper, fitted, reference``.
    """
    reference = reference_weights(analysis, synthetic)
    tables = []
    for mouse in analysis.designs:
        bounds = q_kernel_bounds(_agent_parameters(synthetic, mouse), analysis.n_lags)
        weights = {}
        for name, table in (("fitted", analysis.weights), ("reference", reference)):
            rows = table[(table["mouse"] == mouse) & table["feature"].str.match(r"reward_lag\d+$")]
            weights[name] = rows.set_index("lag")["weight"]
        tables.append(
            bounds.assign(
                mouse=mouse,
                fitted=bounds["lag"].map(weights["fitted"]).to_numpy(),
                reference=bounds["lag"].map(weights["reference"]).to_numpy(),
            )
        )
    columns = ["mouse", "lag", "lower", "upper", "fitted", "reference"]
    return pd.concat(tables, ignore_index=True)[columns]
