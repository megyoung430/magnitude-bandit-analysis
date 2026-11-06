"""The choice logit as a source of value estimates, for the mice and for simulated agents.

Three steps, each a plain function so a notebook can call them one at a time:

1. **Load** (:func:`load_drifting_trial_table`) the drifting problems of the chosen cohorts as one
   trial table, from the newest import-cache snapshot (nothing is read from or written to the
   data share), or :func:`simulate_cohort` a cohort of value-learning agents whose generating
   values and choice rule are known.
2. **Analyse** (:func:`analyse_trial_table`): per mouse, the design matrix
   (:mod:`src.behavior_analysis.choice_logit`), the fit to all of the mouse's trials (for the
   kernels), and leave-one-problem-out validation
   (:mod:`src.behavior_analysis.choice_logit_cv`), which also gives the held-out model values.
3. **Summarise** the held-out values as psychometric curves and stay/switch pairs against
   quantile bins of the value (:func:`model_value_curves`, :func:`model_value_pairs`).

Drawing is in :mod:`src.behavior_visualization.plot_choice_logit`.
"""

import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from src.agents import drifting_simulation as sim
from src.agents.choice_logit_truth import project_utilities, true_utilities
from src.behavior_analysis import choice_logit as cl
from src.behavior_analysis import choice_logit_cv as cv
from src.behavior_analysis import model_value_psychometrics as mvp
from src.behavior_analysis.drifting_trial_table import drifting_trial_table
from src.pipelines.all_cohorts_drifting import check_cohort_ids, drifting_problems, pool_cohorts
from src.pipelines.cohort_analysis import load_cohort

# A mouse with fewer trials than this is not analysed: the weights would mostly be the prior.
MIN_TRIALS = 300
# Towers of the simulated problems; each triple avoids the initiation port B2 and mixes
# plus-shaped ports (same row or column as B2) with the others.
SIMULATED_PORTS = (
    ("A1", "B3", "C2"),
    ("A3", "B1", "C3"),
    ("A2", "B3", "C1"),
    ("A1", "B1", "C3"),
    ("A3", "B3", "C1"),
)
SIMULATED_INITIATION = "B2"
# Magnitudes (pulses) of the three ports on a simulated problem's first trial.
START_MAGNITUDES = (4, 2, 0)


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
        :func:`~src.behavior_analysis.drifting_trial_table.drifting_trial_table`.
    """
    return drifting_trial_table(load_drifting_problems(cohort_ids, data_root, trials_by_cohort))


class SyntheticCohort(NamedTuple):
    """Simulated value-learning agents on the drifting task.

    Attributes:
        trials (pd.DataFrame): Their trial table.
        parameters (pd.DataFrame): ``mouse`` and the generating ``value_agent`` parameters.
        problems (dict): ``{problem: {mouse: sessions}}``, for the analyses that take sessions.
    """

    trials: pd.DataFrame
    parameters: pd.DataFrame
    problems: dict


def _layouts(n_problems: int, n_sessions: int, trials_per_session: int) -> dict:
    """The problems every simulated agent plays: other towers in each, B2 to initiate."""
    ports = [sorted(SIMULATED_PORTS[k % len(SIMULATED_PORTS)]) for k in range(n_problems)]
    return {
        k: sim.MouseLayout(
            p,
            SIMULATED_INITIATION,
            dict(zip(p, START_MAGNITUDES, strict=True)),
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
    agent has its own random stream, so adding agents does not change the others.

    Args:
        parameter_sets (Sequence[Sequence[float]]): ``(learning_rate, beta, kappa, plus_bias)``
            of each setting. ``kappa`` is the stickiness and ``plus_bias`` a known bias towards
            the ports in the row or column of the initiation port.
        mice_per_set (int): Agents per setting. Keyword-only.
        n_problems (int): Problems each agent plays, at most ``len(SIMULATED_PORTS)`` distinct
            tower triples before they repeat. Keyword-only.
        sessions_per_problem (int): Sessions per problem. Keyword-only.
        trials_per_session (int): Trials per session. Keyword-only.
        seed (int): Base seed. Keyword-only.
        drift_rate (float): Per-trial drift attempt probability of the task. Default 0.1.
        block_lengths (Sequence[int]): Block lengths, sampled with replacement. Default 30-35.

    Returns:
        SyntheticCohort: The simulated trials, the generating parameters and the sessions.
    """
    rules = sim.TaskRules(drift_rate, np.asarray(block_lengths))
    layouts = _layouts(n_problems, sessions_per_problem, trials_per_session)
    settings = [p for p in parameter_sets for _ in range(mice_per_set)]
    problems: dict = {}
    for i, parameters in enumerate(settings, start=1):
        mouse = f"sim{i:02d}"
        fit = sim.AgentFit(mouse, "magnitude", tuple(parameters), 0.0, 0)
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
        cv (dict[str, cv.MouseCV]): Leave-one-problem-out scores and held-out values.
        weights (pd.DataFrame): ``choice_logit.weights_table`` of the full fits, all mice.
        comparison (pd.DataFrame): ``choice_logit_cv.model_comparison`` of all mice.
        skipped (dict[str, str]): Mice not analysed and why.
        meta (pd.DataFrame): Trial metadata of all analysed mice, rows aligned with ``chosen``
            and ``values``.
        chosen (np.ndarray): ``(n_trials,)`` chosen arm index.
        values (np.ndarray): ``(n_trials, 3)`` held-out model values.
        n_lags (int): Lags of the design.
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
    n_lags: int
    l2: float


def skip_reason(trials: pd.DataFrame, min_trials: int) -> str | None:
    """Why one mouse's *trials* cannot be validated, or ``None`` if they can."""
    n_problems = trials["problem"].nunique()
    if n_problems < 2:
        return f"{n_problems} problem; leave-one-problem-out needs at least two"
    if len(trials) < min_trials:
        return f"{len(trials)} trials; at least {min_trials} are needed"
    return None


def analyse_mouse(trials: pd.DataFrame, n_lags: int, l2: float):
    """``(design, full-data fit, validation)`` of one mouse."""
    design = cl.design_matrix(trials, n_lags)
    fit = cl.fit_logit(design.X, design.chosen, design.feature_names, l2)
    return design, fit, cv.cross_validate_mouse(design, l2)


def analyse_trial_table(
    trial_table: pd.DataFrame,
    n_lags: int = cl.DEFAULT_LAGS,
    l2: float = cl.DEFAULT_L2,
    min_trials: int = MIN_TRIALS,
) -> LogitAnalysis:
    """Fit and validate the choice logit for every mouse of *trial_table*.

    Args:
        trial_table (pd.DataFrame): A drifting trial table.
        n_lags (int): Reward lags and choice lags. Default ``choice_logit.DEFAULT_LAGS``.
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
        m: analyse_mouse(rows, n_lags, l2) for m, rows in by_mouse.items() if m not in skipped
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
        n_lags=n_lags,
        l2=l2,
    )


# ---------------------------------------------------------------------------
# Value summaries
# ---------------------------------------------------------------------------


def value_rank_table(analysis: LogitAnalysis, values: np.ndarray | None = None) -> pd.DataFrame:
    """:func:`~src.behavior_analysis.model_value_psychometrics.rank_table` of the trials.

    Args:
        analysis (LogitAnalysis): The analysis.
        values (np.ndarray | None): ``(n_trials, 3)`` values to rank instead of the held-out
            model values, rows aligned with ``analysis.meta`` (e.g. :func:`generating_values`).
    """
    chosen_values = analysis.values if values is None else values
    return mvp.rank_table(analysis.meta["mouse"], chosen_values, analysis.chosen)


def model_value_curves(
    analysis: LogitAnalysis, n_bins: int = mvp.DEFAULT_BINS, values: np.ndarray | None = None
) -> dict:
    """P(best) and P(second) against quantile bins of the (model) value, see
    :func:`~src.behavior_analysis.model_value_psychometrics.binned_psychometric_curves`."""
    return mvp.binned_psychometric_curves(value_rank_table(analysis, values), n_bins)


def model_value_pairs(
    analysis: LogitAnalysis, n_bins: int = mvp.DEFAULT_BINS, values: np.ndarray | None = None
) -> pd.DataFrame:
    """Stay/switch trial pairs on the (model) value, see
    :func:`~src.behavior_analysis.model_value_psychometrics.stay_switch_pairs_from_values`."""
    chosen_values = analysis.values if values is None else values
    return mvp.stay_switch_pairs_from_values(analysis.meta, chosen_values, analysis.chosen, n_bins)


# ---------------------------------------------------------------------------
# Simulated agents: what the generating rule implies
# ---------------------------------------------------------------------------


def _agent_parameters(synthetic: SyntheticCohort, mouse: str) -> tuple:
    row = synthetic.parameters.set_index("mouse").loc[mouse]
    return tuple(float(x) for x in row)


def generating_values(analysis: LogitAnalysis, synthetic: SyntheticCohort) -> np.ndarray:
    """The agents' own Q-values before each choice, rows aligned with ``analysis.meta``."""
    return np.concatenate(
        [
            true_utilities(
                synthetic.trials[synthetic.trials["mouse"] == m], _agent_parameters(synthetic, m)
            )[1]
            for m in analysis.designs
        ]
    )


def reference_weights(analysis: LogitAnalysis, synthetic: SyntheticCohort) -> pd.DataFrame:
    """The best linear summary of each agent's generating utility in the logit's features.

    Same layout as ``analysis.weights`` (see
    :func:`~src.agents.choice_logit_truth.project_utilities` for what these weights are).
    """
    tables = []
    for mouse, design in analysis.designs.items():
        trials = synthetic.trials[synthetic.trials["mouse"] == mouse]
        utilities, _ = true_utilities(trials, _agent_parameters(synthetic, mouse))
        beta = project_utilities(utilities, design.X)
        tables.append(cl.weights_table(mouse, design.feature_names, beta))
    return pd.concat(tables, ignore_index=True)
