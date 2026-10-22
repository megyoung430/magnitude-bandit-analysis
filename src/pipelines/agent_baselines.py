"""Baseline agents and the choice regression for the pooled drifting problems.

:func:`run_agent_baselines` answers two questions about the drifting-task figures:

1. *Does magnitude drive choice, beyond avoiding the empty port, repeating the last choice and
   preferring plus ports?* The conditional-logit regression of
   :mod:`src.behavior_analysis.choice_regression`, fitted per mouse.
2. *What would the existing figures look like for an animal that does, or does not, learn from
   magnitude?* Two agents (:mod:`src.agents.drifting_simulation`) are fitted to each mouse, and
   each fitted agent plays that mouse's problems in a task that follows the rules measured from
   the data. The simulated animals then go through the same rank-proportion, psychometric,
   block-change, value-change and port-preference code as the mice, and through the same
   regression.

Outputs, under ``<results_root>/figures/open-field/agent-baselines/cohorts-<ids>/`` and
``<results_root>/tables/open-field/agent-baselines/cohorts-<ids>/``::

    figures: summary.{png,pdf}                 regression weights (mice and both agents) and
                                               the per-mouse likelihood comparison
             {magnitude,binary}-agent/         the pooled drifting figures of each agent, laid
                 choice-stats/ value-psychometrics/ reversal-aligned-arms/ value-change/
                 port-preference/
    tables:  task_rules_block_changes.csv      every block change (which arm was zeroed, ...)
             task_rules_summary.csv            per mouse and overall, plus drift rates
             agent_fits.csv                    fitted parameters and likelihoods
             choice_regression_weights.csv     per animal and feature, all sources
             choice_regression_summary.csv     mean, SE, t, p per source and feature

The mice's own pooled figures are the ones ``run_all_cohorts_drifting`` already writes; compare
them with the agent folders side by side.
"""

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from joblib import Memory, Parallel, delayed

from src.agents import drifting_simulation as sim
from src.behavior_analysis import choice_regression, drifting_task_rules
from src.behavior_analysis.drifting_trial_table import drifting_trial_table
from src.behavior_visualization.plot_agent_baselines import plot_agent_baselines_summary
from src.pipelines.all_cohorts_drifting import (
    check_cohort_ids,
    cohorts_phrase,
    drifting_problems,
    pool_cohorts,
)
from src.pipelines.cohort_analysis import (
    DEFAULT_RESULTS_ROOT,
    DRIFTING_SELECTION,
    TASK,
    load_cohort,
    plot_pooled_drifting_rank,
    pooled_drifting_rank_results,
)
from src.pipelines.open_field import AnalysisSelection, run_drifting_value_analyses
from src.settings import get_settings

AGENT_BASELINES_FOLDER = "agent-baselines"
RANK_FOLDER = "choice-stats"
SOURCES = ("mice", *sim.AGENT_KINDS)


class TaskRuleTables(NamedTuple):
    """The measured task rules and the simulated task built from them.

    Attributes:
        rules (sim.TaskRules): Drift attempt rate and block lengths used in the simulations.
        block_changes (pd.DataFrame): Table of ``drifting_task_rules.block_changes``.
        summary (pd.DataFrame): ``block_change_summary`` plus columns ``observed_drift_rate``
            and ``simulated_drift_attempt_rate`` (the same value on every row).
    """

    rules: sim.TaskRules
    block_changes: pd.DataFrame
    summary: pd.DataFrame


class AgentBaselines(NamedTuple):
    """Everything :func:`run_agent_baselines` computed before writing.

    Attributes:
        task: TaskRuleTables of the mice.
        fits (pd.DataFrame): Table of ``drifting_simulation.fits_table``.
        simulated (dict): ``{kind: {problem: {simulated animal: sessions}}}``.
        weights (pd.DataFrame): Regression weights with a ``source`` column.
        summary (pd.DataFrame): Group regression summary with a ``source`` column.
    """

    task: TaskRuleTables
    fits: pd.DataFrame
    simulated: dict
    weights: pd.DataFrame
    summary: pd.DataFrame


# ---------------------------------------------------------------------------
# Analysis (pure, apart from the optional fit cache)
# ---------------------------------------------------------------------------


def measure_task_rules(trial_table: pd.DataFrame, seed: int = 0) -> TaskRuleTables:
    """Block changes, drift and block lengths of the mice's task, and the simulated task.

    Args:
        trial_table (pd.DataFrame): The mice's drifting trial table.
        seed (int): Seed of the drift-rate calibration. Default 0.

    Returns:
        TaskRuleTables: The rules and the tables documenting them.

    Raises:
        ValueError: If no complete block was observed (block lengths cannot be sampled).
    """
    changes = drifting_task_rules.block_changes(trial_table)
    lengths = drifting_task_rules.complete_block_lengths(changes)
    if len(lengths) == 0:
        raise ValueError("no complete block observed; cannot simulate block lengths")
    observed = drifting_task_rules.drift_counts(trial_table).rate
    attempt = sim.calibrate_drift_rate(observed, seed=seed)
    summary = drifting_task_rules.block_change_summary(changes).assign(
        observed_drift_rate=observed, simulated_drift_attempt_rate=attempt
    )
    return TaskRuleTables(sim.TaskRules(attempt, lengths), changes, summary)


def fit_agents(
    trial_table: pd.DataFrame,
    *,
    n_starts: int,
    seed: int,
    n_jobs: int = 1,
    cache_dir: Path | None = None,
) -> list[sim.AgentFit]:
    """Both agents fitted to every mouse, in parallel over mice and agents.

    Args:
        trial_table (pd.DataFrame): The mice's drifting trial table.
        n_starts (int): Optimiser starts per fit. Keyword-only.
        seed (int): Seed of the starting points (the same for every fit). Keyword-only.
        n_jobs (int): Parallel workers (joblib). Default 1. Keyword-only.
        cache_dir (Path | None): Cache the fits here, keyed on the mouse's trials, the agent
            and the settings. Default ``None``: no cache. Keyword-only.

    Returns:
        list[sim.AgentFit]: One fit per mouse and agent kind, mice in sorted order.
    """
    fit = Memory(cache_dir, verbose=0).cache(sim.fit_agent) if cache_dir else sim.fit_agent
    jobs = [
        delayed(fit)(mouse, rows.reset_index(drop=True), kind, n_starts, seed)
        for mouse, rows in sorted(trial_table.groupby("mouse"), key=lambda item: item[0])
        for kind in sim.AGENT_KINDS
    ]
    return Parallel(n_jobs=n_jobs)(jobs)


def simulate_agents(
    trial_table: pd.DataFrame, fits: list[sim.AgentFit], rules: sim.TaskRules, seed: int
) -> dict:
    """Each fitted agent plays its mouse's problems once.

    Every mouse and agent kind gets its own random stream (seeded from *seed*, the mouse's
    position and the kind), so adding a mouse does not change the others' simulations.

    Args:
        trial_table (pd.DataFrame): The mice's drifting trial table.
        fits (list[sim.AgentFit]): Output of :func:`fit_agents`.
        rules (sim.TaskRules): The simulated task.
        seed (int): Base seed.

    Returns:
        dict: ``{kind: {problem: {"<mouse>-<kind>": sessions}}}``, problems keyed as in
        *trial_table*.
    """
    mice = sorted(trial_table["mouse"].unique())
    simulated: dict = {kind: {} for kind in sim.AGENT_KINDS}
    for f in fits:
        rng = np.random.default_rng([seed, mice.index(f.mouse), sim.AGENT_KINDS.index(f.kind)])
        layouts = sim.mouse_layouts(trial_table[trial_table["mouse"] == f.mouse])
        played = sim.simulate_mouse(f, layouts, rules, rng, f"{f.mouse}-{f.kind}")
        for problem, animals in played.items():
            simulated[f.kind].setdefault(problem, {}).update(animals)
    return simulated


def regression_by_source(
    trial_tables: dict, min_trials: int = 200
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The choice regression for each source (mice and each agent).

    Args:
        trial_tables (dict): ``{source: drifting trial table}``.
        min_trials (int): Animals with fewer trials are left out. Default 200.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: ``(weights, summary)``: the outputs of
        ``fit_by_mouse`` and ``group_summary`` of every source, stacked, with a ``source``
        column first.
    """
    weights, summaries = [], []
    for source, table in trial_tables.items():
        design = choice_regression.choice_design(table)
        w = choice_regression.fit_by_mouse(design, min_trials=min_trials)
        weights.append(w.assign(source=source))
        summaries.append(choice_regression.group_summary(w).assign(source=source))

    def stack(frames):
        out = pd.concat(frames, ignore_index=True)
        return out[["source", *[c for c in out.columns if c != "source"]]]

    return stack(weights), stack(summaries)


def compute_agent_baselines(
    problems: dict,
    *,
    n_starts: int = 5,
    seed: int = 0,
    n_jobs: int = 1,
    cache_dir: Path | None = None,
    min_trials: int = 200,
) -> AgentBaselines:
    """Measure the task, fit and simulate both agents, and run the regression on all sources.

    Args:
        problems (dict): The mice's drifting problems, ``{problem: {mouse: sessions}}`` (e.g.
            ``pool_cohorts(...).problems``).
        n_starts (int): Optimiser starts per agent fit. Default 5. Keyword-only.
        seed (int): Seed of the fits' starting points and of the simulations. Default 0.
            Keyword-only.
        n_jobs (int): Parallel workers for the fits. Default 1. Keyword-only.
        cache_dir (Path | None): Fit cache folder. Default ``None``: no cache. Keyword-only.
        min_trials (int): Animals with fewer trials are left out of the regression. Default
            200. Keyword-only.

    Returns:
        AgentBaselines: The results (nothing is written).
    """
    mice_table = drifting_trial_table(problems)
    task = measure_task_rules(mice_table, seed)
    fits = fit_agents(mice_table, n_starts=n_starts, seed=seed, n_jobs=n_jobs, cache_dir=cache_dir)
    simulated = simulate_agents(mice_table, fits, task.rules, seed)
    tables = {"mice": mice_table}
    tables.update({kind: drifting_trial_table(simulated[kind]) for kind in sim.AGENT_KINDS})
    weights, summary = regression_by_source(tables, min_trials)
    return AgentBaselines(task, sim.fits_table(fits), simulated, weights, summary)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def write_agent_baselines(
    result: AgentBaselines,
    figures_dir: Path,
    tables_dir: Path,
    selection: AnalysisSelection,
    scope_label: str,
) -> None:
    """Write the tables, the summary figure and each agent's pooled drifting figures.

    Args:
        result (AgentBaselines): Output of :func:`compute_agent_baselines`.
        figures_dir (Path): Receives ``summary.{png,pdf}`` and one folder per agent.
        tables_dir (Path): Receives the CSV tables.
        selection (AnalysisSelection): Which drifting analyses to draw for the agents, and the
            rank value bases and pooling modes.
        scope_label (str): Scope named in the figure titles, e.g. ``"magnitude agent, cohorts
            1, 3"``; the agent's name is added in front.
    """
    tables_dir = Path(tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    result.task.block_changes.to_csv(tables_dir / "task_rules_block_changes.csv", index=False)
    result.task.summary.to_csv(tables_dir / "task_rules_summary.csv", index=False)
    result.fits.to_csv(tables_dir / "agent_fits.csv", index=False)
    result.weights.to_csv(tables_dir / "choice_regression_weights.csv", index=False)
    result.summary.to_csv(tables_dir / "choice_regression_summary.csv", index=False)

    figures_dir = Path(figures_dir)
    plot_agent_baselines_summary(result.weights, result.fits, figures_dir / "summary")
    for kind, problems in result.simulated.items():
        out = figures_dir / f"{kind}-agent"
        label = f"{kind} agent, {scope_label}"
        rank = pooled_drifting_rank_results(
            problems, selection.rank_value_bases, selection.rank_pooling_modes
        )
        plot_pooled_drifting_rank(rank, out / RANK_FOLDER, mouse_legend=False)
        agent_tables = run_drifting_value_analyses(
            problems,
            out,
            selection,
            arm_scope_label=label,
            x_vs_plus=True,
            per_mouse_figures=False,
            mouse_legends=False,
        )
        for name, table in agent_tables.items():
            table.to_csv(tables_dir / f"{kind}_agent_{name}.csv", index=False)


def cohorts_folder_name(cohort_ids) -> str:
    """``"cohorts-1-3"`` for cohorts 1 and 3: the output folder of one pooled set."""
    return "cohorts-" + "-".join(str(c) for c in sorted(cohort_ids))


def run_agent_baselines(
    cohort_ids,
    *,
    data_root: Path | None = None,
    results_root: Path = DEFAULT_RESULTS_ROOT,
    selection: AnalysisSelection = DRIFTING_SELECTION,
    n_starts: int = 5,
    seed: int = 0,
    n_jobs: int = 1,
    use_cache: bool = True,
    trials_by_cohort: dict | None = None,
) -> AgentBaselines:
    """Load and pool the cohorts' drifting problems, then compute and write the baselines.

    Figures go to ``<results_root>/figures/open-field/agent-baselines/cohorts-<ids>/`` and tables
    to the matching ``tables/`` folder (see the module docstring for their contents).

    Args:
        cohort_ids (Iterable[int]): Cohorts to pool, e.g. ``[1, 3]``.
        data_root (Path | None): Folder holding ``3x3_field_magnitude_bandit``. Default
            ``None``: the ``BANDIT_DATA_ROOT`` setting. Keyword-only.
        results_root (Path): Folder receiving ``figures/`` and ``tables/``. Keyword-only.
        selection (AnalysisSelection): Drifting analyses drawn for the agents. Keyword-only.
        n_starts (int): Optimiser starts per agent fit. Default 5. Keyword-only.
        seed (int): Seed of the fits and simulations. Default 0. Keyword-only.
        n_jobs (int): Parallel workers for the fits. Default 1. Keyword-only.
        use_cache (bool): Use the import cache, and cache the agent fits next to it. Default
            ``True``. Keyword-only.
        trials_by_cohort (dict | None): Already-loaded ``{cohort id: {overall problem: {mouse:
            sessions}}}``. Default ``None``: load the cohorts. Keyword-only.

    Returns:
        AgentBaselines: What was computed (the files are already written).
    """
    ids = check_cohort_ids(cohort_ids)
    if trials_by_cohort is None:
        trials_by_cohort = {c: load_cohort(c, data_root, use_cache).trials_by_problem for c in ids}
    pooled = pool_cohorts({c: drifting_problems(trials_by_cohort[c]) for c in ids})
    cache_dir = get_settings().cache_dir / "agent_baselines" if use_cache else None
    result = compute_agent_baselines(
        pooled.problems, n_starts=n_starts, seed=seed, n_jobs=n_jobs, cache_dir=cache_dir
    )
    folder = Path(TASK) / AGENT_BASELINES_FOLDER / cohorts_folder_name(ids)
    write_agent_baselines(
        result,
        Path(results_root) / "figures" / folder,
        Path(results_root) / "tables" / folder,
        selection,
        cohorts_phrase(ids),
    )
    return result
