"""The full open-field analysis suite for one or several cohorts, with one call.

:func:`run_cohorts` (or :func:`run_cohort` for one) loads a cohort's raw data, sorts every
mouse's problems into blocked and drifting from the pyControl task name
(:mod:`src.behavior_import.task_types`), numbers them within their type, and runs

- blocked problems: choice probabilities, first leave, number of reversals, true-value rank
  proportions and single sessions;
- drifting problems: number of reversals, true- and believed-value rank proportions and single
  sessions, plus rank proportions pooled over all of a mouse's drifting problems;
- trial and block-length summaries over all problems;
- a problem map and a review table of suspect reversals, as CSV.

Layout under ``results_root`` (``cohort`` is ``cohort-0N``)::

    figures/open-field/<cohort>/{blocked,drifting}/problem-KK/...
    figures/open-field/<cohort>/drifting/all-drifting-problems/choice-stats/
    figures/open-field/<cohort>/trial-block-summaries/all-problems/
    tables/open-field/<cohort>/problem_map.csv
    tables/open-field/<cohort>/suspect_reversals.csv

``KK`` counts a mouse's problems within the type, so drifting problem 1 is the first drifting
problem of every mouse; ``problem_map.csv`` gives the overall problem number it stands for.
"""

from pathlib import Path
from typing import NamedTuple

import pandas as pd

from src.behavior_analysis.get_diagnostic_p_value import pvalue_paired_t_best_vs_second_vs_third
from src.behavior_analysis.get_good_reversal_info import get_good_reversal_info
from src.behavior_analysis.get_rank_counts_by_good_reversal import get_rank_counts_by_value
from src.behavior_analysis.merge_across_problems import merge_windows_by_subject
from src.behavior_analysis.suspect_reversal_table import (
    COLUMNS as SUSPECT_COLUMNS,
)
from src.behavior_analysis.suspect_reversal_table import (
    suspect_reversal_table_by_problem,
    write_suspect_reversal_table,
)
from src.behavior_analysis.trial_block_summary import (
    build_trial_block_tables,
    subject_numbers_from_paths,
)
from src.behavior_import.import_data import session_source_paths
from src.behavior_import.load_trials import load_trials_by_problem
from src.behavior_import.task_types import (
    BLOCKED,
    DRIFTING,
    problem_map_table,
    regroup_by_task_type,
)
from src.behavior_visualization.plot_rank_proportions import plot_rank_proportions
from src.behavior_visualization.plot_trial_block_summary import plot_trial_block_summaries
from src.pipelines.open_field import (
    VALUE_BASIS_LABELS,
    AnalysisSelection,
    ChoiceProbabilitySettings,
    run_problem,
)
from src.settings import REPO_ROOT, get_settings

TASK = "open-field"
EXPERIMENT_FOLDER = "3x3_field_magnitude_bandit"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results"
REVERSAL_TRIAL_WINDOW = 100  # trials per bin of the trial-based reversal-rate plots
POOLED_DRIFTING_FOLDER = "all-drifting-problems"

BLOCKED_SELECTION = AnalysisSelection(
    run_choice_probability_analysis=True,
    run_first_leave_analysis=True,
    run_number_of_reversals_analysis=True,
    run_rank_proportion_analysis=True,
    run_single_session_analysis=True,
    rank_pooling_modes=(True, False),
    rank_value_bases=("true",),
)
# Choice probabilities and first leave look at the reversals of a blocked task; a drifting
# task has none to look at, so only the reversal counts and the rank proportions apply.
DRIFTING_SELECTION = AnalysisSelection(
    run_choice_probability_analysis=False,
    run_first_leave_analysis=False,
    run_number_of_reversals_analysis=True,
    run_rank_proportion_analysis=True,
    run_single_session_analysis=True,
    rank_pooling_modes=(True, False),
    rank_value_bases=("true", "believed"),
)
DEFAULT_CHOICE = ChoiceProbabilitySettings(
    run_all=True,
    skip=True,
    moving_avg=True,
    remove_bad=True,
    split_by_best_change=False,
    split_by_best_change_and_half=False,
    split_by_first_two=True,
    split_by_half=True,
)


class CohortResult(NamedTuple):
    """What :func:`run_cohort` computed (the files are already written)."""

    problem_map: pd.DataFrame
    suspect_reversals: pd.DataFrame
    rank_results: dict  # {(task_type, k): run_problem's rank results}
    pooled_drifting_ranks: dict  # {value_basis: {pool_trials: {...}}}


# ---------------------------------------------------------------------------
# Names, paths and settings
# ---------------------------------------------------------------------------


def cohort_name(cohort_id: int) -> str:
    """``3`` as ``"cohort-03"``."""
    if not isinstance(cohort_id, int) or cohort_id < 1:
        raise ValueError(f"cohort id must be a positive integer, got {cohort_id!r}")
    return f"cohort-{cohort_id:02d}"


def cohort_raw_dir(data_root: Path, cohort_id: int) -> Path:
    """``<data_root>/3x3_field_magnitude_bandit/rawdata/cohort-0N``."""
    return Path(data_root) / EXPERIMENT_FOLDER / "rawdata" / cohort_name(cohort_id)


def selection_for_task_type(
    task_type: str,
    blocked: AnalysisSelection = BLOCKED_SELECTION,
    drifting: AnalysisSelection = DRIFTING_SELECTION,
) -> AnalysisSelection:
    """The analysis selection that applies to *task_type*."""
    if task_type == BLOCKED:
        return blocked
    if task_type == DRIFTING:
        return drifting
    raise ValueError(f"unknown task type {task_type!r}; expected {BLOCKED!r} or {DRIFTING!r}")


def _figures_dir(results_root: Path, cohort: str) -> Path:
    return Path(results_root) / "figures" / TASK / cohort


def _tables_dir(results_root: Path, cohort: str) -> Path:
    return Path(results_root) / "tables" / TASK / cohort


# ---------------------------------------------------------------------------
# Analysis (pure)
# ---------------------------------------------------------------------------


def subject_numbers_from_sessions(subjects_trials_by_problem: dict) -> dict:
    """``{mouse: subject number}`` from the file paths the sessions were imported from.

    Raises:
        ValueError: If a mouse has no session with a recorded file path.
    """
    paths_by_mouse: dict[str, list[str]] = {}
    for subjects in subjects_trials_by_problem.values():
        for mouse, sessions in subjects.items():
            paths = paths_by_mouse.setdefault(mouse, [])
            paths.extend(path for s in sessions.values() for path in session_source_paths(s))
    numbers = {}
    for mouse, paths in paths_by_mouse.items():
        found = subject_numbers_from_paths(paths)
        if mouse not in found:
            raise ValueError(f"no subject number (sub-NN_id-{mouse}) in the file paths of {mouse}")
        numbers[mouse] = found[mouse]
    return numbers


def suspect_table_with_task_types(grouped: dict, problem_map: pd.DataFrame) -> pd.DataFrame:
    """Suspect-reversal rows of every problem, with where each sits in its mouse's numbering.

    Args:
        grouped: ``{(task_type, k): {subject: sessions}}`` from
            :func:`~src.behavior_import.task_types.regroup_by_task_type`.
        problem_map: The cohort's :func:`~src.behavior_import.task_types.problem_map_table`,
            which supplies each mouse's overall problem number.

    Returns:
        The columns ``task_type``, ``problem_in_type``, ``overall_problem``, then
        :data:`~src.behavior_analysis.suspect_reversal_table.COLUMNS`.
    """
    leading = ["task_type", "problem_in_type", "overall_problem"]
    frames = []
    for task_type in (BLOCKED, DRIFTING):
        by_k = {k: subs for (kind, k), subs in grouped.items() if kind == task_type}
        table = suspect_reversal_table_by_problem(by_k)
        frames.append(
            table.rename(columns={"problem": "problem_in_type"}).assign(task_type=task_type)
        )
    stacked = pd.concat([f for f in frames if len(f)] or frames[:1], ignore_index=True)
    overall = problem_map[["mouse", "task_type", "problem_in_type", "overall_problem"]]
    merged = stacked.merge(
        overall,
        how="left",
        left_on=["subject", "task_type", "problem_in_type"],
        right_on=["mouse", "task_type", "problem_in_type"],
    )
    return merged[[*leading, *SUSPECT_COLUMNS]]


def pooled_drifting_rank_results(
    drifting_problems: dict, value_bases: tuple[str, ...], pooling_modes: tuple[bool, ...]
) -> dict:
    """Rank proportions pooled over a mouse's drifting problems.

    Ranks are computed inside each problem before the counts are combined, so a mouse's
    beliefs about the arms reset at every problem boundary.

    Args:
        drifting_problems: ``{k: {subject: sessions}}``.
        value_bases: ``"true"`` and/or ``"believed"``.
        pooling_modes: ``pool_trials_within_subject`` values. ``True`` sums counts within each
            mouse, ``False`` gives equal weight to each block; mice stay the replicates.

    Returns:
        ``{value_basis: {pool_trials: {"problems", "counts", "p_values"}}}``; a basis with no
        rank windows in any problem is left out.
    """
    windows = {
        k: get_good_reversal_info(trials, include_first_block=True)
        for k, trials in drifting_problems.items()
    }
    results = {}
    for value_basis in value_bases:
        counts_by_problem = {
            k: get_rank_counts_by_value(trials, windows[k], value_basis=value_basis)
            for k, trials in drifting_problems.items()
        }
        pooled = merge_windows_by_subject(counts_by_problem)
        if not pooled:
            print(f"[INFO] No rank windows available for {value_basis}; skipping.")
            continue
        results[value_basis] = {
            pool_trials: {
                "problems": list(drifting_problems),
                "counts": pooled,
                "p_values": pvalue_paired_t_best_vs_second_vs_third(
                    pooled, pool_trials_within_subject=pool_trials
                ),
            }
            for pool_trials in pooling_modes
        }
    return results


# ---------------------------------------------------------------------------
# Plotting and writing (the edges)
# ---------------------------------------------------------------------------


def plot_pooled_drifting_rank(results: dict, save_dir: Path) -> None:
    """One rank-proportion figure per value basis and pooling mode of *results*."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for value_basis, by_mode in results.items():
        label = VALUE_BASIS_LABELS[value_basis]
        for pool_trials, result in by_mode.items():
            plot_rank_proportions(
                result["counts"],
                average_across_mice_pvalues=result["p_values"],
                pool_trials_within_subject=pool_trials,
                # plot_rank_proportions appends suffixes to a str save_path
                save_path=str(save_dir / f"Rank Proportions ({label}) (Pool Trials {pool_trials})"),
            )


def _write_problem_map(problem_map: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    problem_map.to_csv(path, index=False)


def _load(cohort_id: int, data_root, use_cache: bool) -> dict:
    root = Path(data_root) if data_root is not None else get_settings().data_root
    return load_trials_by_problem(cohort_raw_dir(root, cohort_id), use_cache=use_cache)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def run_cohort(
    cohort_id: int,
    *,
    data_root: Path | None = None,
    results_root: Path = DEFAULT_RESULTS_ROOT,
    blocked_selection: AnalysisSelection = BLOCKED_SELECTION,
    drifting_selection: AnalysisSelection = DRIFTING_SELECTION,
    choice: ChoiceProbabilitySettings = DEFAULT_CHOICE,
    reversal_trial_window: int = REVERSAL_TRIAL_WINDOW,
    use_cache: bool = True,
    trials_by_problem: dict | None = None,
) -> CohortResult:
    """Run the whole analysis suite for one cohort; see the module docstring for the outputs.

    Args:
        cohort_id: 1 for ``cohort-01``, and so on.
        data_root: Folder holding ``3x3_field_magnitude_bandit`` (default: the
            ``BANDIT_DATA_ROOT`` setting).
        results_root: Folder that receives ``figures/`` and ``tables/``.
        blocked_selection, drifting_selection: Which analyses run for each task type.
        choice: Choice-probability figure toggles and windows.
        reversal_trial_window: Trials per bin of the trial-based reversal-rate plots.
        use_cache: Use the on-disk import cache (keyed on the raw files).
        trials_by_problem: Already-loaded ``{problem: {subject: sessions}}``, to skip the load.

    Raises:
        ValueError: If the cohort has no problems, or a session's task name is not recognised.
    """
    cohort = cohort_name(cohort_id)
    if trials_by_problem is None:
        trials_by_problem = _load(cohort_id, data_root, use_cache)
    if not trials_by_problem:
        raise ValueError(f"No problems found for {cohort}")

    # Classify first: a bad task name stops the run before anything is written.
    grouped = regroup_by_task_type(trials_by_problem)
    problem_map = problem_map_table(trials_by_problem)
    figures, tables = _figures_dir(results_root, cohort), _tables_dir(results_root, cohort)

    _write_problem_map(problem_map, tables / "problem_map.csv")
    suspects = suspect_table_with_task_types(grouped, problem_map)
    write_suspect_reversal_table(suspects, tables / "suspect_reversals.csv")

    rank_results = {}
    for (task_type, k), subjects_trials in grouped.items():
        selection = selection_for_task_type(task_type, blocked_selection, drifting_selection)
        print(f"[INFO] {cohort} {task_type} problem {k}: mice {sorted(subjects_trials)}")
        rank_results[(task_type, k)] = run_problem(
            k,
            subjects_trials,
            task=TASK,
            cohort=cohort,
            output_root=Path(results_root) / "figures",
            selection=selection,
            choice=choice,
            reversal_trial_window=reversal_trial_window,
            task_type=task_type,
        )

    drifting = {k: subs for (kind, k), subs in grouped.items() if kind == DRIFTING}
    pooled = {}
    if drifting:
        pooled = pooled_drifting_rank_results(
            drifting, drifting_selection.rank_value_bases, drifting_selection.rank_pooling_modes
        )
        plot_pooled_drifting_rank(
            pooled, figures / DRIFTING / POOLED_DRIFTING_FOLDER / "choice-stats"
        )
    else:
        print(f"[INFO] {cohort}: no drifting problems; skipping pooled drifting ranks.")

    session_table, block_table = build_trial_block_tables(
        trials_by_problem, subject_numbers_from_sessions(trials_by_problem)
    )
    plot_trial_block_summaries(
        session_table,
        block_table,
        figures / "trial-block-summaries" / "all-problems",
        show=False,
    )
    return CohortResult(problem_map, suspects, rank_results, pooled)


def run_cohorts(cohort_ids, *, continue_on_error: bool = False, **settings) -> dict:
    """:func:`run_cohort` for each id; ``settings`` are passed on to every cohort.

    By default the first failing cohort stops the run. With ``continue_on_error`` the other
    cohorts still run, and the failures are raised together at the end.

    Returns:
        ``{cohort_id: CohortResult}``.

    Raises:
        RuntimeError: Naming each cohort that failed and why.
    """
    results, failures = {}, {}
    for cohort_id in cohort_ids:
        try:
            results[cohort_id] = run_cohort(cohort_id, **settings)
        except Exception as err:  # re-raised below with the cohort named, never dropped
            message = f"{cohort_name(cohort_id)} failed: {type(err).__name__}: {err}"
            if not continue_on_error:
                raise RuntimeError(message) from err
            print(f"[ERROR] {message}")
            failures[cohort_id] = message
    if failures:
        raise RuntimeError(f"{len(failures)} cohort(s) failed:\n" + "\n".join(failures.values()))
    return results
