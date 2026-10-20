"""The full open-field analysis suite for one or several cohorts, with one call.

:func:`run_cohorts` (or :func:`run_cohort` for one) loads a cohort's raw data, sorts every
mouse's problems into blocked and drifting from the pyControl task name
(:mod:`src.behavior_import.task_types`), numbers them within their type, and runs

- blocked problems: choice probabilities, first leave, number of reversals, true-value rank
  proportions, long pokes with maze heatmaps, and single sessions;
- drifting problems: number of reversals, true- and believed-value rank proportions, long pokes
  with maze heatmaps, value heatmaps and psychometric curves, reversal-aligned arm choices,
  value-change choice probabilities, port preference and single sessions, plus the rank
  proportions and the value analyses and port preference pooled over all of a mouse's drifting
  problems;
- trial and block-length summaries over all problems;
- the review tables, as CSV (below).

Layout under ``results_root`` (``cohort`` is ``cohort-0N``)::

    figures/open-field/<cohort>/{blocked,drifting}/problem-KK/{reversal-stats,choice-stats,
        single-sessions,long-pokes}/
    figures/open-field/<cohort>/drifting/problem-KK/{value-psychometrics,reversal-aligned-arms,
        value-change,port-preference}/
    figures/open-field/<cohort>/drifting/all-drifting-problems/{choice-stats,value-psychometrics,
        reversal-aligned-arms,value-change,port-preference,sessions-per-problem}/
    figures/open-field/<cohort>/trial-block-summaries/all-problems/
    (reversal-aligned-arms/ and value-change/ each hold the subfolders mouse-weighted/ and
        event-weighted/, with the figures inside)
    tables/open-field/<cohort>/*.csv

``KK`` counts a mouse's problems within the type, so drifting problem 1 is the first drifting
problem of every mouse; ``problem_map.csv`` gives the overall problem number it stands for.

The tables are for looking at by hand; each row names the mouse, the session (and the file where
it has one), the task type and the problem (number within the type, and overall):

- ``problem_map.csv``: one row per mouse and problem, with its towers and session range;
- ``sessions_per_problem.csv``: one row per mouse and drifting problem with the number of sessions
  it took (see :mod:`src.behavior_analysis.sessions_per_problem`); written from the whole cohort
  on every run, also a partial one, and drawn into ``sessions-per-problem/`` with the pooled
  drifting analyses;
- ``suspect_reversals.csv``: the good reversals and block changes the analyses drop;
- ``suspect_sessions.csv``: sessions the problem numbering flagged (a one-session layout in
  "A A B A" is treated as a settings error), and configuration settings that changed within a
  session;
- ``suspect_starting_reward_mags.csv``: sessions that start on other reward magnitudes than the
  problem's previous session (the last one with trials) ended on, with the most likely cause;
- ``no_trial_sessions.csv``: files and recordings the importer skipped, and sessions without
  trials (one row per session, listing its files), with the reason;
- ``no_second_best.csv``: blocks in which more than one arm is at 0;
- ``poke_trial_summary.csv``, ``port_finding_check.csv``, ``reversal_aligned_arms_counts.csv``,
  ``value_change_events.csv``, ``value_change_bin_counts.csv`` and
  ``value_change_symmetric_exclusions.csv``: the diagnostics the analyses return, over all
  problems (the pooled drifting analyses repeat the per-problem ones and add none).
  ``poke_trial_summary.csv`` has a row for every port-finding-only session whose trials the
  port-finding maps count (``port_finding_only`` true), and ``port_finding_check.csv`` flags a
  mouse and problem with fewer port-finding trials than ``n_ports x found_t_thresh``;
- ``warnings_log.csv``: every distinct Python warning of the latest run (that run only, also
  after a partial one), with the mouse, session, task type and problem its text names. It is
  written even when the run fails, and every table is always written (header only when empty).

Partial runs: ``run_cohort(..., only_problems={("drifting", 2)})``, or
:func:`run_cohort_incremental` / :func:`run_cohorts` with ``last_sessions=N`` or
``since="YYYY-MM-DD"``, regenerate only the problems that contain the selected sessions (a
problem folder holds every mouse that ran it) and leave the other figure folders alone. The pooled
``all-drifting-problems/`` analyses are redone only when a selected problem is drifting; the
review tables and ``trial-block-summaries/`` cover the cohort and are always rewritten. In the
stacked diagnostic tables the rows of the regenerated problems replace the earlier ones and
the other rows stay; if a table is missing, a warning says that only the new rows were written.
``warnings_log.csv`` lists only the warnings of the latest run, partial or not.

Port finding can span sessions. A session that holds only port finding has no bandit trials, so
it is not part of any problem, but its port-finding trials belong to the problem of the mouse's
next session with bandit trials when that session has the same towers; the port-finding maps of
that problem count them (the import log carries these sessions; without a log there are none).

The maze heatmaps outline the choice ports of the problem the mouse ran immediately before, in
the overall problem order and whatever its task type, so drifting problem 1 is outlined with
the ports of the last blocked problem.
"""

import warnings
from collections.abc import Collection, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from src.behavior_analysis.diagnostic_tables import stack_diagnostics
from src.behavior_analysis.get_diagnostic_p_value import pvalue_paired_t_best_vs_second_vs_third
from src.behavior_analysis.get_good_reversal_info import ReversalWarning, get_good_reversal_info
from src.behavior_analysis.get_rank_counts_by_good_reversal import get_rank_counts_by_value
from src.behavior_analysis.merge_across_problems import merge_windows_by_subject
from src.behavior_analysis.no_second_best import no_second_best_table
from src.behavior_analysis.no_trial_sessions import no_trial_sessions_table
from src.behavior_analysis.poke_counts import (
    previous_choice_towers_for_problem,
    previous_problem_choice_towers,
)
from src.behavior_analysis.port_finding_sessions import (
    assign_port_finding_sessions,
    port_finding_sessions_for_problem,
)
from src.behavior_analysis.review_context import ProblemIndex
from src.behavior_analysis.sessions_per_problem import sessions_per_problem_table
from src.behavior_analysis.starting_magnitudes import starting_magnitudes_table
from src.behavior_analysis.suspect_reversal_table import (
    COLUMNS as SUSPECT_COLUMNS,
)
from src.behavior_analysis.suspect_reversal_table import (
    suspect_reversal_table_by_problem,
    write_suspect_reversal_table,
)
from src.behavior_analysis.suspect_sessions import suspect_sessions_table
from src.behavior_analysis.trial_block_summary import (
    build_trial_block_tables,
    subject_numbers_from_paths,
)
from src.behavior_analysis.warnings_log import warnings_log_table
from src.behavior_import.import_data import session_source_paths
from src.behavior_import.import_log import ImportLog
from src.behavior_import.load_trials import ImportedTrials, load_trials_with_log
from src.behavior_import.task_types import (
    BLOCKED,
    DRIFTING,
    number_within_task_type,
    problem_map_table,
    regroup_by_task_type,
)
from src.behavior_import.warning_records import record_warnings
from src.behavior_visualization.plot_rank_proportions import plot_rank_proportions
from src.behavior_visualization.plot_trial_block_summary import plot_trial_block_summaries
from src.pipelines.incremental import (
    NoSessionsSelectedError,
    SessionSelection,
    check_selection_options,
    describe_plan,
    describe_written,
    merge_diagnostic_tables,
    select_recent_sessions,
)
from src.pipelines.open_field import (
    POOLED_DRIFTING_LABEL,
    VALUE_BASIS_LABELS,
    AnalysisSelection,
    ChoiceProbabilitySettings,
    check_selection_for_task_type,
    run_drifting_value_analyses,
    run_problem,
)
from src.pipelines.sessions_per_problem import (
    SESSIONS_PER_PROBLEM_FOLDER,
    write_sessions_per_problem_figures,
)
from src.settings import REPO_ROOT, get_settings

TASK = "open-field"
EXPERIMENT_FOLDER = "3x3_field_magnitude_bandit"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results"
REVERSAL_TRIAL_WINDOW = 100  # trials per bin of the trial-based reversal-rate plots
POOLED_DRIFTING_FOLDER = "all-drifting-problems"
PARTIAL_TABLE_WARNING = (
    "{name}.csv: no earlier table with task_type and problem_in_type columns to merge into, so "
    "only the rows of the regenerated problems {problems} were written; the table is partial"
)
NO_IMPORT_LOG_WARNING = (
    "trials were passed in without an import_log, so what the import skipped or flagged is "
    "unknown: no_trial_sessions.csv is not written, suspect_sessions.csv has no problem "
    "numbering rows, suspect_starting_reward_mags.csv cannot name a session without "
    "trials as a cause, and port-finding-only sessions are missing from the port-finding maps"
)

BLOCKED_SELECTION = AnalysisSelection(
    run_choice_probability_analysis=True,
    run_first_leave_analysis=True,
    run_number_of_reversals_analysis=True,
    run_rank_proportion_analysis=True,
    run_single_session_analysis=True,
    run_poke_analysis=True,
    rank_pooling_modes=(True, False),
    rank_value_bases=("true",),
)
# Choice probabilities and first leave look at the reversals of a blocked task; a drifting
# task has none to look at, so only the reversal counts and the rank proportions apply. The
# value analyses are the drifting task's own.
DRIFTING_SELECTION = AnalysisSelection(
    run_choice_probability_analysis=False,
    run_first_leave_analysis=False,
    run_number_of_reversals_analysis=True,
    run_rank_proportion_analysis=True,
    run_single_session_analysis=True,
    run_poke_analysis=True,
    run_value_psychometric_analysis=True,
    run_reversal_aligned_arm_analysis=True,
    run_value_change_analysis=True,
    run_port_preference_analysis=True,
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
    """What :func:`run_cohort` computed (the files are already written).

    Attributes:
        problem_map (pd.DataFrame): One row per mouse and problem, from
            :func:`~src.behavior_import.task_types.problem_map_table`.
        suspect_reversals (pd.DataFrame): The suspect-reversal review table with ``task_type``,
            ``problem_in_type`` and ``overall_problem`` in front (see
            :func:`suspect_table_with_task_types`).
        rank_results (dict): ``{(task_type, k): rank results of run_problem}``, each
            ``{value_basis: {pool_trials: {"counts": ..., "p_values": ...}}}`` or ``{}``.
        pooled_drifting_ranks (dict): Output of :func:`pooled_drifting_rank_results`:
            ``{value_basis: {pool_trials: {"problems", "counts", "p_values"}}}``; ``{}`` if the
            cohort has no drifting problems.
        review_tables (dict): ``{name: DataFrame}`` of the other tables written next to the
            problem map (``suspect_sessions``, ``warnings_log`` and so on; see the module doc).
    """

    problem_map: pd.DataFrame
    suspect_reversals: pd.DataFrame
    rank_results: dict  # {(task_type, k): run_problem's rank results}
    pooled_drifting_ranks: dict  # {value_basis: {pool_trials: {...}}}
    review_tables: dict  # {table name: DataFrame}


# ---------------------------------------------------------------------------
# Names, paths and settings
# ---------------------------------------------------------------------------


def cohort_name(cohort_id: int) -> str:
    """``3`` as ``"cohort-03"``.

    Args:
        cohort_id (int): Cohort number, 1 or more.

    Returns:
        str: ``"cohort-"`` followed by the number zero-padded to two digits.

    Raises:
        ValueError: If *cohort_id* is not a positive integer.
    """
    if not isinstance(cohort_id, int) or cohort_id < 1:
        raise ValueError(f"cohort id must be a positive integer, got {cohort_id!r}")
    return f"cohort-{cohort_id:02d}"


def cohort_raw_dir(data_root: Path, cohort_id: int) -> Path:
    """``<data_root>/3x3_field_magnitude_bandit/rawdata/cohort-0N``.

    Args:
        data_root (Path): Folder holding the experiment folder ``3x3_field_magnitude_bandit``.
        cohort_id (int): Cohort number, 1 or more.

    Returns:
        Path: The cohort's raw-data folder (not checked for existence).

    Raises:
        ValueError: If *cohort_id* is not a positive integer.
    """
    return Path(data_root) / EXPERIMENT_FOLDER / "rawdata" / cohort_name(cohort_id)


def selection_for_task_type(
    task_type: str,
    blocked: AnalysisSelection = BLOCKED_SELECTION,
    drifting: AnalysisSelection = DRIFTING_SELECTION,
) -> AnalysisSelection:
    """The analysis selection that applies to *task_type*.

    Args:
        task_type (str): ``"blocked"`` or ``"drifting"``.
        blocked (AnalysisSelection): Selection used for blocked problems. Default
            ``BLOCKED_SELECTION``.
        drifting (AnalysisSelection): Selection used for drifting problems. Default
            ``DRIFTING_SELECTION``.

    Returns:
        AnalysisSelection: *blocked* or *drifting* itself, per *task_type*.

    Raises:
        ValueError: For an unknown task type, or a blocked selection that asks for a
            drifting-only analysis.
    """
    if task_type == BLOCKED:
        check_selection_for_task_type(blocked, BLOCKED)
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

    Args:
        subjects_trials_by_problem (dict): ``{problem: {mouse: {session_key: session_dict}}}``;
            the sessions' raw ``"df"`` frames carry the paths (``sub-<N>_id-<mouse>``).

    Returns:
        dict[str, int]: The ``N`` of ``sub-<N>_id-<mouse>`` for every mouse.

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
        grouped (dict): ``{(task_type, k): {subject: sessions}}`` from
            :func:`~src.behavior_import.task_types.regroup_by_task_type`.
        problem_map (pd.DataFrame): The cohort's
            :func:`~src.behavior_import.task_types.problem_map_table`, which supplies each
            mouse's overall problem number.

    Returns:
        pd.DataFrame: One row per suspect reversal, with the columns ``task_type``,
        ``problem_in_type``, ``overall_problem``, then
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
        drifting_problems (dict): ``{k: {subject: sessions}}``, ``k`` the drifting problem
            number within the type.
        value_bases (tuple[str, ...]): ``"true"`` and/or ``"believed"``.
        pooling_modes (tuple[bool, ...]): ``pool_trials_within_subject`` values. ``True`` sums
            counts within each mouse, ``False`` gives equal weight to each block; mice stay the
            replicates.

    Returns:
        dict: ``{value_basis: {pool_trials: {"problems", "counts", "p_values"}}}``, where
        ``"problems"`` lists the ``k`` pooled; a basis with no rank windows in any problem is
        left out.
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


def plot_pooled_drifting_rank(results: dict, save_dir: Path, mouse_legend: bool = True) -> None:
    """One rank-proportion figure per value basis and pooling mode of *results*.

    Args:
        results (dict): As returned by :func:`pooled_drifting_rank_results`.
        save_dir (Path): Folder that receives the figures (created if missing).
        mouse_legend (bool): Name the mice in each figure's legend; their lines are drawn either
            way. Default ``True``.

    Returns:
        None: Returns nothing; the figures are written to *save_dir*.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for value_basis, by_mode in results.items():
        label = VALUE_BASIS_LABELS[value_basis]
        for pool_trials, result in by_mode.items():
            plot_rank_proportions(
                result["counts"],
                average_across_mice_pvalues=result["p_values"],
                pool_trials_within_subject=pool_trials,
                mouse_legend=mouse_legend,
                # plot_rank_proportions appends suffixes to a str save_path
                save_path=str(save_dir / f"Rank Proportions ({label}) (Pool Trials {pool_trials})"),
            )


@contextmanager
def each_reversal_warning_once() -> Iterator[None]:
    """Show each distinct :class:`ReversalWarning` text once while the block runs.

    Several analyses of one problem find the same dropped reversals again, and each says so.
    Other warnings are untouched, and the filter is gone on leaving the block, so a rerun in
    the same session warns again.

    Yields:
        None: Nothing; use as ``with each_reversal_warning_once(): ...``.
    """
    seen: set[str] = set()
    show_warning = warnings.showwarning

    def show_once(message, category, *args, **kwargs):
        if issubclass(category, ReversalWarning):
            if str(message) in seen:
                return
            seen.add(str(message))
        show_warning(message, category, *args, **kwargs)

    with warnings.catch_warnings():
        warnings.showwarning = show_once
        yield


def _previous_ports_by_problem(
    trials_by_problem: dict, grouped: dict, selections: list[AnalysisSelection]
) -> dict:
    """``{(task_type, k): {mouse: previous-problem choice ports or None}}`` for the red outlines.

    The previous problem comes from the overall problem order, before the regrouping by task
    type. Only computed when a selection runs the poke analysis or the single-session figures
    (the defaults do), which need ``choice_towers`` on every session; otherwise it is empty and
    sessions without choice towers still run.
    """
    if not any(s.run_poke_analysis or s.run_single_session_analysis for s in selections):
        return {}
    previous = previous_problem_choice_towers(trials_by_problem)
    return {
        key: previous_choice_towers_for_problem(subjects_trials, previous)
        for key, subjects_trials in grouped.items()
    }


def _write_tables(tables: dict[str, pd.DataFrame], directory: Path) -> None:
    """``<name>.csv`` in *directory* (created) for each table."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(directory / f"{name}.csv", index=False)


def read_existing_tables(directory: Path, names) -> dict[str, pd.DataFrame | None]:
    """``{name: table}`` read from ``<name>.csv`` in *directory*; ``None`` for a missing file.

    Args:
        directory (Path): Folder holding the CSVs.
        names (Iterable[str]): Table names, without ``.csv``.

    Returns:
        dict[str, pd.DataFrame | None]: The table of each name, or ``None`` if its file does not
        exist or has no content at all (not even a header).
    """
    tables: dict[str, pd.DataFrame | None] = {}
    for name in names:
        path = Path(directory) / f"{name}.csv"
        try:
            tables[name] = pd.read_csv(path)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            tables[name] = None
    return tables


def merge_with_earlier_run(new: dict, tables_dir: Path, rerun: frozenset) -> dict:
    """Diagnostic tables of a partial run merged into those the earlier run left on disk.

    Args:
        new (dict): ``{name: rows of the regenerated problems}`` from ``stack_diagnostics``.
        tables_dir (Path): Folder with the earlier run's CSVs.
        rerun (frozenset): ``(task_type, k)`` of the regenerated problems.

    Returns:
        dict[str, pd.DataFrame]: The tables to write: the rows of the regenerated problems
        replaced, all others kept. A table with no usable earlier file holds only the new rows,
        and a warning says it is partial.
    """
    existing = read_existing_tables(tables_dir, new)
    merged, partial = merge_diagnostic_tables(new, existing, rerun)
    for name in partial:
        warnings.warn(PARTIAL_TABLE_WARNING.format(name=name, problems=sorted(rerun)), stacklevel=2)
    return merged


def check_only_problems(only_problems: Collection[tuple[str, int]], grouped: dict) -> frozenset:
    """*only_problems* as a frozenset, after checking that each is a problem of the cohort.

    Args:
        only_problems (Collection[tuple[str, int]]): ``(task_type, k)`` to regenerate.
        grouped (dict): ``{(task_type, k): {mouse: sessions}}`` of the cohort.

    Returns:
        frozenset[tuple[str, int]]: The same problems.

    Raises:
        ValueError: If it is empty (use ``None`` for every problem), or names a problem that
            *grouped* does not have.
    """
    wanted = frozenset(only_problems)
    if not wanted:
        raise ValueError("only_problems is empty: no problems to run (None runs every problem)")
    unknown = sorted(wanted - set(grouped))
    if unknown:
        raise ValueError(f"only_problems names problems the cohort does not have: {unknown}")
    return wanted


def import_review_tables(
    grouped: dict, problem_map: pd.DataFrame, index: ProblemIndex, log: ImportLog | None
) -> dict[str, pd.DataFrame]:
    """The tables that come from the imported sessions and the import log alone.

    Args:
        grouped (dict): ``{(task_type, k): {mouse: sessions}}`` from
            :func:`~src.behavior_import.task_types.regroup_by_task_type`.
        problem_map (pd.DataFrame): The cohort's problem map.
        index (ProblemIndex): Where each session sits in its mouse's problems.
        log (ImportLog | None): What the import skipped and flagged; ``None`` if the sessions
            were not imported here. Then ``no_trial_sessions`` is left out, and
            ``suspect_sessions`` has no numbering rows, rather than being empty like a cohort
            where nothing was skipped or flagged.

    Returns:
        dict[str, pd.DataFrame]: ``problem_map``, ``suspect_sessions``,
        ``suspect_starting_reward_mags``, ``no_trial_sessions`` (not without a log) and
        ``no_second_best``.
    """
    flags, skips = (log.flags, log.skips) if log else ([], [])
    tables = {
        "problem_map": problem_map,
        "suspect_sessions": suspect_sessions_table(flags, grouped, index),
        "suspect_starting_reward_mags": starting_magnitudes_table(grouped, skips, index),
        "no_second_best": no_second_best_table(grouped, index),
    }
    if log:
        tables["no_trial_sessions"] = no_trial_sessions_table(skips, index)
    return tables


def run_warnings_log(
    log: ImportLog | None, run_warnings: list, index: ProblemIndex
) -> pd.DataFrame:
    """The warnings log of a run: those of the import, then those of the run.

    Args:
        log (ImportLog | None): Its ``warnings`` (raised during the import) come first; ``None``
            if the sessions were not imported here.
        run_warnings (list[WarningRecord]): The warnings raised during the run.
        index (ProblemIndex): The cohort's problems, to place the warnings.

    Returns:
        pd.DataFrame: :func:`~src.behavior_analysis.warnings_log.warnings_log_table`.
    """
    return warnings_log_table([*(log.warnings if log else ()), *run_warnings], index)


def load_cohort(cohort_id: int, data_root, use_cache: bool) -> ImportedTrials:
    """Import a cohort's raw files, classifying the task type of every session.

    Args:
        cohort_id (int): 1 for ``cohort-01``, and so on.
        data_root (Path | None): Folder holding ``3x3_field_magnitude_bandit``; ``None`` for the
            ``BANDIT_DATA_ROOT`` setting.
        use_cache (bool): Use the on-disk import cache (keyed on the raw files).

    Returns:
        ImportedTrials: ``(trials_by_problem, import_log)`` of
        :func:`~src.behavior_import.load_trials.load_trials_with_log`.
    """
    root = Path(data_root) if data_root is not None else get_settings().data_root
    # task types are read at import, so a task-type change starts a new problem
    return load_trials_with_log(
        cohort_raw_dir(root, cohort_id), use_cache=use_cache, classify_task_types=True
    )


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
    import_log: ImportLog | None = None,
    only_problems: Collection[tuple[str, int]] | None = None,
) -> CohortResult:
    """Run the analysis suite for one cohort; see the module docstring for the outputs.

    With *only_problems* the run is partial: the figures of those problems are regenerated and
    the rest of ``figures/`` is left as it is. The pooled drifting analyses (which use every
    drifting problem) run only if a regenerated problem is drifting; the review tables and the
    trial and block-length summaries cover the whole cohort and are always rewritten; each
    stacked diagnostic table keeps the rows of the other problems from the earlier run (see
    :func:`merge_with_earlier_run`). ``warnings_log.csv`` lists only this run's warnings.

    Args:
        cohort_id (int): 1 for ``cohort-01``, and so on.
        data_root (Path | None): Folder holding ``3x3_field_magnitude_bandit``. Default
            ``None``: the ``BANDIT_DATA_ROOT`` setting. Keyword-only.
        results_root (Path): Folder that receives ``figures/`` and ``tables/``. Default
            ``DEFAULT_RESULTS_ROOT`` (``<repo>/results``). Keyword-only.
        blocked_selection (AnalysisSelection): Which analyses run for blocked problems.
            Default ``BLOCKED_SELECTION``. Keyword-only.
        drifting_selection (AnalysisSelection): Which analyses run for drifting problems.
            Default ``DRIFTING_SELECTION``. Keyword-only.
        choice (ChoiceProbabilitySettings): Choice-probability figure toggles and windows.
            Default ``DEFAULT_CHOICE``. Keyword-only.
        reversal_trial_window (int): Trials per bin of the trial-based reversal-rate plots.
            Default ``REVERSAL_TRIAL_WINDOW`` (100). Keyword-only.
        use_cache (bool): Use the on-disk import cache (keyed on the raw files). Default
            ``True``. Keyword-only.
        trials_by_problem (dict | None): Already-loaded ``{problem: {subject: sessions}}``, to
            skip the load. Default ``None``: load the cohort. Keyword-only.
        import_log (ImportLog | None): The import log of *trials_by_problem*. Without it,
            passed-in trials have no record of what the import skipped or flagged, so a warning
            says that ``no_trial_sessions.csv`` is not written and ``suspect_sessions.csv`` has
            no numbering rows. Ignored when the cohort is loaded. Default ``None``.
            Keyword-only.
        only_problems (Collection[tuple[str, int]] | None): ``(task_type, k)`` of the problems
            to regenerate (``k`` the number within the type, as in ``problem_map.csv``). Default
            ``None``: every problem. Keyword-only.

    Returns:
        CohortResult: The problem map, suspect-reversal table, per-problem rank results,
        pooled drifting rank results and the other review tables. In a partial run
        ``rank_results`` holds only the regenerated problems and ``pooled_drifting_ranks`` is
        empty when the pooled analyses were skipped; the diagnostic tables are the merged ones.

    Raises:
        ValueError: If the cohort has no problems, a session's task name is not recognised, or
            *only_problems* is empty or names a problem the cohort does not have.
    """
    cohort = cohort_name(cohort_id)
    log = import_log
    if trials_by_problem is None:
        trials_by_problem, log = load_cohort(cohort_id, data_root, use_cache)
    if not trials_by_problem:
        raise ValueError(f"No problems found for {cohort}")

    # Classify and check the selections first: a bad task name or selection stops the run
    # before anything is written.
    selections = [
        selection_for_task_type(kind, blocked_selection, drifting_selection)
        for kind in (BLOCKED, DRIFTING)
    ]
    numbering = number_within_task_type(trials_by_problem)
    grouped = regroup_by_task_type(trials_by_problem, numbering)
    problem_map = problem_map_table(trials_by_problem, numbering)
    index = ProblemIndex.from_cohort(trials_by_problem, problem_map)
    rerun = None if only_problems is None else check_only_problems(only_problems, grouped)
    previous_ports = _previous_ports_by_problem(trials_by_problem, grouped, selections)
    figures, tables = _figures_dir(results_root, cohort), _tables_dir(results_root, cohort)

    run_warnings: list = []
    diagnostic_tables: dict = {}
    try:
        with each_reversal_warning_once(), record_warnings() as run_warnings:
            if log is None:
                warnings.warn(NO_IMPORT_LOG_WARNING, stacklevel=2)
            port_finding = assign_port_finding_sessions(
                trials_by_problem, log.port_finding_sessions if log else {}
            )
            review_tables = import_review_tables(grouped, problem_map, index, log)
            sessions_table = sessions_per_problem_table(trials_by_problem, port_finding)
            review_tables["sessions_per_problem"] = sessions_table
            _write_tables(review_tables, tables)
            suspects = suspect_table_with_task_types(grouped, problem_map)
            write_suspect_reversal_table(suspects, tables / "suspect_reversals.csv")

            rank_results, diagnostics = {}, {}
            for (task_type, k), subjects_trials in grouped.items():
                if rerun is not None and (task_type, k) not in rerun:
                    continue
                selection = selection_for_task_type(
                    task_type, blocked_selection, drifting_selection
                )
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
                    previous_choice_towers=previous_ports.get((task_type, k)),
                    diagnostics=diagnostics.setdefault((task_type, k), {}),
                    port_finding_sessions=port_finding_sessions_for_problem(
                        port_finding, subjects_trials
                    ),
                )

            drifting = {k: subs for (kind, k), subs in grouped.items() if kind == DRIFTING}
            pooled, pooled_ran = {}, False
            if drifting and (rerun is None or any(kind == DRIFTING for kind, _ in rerun)):
                pooled = pooled_drifting_rank_results(
                    drifting,
                    drifting_selection.rank_value_bases,
                    drifting_selection.rank_pooling_modes,
                )
                pooled_dir = figures / DRIFTING / POOLED_DRIFTING_FOLDER
                plot_pooled_drifting_rank(pooled, pooled_dir / "choice-stats")
                run_drifting_value_analyses(
                    drifting, pooled_dir, drifting_selection, arm_scope_label=POOLED_DRIFTING_LABEL
                )
                write_sessions_per_problem_figures(
                    sessions_table,
                    pooled_dir / SESSIONS_PER_PROBLEM_FOLDER,
                    f"cohort {cohort_id}",
                )
                pooled_ran = True
            else:
                why = "no drifting problems" if not drifting else "no drifting problem selected"
                print(f"[INFO] {cohort}: {why}; skipping pooled drifting ranks.")

            session_table, block_table = build_trial_block_tables(
                trials_by_problem, subject_numbers_from_sessions(trials_by_problem)
            )
            plot_trial_block_summaries(
                session_table,
                block_table,
                figures / "trial-block-summaries" / "all-problems",
                show=False,
            )
            # stacking needs every selected problem to have run. It stays inside the warnings
            # block because merging a partial run can warn, and that warning belongs in the log
            # (stacking itself raises no warnings, so a full run is unaffected).
            diagnostic_tables = stack_diagnostics(diagnostics, problem_map)
            if rerun is not None:
                diagnostic_tables = merge_with_earlier_run(diagnostic_tables, tables, rerun)
    finally:
        # a run that fails is the one whose warnings are needed most
        late_tables = {
            **diagnostic_tables,
            "warnings_log": run_warnings_log(log, run_warnings, index),
        }
        _write_tables(late_tables, tables)
    if rerun is not None:
        print(describe_written(cohort, rerun, pooled_ran, late_tables))
    return CohortResult(
        problem_map, suspects, rank_results, pooled, {**review_tables, **late_tables}
    )


def plan_cohort_incremental(
    cohort_id: int,
    *,
    last_sessions: int | None = None,
    since: str | None = None,
    data_root: Path | None = None,
    use_cache: bool = True,
    trials_by_problem: dict | None = None,
) -> SessionSelection:
    """The sessions and problems a partial run of one cohort would regenerate.

    Writes no results or tables (the import cache may be filled).

    Args:
        cohort_id (int): 1 for ``cohort-01``, and so on.
        last_sessions (int | None): Each mouse's N most recent sessions; see
            :func:`~src.pipelines.incremental.select_recent_sessions`. Keyword-only.
        since (str | None): Sessions dated on or after this ``YYYY-MM-DD`` date. Keyword-only.
        data_root (Path | None): As for :func:`run_cohort`. Keyword-only.
        use_cache (bool): As for :func:`run_cohort`. Keyword-only.
        trials_by_problem (dict | None): Already-loaded trials, to skip the load. Keyword-only.

    Returns:
        SessionSelection: The selected sessions and the problems that contain them.

    Raises:
        ValueError: As :func:`~src.pipelines.incremental.select_recent_sessions`, or if the
            cohort has no problems.
        NoSessionsSelectedError: If no session matches.
    """
    if trials_by_problem is None:
        trials_by_problem, _ = load_cohort(cohort_id, data_root, use_cache)
    if not trials_by_problem:
        raise ValueError(f"No problems found for {cohort_name(cohort_id)}")
    grouped = regroup_by_task_type(trials_by_problem, number_within_task_type(trials_by_problem))
    return select_recent_sessions(grouped, last_sessions=last_sessions, since=since)


def run_cohort_incremental(
    cohort_id: int,
    *,
    last_sessions: int | None = None,
    since: str | None = None,
    data_root: Path | None = None,
    use_cache: bool = True,
    trials_by_problem: dict | None = None,
    import_log: ImportLog | None = None,
    **settings,
) -> CohortResult:
    """Regenerate only what the recent sessions of a cohort change.

    Selects the sessions (each mouse's last N, or those since a date), prints which problems
    and pooled scopes will be regenerated and why, then runs :func:`run_cohort` for those
    problems alone (its ``only_problems``). The selection is made on the data the run uses,
    so the import happens once.

    Args:
        cohort_id (int): 1 for ``cohort-01``, and so on.
        last_sessions (int | None): Each mouse's N most recent sessions. Exactly one of this and
            *since* is required. Keyword-only.
        since (str | None): Sessions dated on or after this ``YYYY-MM-DD`` date. Keyword-only.
        data_root (Path | None): As for :func:`run_cohort`. Keyword-only.
        use_cache (bool): As for :func:`run_cohort`. Keyword-only.
        trials_by_problem (dict | None): Already-loaded trials, to skip the load. Keyword-only.
        import_log (ImportLog | None): The log of *trials_by_problem*; ignored when the cohort
            is loaded here. Keyword-only.
        **settings: The remaining keyword arguments of :func:`run_cohort` (``results_root``,
            selections, ...).

    Returns:
        CohortResult: As :func:`run_cohort` returns for a partial run.

    Raises:
        ValueError: As :func:`plan_cohort_incremental`.
        NoSessionsSelectedError: If no session matches; nothing is written.
    """
    log = import_log
    if trials_by_problem is None:
        trials_by_problem, log = load_cohort(cohort_id, data_root, use_cache)
    selection = plan_cohort_incremental(
        cohort_id, last_sessions=last_sessions, since=since, trials_by_problem=trials_by_problem
    )
    print(f"[PLAN] {describe_plan(cohort_name(cohort_id), selection)}")
    return run_cohort(
        cohort_id,
        use_cache=use_cache,
        trials_by_problem=trials_by_problem,
        import_log=log,
        only_problems=selection.problems,
        **settings,
    )


def _run_one_cohort(cohort_id: int, last_sessions, since, dry_run: bool, settings: dict):
    """One cohort of :func:`run_cohorts`: full, partial or dry-run; the result or ``None``."""
    if dry_run:
        plan = plan_cohort_incremental(
            cohort_id,
            last_sessions=last_sessions,
            since=since,
            data_root=settings.get("data_root"),
            use_cache=settings.get("use_cache", True),
        )
        print(f"[PLAN] dry run, nothing written. {describe_plan(cohort_name(cohort_id), plan)}")
        return None
    if last_sessions is not None or since is not None:
        return run_cohort_incremental(
            cohort_id, last_sessions=last_sessions, since=since, **settings
        )
    return run_cohort(cohort_id, **settings)


def run_cohorts(
    cohort_ids,
    *,
    continue_on_error: bool = False,
    last_sessions: int | None = None,
    since: str | None = None,
    dry_run: bool = False,
    **settings,
) -> dict:
    """:func:`run_cohort` for each id; ``settings`` are passed on to every cohort.

    By default the first failing cohort stops the run. With ``continue_on_error`` the other
    cohorts still run, and the failures are raised together at the end.

    With *last_sessions* or *since* every cohort runs :func:`run_cohort_incremental`. A cohort
    with no matching session is skipped with a ``[SKIP]`` line (a date recent for one cohort
    may not reach another), but if no cohort matches at all the run raises.

    Args:
        cohort_ids (Iterable[int]): Cohort numbers to run, in order.
        continue_on_error (bool): Keep going after a failing cohort. Default ``False``.
            Keyword-only.
        last_sessions (int | None): Partial run on each mouse's N most recent sessions.
            Default ``None``. Keyword-only.
        since (str | None): Partial run on the sessions dated on or after this ``YYYY-MM-DD``
            date. Default ``None``. Keyword-only.
        dry_run (bool): Print the plan of the partial run (problems, pooled scopes, sessions)
            and write no results or tables (the import cache may be filled). Needs
            *last_sessions* or *since*. Default ``False``. Keyword-only.
        **settings: Keyword arguments of :func:`run_cohort` (``data_root``, ``results_root``,
            selections, ...), applied to every cohort.

    Returns:
        dict[int, CohortResult]: ``{cohort_id: CohortResult}``; empty for a dry run.

    Raises:
        ValueError: For *dry_run* without a selection, or invalid selection options (checked
            before any cohort is loaded).
        RuntimeError: Naming each cohort that failed and why and, for a partial run, that no
            cohort has a matching session (both are reported when both happen).
    """
    if dry_run and last_sessions is None and since is None:
        raise ValueError("dry_run needs last_sessions or since: a full run has no plan to print")
    if last_sessions is not None or since is not None:
        check_selection_options(last_sessions, since)
    results, failures, unmatched, planned = {}, {}, {}, 0
    for cohort_id in cohort_ids:
        try:
            result = _run_one_cohort(cohort_id, last_sessions, since, dry_run, settings)
            if dry_run:
                planned += 1
            else:
                results[cohort_id] = result
        except NoSessionsSelectedError as err:
            unmatched[cohort_id] = f"{cohort_name(cohort_id)}: {err}"
            print(f"[SKIP] {unmatched[cohort_id]}")
        except Exception as err:  # re-raised below with the cohort named, never dropped
            message = f"{cohort_name(cohort_id)} failed: {type(err).__name__}: {err}"
            if not continue_on_error:
                raise RuntimeError(message) from err
            print(f"[ERROR] {message}")
            failures[cohort_id] = message
    problems = list(failures.values())
    if unmatched and not (results or planned):
        problems.append("no cohort has a matching session:\n" + "\n".join(unmatched.values()))
    if failures:
        problems.insert(0, f"{len(failures)} cohort(s) failed:")
    if problems:
        raise RuntimeError("\n".join(problems))
    return results
