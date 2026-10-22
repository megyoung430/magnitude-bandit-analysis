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
    figures/open-field/<cohort>/drifting/all-drifting-problems-without-<mice>/  (only with
        ``exclude_mice``: the same subfolders, from the drifting problems without those mice, its
        figures marked "without <mice>"; its tables are in ``tables/.../<same name>/``)
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
  on every run, also a partial one, and drawn into ``sessions-per-problem/`` on every run of a
  cohort with drifting problems;
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
review tables, ``trial-block-summaries/`` and ``all-drifting-problems/sessions-per-problem/``
cover the cohort and are always rewritten. In the
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

import re
import shutil
import warnings
from collections.abc import Collection, Iterable, Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from src.behavior_analysis.diagnostic_tables import stack_diagnostics
from src.behavior_analysis.exclude_mice import (
    ExcludedMiceError,
    exclude_mice_from_problems,
    exclude_mice_from_table,
    excluded_mice_description,
    excluded_scope_text,
    excluded_set_name,
    optional_mouse_ids,
)
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
from src.behavior_visualization.plot_helpers import figure_banner
from src.behavior_visualization.plot_rank_proportions import plot_rank_proportions
from src.behavior_visualization.plot_trial_block_summary import plot_trial_block_summaries
from src.pipelines.cache_provenance import (
    cache_sources,
    write_from_cache_notes,
)
from src.pipelines.incremental import (
    NoSessionsSelectedError,
    SessionSelection,
    check_selection_options,
    describe_plan,
    describe_pooled_only_plan,
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
from src.pipelines.sessions_per_problem import (
    TABLE_NAME as SESSIONS_PER_PROBLEM_TABLE,
)
from src.settings import REPO_ROOT, get_settings, get_unchecked_settings

TASK = "open-field"
EXPERIMENT_FOLDER = "3x3_field_magnitude_bandit"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results"
REVERSAL_TRIAL_WINDOW = 100  # trials per bin of the trial-based reversal-rate plots
POOLED_DRIFTING_FOLDER = "all-drifting-problems"
EXCLUDED_MICE_FILE = "excluded_mice.txt"
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
        excluded_pooled_ranks (dict | None): As *pooled_drifting_ranks*, for the pooled set
            without the mice of ``exclude_mice``; ``None`` if no such set was written.
    """

    problem_map: pd.DataFrame
    suspect_reversals: pd.DataFrame
    rank_results: dict  # {(task_type, k): run_problem's rank results}
    pooled_drifting_ranks: dict  # {value_basis: {pool_trials: {...}}}
    review_tables: dict  # {table name: DataFrame}
    excluded_pooled_ranks: dict | None = None  # as pooled_drifting_ranks, without the mice


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


def cohort_id_from_name(name: str) -> int:
    """The cohort number of a folder name such as ``"cohort-01"``; inverse of :func:`cohort_name`.

    Args:
        name (str): A cohort folder name, ``"cohort-"`` followed by a positive number.

    Returns:
        int: The cohort number, e.g. 1.

    Raises:
        ValueError: If *name* is not of that form.
    """
    match = re.fullmatch(r"cohort-(\d+)", name)
    if match is None or int(match.group(1)) < 1:
        raise ValueError(f"expected a cohort name like 'cohort-01', got {name!r}")
    return int(match.group(1))


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


def load_cohort(
    cohort_id: int, data_root, use_cache: bool, from_cache: bool = False
) -> ImportedTrials:
    """Import a cohort's raw files, classifying the task type of every session.

    Args:
        cohort_id (int): 1 for ``cohort-01``, and so on.
        data_root (Path | None): Folder holding ``3x3_field_magnitude_bandit``; ``None`` for the
            ``BANDIT_DATA_ROOT`` setting.
        use_cache (bool): Use the on-disk import cache (keyed on the raw files).
        from_cache (bool): Load the newest cached import of the cohort without reading the raw
            files (see :func:`~src.behavior_import.load_trials.load_trials_with_log`); the data
            root then need not exist. Default ``False``.

    Returns:
        ImportedTrials: ``(trials_by_problem, import_log)`` of
        :func:`~src.behavior_import.load_trials.load_trials_with_log`.
    """
    if data_root is not None:
        root = Path(data_root)
    else:
        # the checked settings test that the data root exists, which touches the share
        root = (get_unchecked_settings() if from_cache else get_settings()).data_root
    # task types are read at import, so a task-type change starts a new problem
    return load_trials_with_log(
        cohort_raw_dir(root, cohort_id),
        use_cache=use_cache,
        classify_task_types=True,
        from_cache=from_cache,
    )


def _load_cohort_for_run(
    cohort_id: int, data_root, use_cache: bool, from_cache: bool
) -> ImportedTrials:
    """Load one cohort, forwarding *from_cache* only when it is enabled."""
    if from_cache:
        return load_cohort(cohort_id, data_root, use_cache, from_cache)
    return load_cohort(cohort_id, data_root, use_cache)


# ---------------------------------------------------------------------------
# The pooled drifting sets
# ---------------------------------------------------------------------------


def checked_exclusion(drifting: dict, exclude_mice: Iterable[str] | None) -> tuple[tuple, dict]:
    """The excluded mouse ids and the drifting problems without them.

    Args:
        drifting (dict): ``{k: {mouse: sessions}}`` of the cohort.
        exclude_mice (Iterable[str] | None): Mice to leave out; ``None`` for no extra set.

    Returns:
        tuple[tuple[str, ...], dict]: The sorted ids and the reduced problems; ``((), {})`` for
        ``None``.

    Raises:
        ValueError: As :func:`~src.behavior_analysis.exclude_mice.exclude_mice_from_problems`
            (also for an empty list).
    """
    ids = optional_mouse_ids(exclude_mice)
    if ids is None:
        return (), {}
    return ids, exclude_mice_from_problems(drifting, ids)


def run_pooled_drifting(
    drifting: dict, pooled_dir: Path, selection: AnalysisSelection, scope_text: str | None = None
) -> tuple[dict, dict]:
    """The pooled drifting analyses over *drifting*, written into *pooled_dir*.

    Args:
        drifting (dict): ``{k: {mouse: sessions}}`` to pool.
        pooled_dir (Path): Folder that receives ``choice-stats`` and the value-analysis folders.
        selection (AnalysisSelection): The drifting selection (analyses, rank bases, pooling).
        scope_text (str | None): Added to the scope named in the titles that carry one, e.g.
            ``"without MY_1_A"``. Default ``None``: the plain pool.

    Returns:
        tuple[dict, dict]: The rank results (as :func:`pooled_drifting_rank_results`) and the
        diagnostic tables of the value analyses.
    """
    label = POOLED_DRIFTING_LABEL if scope_text is None else f"{POOLED_DRIFTING_LABEL} {scope_text}"
    ranks = pooled_drifting_rank_results(
        drifting, selection.rank_value_bases, selection.rank_pooling_modes
    )
    plot_pooled_drifting_rank(ranks, pooled_dir / "choice-stats")
    # x_vs_plus is stated because the default switches on the pool's own label, which this one
    # may extend
    tables = run_drifting_value_analyses(
        drifting, pooled_dir, selection, arm_scope_label=label, x_vs_plus=True
    )
    return ranks, tables


def run_excluded_pooled_set(
    cohort_id: int,
    drifting: dict,
    excluded: tuple[str, ...],
    *,
    results_root: Path,
    selection: AnalysisSelection,
) -> dict:
    """The pooled drifting analyses without the *excluded* mice, in their own folders.

    Writes the figures of ``drifting/all-drifting-problems-without-<mice>/`` and the analyses'
    tables under the same name in the cohort's tables folder. Both folders belong to the set
    and are emptied first, so nothing from an earlier run with other settings remains. Every
    figure written is marked with the exclusion, whichever plot drew it. The tables are kept
    out of the cohort's stacked diagnostic tables, which come from the per-problem runs alone.
    The sessions-per-problem figure and table are written by
    :func:`write_excluded_sessions_per_problem`, after this.

    Args:
        cohort_id (int): 1 for ``cohort-01``, and so on.
        drifting (dict): ``{k: {mouse: sessions}}`` already without the excluded mice (see
            :func:`checked_exclusion`).
        excluded (tuple[str, ...]): The excluded mouse ids, sorted.
        results_root (Path): Folder that receives ``figures/`` and ``tables/``. Keyword-only.
        selection (AnalysisSelection): The drifting selection. Keyword-only.

    Returns:
        dict: The rank results of the reduced pool, as :func:`pooled_drifting_rank_results`.
    """
    cohort = cohort_name(cohort_id)
    name = excluded_set_name(POOLED_DRIFTING_FOLDER, excluded)
    scope = excluded_scope_text(excluded)
    pooled_dir = _figures_dir(results_root, cohort) / DRIFTING / name
    tables_dir = _tables_dir(results_root, cohort) / name
    for folder in (pooled_dir, tables_dir):
        shutil.rmtree(folder, ignore_errors=True)
    with figure_banner(scope):
        ranks, tables = run_pooled_drifting(drifting, pooled_dir, selection, scope)
    _write_tables(tables, tables_dir)
    return ranks


def write_excluded_sessions_per_problem(
    cohort_id: int,
    excluded: tuple[str, ...],
    sessions_table: pd.DataFrame,
    *,
    results_root: Path,
    run_at: datetime,
) -> None:
    """The sessions-per-problem figure and table of the set without the *excluded* mice.

    Redrawn on every run of a cohort with drifting problems, as the plain set's is, because any
    new session can complete a problem. Also writes ``excluded_mice.txt`` into the set's figure
    and table folders (created if missing).

    Args:
        cohort_id (int): 1 for ``cohort-01``, and so on.
        excluded (tuple[str, ...]): The excluded mouse ids, sorted.
        sessions_table (pd.DataFrame): The cohort's whole sessions-per-problem table.
        results_root (Path): Folder that receives ``figures/`` and ``tables/``. Keyword-only.
        run_at (datetime): Time written to ``excluded_mice.txt``. Keyword-only.
    """
    cohort = cohort_name(cohort_id)
    name = excluded_set_name(POOLED_DRIFTING_FOLDER, excluded)
    scope = excluded_scope_text(excluded)
    pooled_dir = _figures_dir(results_root, cohort) / DRIFTING / name
    tables_dir = _tables_dir(results_root, cohort) / name
    sessions = exclude_mice_from_table(sessions_table, excluded)
    with figure_banner(scope):
        write_sessions_per_problem_figures(
            sessions, pooled_dir / SESSIONS_PER_PROBLEM_FOLDER, f"cohort {cohort_id} {scope}"
        )
    _write_tables({SESSIONS_PER_PROBLEM_TABLE: sessions}, tables_dir)
    description = excluded_mice_description(cohort, excluded, run_at)
    for folder in (pooled_dir, tables_dir):
        (folder / EXCLUDED_MICE_FILE).write_text(description)


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
    from_cache: bool = False,
    trials_by_problem: dict | None = None,
    import_log: ImportLog | None = None,
    only_problems: Collection[tuple[str, int]] | None = None,
    exclude_mice: Iterable[str] | None = None,
    pooled_only: bool = False,
    run_at: datetime | None = None,
) -> CohortResult:
    """Run the analysis suite for one cohort; see the module docstring for the outputs.

    With *only_problems* the run is partial: the figures of those problems are regenerated and
    the rest of ``figures/`` is left as it is. The pooled drifting analyses (which use every
    drifting problem) run only if a regenerated problem is drifting; the review tables and the
    trial and block-length summaries cover the whole cohort and are always rewritten; each
    stacked diagnostic table keeps the rows of the other problems from the earlier run (see
    :func:`merge_with_earlier_run`). ``warnings_log.csv`` lists only this run's warnings.

    With *exclude_mice* the pooled drifting analyses are run a second time without those mice,
    into ``drifting/all-drifting-problems-without-<mice>/`` (see
    :func:`run_excluded_pooled_set`), whenever the plain pool is run; the plain pool and the
    stacked diagnostic tables are as without the option. The names are checked against the
    cohort's drifting problems before anything is written.

    With *pooled_only* no problem is run: only the pooled drifting analyses (and the extra set,
    if asked for), the sessions-per-problem figures and the review tables are written. The
    problem folders, ``trial-block-summaries/`` and the stacked diagnostic tables are left as
    they are.

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
        from_cache (bool): Load the newest cached import of the cohort without reading the
            share; sessions added since are missing (see :func:`load_cohort`). Default
            ``False``. Keyword-only.
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
        exclude_mice (Iterable[str] | None): Mouse ids to leave out of the extra pooled set.
            Default ``None``: no extra set. Keyword-only.
        pooled_only (bool): Run only the pooled drifting analyses, not the problems. Default
            ``False``. Keyword-only.
        run_at (datetime | None): The time written to ``excluded_mice.txt``. Default ``None``:
            now. Keyword-only.

    Returns:
        CohortResult: The problem map, suspect-reversal table, per-problem rank results,
        pooled drifting rank results and the other review tables. In a partial run
        ``rank_results`` holds only the regenerated problems and ``pooled_drifting_ranks`` is
        empty when the pooled analyses were skipped; the diagnostic tables are the merged ones.
        In a pooled-only run ``rank_results`` is empty and no diagnostic table is among the
        review tables. ``excluded_pooled_ranks`` holds the extra set's rank results.

    Raises:
        ValueError: If the cohort has no problems, a session's task name is not recognised,
            *only_problems* is empty or names a problem the cohort does not have, *pooled_only*
            is combined with *only_problems* or the cohort has no drifting problem, or
            *exclude_mice* is empty, names a mouse without drifting problems in the cohort, or
            names all of them.
    """
    if pooled_only and only_problems is not None:
        raise ValueError(
            "pooled_only runs no problems, so it cannot be combined with only_problems"
        )
    exclude_mice = optional_mouse_ids(exclude_mice)
    cohort = cohort_name(cohort_id)
    log = import_log
    if trials_by_problem is None:
        trials_by_problem, log = load_cohort(cohort_id, data_root, use_cache, from_cache)
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
    drifting = _drifting_of(grouped)
    if pooled_only and not drifting:
        raise ValueError(f"{cohort} has no drifting problems for a pooled-only run")
    # checked before anything is written, so a mistyped mouse id costs no run
    excluded, drifting_without = checked_exclusion(drifting, exclude_mice)
    # the per-problem red outlines are not needed when no problem is run
    previous_ports = (
        {} if pooled_only else _previous_ports_by_problem(trials_by_problem, grouped, selections)
    )
    figures, tables = _figures_dir(results_root, cohort), _tables_dir(results_root, cohort)
    # written first, so even a run that fails part-way marks its folders as made from the cache
    sources = cache_sources({cohort: log})
    write_from_cache_notes([figures, tables], sources)

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
                if pooled_only or (rerun is not None and (task_type, k) not in rerun):
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

            pooled, pooled_ran, excluded_pooled = {}, False, None
            if drifting and (
                pooled_only or rerun is None or any(kind == DRIFTING for kind, _ in rerun)
            ):
                # the plain pool's tables are not written: the per-problem runs already hold
                # these rows in the stacked diagnostic tables
                pooled, _ = run_pooled_drifting(
                    drifting, figures / DRIFTING / POOLED_DRIFTING_FOLDER, drifting_selection
                )
                if excluded:
                    excluded_pooled = run_excluded_pooled_set(
                        cohort_id,
                        drifting_without,
                        excluded,
                        results_root=results_root,
                        selection=drifting_selection,
                    )
                pooled_ran = True
            else:
                why = "no drifting problems" if not drifting else "no drifting problem selected"
                print(f"[INFO] {cohort}: {why}; skipping pooled drifting ranks.")

            if drifting:
                # redrawn on every run: any new session can complete a problem, drifting or not
                write_sessions_per_problem_figures(
                    sessions_table,
                    figures / DRIFTING / POOLED_DRIFTING_FOLDER / SESSIONS_PER_PROBLEM_FOLDER,
                    f"cohort {cohort_id}",
                )
                if excluded:
                    write_excluded_sessions_per_problem(
                        cohort_id,
                        excluded,
                        sessions_table,
                        results_root=results_root,
                        run_at=run_at or datetime.now(),
                    )
            if not pooled_only:
                session_table, block_table = build_trial_block_tables(
                    trials_by_problem, subject_numbers_from_sessions(trials_by_problem)
                )
                plot_trial_block_summaries(
                    session_table,
                    block_table,
                    figures / "trial-block-summaries" / "all-problems",
                    show=False,
                )
                # stacking needs every selected problem to have run. It stays inside the
                # warnings block because merging a partial run can warn, and that warning
                # belongs in the log (stacking itself raises no warnings, so a full run is
                # unaffected). A pooled-only run has no problem rows to put in the tables, so
                # it leaves the earlier ones alone.
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
    extra_set = excluded_set_name(POOLED_DRIFTING_FOLDER, excluded) if excluded else None
    if pooled_only:
        sets = [POOLED_DRIFTING_FOLDER, *([extra_set] if extra_set else [])]
        written = " and ".join(f"{DRIFTING}/{name}" for name in sets)
        print(f"[DONE] {cohort} pooled-only run wrote {written}; review tables rewritten")
    elif rerun is not None:
        print(describe_written(cohort, rerun, pooled_ran, late_tables, extra_set))
    return CohortResult(
        problem_map,
        suspects,
        rank_results,
        pooled,
        {**review_tables, **late_tables},
        excluded_pooled,
    )


def plan_cohort_incremental(
    cohort_id: int,
    *,
    last_sessions: int | None = None,
    since: str | None = None,
    data_root: Path | None = None,
    use_cache: bool = True,
    from_cache: bool = False,
    trials_by_problem: dict | None = None,
    exclude_mice: Iterable[str] | None = None,
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
        from_cache (bool): As for :func:`run_cohort`. Keyword-only.
        trials_by_problem (dict | None): Already-loaded trials, to skip the load. Keyword-only.
        exclude_mice (Iterable[str] | None): The mice of the run's extra pooled set, checked
            against the cohort here so that a dry run reports a bad name too. Keyword-only.

    Returns:
        SessionSelection: The selected sessions and the problems that contain them.

    Raises:
        ValueError: As :func:`~src.pipelines.incremental.select_recent_sessions`, if the
            cohort has no problems, or for *exclude_mice* as in :func:`run_cohort`.
        NoSessionsSelectedError: If no session matches.
    """
    if trials_by_problem is None:
        trials_by_problem, _ = _load_cohort_for_run(cohort_id, data_root, use_cache, from_cache)
    if not trials_by_problem:
        raise ValueError(f"No problems found for {cohort_name(cohort_id)}")
    grouped = regroup_by_task_type(trials_by_problem, number_within_task_type(trials_by_problem))
    checked_exclusion(_drifting_of(grouped), exclude_mice)
    return select_recent_sessions(grouped, last_sessions=last_sessions, since=since)


def plan_cohort_pooled_only(
    cohort_id: int,
    *,
    exclude_mice: Iterable[str] | None = None,
    data_root: Path | None = None,
    use_cache: bool = True,
    from_cache: bool = False,
    trials_by_problem: dict | None = None,
) -> str:
    """What a pooled-only run of one cohort would regenerate, as text.

    Writes no results or tables (the import cache may be filled).

    Args:
        cohort_id (int): 1 for ``cohort-01``, and so on.
        exclude_mice (Iterable[str] | None): The mice of the extra pooled set. Keyword-only.
        data_root (Path | None): As for :func:`run_cohort`. Keyword-only.
        use_cache (bool): As for :func:`run_cohort`. Keyword-only.
        from_cache (bool): As for :func:`run_cohort`. Keyword-only.
        trials_by_problem (dict | None): Already-loaded trials, to skip the load. Keyword-only.

    Returns:
        str: The plan of :func:`~src.pipelines.incremental.describe_pooled_only_plan`.

    Raises:
        ValueError: If the cohort has no problems or no drifting problem, or for *exclude_mice*
            as in :func:`run_cohort` (so a name that would leave no mouse shows up here).
    """
    cohort = cohort_name(cohort_id)
    if trials_by_problem is None:
        trials_by_problem, _ = _load_cohort_for_run(cohort_id, data_root, use_cache, from_cache)
    if not trials_by_problem:
        raise ValueError(f"No problems found for {cohort}")
    grouped = regroup_by_task_type(trials_by_problem, number_within_task_type(trials_by_problem))
    drifting = _drifting_of(grouped)
    if not drifting:
        raise ValueError(f"{cohort} has no drifting problems for a pooled-only run")
    excluded, _ = checked_exclusion(drifting, exclude_mice)
    return describe_pooled_only_plan(cohort, drifting, excluded, _extra_set_name(excluded or None))


def run_cohort_incremental(
    cohort_id: int,
    *,
    last_sessions: int | None = None,
    since: str | None = None,
    data_root: Path | None = None,
    use_cache: bool = True,
    from_cache: bool = False,
    trials_by_problem: dict | None = None,
    import_log: ImportLog | None = None,
    exclude_mice: Iterable[str] | None = None,
    **settings,
) -> CohortResult:
    """Regenerate only what the recent sessions of a cohort change.

    Selects the sessions (each mouse's last N, or those since a date), prints which problems
    and pooled scopes will be regenerated and why, then runs :func:`run_cohort` for those
    problems alone (its ``only_problems``). The selection is made on the data the run uses,
    so the import happens once. With *exclude_mice* the pooled set without those mice is
    regenerated whenever the plain pooled set is.

    Args:
        cohort_id (int): 1 for ``cohort-01``, and so on.
        last_sessions (int | None): Each mouse's N most recent sessions. Exactly one of this and
            *since* is required. Keyword-only.
        since (str | None): Sessions dated on or after this ``YYYY-MM-DD`` date. Keyword-only.
        data_root (Path | None): As for :func:`run_cohort`. Keyword-only.
        use_cache (bool): As for :func:`run_cohort`. Keyword-only.
        from_cache (bool): As for :func:`run_cohort`. Keyword-only.
        trials_by_problem (dict | None): Already-loaded trials, to skip the load. Keyword-only.
        import_log (ImportLog | None): The log of *trials_by_problem*; ignored when the cohort
            is loaded here. Keyword-only.
        exclude_mice (Iterable[str] | None): Mice to leave out of the extra pooled set, as for
            :func:`run_cohort`. Keyword-only.
        **settings: The remaining keyword arguments of :func:`run_cohort` (``results_root``,
            selections, ...).

    Returns:
        CohortResult: As :func:`run_cohort` returns for a partial run.

    Raises:
        ValueError: As :func:`plan_cohort_incremental`.
        NoSessionsSelectedError: If no session matches; nothing is written.
    """
    exclude_mice = optional_mouse_ids(exclude_mice)
    log = import_log
    if trials_by_problem is None:
        trials_by_problem, log = _load_cohort_for_run(cohort_id, data_root, use_cache, from_cache)
    selection = plan_cohort_incremental(
        cohort_id,
        last_sessions=last_sessions,
        since=since,
        trials_by_problem=trials_by_problem,
        exclude_mice=exclude_mice,
    )
    extra_set = _extra_set_name(exclude_mice)
    print(f"[PLAN] {describe_plan(cohort_name(cohort_id), selection, extra_set)}")
    return run_cohort(
        cohort_id,
        use_cache=use_cache,
        from_cache=from_cache,
        trials_by_problem=trials_by_problem,
        import_log=log,
        only_problems=selection.problems,
        exclude_mice=exclude_mice,
        **settings,
    )


def check_exclusion_in_cohorts(
    cohort_ids: Iterable[int],
    exclude_mice: tuple[str, ...],
    data_root,
    use_cache: bool,
    from_cache: bool = False,
) -> dict[int, ImportedTrials]:
    """Check that every cohort has all of *exclude_mice* among its drifting mice.

    Imports each cohort (from the cache when it is there), so that a wrong name stops a
    multi-cohort run before the first cohort starts its long work.

    Args:
        cohort_ids (Iterable[int]): The cohorts a run will cover.
        exclude_mice (tuple[str, ...]): The mouse ids to leave out.
        data_root (Path | None): As for :func:`run_cohort`.
        use_cache (bool): As for :func:`run_cohort`.
        from_cache (bool): As for :func:`run_cohort`. Default ``False``.

    Returns:
        dict[int, ImportedTrials]: The imported cohorts, for a caller that keeps them.

    Raises:
        ExcludedMiceError: Naming every cohort the mice do not fit, and why.
    """
    problems, loaded = [], {}
    for cohort_id in cohort_ids:
        loaded[cohort_id] = _load_cohort_for_run(cohort_id, data_root, use_cache, from_cache)
        # unpacked rather than read by name, so any (trials, log) pair works
        trials, _ = loaded[cohort_id]
        grouped = regroup_by_task_type(trials, number_within_task_type(trials)) if trials else {}
        try:
            checked_exclusion(_drifting_of(grouped), exclude_mice)
        except ExcludedMiceError as err:
            problems.append(f"{cohort_name(cohort_id)}: {err}")
    if problems:
        raise ExcludedMiceError("\n".join(problems))
    return loaded


def _drifting_of(grouped: dict) -> dict:
    """``{k: {mouse: sessions}}`` of the drifting problems in *grouped*."""
    return {k: subs for (kind, k), subs in grouped.items() if kind == DRIFTING}


def _extra_set_name(exclude_mice: Iterable[str] | None) -> str | None:
    """Folder name of the pooled set without *exclude_mice*, or ``None`` for no such set."""
    if exclude_mice is None:
        return None
    return excluded_set_name(POOLED_DRIFTING_FOLDER, exclude_mice)


def _run_one_cohort(
    cohort_id: int,
    last_sessions,
    since,
    dry_run: bool,
    pooled_only: bool,
    exclude_mice,
    settings: dict,
    loaded: ImportedTrials | None = None,
):
    """One cohort of :func:`run_cohorts`: full, partial or dry-run; the result or ``None``.

    *loaded* is the cohort as already imported, so it is not loaded again.
    """
    name = cohort_name(cohort_id)
    planning = {
        "data_root": settings.get("data_root"),
        "use_cache": settings.get("use_cache", True),
        "from_cache": settings.get("from_cache", False),
    }
    if loaded is not None:
        trials, log = loaded
        planning["trials_by_problem"] = trials
        settings = {**settings, "trials_by_problem": trials, "import_log": log}
    if dry_run and pooled_only:
        plan = plan_cohort_pooled_only(cohort_id, exclude_mice=exclude_mice, **planning)
        print(f"[PLAN] dry run, nothing written. {plan}")
        return None
    if dry_run:
        plan = plan_cohort_incremental(
            cohort_id,
            last_sessions=last_sessions,
            since=since,
            exclude_mice=exclude_mice,
            **planning,
        )
        text = describe_plan(name, plan, _extra_set_name(exclude_mice))
        print(f"[PLAN] dry run, nothing written. {text}")
        return None
    # an option left at its default is not forwarded
    options = {"exclude_mice": exclude_mice} if exclude_mice is not None else {}
    if last_sessions is not None or since is not None:
        return run_cohort_incremental(
            cohort_id, last_sessions=last_sessions, since=since, **options, **settings
        )
    if pooled_only:
        options["pooled_only"] = True
    return run_cohort(cohort_id, **options, **settings)


def run_cohorts(
    cohort_ids,
    *,
    continue_on_error: bool = False,
    last_sessions: int | None = None,
    since: str | None = None,
    dry_run: bool = False,
    pooled_only: bool = False,
    exclude_mice: Iterable[str] | None = None,
    **settings,
) -> dict:
    """:func:`run_cohort` for each id; ``settings`` are passed on to every cohort.

    By default the first failing cohort stops the run. With ``continue_on_error`` the other
    cohorts still run, and the failures are raised together at the end.

    With *last_sessions* or *since* every cohort runs :func:`run_cohort_incremental`. A cohort
    with no matching session is skipped with a ``[SKIP]`` line (a date recent for one cohort
    may not reach another), but if no cohort matches at all the run raises.

    With *pooled_only* every cohort runs only its pooled drifting analyses. *exclude_mice*
    names mice of the cohorts run; each cohort must have all of them among its drifting mice,
    so name the cohort that holds them (``--cohorts 3``), not all four. A generator is read
    once.

    Args:
        cohort_ids (Iterable[int]): Cohort numbers to run, in order.
        continue_on_error (bool): Keep going after a failing cohort. Default ``False``.
            Keyword-only.
        last_sessions (int | None): Partial run on each mouse's N most recent sessions.
            Default ``None``. Keyword-only.
        since (str | None): Partial run on the sessions dated on or after this ``YYYY-MM-DD``
            date. Default ``None``. Keyword-only.
        dry_run (bool): Print the plan of the partial or pooled-only run (problems, pooled
            scopes, sessions) and write no results or tables (the import cache may be filled).
            Needs *last_sessions*, *since* or *pooled_only*. Default ``False``. Keyword-only.
        pooled_only (bool): Run only the pooled drifting analyses of each cohort (see
            :func:`run_cohort`); not with *last_sessions* or *since*. Default ``False``.
            Keyword-only.
        exclude_mice (Iterable[str] | None): Mice to leave out of an extra pooled set (see
            :func:`run_cohort`). Default ``None``. Keyword-only.
        **settings: Keyword arguments of :func:`run_cohort` (``data_root``, ``results_root``,
            selections, ...), applied to every cohort.

    Returns:
        dict[int, CohortResult]: ``{cohort_id: CohortResult}``; empty for a dry run.

    Raises:
        ValueError: For *dry_run* without a selection or *pooled_only*, *pooled_only* with a
            selection, an invalid mouse id, or invalid selection options (all checked before any
            cohort is loaded).
        ExcludedMiceError: If a cohort lacks one of *exclude_mice* among its drifting mice or
            would be left with none; every cohort is checked, and every misfit named, before
            the first one runs. Raised for a dry run too.
        RuntimeError: Naming each cohort that failed and why and, for a partial run, that no
            cohort has a matching session (both are reported when both happen).
    """
    selecting = last_sessions is not None or since is not None
    if pooled_only and selecting:
        raise ValueError(
            "pooled_only runs no problems, so it cannot be combined with last_sessions or since"
        )
    if dry_run and not (selecting or pooled_only):
        raise ValueError(
            "dry_run needs last_sessions, since or pooled_only: a full run has no plan to print"
        )
    if selecting:
        check_selection_options(last_sessions, since)
    exclude_mice = optional_mouse_ids(exclude_mice)  # once, so a generator reaches every cohort
    from_cache = settings.get("from_cache", False)
    preloaded = {}
    if exclude_mice is not None:
        checked = check_exclusion_in_cohorts(
            cohort_ids,
            exclude_mice,
            settings.get("data_root"),
            settings.get("use_cache", True),
            from_cache,
        )
        # A from-cache load cannot change between the check and the run, and loading again
        # would repeat its warning; a load from the server is redone so that it is current.
        # Otherwise the cohorts would all be held in memory for the whole run.
        preloaded = checked if from_cache else {}
    results, failures, unmatched, planned = {}, {}, {}, 0
    for cohort_id in cohort_ids:
        try:
            result = _run_one_cohort(
                cohort_id,
                last_sessions,
                since,
                dry_run,
                pooled_only,
                exclude_mice,
                settings,
                preloaded.get(cohort_id),
            )
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
