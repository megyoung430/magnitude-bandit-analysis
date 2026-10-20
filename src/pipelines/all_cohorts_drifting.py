"""The pooled drifting-problem analyses over several cohorts at once.

:func:`run_all_cohorts_drifting` writes what ``run_cohort`` writes into
``drifting/all-drifting-problems/`` of one cohort, but for the drifting problems of the chosen
cohorts together, in one folder that belongs to no cohort::

    figures/open-field/all-cohorts/{choice-stats,value-psychometrics,reversal-aligned-arms,
        value-change,port-preference,sessions-per-problem}/
    figures/open-field/all-cohorts/{reversal-aligned-arms,value-change}/{mouse-weighted,
        event-weighted}/   (their figures; nothing at the top level of those two folders)
    figures/open-field/all-cohorts/cohorts.txt
    tables/open-field/all-cohorts/{reversal_aligned_arms_counts,value_change_events,
        value_change_bin_counts,value_change_symmetric_exclusions,sessions_per_problem}.csv

Problems stay separate: each cohort's drifting problems are keyed ``(cohort id, number within
the type)``, so problem 1 of cohort 1 and problem 1 of cohort 3 are two problems, and every
analysis computes inside a problem (beliefs reset, windows and events end at its boundaries)
before the mice's results are combined. A mouse keeps its own id, so mouse ids must differ
between cohorts; a shared one raises.

``sessions-per-problem/`` holds the number of sessions each mouse ran in each drifting problem,
one line per mouse, and with several cohorts a second figure with one mean line per cohort;
``sessions_per_problem.csv`` has the cohort in front of the columns of the cohort runner's table.
A mouse's latest problem may still be running (see
:mod:`src.behavior_analysis.sessions_per_problem`), and the port-finding-only sessions of the
cohorts' import logs are counted in their own column.

With many mice, the figures leave out what the cohort folders already hold per mouse: the
per-mouse heatmaps, and the legends that name each mouse (their thin lines are still drawn).
"""

import shutil
import warnings
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from src.behavior_analysis.port_finding_sessions import assign_port_finding_sessions
from src.behavior_analysis.sessions_per_problem import sessions_per_problem_table, with_cohort
from src.behavior_import.import_log import ImportLog
from src.behavior_import.task_types import DRIFTING, number_within_task_type, regroup_by_task_type
from src.pipelines.cohort_analysis import (
    DEFAULT_RESULTS_ROOT,
    DRIFTING_SELECTION,
    TASK,
    cohort_name,
    each_reversal_warning_once,
    load_cohort,
    plot_pooled_drifting_rank,
    pooled_drifting_rank_results,
)
from src.pipelines.open_field import (
    PORT_PREFERENCE_FOLDER,
    REVERSAL_ARMS_FOLDER,
    VALUE_CHANGE_FOLDER,
    VALUE_PSYCHOMETRICS_FOLDER,
    AnalysisSelection,
    run_drifting_value_analyses,
)
from src.pipelines.sessions_per_problem import (
    SESSIONS_PER_PROBLEM_FOLDER,
    TABLE_NAME,
    write_sessions_per_problem_figures,
)

ALL_COHORTS_FOLDER = "all-cohorts"
COHORTS_FILE = "cohorts.txt"
RANK_FOLDER = "choice-stats"
# The folders and tables this pipeline writes; a rerun clears exactly these.
FIGURE_FOLDERS = (
    RANK_FOLDER,
    VALUE_PSYCHOMETRICS_FOLDER,
    REVERSAL_ARMS_FOLDER,
    VALUE_CHANGE_FOLDER,
    PORT_PREFERENCE_FOLDER,
    SESSIONS_PER_PROBLEM_FOLDER,
)
TABLE_NAMES = (
    "reversal_aligned_arms_counts",
    "value_change_events",
    "value_change_bin_counts",
    "value_change_symmetric_exclusions",
    TABLE_NAME,
)


class PooledCohorts(NamedTuple):
    """The drifting problems of several cohorts as one dict.

    Attributes:
        problems (dict): ``{(cohort id, k): {mouse: sessions}}``, ``k`` the drifting problem
            number within its cohort.
        mice_by_cohort (dict): ``{cohort id: sorted mice that have drifting problems}``.
    """

    problems: dict
    mice_by_cohort: dict


class AllCohortsResult(NamedTuple):
    """What :func:`run_all_cohorts_drifting` computed (the files are already written).

    Attributes:
        figures_dir (Path): ``<results_root>/figures/open-field/all-cohorts``.
        tables_dir (Path): ``<results_root>/tables/open-field/all-cohorts``.
        mice_by_cohort (dict): As in :class:`PooledCohorts`.
        rank_results (dict): Output of ``pooled_drifting_rank_results``.
        tables (dict): ``{name: DataFrame}`` of the tables written: the diagnostic tables and
            ``sessions_per_problem``.
    """

    figures_dir: Path
    tables_dir: Path
    mice_by_cohort: dict
    rank_results: dict
    tables: dict


# ---------------------------------------------------------------------------
# Analysis (pure)
# ---------------------------------------------------------------------------


def cohorts_phrase(cohort_ids) -> str:
    """``"cohort 3"`` or ``"cohorts 1, 3, 4"``."""
    ids = list(cohort_ids)
    return f"cohort {ids[0]}" if len(ids) == 1 else "cohorts " + ", ".join(map(str, ids))


def pooled_scope_label(cohort_ids) -> str:
    """The scope named in the figure titles, e.g. ``"all drifting problems, cohorts 1, 3, 4"``."""
    return f"all drifting problems, {cohorts_phrase(cohort_ids)}"


def check_cohort_ids(cohort_ids) -> list[int]:
    """The cohort ids, sorted.

    Raises:
        ValueError: If there are none, one is not a positive integer, or one is repeated.
    """
    ids = list(cohort_ids)
    if not ids:
        raise ValueError("no cohorts given; pass at least one cohort id")
    for cohort_id in ids:
        cohort_name(cohort_id)  # raises for anything but a positive integer
    repeated = sorted({c for c in ids if ids.count(c) > 1})
    if repeated:
        raise ValueError(f"cohort ids must be listed once each; repeated: {repeated}")
    return sorted(ids)


def drifting_problems(trials_by_problem: dict) -> dict:
    """The drifting problems of one cohort, numbered within the task type.

    Args:
        trials_by_problem (dict): ``{overall problem: {mouse: sessions}}`` as imported with the
            task types classified.

    Returns:
        dict: ``{k: {mouse: sessions}}``, ``k`` the drifting problem number; blocked problems are
        left out. Empty if the cohort has no drifting problem.
    """
    grouped = regroup_by_task_type(trials_by_problem, number_within_task_type(trials_by_problem))
    return {k: subjects for (kind, k), subjects in grouped.items() if kind == DRIFTING}


def pool_cohorts(drifting_by_cohort: dict) -> PooledCohorts:
    """Combine the drifting problems of several cohorts, keeping every problem separate.

    Args:
        drifting_by_cohort (dict): ``{cohort id: {k: {mouse: sessions}}}``, each value as
            returned by :func:`drifting_problems`.

    Returns:
        PooledCohorts: The problems keyed ``(cohort id, k)`` and the mice of each cohort.

    Raises:
        ValueError: If a cohort has no drifting problems, or a mouse id occurs in two cohorts
            (it would be pooled as one animal).
    """
    for cohort_id, problems in drifting_by_cohort.items():
        if not problems:
            raise ValueError(f"cohort {cohort_id} has no drifting problems to pool")
    mice_by_cohort = {
        cohort_id: sorted({m for subjects in problems.values() for m in subjects})
        for cohort_id, problems in drifting_by_cohort.items()
    }
    owners: dict[str, list] = {}
    for cohort_id, mice in mice_by_cohort.items():
        for mouse in mice:
            owners.setdefault(mouse, []).append(cohort_id)
    shared = {mouse: ids for mouse, ids in owners.items() if len(ids) > 1}
    if shared:
        detail = "; ".join(
            f"{m!r} is in {cohorts_phrase(ids)}" for m, ids in sorted(shared.items())
        )
        raise ValueError(f"mouse ids must be unique across cohorts, but {detail}")
    problems = {
        (cohort_id, k): subjects
        for cohort_id, by_k in drifting_by_cohort.items()
        for k, subjects in by_k.items()
    }
    return PooledCohorts(problems, mice_by_cohort)


def _mice(n: int) -> str:
    return f"{n} mouse" if n == 1 else f"{n} mice"


def cohorts_description(mice_by_cohort: dict, run_at: datetime) -> str:
    """The text of ``cohorts.txt``: which cohorts were pooled, how many mice, and when."""
    lines = [
        f"Drifting problems pooled over {cohorts_phrase(mice_by_cohort)}",
        f"Run at {run_at:%Y-%m-%d %H:%M:%S}",
        "Mice with drifting problems:",
        *(f"{cohort_name(c)}: {_mice(len(mice))}" for c, mice in mice_by_cohort.items()),
        f"Total: {_mice(sum(map(len, mice_by_cohort.values())))}",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# I/O and plotting (the edges)
# ---------------------------------------------------------------------------


def _clear_previous_outputs(figures_dir: Path, tables_dir: Path) -> None:
    """Remove what an earlier run wrote, so a rerun with other cohorts leaves no stale figure.

    Only the folders and tables this pipeline owns are touched.
    """
    for folder in FIGURE_FOLDERS:
        shutil.rmtree(figures_dir / folder, ignore_errors=True)
    for name in TABLE_NAMES:
        (tables_dir / f"{name}.csv").unlink(missing_ok=True)


def _inputs_for_cohorts(
    cohort_ids, trials_by_cohort, import_logs, data_root, use_cache
) -> tuple[dict, dict]:
    """``({cohort id: trials by problem}, {cohort id: ImportLog or None})``, passed in or loaded.

    Raises:
        ValueError: If trials are passed in but lack a cohort, or logs are given for a cohort that
            is not run.
    """
    if trials_by_cohort is None:
        loaded = {c: load_cohort(c, data_root, use_cache) for c in cohort_ids}
        return {c: trials for c, (trials, _) in loaded.items()}, {
            c: log for c, (_, log) in loaded.items()
        }
    missing = [c for c in cohort_ids if c not in trials_by_cohort]
    if missing:
        raise ValueError(f"no trials given for {cohorts_phrase(missing)}")
    extra = sorted(set(import_logs or {}) - set(cohort_ids))
    if extra:
        raise ValueError(f"import logs given for {cohorts_phrase(extra)}, which is not run")
    logs = import_logs or {}
    unlogged = [c for c in cohort_ids if c not in logs]
    if unlogged:
        warnings.warn(
            f"trials were passed in without import_logs for {cohorts_phrase(unlogged)}, so their "
            "port-finding-only sessions are unknown: sessions_per_problem.csv counts none",
            stacklevel=3,
        )
    return {c: trials_by_cohort[c] for c in cohort_ids}, {c: logs.get(c) for c in cohort_ids}


def _sessions_per_problem(trials_by_cohort: dict, logs: dict) -> pd.DataFrame:
    """The stacked sessions-per-problem tables of the cohorts, the cohort name in front.

    A cohort without an import log has no port-finding-only sessions.
    """
    tables = []
    for cohort_id, trials in trials_by_cohort.items():
        log = logs[cohort_id]
        assignment = assign_port_finding_sessions(trials, log.port_finding_sessions if log else {})
        tables.append(
            with_cohort(sessions_per_problem_table(trials, assignment), cohort_name(cohort_id))
        )
    return pd.concat(tables, ignore_index=True)


def run_all_cohorts_drifting(
    cohort_ids,
    *,
    data_root: Path | None = None,
    results_root: Path = DEFAULT_RESULTS_ROOT,
    drifting_selection: AnalysisSelection = DRIFTING_SELECTION,
    use_cache: bool = True,
    trials_by_cohort: dict | None = None,
    import_logs: dict[int, ImportLog] | None = None,
    run_at: datetime | None = None,
) -> AllCohortsResult:
    """Run the all-drifting-problems analyses over the drifting problems of several cohorts.

    See the module docstring for the outputs. Anything an earlier run left in the six figure
    folders and five tables of the output is removed first, so the files always belong to the
    cohorts named in ``cohorts.txt``.

    Args:
        cohort_ids (Iterable[int]): The cohorts to pool, e.g. ``[1, 3, 4]``.
        data_root (Path | None): Folder holding ``3x3_field_magnitude_bandit``. Default ``None``:
            the ``BANDIT_DATA_ROOT`` setting. Keyword-only.
        results_root (Path): Folder that receives ``figures/`` and ``tables/``. Default
            ``DEFAULT_RESULTS_ROOT`` (``<repo>/results``). Keyword-only.
        drifting_selection (AnalysisSelection): Which drifting analyses run, and the rank value
            bases and pooling modes. Default ``DRIFTING_SELECTION``. Keyword-only.
        use_cache (bool): Use the on-disk import cache. Default ``True``. Keyword-only.
        trials_by_cohort (dict | None): Already-loaded ``{cohort id: {overall problem: {mouse:
            sessions}}}``, to skip the load. Default ``None``: load the cohorts. Keyword-only.
        import_logs (dict[int, ImportLog] | None): ``{cohort id: import log}`` of the passed-in
            *trials_by_cohort*, for their port-finding-only sessions; a cohort without one has
            none. Ignored when the cohorts are loaded. Default ``None``. Keyword-only.
        run_at (datetime | None): The time written to ``cohorts.txt``. Default ``None``: now.
            Keyword-only.

    Returns:
        AllCohortsResult: The output folders, mice per cohort, rank results and tables.

    Raises:
        ValueError: For no, repeated or invalid cohort ids, a cohort without trials or without
            drifting problems, a mouse id shared by two cohorts, or a selection without rank
            value bases.
    """
    ids = check_cohort_ids(cohort_ids)
    if drifting_selection.rank_value_bases is None:
        raise ValueError("drifting_selection.rank_value_bases must name the value bases to pool")
    imported, logs = _inputs_for_cohorts(ids, trials_by_cohort, import_logs, data_root, use_cache)
    pooled = pool_cohorts({c: drifting_problems(trials) for c, trials in imported.items()})
    figures_dir = Path(results_root) / "figures" / TASK / ALL_COHORTS_FOLDER
    tables_dir = Path(results_root) / "tables" / TASK / ALL_COHORTS_FOLDER
    label = pooled_scope_label(ids)
    print(f"[INFO] {label}: {len(pooled.problems)} problems, {pooled.mice_by_cohort}")

    _clear_previous_outputs(figures_dir, tables_dir)
    with each_reversal_warning_once():
        rank_results = pooled_drifting_rank_results(
            pooled.problems,
            drifting_selection.rank_value_bases,
            drifting_selection.rank_pooling_modes,
        )
        plot_pooled_drifting_rank(rank_results, figures_dir / RANK_FOLDER, mouse_legend=False)
        tables = run_drifting_value_analyses(
            pooled.problems,
            figures_dir,
            drifting_selection,
            arm_scope_label=label,
            x_vs_plus=True,
            per_mouse_figures=False,
            mouse_legends=False,
        )
        tables[TABLE_NAME] = _sessions_per_problem(imported, logs)
        write_sessions_per_problem_figures(
            tables[TABLE_NAME],
            figures_dir / SESSIONS_PER_PROBLEM_FOLDER,
            cohorts_phrase(ids),
            mouse_legend=False,
            by_cohort=len(ids) > 1,
        )
    tables_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(tables_dir / f"{name}.csv", index=False)
    figures_dir.mkdir(parents=True, exist_ok=True)
    (figures_dir / COHORTS_FILE).write_text(
        cohorts_description(pooled.mice_by_cohort, run_at or datetime.now())
    )
    return AllCohortsResult(figures_dir, tables_dir, pooled.mice_by_cohort, rank_results, tables)
