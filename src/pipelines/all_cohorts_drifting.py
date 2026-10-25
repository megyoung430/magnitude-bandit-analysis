"""The pooled drifting-problem analyses over several cohorts at once.

:func:`run_all_cohorts_drifting` writes what ``run_cohort`` writes into
``drifting/all-drifting-problems/`` of one cohort, but for the drifting problems of the chosen
cohorts together, in one folder that belongs to no cohort::

    figures/open-field/all-cohorts/{choice-stats,value-psychometrics,reversal-aligned-arms,
        value-change,port-preference,sessions-per-problem,port-finding-summary}/
    figures/open-field/all-cohorts/{reversal-aligned-arms,value-change}/{mouse-weighted,
        event-weighted}/   (their figures; nothing at the top level of those two folders)
    figures/open-field/all-cohorts/cohorts.txt
    tables/open-field/all-cohorts/{reversal_aligned_arms_counts,value_change_events,
        value_change_bin_counts,value_change_symmetric_exclusions,sessions_per_problem}.csv

With ``exclude_mice`` a second set is written next to it, from the same problems without those
mice, into ``figures/open-field/all-cohorts-without-<mice>/`` and
``tables/open-field/all-cohorts-without-<mice>/`` (same layout, every figure's caption starting
"without <mice>", plus ``excluded_mice.txt``); the plain set is as without the option.

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

``port-finding-summary/`` (when the selection runs the poke analysis) holds the port-finding
heatmaps pooled by port layout, one figure per layout and poke kind (the same layout is often a
different problem number in each cohort), and the port-category fractions of all the drifting
problems (see :mod:`src.pipelines.port_finding_summary`).

With many mice, the figures leave out what the cohort folders already hold per mouse: the
per-mouse heatmaps, and the legends that name each mouse (their thin lines are still drawn).
"""

import shutil
import warnings
from collections.abc import Iterable
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from src.behavior_analysis.exclude_mice import (
    exclude_mice_from_problems,
    exclude_mice_from_table,
    excluded_mice_description,
    excluded_scope_text,
    excluded_set_name,
    optional_mouse_ids,
)
from src.behavior_analysis.port_finding_sessions import assign_port_finding_sessions
from src.behavior_analysis.port_finding_summary import (
    MousePortFinding,
    cohort_port_finding_summaries,
)
from src.behavior_analysis.sessions_per_problem import sessions_per_problem_table, with_cohort
from src.behavior_import.import_log import ImportLog
from src.behavior_import.task_types import DRIFTING, number_within_task_type, regroup_by_task_type
from src.behavior_visualization.plot_helpers import figure_banner
from src.pipelines.cache_provenance import (
    cache_sources,
    write_from_cache_notes,
)
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
from src.pipelines.port_finding_summary import (
    PORT_FINDING_SUMMARY_FOLDER,
    write_category_fractions,
    write_layout_port_finding_heatmaps,
)
from src.pipelines.sessions_per_problem import (
    SESSIONS_PER_PROBLEM_FOLDER,
    TABLE_NAME,
    write_sessions_per_problem_figures,
)

ALL_COHORTS_FOLDER = "all-cohorts"
COHORTS_FILE = "cohorts.txt"
EXCLUDED_MICE_FILE = "excluded_mice.txt"
RANK_FOLDER = "choice-stats"
# The folders and tables this pipeline writes; a rerun clears exactly these.
FIGURE_FOLDERS = (
    RANK_FOLDER,
    VALUE_PSYCHOMETRICS_FOLDER,
    REVERSAL_ARMS_FOLDER,
    VALUE_CHANGE_FOLDER,
    PORT_PREFERENCE_FOLDER,
    SESSIONS_PER_PROBLEM_FOLDER,
    PORT_FINDING_SUMMARY_FOLDER,
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
        without (AllCohortsResult | None): The same for the set without the ``exclude_mice``
            (``figures_dir`` and ``tables_dir`` are its own folders); ``None`` if none was asked
            for.
    """

    figures_dir: Path
    tables_dir: Path
    mice_by_cohort: dict
    rank_results: dict
    tables: dict
    without: "AllCohortsResult | None" = None


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
    cohort_ids, trials_by_cohort, import_logs, data_root, use_cache, from_cache=False
) -> tuple[dict, dict]:
    """``({cohort id: trials by problem}, {cohort id: ImportLog or None})``, passed in or loaded.

    Raises:
        ValueError: If trials are passed in but lack a cohort, or logs are given for a cohort that
            is not run.
    """
    if trials_by_cohort is None:
        loaded = {c: load_cohort(c, data_root, use_cache, from_cache) for c in cohort_ids}
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


def _mice_by_cohort(problems: dict) -> dict:
    """``{cohort id: sorted mice}`` over problems keyed ``(cohort id, k)``."""
    found: dict[int, set] = {}
    for (cohort_id, _), subjects in problems.items():
        found.setdefault(cohort_id, set()).update(subjects)
    return {cohort_id: sorted(mice) for cohort_id, mice in sorted(found.items())}


def _draw_pool(
    problems: dict,
    sessions: pd.DataFrame,
    figures_dir: Path,
    selection: AnalysisSelection,
    shown_ids: list[int],
    label: str,
    scope: str | None,
    port_finding: dict[tuple[str, int, str], MousePortFinding],
) -> tuple[dict, dict]:
    """Draw every figure of the pool into *figures_dir*; return the rank results and tables."""
    rank_results = pooled_drifting_rank_results(
        problems, selection.rank_value_bases, selection.rank_pooling_modes
    )
    plot_pooled_drifting_rank(rank_results, figures_dir / RANK_FOLDER, mouse_legend=False)
    tables = run_drifting_value_analyses(
        problems,
        figures_dir,
        selection,
        arm_scope_label=label,
        x_vs_plus=True,
        per_mouse_figures=False,
        mouse_legends=False,
    )
    tables[TABLE_NAME] = sessions
    write_sessions_per_problem_figures(
        sessions,
        figures_dir / SESSIONS_PER_PROBLEM_FOLDER,
        cohorts_phrase(shown_ids) + (f" {scope}" if scope else ""),
        mouse_legend=False,
        by_cohort=len(shown_ids) > 1,
    )
    if port_finding:
        _draw_port_finding_summary(port_finding, figures_dir / PORT_FINDING_SUMMARY_FOLDER, label)
    return rank_results, tables


def _draw_port_finding_summary(entries: dict, out_dir: Path, label: str) -> None:
    """The per-layout heatmaps and the port-category fractions of the pooled drifting problems."""
    write_layout_port_finding_heatmaps(entries, out_dir, scope=label)
    write_category_fractions(
        [({"cohort": c, "task_type": DRIFTING, "k": k}, s) for (c, k, _), s in entries.items()],
        out_dir,
        scope=label,
        mouse_columns=("cohort", "mouse"),
    )


def _port_finding_entries(trials_by_cohort: dict, logs: dict) -> dict:
    """``{(cohort name, k, mouse): MousePortFinding}`` over the cohorts' drifting problems.

    A cohort without an import log has no port-finding-only sessions.
    """
    entries = {}
    for cohort_id, trials in trials_by_cohort.items():
        log = logs[cohort_id]
        found = log.port_finding_sessions if log else {}
        by_problem = cohort_port_finding_summaries(trials, found, task_types=[DRIFTING])
        entries |= {
            (cohort_name(cohort_id), k, mouse): summary
            for (_, k), mice in by_problem.items()
            for mouse, summary in mice.items()
        }
    return entries


def _write_pool(
    problems: dict,
    sessions: pd.DataFrame,
    *,
    ids: list[int],
    figures_dir: Path,
    tables_dir: Path,
    selection: AnalysisSelection,
    run_at: datetime,
    excluded: tuple[str, ...] = (),
    sources: dict | None = None,
    port_finding: dict | None = None,
) -> AllCohortsResult:
    """Run and write the all-drifting-problems analyses of *problems* into the two folders.

    Anything an earlier run left in the folders this pipeline owns is removed first. With
    *excluded* the scope text and the first line of every figure's caption say which mice are
    left out, and ``excluded_mice.txt`` lists them.

    Args:
        problems (dict): ``{(cohort id, k): {mouse: sessions}}`` to pool.
        sessions (pd.DataFrame): The stacked sessions-per-problem table of the problems' mice.
        ids (list[int]): The cohorts named in the captions.
        figures_dir (Path): Folder that receives the figures.
        tables_dir (Path): Folder that receives the tables.
        selection (AnalysisSelection): Which drifting analyses run, and the rank settings.
        run_at (datetime): The time written to ``cohorts.txt``.
        excluded (tuple[str, ...]): Mice left out of *problems*; empty for the plain pool.
        sources (dict | None): ``{cohort: CacheSource}`` of cohorts loaded from the cache
            without the server; each folder then gets ``from-cache.txt`` saying so. Empty or
            ``None``: the folders' notes are removed. Default None.
        port_finding (dict | None): ``{(cohort name, k, mouse): MousePortFinding}`` of the
            problems' mice, for ``port-finding-summary/``; empty or ``None`` draws none (the
            mice of *excluded* are dropped from it). Default None.

    Returns:
        AllCohortsResult: The folders, the mice per cohort of *problems*, rank results and tables.
    """
    mice_by_cohort = _mice_by_cohort(problems)
    shown_ids = list(mice_by_cohort)  # a cohort whose mice are all excluded is not in the pool
    scope = excluded_scope_text(excluded) if excluded else None
    label = pooled_scope_label(shown_ids) + (f" {scope}" if scope else "")
    print(f"[INFO] {label}: {len(problems)} problems, {mice_by_cohort}")

    _clear_previous_outputs(figures_dir, tables_dir)
    sources = sources or {}
    write_from_cache_notes([figures_dir, tables_dir], sources)
    with each_reversal_warning_once(), figure_banner(scope) if scope else nullcontext():
        kept = {key: v for key, v in (port_finding or {}).items() if key[2] not in excluded}
        rank_results, tables = _draw_pool(
            problems, sessions, figures_dir, selection, shown_ids, label, scope, kept
        )
    tables_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(tables_dir / f"{name}.csv", index=False)
    figures_dir.mkdir(parents=True, exist_ok=True)
    (figures_dir / COHORTS_FILE).write_text(cohorts_description(mice_by_cohort, run_at))
    if excluded:
        description = excluded_mice_description(cohorts_phrase(ids), excluded, run_at)
        for folder in (figures_dir, tables_dir):
            (folder / EXCLUDED_MICE_FILE).write_text(description)
    return AllCohortsResult(figures_dir, tables_dir, mice_by_cohort, rank_results, tables)


def run_all_cohorts_drifting(
    cohort_ids,
    *,
    data_root: Path | None = None,
    results_root: Path = DEFAULT_RESULTS_ROOT,
    drifting_selection: AnalysisSelection = DRIFTING_SELECTION,
    use_cache: bool = True,
    from_cache: bool = False,
    trials_by_cohort: dict | None = None,
    import_logs: dict[int, ImportLog] | None = None,
    run_at: datetime | None = None,
    exclude_mice: Iterable[str] | None = None,
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
        from_cache (bool): Load the newest cached import of each cohort without reading the
            share; sessions added since are missing (see
            :func:`~src.pipelines.cohort_analysis.load_cohort`). Default ``False``.
            Keyword-only.
        trials_by_cohort (dict | None): Already-loaded ``{cohort id: {overall problem: {mouse:
            sessions}}}``, to skip the load. Default ``None``: load the cohorts. Keyword-only.
        import_logs (dict[int, ImportLog] | None): ``{cohort id: import log}`` of the passed-in
            *trials_by_cohort*, for their port-finding-only sessions; a cohort without one has
            none. Ignored when the cohorts are loaded. Default ``None``. Keyword-only.
        run_at (datetime | None): The time written to ``cohorts.txt``. Default ``None``: now.
            Keyword-only.
        exclude_mice (Iterable[str] | None): Mouse ids (of any of the cohorts) to leave out of
            a second set, written to ``all-cohorts-without-<mice>/``. Default ``None``: no
            second set. Keyword-only.

    Returns:
        AllCohortsResult: The output folders, mice per cohort, rank results and tables; its
        ``without`` holds the same for the second set.

    Raises:
        ValueError: For no, repeated or invalid cohort ids, a cohort without trials or without
            drifting problems, a mouse id shared by two cohorts, a selection without rank
            value bases, or *exclude_mice* that is empty, names a mouse of no cohort or all of
            them (checked before anything is written).
    """
    ids = check_cohort_ids(cohort_ids)
    if drifting_selection.rank_value_bases is None:
        raise ValueError("drifting_selection.rank_value_bases must name the value bases to pool")
    imported, logs = _inputs_for_cohorts(
        ids, trials_by_cohort, import_logs, data_root, use_cache, from_cache
    )
    pooled = pool_cohorts({c: drifting_problems(trials) for c, trials in imported.items()})
    excluded = optional_mouse_ids(exclude_mice) or ()  # a tuple, so a generator is read once
    problems_without = exclude_mice_from_problems(pooled.problems, excluded) if excluded else {}
    run_at = run_at or datetime.now()
    sources = cache_sources({cohort_name(c): log for c, log in logs.items()})
    sessions = _sessions_per_problem(imported, logs)
    port_finding = (
        _port_finding_entries(imported, logs) if drifting_selection.run_poke_analysis else {}
    )
    root = Path(results_root)
    folders = {kind: root / kind / TASK / ALL_COHORTS_FOLDER for kind in ("figures", "tables")}

    plain = _write_pool(
        pooled.problems,
        sessions,
        ids=ids,
        figures_dir=folders["figures"],
        tables_dir=folders["tables"],
        selection=drifting_selection,
        run_at=run_at,
        sources=sources,
        port_finding=port_finding,
    )
    if not excluded:
        return plain
    name = excluded_set_name(ALL_COHORTS_FOLDER, excluded)
    without = _write_pool(
        problems_without,
        exclude_mice_from_table(sessions, excluded),
        ids=ids,
        figures_dir=root / "figures" / TASK / name,
        tables_dir=root / "tables" / TASK / name,
        selection=drifting_selection,
        run_at=run_at,
        excluded=excluded,
        sources=sources,
        port_finding=port_finding,
    )
    return plain._replace(without=without)
