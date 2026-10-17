"""Entry point: one figure per drifting session to check the best-decrease rank-swap windows.

:func:`run_cohort_diagnostic` loads a cohort the way the cohort runner does, keeps the drifting
problems and writes ``<results_root>/figures/value-change-diagnostic/cohort-XX/problem-KK/
<mouse>/<session>.png`` plus an ``index.txt`` listing every session's events. Sessions without
events get a figure too, marked as such.

It also writes ``mouse_curves_<scope>.csv`` for each drifting problem (``problem-KK``) and for
all of them pooled (``all-drifting-problems``): the per-mouse choice probabilities whose mean
the mouse-weighted value-change figure plots, for the same events, window and both truncation
modes (see :func:`src.behavior_analysis.value_change_mouse_curves.mouse_curves_table`).
"""

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.behavior_analysis.value_change_diagnostic import (
    DIRECTION,
    POST,
    PRE,
    RANK,
    RANK_FILTER,
    EventDiagnostic,
    diagnose_session,
)
from src.behavior_analysis.value_change_mouse_curves import mouse_curves_table
from src.behavior_analysis.value_change_windows import (
    ValueChangeWindows,
    get_value_change_windows,
    merge_value_change_windows,
    summarize_windows,
    truncate_windows_at_rank_swap,
)
from src.behavior_import.load_trials import load_trials_with_log
from src.behavior_import.session_order import ordered_session_keys
from src.behavior_import.task_types import DRIFTING, number_within_task_type, regroup_by_task_type
from src.behavior_visualization.plot_value_change_diagnostic import (
    NO_EVENTS_TEXT,
    plot_value_change_diagnostic,
)
from src.pipelines.cohort_analysis import (
    DEFAULT_RESULTS_ROOT,
    POOLED_DRIFTING_FOLDER,
    cohort_name,
    cohort_raw_dir,
)
from src.settings import get_settings

DIAGNOSTIC_FOLDER = "value-change-diagnostic"
# PNG only, at a resolution that is readable but quick to page through.
FIGURE_DPI = 100


def session_figure_path(cohort_dir: Path, problem: int, mouse: str, session_key: str) -> Path:
    """``<cohort_dir>/problem-KK/<mouse>/<session_key>.png``."""
    return cohort_dir / f"problem-{problem:02d}" / mouse / f"{session_key}.png"


def index_entry(session_key: str, diagnostics: list[EventDiagnostic]) -> str:
    """One index line: the session, its number of events and their t = 0 trials."""
    if not diagnostics:
        return f"{session_key}: {NO_EVENTS_TEXT}"
    trials = ", ".join(str(d.trial_idx) for d in diagnostics)
    if len(diagnostics) == 1:
        return f"{session_key}: 1 event, t=0 at trial {trials}"
    return f"{session_key}: {len(diagnostics)} events, t=0 at trials {trials}"


def format_index(
    cohort: str, entries: Mapping[tuple[int, str], list[str]], csv_names: Sequence[str] = ()
) -> str:
    """The text of ``index.txt``: sessions grouped by problem and mouse, then the CSV files.

    Args:
        cohort (str): Cohort folder name, for the heading.
        entries (Mapping[tuple[int, str], list[str]]): ``{(problem, mouse): [index_entry]}`` in
            the order to print.
        csv_names (Sequence[str]): File names of the per-mouse curve CSVs to list. Default none.
    """
    lines = [f"{cohort}: best-decrease rank-swap events per drifting session", ""]
    for problem in sorted({problem for problem, _ in entries}):
        lines.append(f"problem-{problem:02d}")
        for (p, mouse), mouse_entries in entries.items():
            if p == problem:
                lines.append(f"  {mouse}")
                lines.extend(f"    {entry}" for entry in mouse_entries)
    if csv_names:
        lines += ["", "per-mouse choice probabilities behind the mouse-weighted plot (both modes)"]
        lines += [f"  {name}" for name in csv_names]
    return "\n".join(lines) + "\n"


def _write_session(mouse: str, key: str, session: Mapping, path: Path) -> list[EventDiagnostic]:
    """Diagnose one session and save its figure at *path*."""
    diagnostics = diagnose_session(mouse, key, session)
    fig = plot_value_change_diagnostic(session, diagnostics, title=f"{mouse} | {key}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    finally:
        plt.close(fig)
    return diagnostics


def scope_windows(problems: Mapping[int, Mapping]) -> ValueChangeWindows:
    """Untruncated best-decrease rank-swap events of *problems*, merged per mouse.

    The window and configuration are those of the diagnostic figures and the value-change
    figures; a mouse's events are pooled across the problems.

    Args:
        problems (Mapping[int, Mapping]): ``{k: {mouse: {session_key: session}}}``.
    """
    per_problem = [
        get_value_change_windows(problems[k], RANK, DIRECTION, RANK_FILTER, pre=PRE, post=POST)
        for k in sorted(problems)
    ]
    return merge_value_change_windows(per_problem)


def scope_mouse_curves(scope: str, problems: Mapping[int, Mapping]) -> pd.DataFrame:
    """The per-mouse curve table of *problems* for both truncation modes, full window first."""
    full = scope_windows(problems)
    tables = [
        mouse_curves_table(summarize_windows(windows), scope=scope)
        for windows in (full, truncate_windows_at_rank_swap(full))
    ]
    return pd.concat(tables, ignore_index=True)


def mouse_curves_path(cohort_dir: Path, scope: str) -> Path:
    """``<cohort_dir>/mouse_curves_<scope>.csv``."""
    return cohort_dir / f"mouse_curves_{scope}.csv"


def write_mouse_curves(drifting: Mapping[int, Mapping], cohort_dir: Path) -> list[Path]:
    """Write one per-mouse curves CSV per drifting problem and one for all of them pooled.

    Args:
        drifting (Mapping[int, Mapping]): ``{k: {mouse: {session_key: session}}}``.
        cohort_dir (Path): The cohort's output folder (must exist).

    Returns:
        list[Path]: The CSVs, the problems in order and the pooled one last.
    """
    scopes = {f"problem-{k:02d}": {k: drifting[k]} for k in sorted(drifting)}
    scopes[POOLED_DRIFTING_FOLDER] = dict(drifting)
    paths = []
    for scope, problems in scopes.items():
        path = mouse_curves_path(cohort_dir, scope)
        scope_mouse_curves(scope, problems).to_csv(path, index=False)
        paths.append(path)
    return paths


def write_cohort_diagnostics(drifting: Mapping[int, Mapping], cohort_dir: Path) -> str:
    """Write the session figures and per-mouse curve CSVs under *cohort_dir*; return the index.

    Args:
        drifting (Mapping[int, Mapping]): ``{k: {mouse: {session_key: session}}}``, the drifting
            problems numbered within their type.
        cohort_dir (Path): The cohort's output folder.

    Raises:
        ValueError: As :func:`diagnose_session`, naming the mouse and session.
    """
    entries: dict[tuple[int, str], list[str]] = {}
    for k, subjects in drifting.items():
        for mouse, sessions in subjects.items():
            for key in ordered_session_keys(mouse, sessions):
                if not sessions[key].get("trial_variables"):
                    entries.setdefault((k, mouse), []).append(f"{key}: no trials, no figure")
                    continue
                path = session_figure_path(cohort_dir, k, mouse, key)
                diagnostics = _write_session(mouse, key, sessions[key], path)
                entries.setdefault((k, mouse), []).append(index_entry(key, diagnostics))
    csv_names = [path.name for path in write_mouse_curves(drifting, cohort_dir)]
    return format_index(cohort_dir.name, entries, csv_names)


def run_cohort_diagnostic(
    cohort_id: int,
    *,
    data_root: Path | None = None,
    results_root: Path = DEFAULT_RESULTS_ROOT,
    use_cache: bool = True,
) -> Path:
    """Load a cohort, write the diagnostic figures of its drifting problems and the index.

    Args:
        cohort_id (int): Cohort number.
        data_root (Path | None): Folder holding the experiment folder. Default None, meaning
            ``get_settings().data_root``.
        results_root (Path): Output root. Default ``DEFAULT_RESULTS_ROOT``.
        use_cache (bool): Whether the import may use its on-disk cache. Default True.

    Returns:
        Path: The cohort's output folder.

    Raises:
        ValueError: If the cohort has no drifting problem.
    """
    root = get_settings().data_root if data_root is None else data_root
    trials = load_trials_with_log(
        cohort_raw_dir(root, cohort_id), use_cache=use_cache, classify_task_types=True
    ).trials_by_problem
    grouped = regroup_by_task_type(trials, number_within_task_type(trials))
    drifting = {k: subjects for (kind, k), subjects in grouped.items() if kind == DRIFTING}
    cohort = cohort_name(cohort_id)
    if not drifting:
        raise ValueError(f"{cohort} has no drifting problems")
    cohort_dir = results_root / "figures" / DIAGNOSTIC_FOLDER / cohort
    cohort_dir.mkdir(parents=True, exist_ok=True)
    (cohort_dir / "index.txt").write_text(write_cohort_diagnostics(drifting, cohort_dir))
    return cohort_dir
