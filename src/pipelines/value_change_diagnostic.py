"""Entry point: one figure per drifting session to check the best-decrease rank-swap windows.

:func:`run_cohort_diagnostic` loads a cohort the way the cohort runner does, keeps the drifting
problems and writes ``<results_root>/figures/value-change-diagnostic/cohort-XX/problem-KK/
<mouse>/<session>.png`` plus an ``index.txt`` listing every session's events. Sessions without
events get a figure too, marked as such.
"""

from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt

from src.behavior_analysis.value_change_diagnostic import EventDiagnostic, diagnose_session
from src.behavior_import.load_trials import load_trials_with_log
from src.behavior_import.session_order import ordered_session_keys
from src.behavior_import.task_types import DRIFTING, number_within_task_type, regroup_by_task_type
from src.behavior_visualization.plot_value_change_diagnostic import (
    NO_EVENTS_TEXT,
    plot_value_change_diagnostic,
)
from src.pipelines.cohort_analysis import DEFAULT_RESULTS_ROOT, cohort_name, cohort_raw_dir
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
    noun = "event" if len(diagnostics) == 1 else "events"
    return f"{session_key}: {len(diagnostics)} {noun}, t=0 at trial{'' if len(diagnostics) == 1 else 's'} {trials}"


def format_index(cohort: str, entries: Mapping[tuple[int, str], list[str]]) -> str:
    """The text of ``index.txt``: sessions grouped by problem and mouse.

    Args:
        cohort (str): Cohort folder name, for the heading.
        entries (Mapping[tuple[int, str], list[str]]): ``{(problem, mouse): [index_entry]}`` in
            the order to print.
    """
    lines = [f"{cohort}: best-decrease rank-swap events per drifting session", ""]
    for problem in sorted({problem for problem, _ in entries}):
        lines.append(f"problem-{problem:02d}")
        for (p, mouse), mouse_entries in entries.items():
            if p == problem:
                lines.append(f"  {mouse}")
                lines.extend(f"    {entry}" for entry in mouse_entries)
    return "\n".join(lines) + "\n"


def _write_session(
    mouse: str, key: str, session: Mapping, path: Path
) -> list[EventDiagnostic]:
    """Diagnose one session and save its figure at *path*."""
    diagnostics = diagnose_session(mouse, key, session)
    fig = plot_value_change_diagnostic(session, diagnostics, title=f"{mouse} | {key}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    finally:
        plt.close(fig)
    return diagnostics


def write_cohort_diagnostics(drifting: Mapping[int, Mapping], cohort_dir: Path) -> str:
    """Write every session's figure under *cohort_dir*; return the index text.

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
    return format_index(cohort_dir.name, entries)


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
