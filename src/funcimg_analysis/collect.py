"""Reward responses across every run of one destriping variant, and each run's top cells.

Runs are found under ``derivatives/<subject>/ses-*/funcimg/run-*/<variant>/suite2p/plane0``. A run
contributes only with an extracted plane and a ``frame_times.npz``; any other run is listed with
the reason it was left out, never dropped silently.
"""

import re
from collections.abc import Callable, Collection
from pathlib import Path

import numpy as np

from src.behavior_analysis.outcome_events import OutcomeParseError, load_outcome_events
from src.funcimg_analysis.event_frames import (
    pooled_mean_sem,
    top_cells_by_max,
    top_cells_by_score,
)
from src.funcimg_analysis.frame_times import (
    FRAME_TIMES_FILE,
    RunMismatchError,
    check_run_source,
    load_frame_times,
)
from src.funcimg_analysis.run_responses import (
    EXTRACTION_RECORD,
    PULSE_CATEGORIES,
    VIEWS,
    plane_views,
    pulse_responses,
)

RANKINGS = ("max_dff", "p_cell")
SESSION = re.compile(r"^ses-(\d+)_")


def variant_runs(
    subject_dir,
    first_session: int,
    last_session: int,
    variant: str,
    runs: Collection[str] | None = None,
):
    """``(session, run dir, plane dir)`` of each run folder of *variant* in range, in order.

    Args:
        subject_dir (str | Path): ``derivatives/<cohort>/<subject>``.
        first_session (int): First session number to include.
        last_session (int): Last session number to include.
        variant (str): ``"without_destriping"`` or ``"with_destriping"``.
        runs (Collection[str] | None): Run folder names to keep, e.g. ``("run-02",)``, in every
            session of the range. Default None, meaning every run.

    Returns:
        list[tuple[str, Path, Path]]: Sorted by session number, then run name.

    Raises:
        TypeError: If *runs* is a single string rather than a collection of names.
        ValueError: If *runs* names no run of *variant* in the session range (the runs that
            exist are listed).
    """
    if isinstance(runs, str):
        raise TypeError(f"runs must be a tuple of run names, e.g. ({runs!r},), not a string")
    found = []
    for session_dir in sorted(Path(subject_dir).glob("ses-*")):
        match = SESSION.match(session_dir.name)
        if not match or not first_session <= int(match.group(1)) <= last_session:
            continue
        for run_dir in sorted((session_dir / "funcimg").glob("run-*")):
            plane_dir = run_dir / variant / "suite2p" / "plane0"
            if (plane_dir / "ops.npy").is_file():
                found.append((session_dir.name, run_dir, plane_dir))
    found.sort(key=lambda r: (int(SESSION.match(r[0]).group(1)), r[1].name))
    if runs is None:
        return found
    kept = [r for r in found if r[1].name in set(runs)]
    if found and not kept:
        available = ", ".join(sorted({r[1].name for r in found}))
        raise ValueError(
            f"no run named {', '.join(sorted(runs))} in sessions {first_session}-{last_session} "
            f"({variant}); the runs there are {available}"
        )
    return kept


def _missing(run_dir: Path, plane_dir: Path) -> str | None:
    if not (run_dir / FRAME_TIMES_FILE).is_file():
        return f"no {FRAME_TIMES_FILE}"
    if not (plane_dir / EXTRACTION_RECORD).is_file():
        return "no Cellpose extraction"
    return None


def collect_pulse_responses(
    subject_dir,
    first_session: int,
    last_session: int,
    variant: str,
    window_s: tuple[float, float],
    phases: tuple[str, ...],
    load_outcomes: Callable = load_outcome_events,
    *,
    runs: Collection[str] | None = None,
) -> tuple[dict, list[dict], list[tuple[str, str, str]]]:
    """Pool every cell's reward response across the runs of *variant*.

    Args:
        subject_dir (str | Path): ``derivatives/<cohort>/<subject>``.
        first_session (int): First session number to include.
        last_session (int): Last session number to include.
        variant (str): ``"without_destriping"`` or ``"with_destriping"``.
        window_s (tuple[float, float]): Window around the outcome (s).
        phases (tuple[str, ...]): Trial phases to include.
        load_outcomes (Callable): Reads a behaviour file's outcome table (for tests).
        runs (Collection[str] | None): Run folder names to include, e.g. ``("run-02",)``; runs
            not named are neither pooled nor listed as left out. Default None, meaning every run.
            Keyword-only.

    Returns:
        tuple: ``{view: {pulses: {"mean", "sem", "n_cells", "n_runs"}}}`` pooled over cells;
        one result per included run (see ``pulse_responses``, plus ``session`` and ``run``);
        and ``(session, run, reason)`` for each run left out.
    """
    found = variant_runs(subject_dir, first_session, last_session, variant, runs)
    runs, skipped = [], []
    for session, run_dir, plane_dir in found:
        reason = _missing(run_dir, plane_dir)
        if reason is None:
            try:
                frames = load_frame_times(run_dir)
                check_run_source(frames.tiff.name, run_dir, variants=(variant,))
                views, _ = plane_views(plane_dir)
                result = pulse_responses(
                    views, frames.time_s, load_outcomes(frames.tsv), window_s, phases
                )
                runs.append({"session": session, "run": run_dir.name, **result})
                continue
            except (RunMismatchError, OutcomeParseError, FileNotFoundError) as error:
                reason = str(error)  # anything else is a bug and stops the analysis
        skipped.append((session, run_dir.name, reason))
    _check_same_window(runs)
    pooled = {
        view: pooled_mean_sem([run["means"][view] for run in runs], PULSE_CATEGORIES)
        for view in VIEWS
    }
    return pooled, runs, skipped


def _check_same_window(runs: list[dict]) -> None:
    shapes = {tuple(run["offsets"]) for run in runs}
    if len(shapes) > 1:
        raise ValueError("Runs have different frame periods, so their windows cannot be pooled.")


def top_cells(
    plane_dir,
    run_dir,
    n: int,
    load_outcomes: Callable = load_outcome_events,
    phases: tuple[str, ...] = ("task",),
    rank_by: str = "max_dff",
) -> dict:
    """The *n* top-ranked cells of a run, with what a figure needs.

    Args:
        plane_dir: The run's ``suite2p/plane0`` folder.
        run_dir: The run's derivatives folder.
        n (int): Number of cells.
        load_outcomes (Callable): Reads a behaviour file's outcome table (for tests).
        phases (tuple[str, ...]): Trial phases whose outcomes are marked.
        rank_by (str): ``"max_dff"`` for the largest green ΔF/F0 anywhere in the trace, or
            ``"p_cell"`` for Suite2p's classifier probability (``iscell.npy`` column 1, advisory
            for Cellpose ROIs). Ties go to the lower ROI index.

    Raises:
        ValueError: If *rank_by* is neither of those.

    Returns:
        dict: ``cells`` (ROI indices, best first), ``rank_by``, ``p_cell`` and ``peaks`` (their
        classifier probability and max ΔF/F0), ``dff``
        (``(n, n_frames)`` green ΔF/F0), ``time_s`` and ``period_s`` of the frames, ``labels``
        and ``mean_image`` of the plane, and ``outcomes`` (outcome time and pulses per task
        trial) to mark on the traces.
    """
    if rank_by not in RANKINGS:
        raise ValueError(f"rank_by must be one of {RANKINGS}, not {rank_by!r}.")
    frames = load_frame_times(run_dir)
    check_run_source(frames.tiff.name, run_dir, variants=(Path(plane_dir).parents[1].name,))
    views, traces = plane_views(plane_dir)
    dff = views[("green", "dF/F0")]
    if dff.shape[1] != len(frames.time_s):
        raise RunMismatchError(
            f"{plane_dir} has {dff.shape[1]} frames but there are {len(frames.time_s)} frame times."
        )
    p_cell = np.asarray(traces.iscell, dtype=float)[:, 1]
    cells = top_cells_by_max(dff, n) if rank_by == "max_dff" else top_cells_by_score(dff, p_cell, n)
    outcomes = load_outcomes(frames.tsv)
    return {
        "cells": cells,
        "rank_by": rank_by,
        "p_cell": p_cell[cells],
        "peaks": np.nanmax(dff[cells], axis=1),
        "dff": dff[cells],
        "time_s": frames.time_s,
        "period_s": frames.period_s,
        "labels": traces.labels,
        "mean_image": np.asarray(traces.ops["meanImg"], dtype=float),
        "outcomes": outcomes[outcomes["phase"].isin(phases)][["outcome_time", "num_rew_pulses"]],
    }
