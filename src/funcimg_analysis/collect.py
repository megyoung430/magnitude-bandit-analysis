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
from src.funcimg_analysis.background_io import BackgroundUnavailableError, load_background
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
    BACKGROUND_MEASURES,
    EXTRACTION_RECORD,
    MEASURES,
    PULSE_CATEGORIES,
    VIEWS,
    check_measures,
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


def collect_each_run(
    found: list,
    variant: str,
    load_outcomes: Callable,
    analyse: Callable[..., dict],
    also_skip: tuple[type[Exception], ...] = (),
) -> tuple[list[dict], list[tuple[str, str, str]]]:
    """Run *analyse* on every found run that can be analysed; list the others and why.

    *analyse* is called as ``analyse(session, run_dir, plane_dir, frames, outcomes)`` and
    returns the run's result dict, to which ``session`` and ``run`` are added. A run is left
    out, with the reason, if it lacks frame times or an extraction, belongs to another
    acquisition, has an unreadable behaviour file or lacks what *analyse* reports as missing
    (a background, or any exception class in *also_skip*); anything else is a bug and stops the
    analysis.
    """
    included, skipped = [], []
    for session, run_dir, plane_dir in found:
        reason = _missing(run_dir, plane_dir)
        if reason is None:
            try:
                frames = load_frame_times(run_dir)
                check_run_source(frames.tiff.name, run_dir, variants=(variant,))
                result = analyse(session, run_dir, plane_dir, frames, load_outcomes(frames.tsv))
                included.append({"session": session, "run": run_dir.name, **result})
                continue
            except (
                RunMismatchError,
                OutcomeParseError,
                FileNotFoundError,
                BackgroundUnavailableError,
                *also_skip,
            ) as error:
                reason = str(error)
        skipped.append((session, run_dir.name, reason))
    return included, skipped


def _collect_runs(
    found: list,
    variant: str,
    window_s: tuple[float, float],
    phases: tuple[str, ...],
    load_outcomes: Callable,
    measures: tuple[str, ...],
    load_views: Callable[[Path], tuple[dict, dict]],
) -> tuple[list[dict], list[tuple[str, str, str]]]:
    """Align the views of every found run to its outcomes; list the runs left out and why.

    *load_views* maps a plane folder to the views to align (``{(channel, measure): (n_rows,
    n_frames)}``) and extra entries to keep in the run's result.
    """

    def align(session, run_dir, plane_dir, frames, outcomes):
        views, extra = load_views(plane_dir)
        result = pulse_responses(
            views, frames.time_s, outcomes, window_s, phases, measures=measures
        )
        return {**result, **extra}

    included, skipped = collect_each_run(found, variant, load_outcomes, align)
    check_same_window(included)
    return included, skipped


def pool_runs(runs: list[dict], measures: tuple[str, ...]) -> dict:
    """Pool *runs* for *measures*: ``{view: {pulses: {"mean", "sem", "n_cells", "n_runs"}}}``.

    Raises:
        ValueError: If *measures* is empty or has an unknown name.
    """
    check_measures(measures)
    return {
        view: pooled_mean_sem([run["means"][view] for run in runs], PULSE_CATEGORIES)
        for view in VIEWS
        if view[1] in measures
    }


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
    measures: tuple[str, ...] = MEASURES,
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
        measures (tuple[str, ...]): Measures to compute and pool (see ``pulse_responses``).
            Keyword-only.

    Returns:
        tuple: ``{view: {pulses: {"mean", "sem", "n_cells", "n_runs"}}}`` pooled over cells;
        one result per included run (see ``pulse_responses``, plus ``session`` and ``run``);
        and ``(session, run, reason)`` for each run left out.

    Raises:
        ValueError: If *measures* is empty or has an unknown name (the valid ones are listed).
    """
    check_measures(measures)
    found = variant_runs(subject_dir, first_session, last_session, variant, runs)
    included, skipped = _collect_runs(
        found,
        variant,
        window_s,
        phases,
        load_outcomes,
        measures,
        lambda plane_dir: (plane_views(plane_dir)[0], {}),
    )
    return pool_runs(included, measures), included, skipped


def _with_trial_sem(pooled: dict, runs: list[dict]) -> dict:
    """Mark each pooled SEM as across runs, or across trials where one run alone contributes.

    With a single run the SEM across runs is undefined, so the SEM across that run's trials
    of the pulse count is shown instead, and ``sem_kind`` says which it is.
    """
    for view, per_pulse in pooled.items():
        for pulses, stats in per_pulse.items():
            stats["sem_kind"] = "runs"
            if stats["n_runs"] == 1:
                run = next(
                    r for r in runs if np.isfinite(r["means"][view].get(pulses, np.nan)).any()
                )
                stats["sem"] = run["sems"][view][pulses][0]
                stats["sem_kind"] = "trials"
    return pooled


def collect_background_responses(
    subject_dir,
    first_session: int,
    last_session: int,
    variant: str,
    window_s: tuple[float, float],
    phases: tuple[str, ...],
    load_outcomes: Callable = load_outcome_events,
    *,
    runs: Collection[str] | None = None,
    measures: tuple[str, ...] = BACKGROUND_MEASURES,
    roi_margin_px: int = 0,
    cache_dir=None,
) -> tuple[dict, list[dict], list[tuple[str, str, str]]]:
    """Pool the reward response of each run's background (all non-ROI pixels as one ROI).

    The background F goes through ``pulse_responses`` exactly like a cell's F (same frame times,
    window, phases, pulse categories and robust z), so the two are directly comparable. Runs
    are pooled one trace each.

    Args:
        subject_dir (str | Path): ``derivatives/<cohort>/<subject>``.
        first_session (int): First session number to include.
        last_session (int): Last session number to include.
        variant (str): ``"without_destriping"`` or ``"with_destriping"``.
        window_s (tuple[float, float]): Window around the outcome (s).
        phases (tuple[str, ...]): Trial phases to include.
        load_outcomes (Callable): Reads a behaviour file's outcome table (for tests).
        runs (Collection[str] | None): Run folder names to include. Default None, every run.
        measures (tuple[str, ...]): ``"F"`` and/or ``"F z"``.
        roi_margin_px (int): Pixels excluded around every ROI. Default 0.
        cache_dir (str | Path | None): Trace cache (see ``load_background``).

    Returns:
        tuple: ``{view: {pulses: {"mean", "sem", "n_cells", "n_runs", "sem_kind"}}}`` with
        ``n_cells`` the number of runs and ``sem_kind`` ``"runs"``, or ``"trials"`` where a
        single run contributes; one result per run (``pulse_responses`` output plus ``session``,
        ``run``, ``n_pixels``, ``fraction_of_fov``, ``mask``, ``labels``, ``mean_images``); and
        ``(session, run, reason)`` for each run left out.

    Raises:
        ValueError: If *measures* is empty or has a name other than ``F`` and ``F z``.
    """
    check_measures(measures, BACKGROUND_MEASURES)

    def load_views(plane_dir):
        background = load_background(plane_dir, roi_margin_px, cache_dir)
        views = {view: trace[None, :] for view, trace in background["F"].items()}
        extra = {
            "n_pixels": background["n_pixels"],
            "fraction_of_fov": float(background["mask"].mean()),
            "mask": background["mask"],
            "labels": background["labels"],
            "mean_images": background["mean_images"],
        }
        return views, extra

    found = variant_runs(subject_dir, first_session, last_session, variant, runs)
    included, skipped = _collect_runs(
        found, variant, window_s, phases, load_outcomes, measures, load_views
    )
    return _with_trial_sem(pool_runs(included, measures), included), included, skipped


PERIOD_TOLERANCE = 0.01  # relative; a run's median frame interval varies by far less


def check_same_window(runs: list[dict]) -> None:
    """Raise unless the runs share their window offsets and (nearly) their frame period.

    The pooled figures label the x axis with the first run's period, so runs with different
    periods would be drawn on the wrong time axis.
    """
    shapes = {tuple(run["offsets"]) for run in runs}
    periods = np.array([run["period_s"] for run in runs], dtype=float)
    if len(shapes) > 1 or (len(periods) and np.ptp(periods) > PERIOD_TOLERANCE * periods.mean()):
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
