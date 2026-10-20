"""Reward responses across every run of one destriping variant, and each run's top cells.

Runs are found under ``derivatives/<subject>/ses-*/funcimg/run-*/<variant>/suite2p/plane0``. A run
contributes only with an extracted plane and a ``frame_times.npz``; any other run is listed with
the reason it was left out, never dropped silently.
"""

import re
from collections.abc import Callable
from pathlib import Path

import numpy as np

from src.behavior_analysis.outcome_events import load_outcome_events
from src.funcimg_analysis.event_frames import pooled_mean_sem, top_cells_by_max
from src.funcimg_analysis.frame_times import FRAME_TIMES_FILE, load_frame_times
from src.funcimg_analysis.run_responses import (
    EXTRACTION_RECORD,
    PULSE_CATEGORIES,
    VIEWS,
    plane_views,
    pulse_responses,
)

SESSION = re.compile(r"^ses-(\d+)_")


def variant_runs(subject_dir, first_session: int, last_session: int, variant: str):
    """``(session, run dir, plane dir)`` of each run folder of *variant* in range, in order."""
    found = []
    for session_dir in sorted(Path(subject_dir).glob("ses-*")):
        match = SESSION.match(session_dir.name)
        if not match or not first_session <= int(match.group(1)) <= last_session:
            continue
        for run_dir in sorted((session_dir / "funcimg").glob("run-*")):
            plane_dir = run_dir / variant / "suite2p" / "plane0"
            if (plane_dir / "ops.npy").is_file():
                found.append((session_dir.name, run_dir, plane_dir))
    return sorted(found, key=lambda r: (int(SESSION.match(r[0]).group(1)), r[1].name))


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

    Returns:
        tuple: ``{view: {pulses: {"mean", "sem", "n_cells", "n_runs"}}}`` pooled over cells;
        one result per included run (see ``pulse_responses``, plus ``session`` and ``run``);
        and ``(session, run, reason)`` for each run left out.
    """
    runs, skipped = [], []
    for session, run_dir, plane_dir in variant_runs(
        subject_dir, first_session, last_session, variant
    ):
        reason = _missing(run_dir, plane_dir)
        if reason is None:
            try:
                frames = load_frame_times(run_dir)
                views, _ = plane_views(plane_dir)
                result = pulse_responses(
                    views, frames.time_s, load_outcomes(frames.tsv), window_s, phases
                )
                runs.append({"session": session, "run": run_dir.name, **result})
                continue
            except (ValueError, FileNotFoundError) as error:
                reason = str(error)
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


def top_cells(plane_dir, run_dir, n: int, load_outcomes: Callable = load_outcome_events) -> dict:
    """The *n* cells of a run with the largest ΔF/F0 (green), with what a figure needs.

    Returns:
        dict: ``cells`` (ROI indices, largest first), ``peaks`` (their max ΔF/F0), ``dff``
        (``(n, n_frames)`` green ΔF/F0), ``time_s`` and ``period_s`` of the frames, ``labels``
        and ``mean_image`` of the plane, and ``outcomes`` (outcome time and pulses per task
        trial) to mark on the traces.
    """
    frames = load_frame_times(run_dir)
    views, traces = plane_views(plane_dir)
    dff = views[("green", "dF/F0")]
    if dff.shape[1] != len(frames.time_s):
        raise ValueError(
            f"{plane_dir} has {dff.shape[1]} frames but there are {len(frames.time_s)} frame times."
        )
    cells = top_cells_by_max(dff, n)
    outcomes = load_outcomes(frames.tsv)
    return {
        "cells": cells,
        "peaks": np.nanmax(dff[cells], axis=1),
        "dff": dff[cells],
        "time_s": frames.time_s,
        "period_s": frames.period_s,
        "labels": traces.labels,
        "mean_image": np.asarray(traces.ops["meanImg"], dtype=float),
        "outcomes": outcomes[outcomes["phase"] == "task"][["outcome_time", "num_rew_pulses"]],
    }
