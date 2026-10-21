"""Shared-signal analysis across the runs of one destriping variant (I/O around the pure analysis).

Each run's traces are read (cells, neuropil, and the cached background with its two halves), the
pure analysis of :mod:`src.funcimg_analysis.shared_signal_run` is run on them, and the result is
cached on disk keyed on the content of everything it was computed from.
"""

import zlib
from collections.abc import Callable, Collection

import numpy as np

from src.behavior_analysis.outcome_events import load_outcome_events
from src.funcimg_analysis.background_io import load_background
from src.funcimg_analysis.collect import (
    check_same_window,
    collect_each_run,
    pool_runs,
    variant_runs,
)
from src.funcimg_analysis.run_responses import EXTRACTION_RECORD, check_measures, plane_views
from src.funcimg_analysis.run_statistics import DEFAULT_NULL_KINDS, check_null_kinds
from src.funcimg_analysis.shared_cache import CACHE_SUBDIR, cached_result, content_key
from src.funcimg_analysis.shared_signal_run import (
    Z_MEASURES,
    SharedSignalUnavailableError,
    analyse_shared_signal,
)


def _cell_rows(traces, cells_only: bool) -> np.ndarray:
    """ROI indices to analyse: all, or those Suite2p's classifier calls cells (``iscell[:, 0]``)."""
    iscell = np.asarray(traces.iscell)[:, 0]
    return np.flatnonzero(iscell > 0) if cells_only else np.arange(len(iscell))


def run_seed(seed: int, session: str, run: str) -> list[int]:
    """Entropy fixed by the seed and the run's name, so a run's chance levels do not depend on
    which other runs are collected."""
    return [seed, zlib.crc32(f"{session}/{run}".encode())]


def _neuropil(traces, rois: np.ndarray) -> dict:
    """Each cell's own neuropil trace per channel, ``{(channel, "F"): (n_cells, n_frames)}``."""
    out = {("green", "F"): traces.fneu[rois]}
    if traces.fneu_chan2 is not None:
        out[("red", "F")] = traces.fneu_chan2[rois]
    return out


def collect_shared_signal(
    subject_dir,
    first_session: int,
    last_session: int,
    variant: str,
    window_s: tuple[float, float],
    phases: tuple[str, ...],
    load_outcomes: Callable = load_outcome_events,
    *,
    runs: Collection[str] | None = None,
    measures: tuple[str, ...] = Z_MEASURES,
    roi_margin_px: int = 0,
    cache_dir=None,
    progress: Callable[[str], None] | None = None,
    results_cache_dir=None,
    cells_only: bool = False,
    detrend_window_s: float = 60.0,
    n_shifts: int = 100,
    min_shift_s: float = 30.0,
    null_kinds: tuple[str, ...] = DEFAULT_NULL_KINDS,
    fit_exclude_outcomes: bool = True,
    fit_exclude_window_s: tuple[float, float] | None = None,
    seed: int = 0,
) -> tuple[list[dict], list[tuple[str, str, str]]]:
    """Do the cells of each run share signal beyond the background, and does the response survive?

    Per run, channel and measure (see ``analyse_shared_signal``): correlations of the cells with
    the background and with each other, partial correlations given the background and given
    the background plus neuropil, the background's reliability, chance levels from circular
    shifts, R^2 on the background, and the reward response of the detrended cells next to that
    of the cells with the background's fit removed.

    Args:
        subject_dir (str | Path): ``derivatives/<cohort>/<subject>``.
        first_session (int): First session number to include.
        last_session (int): Last session number to include.
        variant (str): ``"without_destriping"`` or ``"with_destriping"``.
        window_s (tuple[float, float]): Window around the outcome (s).
        phases (tuple[str, ...]): Trial phases to include.
        load_outcomes (Callable): Reads a behaviour file's outcome table (for tests).
        runs (Collection[str] | None): Run folder names to include. Default None, every run.
        measures (tuple[str, ...]): ``"F z"`` and/or ``"dF/F0 z"``.
        roi_margin_px (int): Pixels excluded around every ROI from the background.
        cache_dir (str | Path | None): Background trace cache (see ``load_background``).
        progress (Callable[[str], None] | None): Receives progress lines while the background
            binaries are read (see ``load_background``); e.g. ``print``. Default None.
        results_cache_dir (str | Path | None): Where each run's result is cached. Default None,
            meaning ``<settings cache_dir>/funcimg_shared`` (git-ignored). Never beside the data.
        cells_only (bool): Use only ROIs with ``iscell[:, 0] > 0``. Default False, every ROI.
        detrend_window_s (float): Running-percentile baseline length (s).
        n_shifts (int): Circular shifts per chance level.
        min_shift_s (float): Smallest circular shift (s), in both directions.
        null_kinds (tuple[str, ...]): Which chance levels to compute (see ``run_statistics``).
        fit_exclude_outcomes (bool): Fit the background's slope outside the outcome windows.
        fit_exclude_window_s (tuple[float, float] | None): The window left out of that fit around
            each outcome. Default None, meaning *window_s*.
        seed (int): Seeds the shifts and splits; each run draws its own from the seed and its name.

    Returns:
        tuple: One result per run (``analyse_shared_signal`` output plus ``session``, ``run``
        and ``rois``, the ROI indices used), and ``(session, run, reason)`` for each run left
        out (including runs with fewer than two ROIs).

    Raises:
        ValueError: If *measures* is empty or has a name other than ``F z`` and ``dF/F0 z``, or
            *null_kinds* has an unknown name.
    """
    from mini2p_preprocessing.plane import load_plane_traces

    check_measures(measures, Z_MEASURES)
    check_null_kinds(null_kinds)
    if results_cache_dir is None:
        from src.settings import get_settings

        results_cache_dir = get_settings().cache_dir / CACHE_SUBDIR
    settings = {
        "window_s": tuple(window_s),
        "phases": tuple(phases),
        "measures": tuple(measures),
        "detrend_window_s": detrend_window_s,
        "min_shift_s": min_shift_s,
        "n_shifts": n_shifts,
        "null_kinds": tuple(null_kinds),
        "fit_exclude_outcomes": fit_exclude_outcomes,
        "fit_exclude_window_s": fit_exclude_window_s,
        "cells_only": cells_only,
        "roi_margin_px": roi_margin_px,
    }

    def analyse(session, run_dir, plane_dir, frames, outcomes):
        traces = load_plane_traces(plane_dir)
        rois = _cell_rows(traces, cells_only)
        background = load_background(plane_dir, roi_margin_px, cache_dir, progress=progress)
        entropy = run_seed(seed, session, run_dir.name)
        key = content_key(
            {
                # the raw traces and the settings that make ΔF/F0 stand for the views
                "f": traces.f[rois],
                "fneu": traces.fneu[rois],
                "f_chan2": None if traces.f_chan2 is None else traces.f_chan2[rois],
                "fneu_chan2": None if traces.fneu_chan2 is None else traces.fneu_chan2[rois],
                "extraction": (plane_dir / EXTRACTION_RECORD).read_text(),
                "rois": rois,
                "background": background["F"],
                "halves": background["F_halves"],
                "frame_times": frames.time_s,
                "outcomes": outcomes[["outcome_time", "num_rew_pulses", "phase"]],
                "settings": settings,
                "entropy": entropy,
            }
        )

        def compute():
            views, _ = plane_views(plane_dir)
            result = analyse_shared_signal(
                {view: trace[rois] for view, trace in views.items()},
                background["F"],
                frames.time_s,
                outcomes,
                neuropil=_neuropil(traces, rois),
                background_halves=background["F_halves"],
                rng=np.random.default_rng(entropy),
                window_s=window_s,
                phases=phases,
                measures=measures,
                detrend_window_s=detrend_window_s,
                min_shift_s=min_shift_s,
                n_shifts=n_shifts,
                null_kinds=null_kinds,
                fit_exclude_outcomes=fit_exclude_outcomes,
                fit_exclude_window_s=fit_exclude_window_s,
            )
            return {**result, "rois": rois}

        return cached_result(results_cache_dir, key, compute)

    found = variant_runs(subject_dir, first_session, last_session, variant, runs)
    included, skipped = collect_each_run(
        found, variant, load_outcomes, analyse, also_skip=(SharedSignalUnavailableError,)
    )
    check_same_window([r["original"] for r in included])
    return included, skipped


def pool_shared_signal(runs: list[dict], measures: tuple[str, ...]) -> tuple[dict, dict]:
    """Pool the original and the background-removed responses across runs.

    Returns:
        tuple[dict, dict]: ``pool_runs`` output for the detrended cells and for the residuals.

    Raises:
        ValueError: If the runs differ in window or frame period (their time axes would differ).
    """
    check_same_window([r["original"] for r in runs])
    return (
        pool_runs([r["original"] for r in runs], measures),
        pool_runs([r["residual"] for r in runs], measures),
    )
