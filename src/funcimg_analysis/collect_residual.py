"""Background-corrected ΔF/F0 across the runs of one destriping variant (I/O around the pure code).

Each run's raw F, background and Suite2p ΔF/F0 are read, the correction of
:mod:`src.funcimg_analysis.residual_dff` is applied per channel, and both ΔF/F0 measures of the
cells that survive the F0 check are aligned to the outcomes exactly like section 2's responses.
A run's result is cached on disk keyed on the content of everything it is computed from.
"""

from collections.abc import Callable, Collection

import numpy as np

from src.behavior_analysis.outcome_events import load_outcome_events
from src.funcimg_analysis.background_io import load_background
from src.funcimg_analysis.collect import check_same_window, collect_each_run, variant_runs
from src.funcimg_analysis.event_frames import pooled_mean_sem, quiet_empty_means
from src.funcimg_analysis.frame_times import RunMismatchError
from src.funcimg_analysis.residual_dff import MIN_F0_FRACTION, residual_dff_channel
from src.funcimg_analysis.run_responses import (
    EXTRACTION_RECORD,
    PULSE_CATEGORIES,
    load_fluorescence_settings,
    plane_views,
    pulse_responses,
)
from src.funcimg_analysis.shared_cache import CACHE_SUBDIR, cached_result, content_key
from src.funcimg_analysis.shared_signal_run import SharedSignalUnavailableError, fit_frames

CHANNELS = ("green", "red")
MEASURE = "dF/F0 z"  # both ΔF/F0 measures are compared as robust z, baseline-subtracted
N_EXAMPLE_CELLS = 4


class NoUsableCellsError(Exception):
    """Every ROI of both channels is excluded by the F0 check, so nothing is left to compare."""


def example_cells(slope: np.ndarray, kept: np.ndarray, n: int = N_EXAMPLE_CELLS) -> list[int]:
    """Kept cells spread evenly over the ranking of the slopes, from the largest to the smallest.

    The extremes show where the correction matters most and the middle where it matters little,
    so a sanity check is not drawn only from the cells that change. No kept cell gives ``[]``.
    """
    candidates = np.flatnonzero(kept)
    if not len(candidates):
        return []
    order = candidates[np.argsort(-slope[candidates], kind="stable")]
    positions = np.unique(np.round(np.linspace(0, len(order) - 1, min(n, len(order)))).astype(int))
    return [int(order[i]) for i in positions]


def _channel_results(
    plane_dir, traces, background, fit_mask, min_f0_fraction, detrend_window_s
) -> dict[str, dict]:
    """The correction of both channels; each says whether it has a usable cell.

    Raises:
        NoUsableCellsError: If neither channel has a usable cell.
    """
    settings = load_fluorescence_settings(plane_dir)
    raw = {"green": traces.f, "red": traces.f_chan2}
    channels = {}
    for channel in CHANNELS:
        result = residual_dff_channel(
            raw[channel],
            background["F"][(channel, "F")],
            fit_mask,
            traces.fs,
            settings,
            min_f0_fraction,
            detrend_window_s,
        )
        channels[channel] = {**result, "usable": bool(result["kept"].any())}
    if not any(c["usable"] for c in channels.values()):
        raise NoUsableCellsError(
            "no usable cell in either channel: "
            + "; ".join(
                f"{channel} ROI {roi}: {reason}"
                for channel, c in channels.items()
                for roi, reason in c["excluded"].items()
            )
        )
    return channels


def _aligned(views, channels, frames, outcomes, window_s, phases) -> dict:
    """Both ΔF/F0 measures of the kept cells of each usable channel, aligned to the outcomes."""
    usable = [c for c in CHANNELS if channels[c]["usable"]]
    kept = {channel: channels[channel]["kept"] for channel in usable}
    suite2p = {(c, "dF/F0"): views[(c, "dF/F0")][kept[c]] for c in usable}
    corrected = {(c, "dF/F0"): channels[c]["dff"][kept[c]] for c in usable}
    align = dict(window_s=window_s, phases=phases, measures=(MEASURE,))
    return {
        "suite2p": pulse_responses(suite2p, frames.time_s, outcomes, **align),
        "corrected": pulse_responses(corrected, frames.time_s, outcomes, **align),
    }


def _examples(views, channels) -> dict:
    out = {}
    for channel in CHANNELS:
        cells = example_cells(channels[channel]["slope"], channels[channel]["kept"])
        out[channel] = {
            "cells": cells,
            "suite2p": views[(channel, "dF/F0")][cells],
            "corrected": channels[channel]["dff"][cells],
        }
    return out


def pool_residual(runs: list[dict], which: str) -> dict:
    """Pool ``"suite2p"`` or ``"corrected"`` responses of the runs, each channel over its runs.

    A channel that no run has a usable cell for is absent from the result.

    Returns:
        dict: ``{(channel, "dF/F0 z"): {pulses: {"mean", "sem", "n_cells", "n_runs"}}}``.
    """
    pooled = {}
    for channel in CHANNELS:
        view = (channel, MEASURE)
        having = [run[which]["means"][view] for run in runs if view in run[which]["means"]]
        if having:
            pooled[view] = pooled_mean_sem(having, PULSE_CATEGORIES)
    return pooled


def collect_residual_dff(
    subject_dir,
    first_session: int,
    last_session: int,
    variant: str,
    window_s: tuple[float, float],
    phases: tuple[str, ...],
    load_outcomes: Callable = load_outcome_events,
    *,
    runs: Collection[str] | None = None,
    roi_margin_px: int = 0,
    cache_dir=None,
    progress: Callable[[str], None] | None = None,
    results_cache_dir=None,
    fit_exclude_outcomes: bool = True,
    fit_exclude_window_s: tuple[float, float] | None = None,
    min_f0_fraction: float = MIN_F0_FRACTION,
    detrend_window_s: float = 60.0,
) -> tuple[list[dict], list[tuple[str, str, str]]]:
    """Background-corrected ΔF/F0 of every run, next to Suite2p's, aligned to the outcomes.

    Args:
        subject_dir (str | Path): ``derivatives/<cohort>/<subject>``.
        first_session (int): First session number to include.
        last_session (int): Last session number to include.
        variant (str): ``"without_destriping"`` or ``"with_destriping"``.
        window_s (tuple[float, float]): Window around the outcome (s).
        phases (tuple[str, ...]): Trial phases to include.
        load_outcomes (Callable): Reads a behaviour file's outcome table (for tests).
        runs (Collection[str] | None): Run folder names to include. Default None, every run.
        roi_margin_px (int): Pixels excluded around every ROI from the background.
        cache_dir (str | Path | None): Background trace cache (see ``load_background``).
        progress (Callable[[str], None] | None): Receives progress lines while the background
            binaries are read; e.g. ``print``.
        results_cache_dir (str | Path | None): Where each run's result is cached. Default None,
            meaning ``<settings cache_dir>/funcimg_shared`` (git-ignored). Never beside the data.
        fit_exclude_outcomes (bool): Fit each cell's slope outside the outcome windows only.
        fit_exclude_window_s (tuple[float, float] | None): The window left out around each
            outcome. Default None, meaning *window_s*.
        min_f0_fraction (float): A cell whose F0 is not positive or is below this fraction of its
            median F anywhere is excluded.
        detrend_window_s (float): Running-percentile baseline (s) removed from F and the
            background before each cell's slope is fitted (the shared-signal detrend).

    Returns:
        tuple: One result per run and ``(session, run, reason)`` for each run left out. A result
        has ``channels`` (``{channel: slope, r_after, kept, excluded}`` per ROI), ``suite2p`` and
        ``corrected`` (``pulse_responses`` of the kept cells' ΔF/F0 z), ``examples`` (ΔF/F0 of a
        few cells under both corrections), ``time_s``, ``rois``, ``fit_fraction`` and
        ``min_f0_fraction``.
    """
    from mini2p_preprocessing.plane import load_plane_traces

    if results_cache_dir is None:
        from src.settings import get_settings

        results_cache_dir = get_settings().cache_dir / CACHE_SUBDIR
    settings = {
        "window_s": tuple(window_s),
        "phases": tuple(phases),
        "fit_exclude_outcomes": fit_exclude_outcomes,
        "fit_exclude_window_s": fit_exclude_window_s,
        "min_f0_fraction": min_f0_fraction,
        "detrend_window_s": detrend_window_s,
        "roi_margin_px": roi_margin_px,
        "kind": "residual_dff",
    }

    def analyse(session, run_dir, plane_dir, frames, outcomes):
        traces = load_plane_traces(plane_dir)
        if traces.f_chan2 is None:
            raise FileNotFoundError(f"{plane_dir} has no red-channel traces (F_chan2.npy).")
        background = load_background(plane_dir, roi_margin_px, cache_dir, progress=progress)
        key = content_key(
            {
                "f": traces.f,
                "fneu": traces.fneu,
                "f_chan2": traces.f_chan2,
                "fneu_chan2": traces.fneu_chan2,
                "extraction": (plane_dir / EXTRACTION_RECORD).read_text(),
                "background": background["F"],
                "frame_times": frames.time_s,
                "outcomes": outcomes[["outcome_time", "num_rew_pulses", "phase"]],
                "settings": settings,
            }
        )

        def compute():
            views, _ = plane_views(plane_dir)
            n_frames = len(frames.time_s)
            if traces.f.shape[1] != n_frames:
                raise RunMismatchError(
                    f"{plane_dir} has {traces.f.shape[1]} frames but there are {n_frames} "
                    "frame times."
                )
            if fit_exclude_outcomes:
                fit_mask = fit_frames(
                    frames.time_s,
                    outcomes,
                    phases,
                    fit_exclude_window_s or window_s,
                    frames.period_s,
                )
            else:
                fit_mask = np.ones(n_frames, dtype=bool)
            channels = _channel_results(
                plane_dir, traces, background, fit_mask, min_f0_fraction, detrend_window_s
            )
            return {
                "channels": channels,
                **_aligned(views, channels, frames, outcomes, window_s, phases),
                "examples": _examples(views, channels),
                "time_s": frames.time_s,
                "rois": np.arange(len(traces.f)),
                "fit_fraction": float(fit_mask.mean()),
                "min_f0_fraction": min_f0_fraction,
            }

        return cached_result(results_cache_dir, key, compute)

    found = variant_runs(subject_dir, first_session, last_session, variant, runs)
    included, skipped = collect_each_run(
        found,
        variant,
        load_outcomes,
        analyse,
        also_skip=(SharedSignalUnavailableError, NoUsableCellsError),
    )
    check_same_window([r["suite2p"] for r in included])
    return included, skipped


def summary_table(runs: list[dict]) -> list[dict]:
    """One row per run and channel: cells, exclusions, median slope and residual correlation."""
    rows = []
    for run in runs:
        for channel in CHANNELS:
            result = run["channels"][channel]
            kept = result["kept"]
            with quiet_empty_means():
                rows.append(
                    {
                        "session": run["session"],
                        "run": run["run"],
                        "channel": channel,
                        "n cells": len(kept),
                        "n excluded": int((~kept).sum()),
                        "median b": float(np.nanmedian(result["slope"][kept])),
                        "median r(F_corr, bg)": float(np.nanmedian(result["r_after"][kept])),
                        "fitted on": f"{100 * run['fit_fraction']:.0f}% of frames",
                    }
                )
    return rows


def excluded_cells(runs: list[dict]) -> list[tuple[str, str, str, int, str]]:
    """``(session, run, channel, ROI, reason)`` of every excluded cell, for listing."""
    return [
        (run["session"], run["run"], channel, int(roi), reason)
        for run in runs
        for channel in CHANNELS
        for roi, reason in run["channels"][channel]["excluded"].items()
    ]
