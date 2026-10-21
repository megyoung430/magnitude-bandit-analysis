"""Every shared-signal statistic of one run, channel and measure, and the table row (pure)."""

import numpy as np

from src.funcimg_analysis.event_frames import quiet_empty_means
from src.funcimg_analysis.shared_signal import (
    background_reliability,
    fit_background,
    leave_one_out_mean,
    leave_one_out_reliability,
    partial_row_correlation,
    row_correlation,
)
from src.funcimg_analysis.shared_signal_null import (
    NULL_PERCENTILE,
    null_partial_r,
    null_r,
    run_null_percentile,
)

# The nulls that are drawn or tabulated; the rest are opt-in because each costs a full pass.
DEFAULT_NULL_KINDS = ("r_bg", "r_loo", "partial_average", "partial_neuropil_average")
OPTIONAL_NULL_KINDS = ("partial_background", "r_bg_avg")
NULL_KINDS = DEFAULT_NULL_KINDS + OPTIONAL_NULL_KINDS


def check_null_kinds(null_kinds: tuple[str, ...]) -> None:
    """Raise unless every name in *null_kinds* is a known chance level.

    Raises:
        TypeError: If *null_kinds* is a single string.
        ValueError: If a name is unknown (the valid ones are listed).
    """
    if isinstance(null_kinds, str):
        raise TypeError(
            f"null_kinds must be a tuple of names, e.g. ({null_kinds!r},), not a string."
        )
    unknown = [kind for kind in null_kinds if kind not in NULL_KINDS]
    if unknown:
        raise ValueError(f"unknown null_kinds {unknown}; the valid ones are {list(NULL_KINDS)}.")


def _compute_nulls(
    kinds: tuple[str, ...],
    z: np.ndarray,
    average: np.ndarray,
    background: np.ndarray,
    neuropil_controls: list[np.ndarray] | None,
    offsets: np.ndarray,
) -> dict[str, np.ndarray]:
    """The requested chance levels, keyed ``null_<kind>``."""
    if "partial_neuropil_average" in kinds and neuropil_controls is None:
        raise ValueError("the null 'partial_neuropil_average' needs neuropil_z.")
    computers = {
        "r_bg": lambda: null_r(z, background, offsets),
        "r_loo": lambda: null_r(z, average, offsets),
        "partial_average": lambda: null_partial_r(z, average, background, offsets, "average"),
        "partial_neuropil_average": lambda: null_partial_r(
            z, average, neuropil_controls, offsets, "average"
        ),
        "partial_background": lambda: null_partial_r(z, average, background, offsets, "background"),
        "r_bg_avg": lambda: null_r(np.nanmean(z, axis=0)[None, :], background, offsets)[:, 0],
    }
    return {f"null_{kind}": computers[kind]() for kind in kinds}


def shared_signal_stats(
    z: np.ndarray,
    background_z: np.ndarray,
    offsets: np.ndarray,
    *,
    rng: np.random.Generator,
    neuropil_z: np.ndarray | None = None,
    background_halves_z: np.ndarray | None = None,
    null_kinds: tuple[str, ...] = DEFAULT_NULL_KINDS,
    n_splits: int = 10,
) -> dict:
    """Every statistic of one run, channel and measure.

    Args:
        z (np.ndarray): ``(n_cells, n_frames)`` detrended robust-z traces, at least two cells.
        background_z (np.ndarray): ``(n_frames,)`` the background, detrended and z-scored.
        offsets (np.ndarray): Circular shifts (frames) for the chance levels.
        rng (np.random.Generator): Draws the splits of the reliability estimates.
        neuropil_z (np.ndarray | None): ``(n_cells, n_frames)`` each cell's own neuropil, detrended
            and z-scored; enables the neuropil-controlled partial r. Default None (NaN).
        background_halves_z (np.ndarray | None): ``(2, n_frames)`` two disjoint sets of background
            pixels, detrended and z-scored; enables the background's split-half reliability.
            Default None (NaN).
        null_kinds (tuple[str, ...]): The chance levels to compute (``DEFAULT_NULL_KINDS``, plus
            the opt-in ``OPTIONAL_NULL_KINDS``); each is ``(n_shifts, n_cells)`` under
            ``null_<kind>`` (``null_r_bg_avg`` is ``(n_shifts,)``).
        n_splits (int): Random splits for the reliability of the leave-one-out averages.

    Returns:
        dict: ``n_cells``; ``n_undefined`` (cells whose r with the background or the average is
        undefined); ``r_bg_avg``; per cell ``r_loo``, ``r_bg``, ``loo_reliability``,
        ``r_loo_corrected`` (r_loo divided by the square root of the average's reliability, at
        most 1), ``partial_r`` (given the background), ``partial_r_neuropil`` (given the
        background and the cell's and the other cells' neuropil), ``r2`` and ``slope``;
        ``bg_half_r`` and ``bg_reliability``; and the requested ``null_*``.

    Raises:
        ValueError: If the background (or the neuropil) and the cells differ in their number of
            frames or the neuropil in its number of cells, or *null_kinds* has an unknown name.
    """
    check_null_kinds(null_kinds)
    z, background_z = np.asarray(z, dtype=float), np.asarray(background_z, dtype=float)
    if z.shape[1] != background_z.shape[0]:
        raise ValueError(
            f"cells have {z.shape[1]} frames but the background has {background_z.shape[0]}."
        )
    if neuropil_z is not None and np.shape(neuropil_z) != z.shape:
        raise ValueError(f"neuropil has shape {np.shape(neuropil_z)} but the cells {z.shape}.")
    with quiet_empty_means():
        average = np.nanmean(z, axis=0)
    loo = leave_one_out_mean(z)
    r_bg, r_loo = row_correlation(z, background_z), row_correlation(z, loo)
    # with two cells the average is one cell and cannot be split: no reliability, no correction
    reliability = (
        leave_one_out_reliability(z, rng, n_splits) if len(z) >= 3 else np.full(len(z), np.nan)
    )
    fit = fit_background(z, background_z)
    controls = None
    partial_neuropil = np.full(len(z), np.nan)
    if neuropil_z is not None:
        controls = [background_z, neuropil_z, leave_one_out_mean(neuropil_z)]
        partial_neuropil = partial_row_correlation(z, loo, controls)
    half_r, bg_reliability = (
        (np.nan, np.nan)
        if background_halves_z is None
        else background_reliability(*background_halves_z)
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        corrected = np.clip(r_loo / np.sqrt(reliability), -1, 1)
    return {
        "n_cells": z.shape[0],
        "n_undefined": int((~(np.isfinite(r_bg) & np.isfinite(r_loo))).sum()),
        "r_bg_avg": float(row_correlation(average, background_z)[0]),
        "r_loo": r_loo,
        "r_bg": r_bg,
        "loo_reliability": reliability,
        "r_loo_corrected": corrected,
        "partial_r": partial_row_correlation(z, loo, background_z),
        "partial_r_neuropil": partial_neuropil,
        "r2": fit["r2"],
        "slope": fit["slope"],
        "bg_half_r": half_r,
        "bg_reliability": bg_reliability,
        **_compute_nulls(null_kinds, z, loo, background_z, controls, offsets),
    }


def _null_level(stats: dict, key: str) -> float:
    """The null's 95th percentile of the run's median, or NaN if that null was not computed."""
    if key not in stats:
        return float("nan")
    return run_null_percentile(stats[key], NULL_PERCENTILE, np.nanmedian)


def summary_row(stats: dict) -> dict:
    """The numbers of one run, channel and measure for the printed table.

    The null columns are NaN when that chance level was not computed (``null_kinds``).
    """
    with quiet_empty_means():
        return {
            "cells": stats["n_cells"],
            "undefined": stats["n_undefined"],
            "r(bg, avg)": stats["r_bg_avg"],
            "median r(cell, LOO)": float(np.nanmedian(stats["r_loo"])),
            "median r(cell, LOO), corrected": float(np.nanmedian(stats["r_loo_corrected"])),
            "median r(cell, bg)": float(np.nanmedian(stats["r_bg"])),
            "median partial r": float(np.nanmedian(stats["partial_r"])),
            "null 95th (median partial r)": _null_level(stats, "null_partial_average"),
            "median partial r, neuropil-controlled": float(
                np.nanmedian(stats["partial_r_neuropil"])
            ),
            "null 95th (neuropil-controlled)": _null_level(stats, "null_partial_neuropil_average"),
            "bg split-half r": stats["bg_half_r"],
            "bg reliability": stats["bg_reliability"],
            "median R2": float(np.nanmedian(stats["r2"])),
        }
