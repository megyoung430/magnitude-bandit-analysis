"""Do cells share signal beyond the field-wide background? (pure)

Traces are detrended (a slow running-percentile baseline subtracted) and robust-z-scored first,
so that a slow drift or a different scale does not create correlations. Every correlation uses
the frames that are finite in all the traces involved; frames without a pyControl time are
therefore included, since the question is about the traces, not about the events.

The comparison that matters is a cell against the *leave-one-out* average of the other cells
versus the same cell against the background: both sides use one noisy cell and one cleaner
reference, whereas the whole-FOV average would be mostly background pixels.
"""

import numpy as np
from scipy.stats import wilcoxon

from src.funcimg_analysis.event_frames import quiet_empty_means, robust_z

DETREND_PERCENTILE = 10.0
MIN_WINDOW_FRAMES = 3
MIN_FRAMES_FOR_CORRELATION = 3
NULL_PERCENTILE = 95.0
SHIFTED_TRACES = ("background", "average")


def window_frames_from_s(window_s: float, period_s: float) -> int:
    """An odd number of frames (at least 3) spanning about *window_s* seconds."""
    return max(MIN_WINDOW_FRAMES, int(round(window_s / period_s)) | 1)


# --- detrending ---------------------------------------------------------------------------------


def running_percentile(
    traces: np.ndarray, window_frames: int, percentile: float, stride_frames: int | None = None
) -> np.ndarray:
    """A centred running percentile of each row, NaN-safe, evaluated on a coarse grid.

    The percentile is taken over ``2 * (window_frames // 2) + 1`` frames around each grid frame
    (clipped at the ends of the trace) and linearly interpolated between grid frames, which
    keeps a 60 s window over tens of thousands of frames cheap. A stretch of NaN longer than the
    window leaves NaN in the baseline there.

    Args:
        traces (np.ndarray): ``(n_rows, n_frames)``.
        window_frames (int): Window length; between 3 and ``n_frames``.
        percentile (float): 0-100.
        stride_frames (int | None): Spacing of the grid. Default ``window_frames // 10`` (at least
            1); 1 evaluates every frame.

    Returns:
        np.ndarray: ``(n_rows, n_frames)`` baseline.

    Raises:
        ValueError: If the window is shorter than 3 frames or longer than the trace.
    """
    traces = np.asarray(traces, dtype=float)
    n_frames = traces.shape[1]
    if not MIN_WINDOW_FRAMES <= window_frames <= n_frames:
        raise ValueError(
            f"detrend window of {window_frames} frames must be between {MIN_WINDOW_FRAMES} and "
            f"the {n_frames} frames of the trace."
        )
    stride = max(1, window_frames // 10) if stride_frames is None else stride_frames
    half = window_frames // 2
    grid = np.unique(np.append(np.arange(0, n_frames, stride), n_frames - 1))
    with quiet_empty_means():
        at_grid = np.stack(
            [
                np.nanpercentile(traces[:, max(0, c - half) : c + half + 1], percentile, axis=1)
                for c in grid
            ],
            axis=1,
        )
    return _interpolate_grid(at_grid, grid, n_frames)


def _interpolate_grid(at_grid: np.ndarray, grid: np.ndarray, n_frames: int) -> np.ndarray:
    """Linear interpolation of ``(n_rows, len(grid))`` to every frame; NaN stays local."""
    frames = np.arange(n_frames)
    upper = np.searchsorted(grid, frames)
    exact = grid[upper] == frames
    lower = np.where(exact, upper, upper - 1)
    spacing = np.where(exact, 1, grid[upper] - grid[lower])  # avoids 0 / 0 on grid frames
    weight = np.where(exact, 0.0, (frames - grid[lower]) / spacing)
    mixed = at_grid[:, lower] * (1 - weight) + at_grid[:, upper] * weight
    return np.where(weight == 0, at_grid[:, lower], mixed)  # 0 * NaN must not spread


def detrend(
    traces: np.ndarray, window_frames: int, percentile: float = DETREND_PERCENTILE
) -> np.ndarray:
    """Each row minus its running *percentile* baseline (see :func:`running_percentile`)."""
    traces = np.asarray(traces, dtype=float)
    return traces - running_percentile(traces, window_frames, percentile)


def detrended_z(
    traces: np.ndarray, window_frames: int, percentile: float = DETREND_PERCENTILE
) -> tuple[np.ndarray, np.ndarray]:
    """Detrend, then robust z-score each row (median / MAD), so scales do not matter.

    Returns:
        tuple[np.ndarray, np.ndarray]: The z traces and which rows have no defined scale (all NaN).
    """
    return robust_z(detrend(traces, window_frames, percentile))


# --- correlations -------------------------------------------------------------------------------


def _as_rows(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """*a* and *b* as float ``(n, n_frames)`` arrays; a single trace is broadcast over rows."""
    a, b = np.atleast_2d(np.asarray(a, dtype=float)), np.atleast_2d(np.asarray(b, dtype=float))
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"traces have {a.shape[1]} and {b.shape[1]} frames; they must match.")
    if a.shape[0] != b.shape[0] and 1 not in (a.shape[0], b.shape[0]):
        raise ValueError(f"cannot pair {a.shape[0]} traces with {b.shape[0]}.")
    n = max(a.shape[0], b.shape[0])
    return np.broadcast_to(a, (n, a.shape[1])), np.broadcast_to(b, (n, b.shape[1]))


def _moments(a: np.ndarray, b: np.ndarray) -> dict[str, np.ndarray]:
    """Per-row count, means, variances and covariance over frames finite in both rows."""
    a, b = _as_rows(a, b)
    ok = np.isfinite(a) & np.isfinite(b)
    n = ok.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_a = np.where(ok, a, 0).sum(axis=1) / n
        mean_b = np.where(ok, b, 0).sum(axis=1) / n
        da = np.where(ok, a - mean_a[:, None], 0)
        db = np.where(ok, b - mean_b[:, None], 0)
        return {
            "n": n,
            "mean_a": mean_a,
            "mean_b": mean_b,
            "var_a": (da**2).sum(axis=1) / n,
            "var_b": (db**2).sum(axis=1) / n,
            "cov": (da * db).sum(axis=1) / n,
        }


def _has_spread(mean: np.ndarray, variance: np.ndarray) -> np.ndarray:
    return np.sqrt(variance) > 1e-12 * np.abs(mean)


def row_correlation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson r per row over the frames finite in both; *b* may be a single trace.

    Args:
        a (np.ndarray): ``(n, n_frames)``.
        b (np.ndarray): ``(n, n_frames)`` or ``(n_frames,)``.

    Returns:
        np.ndarray: ``(n,)``; NaN where a row has fewer than 3 shared finite frames or no
        variance.

    Raises:
        ValueError: If the traces differ in their number of frames.
    """
    m = _moments(a, b)
    defined = (
        (m["n"] >= MIN_FRAMES_FOR_CORRELATION)
        & _has_spread(m["mean_a"], m["var_a"])
        & _has_spread(m["mean_b"], m["var_b"])
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        r = m["cov"] / np.sqrt(m["var_a"] * m["var_b"])
    return np.where(defined, np.clip(r, -1, 1), np.nan)


def partial_row_correlation(a: np.ndarray, b: np.ndarray, control: np.ndarray) -> np.ndarray:
    """Correlation of *a* and *b* per row after regressing *control* (with intercept) out of both.

    Equal to correlating the two residual traces. All three are restricted to the frames finite
    in all of them, so the control is removed from exactly the frames that are correlated.

    Args:
        a (np.ndarray): ``(n, n_frames)``.
        b (np.ndarray): ``(n, n_frames)`` or ``(n_frames,)``.
        control (np.ndarray): ``(n_frames,)`` or ``(n, n_frames)``, e.g. the background.

    Returns:
        np.ndarray: ``(n,)``; NaN where undefined (e.g. a row perfectly explained by the control).
    """
    a, b = _as_rows(a, b)
    c, _ = _as_rows(control, a)
    ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
    a, b, c = (np.where(ok, x, np.nan) for x in (a, b, c))
    r_ab, r_ac, r_bc = row_correlation(a, b), row_correlation(a, c), row_correlation(b, c)
    with np.errstate(invalid="ignore", divide="ignore"):
        partial = (r_ab - r_ac * r_bc) / np.sqrt((1 - r_ac**2) * (1 - r_bc**2))
    return np.clip(partial, -1, 1)


def leave_one_out_mean(z: np.ndarray) -> np.ndarray:
    """For each cell, the mean over the *other* cells at every frame (NaN ignored).

    Raises:
        ValueError: If there are fewer than two cells.
    """
    z = np.asarray(z, dtype=float)
    if z.shape[0] < 2:
        raise ValueError("A leave-one-out average needs at least two cells.")
    finite = np.isfinite(z)
    values = np.where(finite, z, 0.0)
    total, count = values.sum(axis=0), finite.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (total - values) / (count - finite)


# --- regression on the background ---------------------------------------------------------------


def fit_background(cells: np.ndarray, background: np.ndarray) -> dict[str, np.ndarray]:
    """OLS of each cell on the background, ``cell ~ intercept + slope * background``.

    Fitted on the frames finite in both. Returns ``slope``, ``intercept`` and ``r2`` per cell
    (NaN where the cell or the background has no variance there).
    """
    m = _moments(cells, background)
    defined = (
        (m["n"] >= MIN_FRAMES_FOR_CORRELATION)
        & _has_spread(m["mean_a"], m["var_a"])
        & _has_spread(m["mean_b"], m["var_b"])
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        slope = np.where(defined, m["cov"] / m["var_b"], np.nan)
        intercept = m["mean_a"] - slope * m["mean_b"]
    return {"slope": slope, "intercept": intercept, "r2": row_correlation(cells, background) ** 2}


def remove_background(cells: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Each cell minus ``slope * background``; the cell's own offset is kept.

    Subtracting only the slope term (not the intercept) leaves the cell's baseline where it was,
    so the usual robust-z and pre-event baseline subtraction apply to the result unchanged.
    """
    slope = fit_background(cells, background)["slope"]
    return np.asarray(cells, dtype=float) - slope[:, None] * np.asarray(background, dtype=float)


# --- chance levels ------------------------------------------------------------------------------


def shift_offsets(
    n_frames: int, min_shift_frames: int, n_shifts: int, rng: np.random.Generator
) -> np.ndarray:
    """Random circular shifts, each at least *min_shift_frames* from zero in both directions.

    Raises:
        ValueError: If a setting is below 1 or the trace is shorter than twice the minimum.
    """
    if min_shift_frames < 1 or n_shifts < 1:
        raise ValueError("min_shift_frames and n_shifts must be at least 1.")
    if n_frames < 2 * min_shift_frames:
        raise ValueError(
            f"{n_frames} frames are too few for circular shifts of at least {min_shift_frames} "
            "frames in both directions."
        )
    return rng.integers(min_shift_frames, n_frames - min_shift_frames + 1, size=n_shifts)


def null_r(cells: np.ndarray, other: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """r(cell, *other* circularly shifted by each offset): ``(n_shifts, n_cells)``."""
    other = np.asarray(other, dtype=float)
    return np.stack([row_correlation(cells, np.roll(other, k, axis=-1)) for k in offsets])


def null_partial_r(
    cells: np.ndarray,
    average: np.ndarray,
    background: np.ndarray,
    offsets: np.ndarray,
    shifted: str,
) -> np.ndarray:
    """Partial r(cell, average | background) with one of the two reference traces shifted.

    Args:
        cells (np.ndarray): ``(n_cells, n_frames)``.
        average (np.ndarray): Leave-one-out averages, ``(n_cells, n_frames)``.
        background (np.ndarray): ``(n_frames,)``.
        offsets (np.ndarray): Circular shifts (frames).
        shifted (str): ``"background"`` (the control is wrong, so little is removed) or
            ``"average"`` (the cell-to-average relation is broken while the control is real).

    Returns:
        np.ndarray: ``(n_shifts, n_cells)``.

    Raises:
        ValueError: If *shifted* is neither of those.
    """
    if shifted not in SHIFTED_TRACES:
        raise ValueError(f"shifted must be one of {SHIFTED_TRACES}, not {shifted!r}.")
    background, average = np.asarray(background, dtype=float), np.asarray(average, dtype=float)
    if shifted == "background":
        return np.stack(
            [partial_row_correlation(cells, average, np.roll(background, k)) for k in offsets]
        )
    return np.stack(
        [partial_row_correlation(cells, np.roll(average, k, axis=-1), background) for k in offsets]
    )


def null_percentile(null: np.ndarray, q: float = NULL_PERCENTILE, statistic=None):
    """The *q*-th percentile of a null ``(n_shifts, n_cells)``.

    Args:
        null (np.ndarray): One row per shift.
        q (float): Percentile.
        statistic (Callable | None): Reduces across cells within each shift (called as
            ``statistic(null, axis=1)``, e.g. ``np.nanmedian``), giving one number for the run.
            Default None, meaning a percentile per cell across shifts.

    Returns:
        np.ndarray | float: ``(n_cells,)``, or a float with *statistic*.
    """
    with quiet_empty_means():
        if statistic is None:
            return np.nanpercentile(null, q, axis=0)
        return float(np.nanpercentile(statistic(null, axis=1), q))


# --- comparison across cells --------------------------------------------------------------------


def paired_comparison(a: np.ndarray, b: np.ndarray) -> dict:
    """Median of ``a - b`` and a Wilcoxon signed-rank p over the cells with both values.

    Cells within a run are not independent, so the p value is descriptive, not a test of
    a population. Zero differences count half on each side (``zsplit``), so identical inputs
    give p = 1 instead of an error.

    Returns:
        dict: ``median_difference``, ``p`` and ``n`` (pairs used; NaN statistics when n is 0).
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    n = int(ok.sum())
    if n == 0:
        return {"median_difference": np.nan, "p": np.nan, "n": 0}
    difference = a[ok] - b[ok]
    p = float(wilcoxon(difference, zero_method="zsplit").pvalue)
    return {"median_difference": float(np.median(difference)), "p": p, "n": n}


# --- all statistics of one run ------------------------------------------------------------------


def shared_signal_stats(z: np.ndarray, background_z: np.ndarray, offsets: np.ndarray) -> dict:
    """Every statistic of one run, channel and measure.

    Args:
        z (np.ndarray): ``(n_cells, n_frames)`` detrended robust-z traces.
        background_z (np.ndarray): ``(n_frames,)`` the background, detrended and z-scored.
        offsets (np.ndarray): Circular shifts (frames) for the chance levels.

    Returns:
        dict: ``n_cells``, ``n_undefined`` (cells whose r with the background or the average
        is undefined), ``r_bg_avg`` and its null ``null_r_bg_avg`` ``(n_shifts,)``; per cell
        ``r_loo``, ``r_bg``, ``partial_r``, ``r2`` and ``slope``; their nulls ``null_r_bg``,
        ``null_r_loo``, ``null_partial_background`` and ``null_partial_average``, each
        ``(n_shifts, n_cells)``.

    Raises:
        ValueError: If the background and the cells differ in their number of frames.
    """
    z, background_z = np.asarray(z, dtype=float), np.asarray(background_z, dtype=float)
    if z.shape[1] != background_z.shape[0]:
        raise ValueError(
            f"cells have {z.shape[1]} frames but the background has {background_z.shape[0]}."
        )
    with quiet_empty_means():
        average = np.nanmean(z, axis=0)
    loo = leave_one_out_mean(z)
    r_bg, r_loo = row_correlation(z, background_z), row_correlation(z, loo)
    fit = fit_background(z, background_z)
    return {
        "n_cells": z.shape[0],
        "n_undefined": int((~(np.isfinite(r_bg) & np.isfinite(r_loo))).sum()),
        "r_bg_avg": float(row_correlation(average, background_z)[0]),
        "null_r_bg_avg": null_r(average, background_z, offsets)[:, 0],
        "r_loo": r_loo,
        "r_bg": r_bg,
        "partial_r": partial_row_correlation(z, loo, background_z),
        "r2": fit["r2"],
        "slope": fit["slope"],
        "null_r_bg": null_r(z, background_z, offsets),
        "null_r_loo": null_r(z, loo, offsets),
        "null_partial_background": null_partial_r(z, loo, background_z, offsets, "background"),
        "null_partial_average": null_partial_r(z, loo, background_z, offsets, "average"),
    }


def summary_row(stats: dict) -> dict:
    """The numbers of one run, channel and measure for the printed table."""
    with quiet_empty_means():
        return {
            "cells": stats["n_cells"],
            "undefined": stats["n_undefined"],
            "r(bg, avg)": stats["r_bg_avg"],
            "median r(cell, LOO)": float(np.nanmedian(stats["r_loo"])),
            "median r(cell, bg)": float(np.nanmedian(stats["r_bg"])),
            "median partial r": float(np.nanmedian(stats["partial_r"])),
            "null 95th (median partial r)": null_percentile(
                stats["null_partial_average"], NULL_PERCENTILE, np.nanmedian
            ),
            "median R2": float(np.nanmedian(stats["r2"])),
        }


# --- responses ----------------------------------------------------------------------------------


def mean_response(
    pooled: dict, offsets: np.ndarray, period_s: float, range_s: tuple[float, float] = (0.0, 2.0)
) -> dict:
    """Mean of each pooled response over the frames starting within *range_s* of the outcome.

    Args:
        pooled (dict): ``{view: {pulses: {"mean": (n_window,), ...}}}``.
        offsets (np.ndarray): Window offsets (frames from the outcome's frame).
        period_s (float): Frame period (s).
        range_s (tuple[float, float]): Inclusive range of frame start times (s).

    Returns:
        dict: ``{view: {pulses: mean}}``.

    Raises:
        ValueError: If no frame of the window starts in the range.
    """
    seconds = np.asarray(offsets) * period_s
    inside = (seconds >= range_s[0]) & (seconds <= range_s[1])
    if not inside.any():
        raise ValueError(f"no frames of the window start within {range_s} s of the outcome.")
    return {
        view: {
            pulses: float(np.nanmean(stats["mean"][inside])) for pulses, stats in per_pulse.items()
        }
        for view, per_pulse in pooled.items()
    }
