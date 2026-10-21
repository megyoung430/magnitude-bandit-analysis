"""Building blocks for asking whether cells share signal beyond the field-wide background (pure).

Traces are detrended (a slow running-percentile baseline subtracted) and robust-z-scored first,
so that a slow drift or a different scale does not create correlations. Every correlation uses
the frames that are finite in all the traces involved; frames without a pyControl time are
therefore included, since the question is about the traces, not about the events.

The comparison that matters is a cell against the *leave-one-out* average of the other cells
versus the same cell against the background: both sides use one noisy cell and one cleaner
reference, whereas the whole-FOV average would be mostly background pixels.

Chance levels live in :mod:`src.funcimg_analysis.shared_signal_null`, the statistics of a whole
run in :mod:`src.funcimg_analysis.run_statistics`.
"""

from collections.abc import Sequence

import numpy as np
from scipy.stats import wilcoxon

from src.funcimg_analysis.event_frames import quiet_empty_means, robust_z

DETREND_PERCENTILE = 10.0
MIN_WINDOW_FRAMES = 3
MIN_FRAMES_FOR_CORRELATION = 3
SINGULAR_DETERMINANT = 1e-10  # below this a correlation matrix has no usable inverse


def window_frames_from_s(window_s: float, period_s: float) -> int:
    """An odd number of frames (at least 3) spanning about *window_s* seconds."""
    return max(MIN_WINDOW_FRAMES, int(round(window_s / period_s)) | 1)


# --- detrending ---------------------------------------------------------------------------------


def running_percentile(
    traces: np.ndarray, window_frames: int, percentile: float, stride_frames: int | None = None
) -> np.ndarray:
    """A centred running percentile of each row, NaN-safe, evaluated on a coarse grid.

    The percentile is taken over ``2 * (window_frames // 2) + 1`` frames around each grid frame
    and linearly interpolated between grid frames, which keeps a 60 s window over tens of
    thousands of frames cheap. Within half a window of either end of the trace the window is
    clipped to the frames that exist, so the baseline there is estimated from about half as many
    frames, all on one side, and is less reliable (it follows the first or last stretch of the
    trace rather than being centred on it). A stretch of NaN longer than the window leaves NaN
    in the baseline there.

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
    at_grid = np.stack(
        [_window_percentile(traces[:, max(0, c - half) : c + half + 1], percentile) for c in grid],
        axis=1,
    )
    return _interpolate_grid(at_grid, grid, n_frames)


def _window_percentile(window: np.ndarray, percentile: float) -> np.ndarray:
    """Per-row percentile of ``(n_rows, n_window)``, ignoring NaN.

    Rows without NaN take numpy's vectorised ``percentile``; ``nanpercentile`` loops over rows
    and is an order of magnitude slower, so it is kept for the rows that need it.
    """
    out = np.full(len(window), np.nan)
    clean = np.isfinite(window).all(axis=1)
    if clean.any():
        out[clean] = np.percentile(window[clean], percentile, axis=1)
    if not clean.all():
        with quiet_empty_means():
            out[~clean] = np.nanpercentile(window[~clean], percentile, axis=1)
    return out


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


def as_rows(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
    a, b = as_rows(a, b)
    ok = np.isfinite(a) & np.isfinite(b)
    n = ok.sum(axis=1)
    if ok.all():  # the usual case: no masking, far fewer temporaries
        mean_a, mean_b = a.mean(axis=1), b.mean(axis=1)
        da, db = a - mean_a[:, None], b - mean_b[:, None]
        return {
            "n": n,
            "mean_a": mean_a,
            "mean_b": mean_b,
            "var_a": np.einsum("ij,ij->i", da, da) / n,
            "var_b": np.einsum("ij,ij->i", db, db) / n,
            "cov": np.einsum("ij,ij->i", da, db) / n,
        }
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


def has_spread(mean: np.ndarray, variance: np.ndarray) -> np.ndarray:
    """Whether a row varies by more than rounding error of its mean."""
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
        & has_spread(m["mean_a"], m["var_a"])
        & has_spread(m["mean_b"], m["var_b"])
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        r = m["cov"] / np.sqrt(m["var_a"] * m["var_b"])
    return np.where(defined, np.clip(r, -1, 1), np.nan)


def _common_frames(variables: Sequence[np.ndarray]) -> list[np.ndarray]:
    """The traces as ``(n, n_frames)`` arrays, NaN wherever any of them is not finite."""
    rows = [np.atleast_2d(np.asarray(v, dtype=float)) for v in variables]
    if len({r.shape[1] for r in rows}) != 1:
        raise ValueError(f"traces have different numbers of frames: {[r.shape[1] for r in rows]}.")
    n = max(r.shape[0] for r in rows)
    if any(r.shape[0] not in (1, n) for r in rows):
        raise ValueError(f"cannot pair traces of {[r.shape[0] for r in rows]} rows.")
    rows = [np.broadcast_to(r, (n, r.shape[1])) for r in rows]
    ok = np.logical_and.reduce([np.isfinite(r) for r in rows])
    return [np.where(ok, r, np.nan) for r in rows]


def correlation_matrices(variables: Sequence[np.ndarray]) -> np.ndarray:
    """Per-row correlation matrix of several traces over the frames finite in all of them.

    Args:
        variables (Sequence[np.ndarray]): Each ``(n, n_frames)`` or ``(n_frames,)``.

    Returns:
        np.ndarray: ``(n, m, m)`` for *m* variables; NaN where a pair is undefined.
    """
    rows = _common_frames(variables)
    m = len(rows)
    out = np.ones((rows[0].shape[0], m, m))
    for i in range(m):
        for j in range(i + 1, m):
            out[:, i, j] = out[:, j, i] = row_correlation(rows[i], rows[j])
    return out


def partial_from_matrix(matrix: np.ndarray) -> np.ndarray:
    """Partial correlation of variables 0 and 1 given all others, from correlation matrices.

    Args:
        matrix (np.ndarray): ``(..., m, m)`` correlation matrices, ``m >= 3``.

    Returns:
        np.ndarray: ``(...)``; NaN where the matrix is undefined or (nearly) singular, e.g. a
        cell that is exactly a linear function of a control.
    """
    matrix = np.asarray(matrix, dtype=float)
    with np.errstate(invalid="ignore"):
        determinant = np.linalg.det(matrix)  # NaN for a matrix with an undefined pair
    unusable = ~(np.abs(determinant) > SINGULAR_DETERMINANT)  # also True for NaN
    safe = np.where(unusable[..., None, None], np.eye(matrix.shape[-1]), matrix)
    precision = np.linalg.inv(safe)
    with np.errstate(invalid="ignore", divide="ignore"):
        partial = -precision[..., 0, 1] / np.sqrt(precision[..., 0, 0] * precision[..., 1, 1])
    return np.where(unusable, np.nan, np.clip(partial, -1, 1))


def partial_row_correlation(
    a: np.ndarray, b: np.ndarray, control: np.ndarray | Sequence[np.ndarray]
) -> np.ndarray:
    """Correlation of *a* and *b* per row after regressing the control(s) (with intercept) out.

    Equal to correlating the two residual traces. All traces are restricted to the frames finite
    in all of them, so the controls are removed from exactly the frames that are correlated.

    Args:
        a (np.ndarray): ``(n, n_frames)``.
        b (np.ndarray): ``(n, n_frames)`` or ``(n_frames,)``.
        control (np.ndarray | Sequence[np.ndarray]): One control, or a list/tuple of them. Each
            is ``(n_frames,)`` (shared, e.g. the background) or ``(n, n_frames)`` (one per row,
            e.g. each cell's own neuropil).

    Returns:
        np.ndarray: ``(n,)``; NaN where undefined (e.g. a row perfectly explained by the control).
    """
    controls = list(control) if isinstance(control, list | tuple) else [control]
    return partial_from_matrix(correlation_matrices([a, b, *controls]))


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


# --- reliabilities ------------------------------------------------------------------------------


def spearman_brown(r_half: np.ndarray) -> np.ndarray:
    """Reliability of the whole from the correlation of its two halves, ``2r / (1 + r)``.

    NaN where the halves do not correlate positively (there is no reliability to speak of), and
    never above 1.
    """
    r = np.asarray(r_half, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        stepped = np.minimum(2 * r / (1 + r), 1.0)
    return np.where(r > 0, stepped, np.nan)


def background_reliability(half_1: np.ndarray, half_2: np.ndarray) -> tuple[float, float]:
    """Split-half reliability of the background from two non-overlapping sets of its pixels.

    Args:
        half_1 (np.ndarray): ``(n_frames,)`` detrended z trace of the first set of pixels.
        half_2 (np.ndarray): The same for the second set.

    Returns:
        tuple[float, float]: r(half 1, half 2) and the Spearman-Brown reliability of the whole
        background (NaN if r is not positive).

    Raises:
        ValueError: If the traces differ in their number of frames.
    """
    r_halves = float(row_correlation(half_1[None, :], half_2)[0])
    return r_halves, float(spearman_brown(np.array(r_halves))[()])


def leave_one_out_reliability(
    z: np.ndarray, rng: np.random.Generator, n_splits: int = 10
) -> np.ndarray:
    """Reliability of each cell's leave-one-out average, by split halves of the other cells.

    For each cell the other cells are split at random into two halves that differ by at most one
    cell, the halves are averaged, and r(half 1, half 2) is stepped up by the Spearman-Brown
    formula. The correlation is averaged over *n_splits* random splits first. The reliability
    says how much of the average's variance is signal common to the cells rather than the
    private noise of the few cells averaged, and is what attenuates r(cell, average).

    Args:
        z (np.ndarray): ``(n_cells, n_frames)`` detrended z traces.
        rng (np.random.Generator): Draws the splits.
        n_splits (int): Number of random splits.

    Returns:
        np.ndarray: ``(n_cells,)``, NaN where the halves do not correlate positively.

    Raises:
        ValueError: If there are fewer than three cells (the others cannot be split).
    """
    z = np.asarray(z, dtype=float)
    n = z.shape[0]
    if n < 3:
        raise ValueError("A split-half reliability of the other cells needs at least three cells.")
    finite = np.isfinite(z)
    values = np.where(finite, z, 0.0)
    all_finite = bool(finite.all())
    half_size = (n - 1) // 2
    r_splits = []
    for _ in range(n_splits):
        # row i marks, among the cells other than i, the half_size cells that form half A
        draw = rng.random((n, n))
        np.fill_diagonal(draw, np.inf)
        in_a = (np.argsort(np.argsort(draw, axis=1), axis=1) < half_size).astype(float)
        in_b = 1.0 - in_a
        np.fill_diagonal(in_b, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            if all_finite:
                mean_a, mean_b = in_a @ values / half_size, in_b @ values / (n - 1 - half_size)
            else:
                mean_a = in_a @ values / (in_a @ finite)
                mean_b = in_b @ values / (in_b @ finite)
        r_splits.append(row_correlation(mean_a, mean_b))
    with quiet_empty_means():
        mean_r = np.nanmean(np.stack(r_splits), axis=0)
    return spearman_brown(mean_r)


# --- regression on the background ---------------------------------------------------------------


def fit_background(
    cells: np.ndarray, background: np.ndarray, fit_mask: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    """OLS of each cell on the background, ``cell ~ intercept + slope * background``.

    Fitted on the frames finite in both, and in *fit_mask* when given. Returns ``slope``,
    ``intercept`` and ``r2`` per cell (NaN where the cell or the background has no variance
    over the frames used).

    Args:
        cells (np.ndarray): ``(n_cells, n_frames)``.
        background (np.ndarray): ``(n_frames,)``.
        fit_mask (np.ndarray | None): Boolean ``(n_frames,)``, True for the frames to fit on.
            Default None, every frame.
    """
    cells, background = np.asarray(cells, dtype=float), np.asarray(background, dtype=float)
    if fit_mask is not None:
        outside = ~np.asarray(fit_mask, dtype=bool)
        cells, background = np.where(outside, np.nan, cells), np.where(outside, np.nan, background)
    m = _moments(cells, background)
    defined = (
        (m["n"] >= MIN_FRAMES_FOR_CORRELATION)
        & has_spread(m["mean_a"], m["var_a"])
        & has_spread(m["mean_b"], m["var_b"])
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        slope = np.where(defined, m["cov"] / m["var_b"], np.nan)
        intercept = m["mean_a"] - slope * m["mean_b"]
    return {"slope": slope, "intercept": intercept, "r2": row_correlation(cells, background) ** 2}


def remove_background(
    cells: np.ndarray, background: np.ndarray, fit_mask: np.ndarray | None = None
) -> np.ndarray:
    """Each cell minus ``slope * background``, with the slope fitted on the frames in *fit_mask*.

    Subtracting only the slope term (not the intercept) leaves the cell's baseline where it was,
    so the usual robust-z and pre-event baseline subtraction apply to the result unchanged. The
    slope is applied to every frame, also those left out of the fit. Fitting outside the outcome
    windows keeps a reward response that the background happens to share from pulling the slope
    towards itself.
    """
    slope = fit_background(cells, background, fit_mask)["slope"]
    return np.asarray(cells, dtype=float) - slope[:, None] * np.asarray(background, dtype=float)


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
