"""Chance levels from circular shifts: a trace is shifted in time against the others (pure).

A shift keeps a trace's own autocorrelation but breaks its alignment with the others, so the
statistics recomputed on shifted traces show what the alignment-free structure alone produces.
A shifted *background* leaves the real relation of a cell to the other cells intact and shows how
little an unrelated control removes; shifting the *average of the other cells* breaks that
relation while the control stays real, which tests partial r = 0. An imperfect control leaves
residual field signal, so beating the null does not by itself show that cells share signal.

For rows without missing frames all shifts are computed at once from FFT cross-correlations
(``O(T log T)`` per row instead of ``O(T)`` per row and shift); rows with a missing frame, whose
frames in common change with every shift, go through the plain loop and give identical numbers.
"""

from collections.abc import Callable, Sequence

import numpy as np

from src.funcimg_analysis.event_frames import quiet_empty_means
from src.funcimg_analysis.shared_signal import (
    as_rows,
    correlation_matrices,
    has_spread,
    partial_from_matrix,
    partial_row_correlation,
    row_correlation,
)

NULL_PERCENTILE = 95.0
SHIFTED_TRACES = ("background", "average")


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


def _standardised(rows: np.ndarray) -> np.ndarray:
    """Rows with mean 0 and unit population variance; NaN where a row has no spread."""
    mean = rows.mean(axis=-1, keepdims=True)
    std = rows.std(axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(has_spread(mean, std**2), (rows - mean) / std, np.nan)


def shifted_correlations(
    shifted: np.ndarray, fixed: Sequence[np.ndarray], offsets: np.ndarray
) -> np.ndarray:
    """r(fixed trace, *shifted* rolled by each offset) for rows without missing frames.

    Args:
        shifted (np.ndarray): ``(n_frames,)`` shared by every row, or ``(n, n_frames)``.
        fixed (Sequence[np.ndarray]): Traces ``(n, n_frames)``, correlated with the shifted one.
        offsets (np.ndarray): Circular shifts ``k``; ``np.roll(shifted, k)`` is correlated.

    Returns:
        np.ndarray: ``(len(offsets), n, len(fixed))``.
    """
    n_frames = np.shape(shifted)[-1]
    spectrum = np.conj(np.fft.rfft(_standardised(np.asarray(shifted, dtype=float)), axis=-1))
    out = np.empty((len(offsets), len(fixed[0]), len(fixed)))
    for j, trace in enumerate(fixed):
        product = np.fft.rfft(_standardised(np.asarray(trace, dtype=float)), axis=-1) * spectrum
        # (1/T) sum_t shifted(t) * fixed(t + k) equals r(fixed, np.roll(shifted, k)) at lag k
        correlation = np.fft.irfft(product, n=n_frames, axis=-1) / n_frames
        out[:, :, j] = correlation[:, np.asarray(offsets)].T
    return out


def _clean_rows(rows: Sequence[np.ndarray]) -> np.ndarray:
    """Which rows are finite in every trace (so that a shift cannot change the frames in common)."""
    return np.logical_and.reduce([np.isfinite(r).all(axis=-1) for r in rows])


def null_r(cells: np.ndarray, other: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """r(cell, *other* circularly shifted by each offset): ``(n_shifts, n_cells)``.

    Args:
        cells (np.ndarray): ``(n_cells, n_frames)``.
        other (np.ndarray): ``(n_frames,)`` shared, or ``(n_cells, n_frames)`` one per cell.
        offsets (np.ndarray): Circular shifts (frames).
    """
    cells, other_rows = as_rows(cells, other)
    other = np.asarray(other, dtype=float)
    clean = _clean_rows([cells, other_rows])
    out = np.full((len(offsets), len(cells)), np.nan)
    if clean.any():
        shifted = other if other.ndim == 1 else other[clean]
        out[:, clean] = shifted_correlations(shifted, [cells[clean]], offsets)[:, :, 0]
    if not clean.all():
        rest = ~clean
        out[:, rest] = np.stack(
            [row_correlation(cells[rest], np.roll(other_rows[rest], k, axis=-1)) for k in offsets]
        )
    return out


def null_partial_r(
    cells: np.ndarray,
    average: np.ndarray,
    controls: np.ndarray | Sequence[np.ndarray],
    offsets: np.ndarray,
    shifted: str,
) -> np.ndarray:
    """Partial r(cell, average | controls) with one of the reference traces shifted.

    Args:
        cells (np.ndarray): ``(n_cells, n_frames)``.
        average (np.ndarray): Leave-one-out averages, ``(n_cells, n_frames)``.
        controls (np.ndarray | Sequence[np.ndarray]): The background ``(n_frames,)``, or a list
            that starts with it, followed by e.g. each cell's neuropil.
        offsets (np.ndarray): Circular shifts (frames).
        shifted (str): ``"background"`` (the first control, the background, is shifted and the
            rest stay, so the control is wrong and little is removed) or ``"average"`` (the
            cell-to-average relation is broken while the controls are real).

    Returns:
        np.ndarray: ``(n_shifts, n_cells)``.

    Raises:
        ValueError: If *shifted* is neither of those.
    """
    if shifted not in SHIFTED_TRACES:
        raise ValueError(f"shifted must be one of {SHIFTED_TRACES}, not {shifted!r}.")
    controls = list(controls) if isinstance(controls, list | tuple) else [controls]
    variables = [np.asarray(v, dtype=float) for v in (cells, average, *controls)]
    index = 1 if shifted == "average" else 2
    rows = [np.broadcast_to(np.atleast_2d(v), (len(cells), v.shape[-1])) for v in variables]
    clean = _clean_rows(rows)
    out = np.full((len(offsets), len(cells)), np.nan)
    if clean.any():
        out[:, clean] = _batched_partial(variables, rows, clean, index, offsets)
    if not clean.all():
        rest = ~clean
        out[:, rest] = np.stack([_one_shift_partial(rows, rest, index, k) for k in offsets])
    return out


def _one_shift_partial(rows: list[np.ndarray], keep: np.ndarray, index: int, k: int) -> np.ndarray:
    """The partial correlation of the *keep* rows with trace *index* rolled by *k* (plain path)."""
    traces = [r[keep] for r in rows]
    traces[index] = np.roll(traces[index], k, axis=-1)
    return partial_row_correlation(traces[0], traces[1], traces[2:])


def _batched_partial(
    variables: list[np.ndarray],
    rows: list[np.ndarray],
    clean: np.ndarray,
    index: int,
    offsets: np.ndarray,
) -> np.ndarray:
    """All shifts of the partial correlation for rows without missing frames: ``(S, n_clean)``."""
    subset = [r[clean] for r in rows]
    matrices = correlation_matrices(subset)  # pairs that no shift changes
    others = [j for j in range(len(subset)) if j != index]
    shifted = variables[index] if variables[index].ndim == 1 else subset[index]
    across = shifted_correlations(shifted, [subset[j] for j in others], offsets)
    full = np.broadcast_to(matrices, (len(offsets), *matrices.shape)).copy()
    for position, j in enumerate(others):
        full[:, :, index, j] = full[:, :, j, index] = across[:, :, position]
    return partial_from_matrix(full)


def null_percentile(null: np.ndarray, q: float = NULL_PERCENTILE) -> np.ndarray:
    """The *q*-th percentile across shifts of a null ``(n_shifts, n_cells)``, per cell."""
    with quiet_empty_means():
        return np.nanpercentile(null, q, axis=0)


def run_null_percentile(
    null: np.ndarray,
    q: float = NULL_PERCENTILE,
    statistic: Callable[..., np.ndarray] = np.nanmedian,
) -> float:
    """The *q*-th percentile across shifts of a statistic of the cells, one number for the run.

    Args:
        null (np.ndarray): ``(n_shifts, n_cells)``.
        q (float): Percentile.
        statistic (Callable): Reduces across cells within each shift, called as
            ``statistic(null, axis=1)``. Default the median, the level the run's median
            has to exceed.
    """
    with quiet_empty_means():
        return float(np.nanpercentile(statistic(null, axis=1), q))
