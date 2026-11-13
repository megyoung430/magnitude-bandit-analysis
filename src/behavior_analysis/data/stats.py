"""Shared NaN-aware summary statistics for the analysis modules.

A standard error needs at least two values: with fewer it is NaN, never 0, and
NaN entries count towards neither the spread nor n. Plots then draw the mean
and omit only the error bar.
"""

import warnings

import numpy as np


def nan_mean(stack: np.ndarray, axis: int = 0) -> np.ndarray:
    """Mean over the non-NaN entries along *axis*; NaN, without a warning, where all are NaN.

    Args:
        stack (np.ndarray): Float array, e.g. ``(n_rows, n_columns)``.
        axis (int): Axis to average over. Default 0.

    Returns:
        np.ndarray: *stack* with *axis* removed.
    """
    n_valid = np.sum(~np.isnan(stack), axis=axis)
    total = np.nansum(stack, axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n_valid > 0, total / np.maximum(n_valid, 1), np.nan)


def nan_standard_error(stack: np.ndarray) -> np.ndarray:
    """Per-column standard error of the mean over the non-NaN rows of *stack*.

    NaN in columns with fewer than two non-NaN rows (the sample SD needs two).
    A 1-D *stack* is a single column: the result is one 0-d value over all
    elements, not one value per element.

    Args:
        stack (np.ndarray): Float array of shape ``(n_rows, n_columns)`` (or 1-D, one column).

    Returns:
        np.ndarray: Standard error per column, shape ``(n_columns,)``.
    """
    n_valid = np.sum(~np.isnan(stack), axis=0)
    with warnings.catch_warnings():
        # nanstd warns for columns with < 2 values; those columns are set to NaN below.
        warnings.filterwarnings(
            "ignore", message="Degrees of freedom <= 0", category=RuntimeWarning
        )
        se = np.nanstd(stack, axis=0, ddof=1) / np.sqrt(n_valid)
    return np.where(n_valid >= 2, se, np.nan)


def mean_se_n(values: list) -> tuple[float, float, int]:
    """Mean, standard error and count over the non-NaN *values*.

    The standard error is NaN (never 0) with fewer than two values; the mean is
    NaN, with a count of 0, when there are none.

    Args:
        values (list): Numbers (or any array-like convertible to float); NaN entries are ignored.

    Returns:
        tuple[float, float, int]: ``(mean, standard error, count of non-NaN values)``.
    """
    vals = np.asarray(values, dtype=float)
    vals = vals[~np.isnan(vals)]
    n = int(vals.size)
    if n == 0:
        return float("nan"), float("nan"), n
    return float(vals.mean()), float(nan_standard_error(vals[:, None])[0]), n
