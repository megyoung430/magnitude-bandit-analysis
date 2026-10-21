"""Do cells have private calcium transients? Skewness and one-sided excursions of ΔF/F0 (pure).

A calcium indicator's transients rise fast and decay slowly, so a cell with them has a
positively skewed ΔF/F0 and more excursions above ``+z`` than below ``-z``; noise is symmetric.
The red channel holds no calcium indicator, so the same statistics on the red trace of the same
cells are the reference for what noise and artefacts alone produce. Each trace is detrended (a
slow running-percentile baseline subtracted) and robust-z-scored first, so a drift or a scale
does not matter.

Excursion *events* are threshold crossings (the trace passing ``+z`` upwards, or ``-z``
downwards) at least a minimum gap apart, so a transient that wobbles around the threshold counts
once.
"""

import math

import numpy as np

from src.funcimg_analysis.event_frames import frame_period_s, quiet_empty_means
from src.funcimg_analysis.shared_signal import detrended_z, window_frames_from_s

SKEW_PERCENTILES = (1, 99)
# The median, not the low percentile used for the correlations: a low percentile follows the
# noise's lower tail, so subtracting it makes positive excursions larger than negative ones even
# for symmetric noise. Transients are sparse, so the median still sits on the baseline.
DETREND_PERCENTILE = 50.0
SIGNS = ("positive", "negative")


def count_events(z: np.ndarray, sign: str, threshold: float, gap_frames: int) -> int:
    """Threshold crossings of one trace, at least *gap_frames* apart.

    Args:
        z (np.ndarray): ``(n_frames,)`` robust z trace (NaN allowed).
        sign (str): ``"positive"`` (crossing above ``+threshold``) or ``"negative"`` (below
            ``-threshold``).
        threshold (float): In units of the trace's robust SD.
        gap_frames (int): A crossing within this many frames of the previous counted one is part
            of it.

    Raises:
        ValueError: If *sign* is neither of those.
    """
    if sign not in SIGNS:
        raise ValueError(f"sign must be one of {SIGNS}, not {sign!r}.")
    beyond = z > threshold if sign == "positive" else z < -threshold
    crossings = np.flatnonzero(beyond[1:] & ~beyond[:-1]) + 1
    if beyond[0]:
        crossings = np.concatenate([[0], crossings])
    count, last = 0, None
    for frame in crossings:  # sequential: each crossing depends on the last one counted
        if last is None or frame - last >= gap_frames:
            count, last = count + 1, frame
    return count


def skewness(z: np.ndarray) -> np.ndarray:
    """Per-row skewness over the finite frames, ``mean(d^3) / mean(d^2)^1.5``; NaN without data."""
    z = np.asarray(z, dtype=float)
    with quiet_empty_means(), np.errstate(invalid="ignore", divide="ignore"):
        d = z - np.nanmean(z, axis=1, keepdims=True)
        return np.nanmean(d**3, axis=1) / np.nanmean(d**2, axis=1) ** 1.5


def robust_skewness(z: np.ndarray) -> np.ndarray:
    """Per-row ``((p99 - median) - (median - p1)) / MAD``: skewness that one spike cannot move.

    The MAD is the plain (unscaled) median absolute deviation, so for normal data the
    percentile spread is about 4.7 MADs wide and the value is 0 when the tails are symmetric.
    """
    z = np.asarray(z, dtype=float)
    with quiet_empty_means(), np.errstate(invalid="ignore", divide="ignore"):
        median = np.nanmedian(z, axis=1)
        mad = np.nanmedian(np.abs(z - median[:, None]), axis=1)
        low, high = np.nanpercentile(z, SKEW_PERCENTILES, axis=1)
        return ((high - median) - (median - low)) / mad


def transient_statistics(
    dff: np.ndarray,
    period_s: float,
    *,
    detrend_window_s: float,
    z_threshold: float,
    min_gap_s: float,
) -> dict[str, np.ndarray]:
    """Skewness and excursion rates of each cell's detrended, robust-z ΔF/F0.

    Args:
        dff (np.ndarray): ``(n_cells, n_frames)``; a cell with no finite value has NaN statistics.
        period_s (float): Frame period (s).
        detrend_window_s (float): Length of the running-percentile baseline (s).
        z_threshold (float): Excursion threshold in robust SD.
        min_gap_s (float): Smallest separation of two counted events (s).

    Returns:
        dict[str, np.ndarray]: Per cell ``skew``, ``robust_skew``, ``rate_pos`` and ``rate_neg``
        (events per minute of frames with data) and ``rate_diff`` (positive minus negative).

    Raises:
        ValueError: If the trace is shorter than the detrend window.
    """
    z, _ = detrended_z(dff, window_frames_from_s(detrend_window_s, period_s), DETREND_PERCENTILE)
    gap = math.ceil(min_gap_s / period_s)
    minutes = np.isfinite(z).sum(axis=1) * period_s / 60
    rates = {}
    for key, sign in (("rate_pos", "positive"), ("rate_neg", "negative")):
        counts = np.array([count_events(row, sign, z_threshold, gap) for row in z], dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            rates[key] = np.where(minutes > 0, counts / minutes, np.nan)
    return {
        "skew": skewness(z),
        "robust_skew": robust_skewness(z),
        **rates,
        "rate_diff": rates["rate_pos"] - rates["rate_neg"],
    }


def run_transients(
    run: dict, *, detrend_window_s: float, z_threshold: float, min_gap_s: float
) -> dict:
    """Transient statistics of both channels of a ``collect_residual_dff`` run.

    Returns:
        dict: ``green`` and ``red`` (see :func:`transient_statistics`) and ``period_s``.
    """
    period = frame_period_s(run["time_s"])
    stats = {
        channel: transient_statistics(
            run["channels"][channel]["dff"],
            period,
            detrend_window_s=detrend_window_s,
            z_threshold=z_threshold,
            min_gap_s=min_gap_s,
        )
        for channel in ("green", "red")
    }
    return {**stats, "period_s": period}


def most_skewed(skew: np.ndarray, n: int) -> np.ndarray:
    """The *n* cells with the highest skewness, highest first; cells without one are left out."""
    skew = np.asarray(skew, dtype=float)
    order = np.argsort(-np.nan_to_num(skew, nan=-np.inf), kind="stable")
    return np.array([int(i) for i in order if np.isfinite(skew[i])][:n], dtype=int)


def channel_comparison(green: dict, red: dict) -> dict:
    """Green against the red reference: skewness, how many green cells exceed red, event rates.

    Args:
        green (dict): :func:`transient_statistics` of the green cells.
        red (dict): The same for the red channel.

    Returns:
        dict: Median skewness of each channel, the fraction of green cells whose skewness is above
        the 95th percentile of the red cells' (5 % expected if green behaved like red), and the
        median positive and negative event rates (per minute) of each channel.
    """
    with quiet_empty_means():
        red_95 = np.nanpercentile(red["skew"], 95)
        green_skew = green["skew"][np.isfinite(green["skew"])]
        table = {
            "n cells green": len(green_skew),
            "n cells red": int(np.isfinite(red["skew"]).sum()),
            "median skew green": float(np.nanmedian(green["skew"])),
            "median skew red": float(np.nanmedian(red["skew"])),
            "fraction green above red 95th": float(np.mean(green_skew > red_95)),
        }
        for label, key in (("positive", "rate_pos"), ("negative", "rate_neg")):
            for channel, stats in (("green", green), ("red", red)):
                table[f"median {label} rate {channel}"] = float(np.nanmedian(stats[key]))
    return table
