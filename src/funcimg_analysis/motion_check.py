"""Does an artefact period explain the weak transient check, and do cells appear without it? (pure)

The red channel holds no calcium indicator, so a signal that all cells share in red is an
artefact (brain motion, blood flow). This module puts that shared signal next to the
registration's motion, suggests where an early artefact period ends, and redoes the transient
check of :mod:`transients` on the frames after it.

Times are seconds from the run's first frame (frame index times the frame period), so frames
without a pyControl time are kept and placed correctly.
"""

import numpy as np

from src.funcimg_analysis.event_frames import quiet_empty_means
from src.funcimg_analysis.frame_times import RunMismatchError
from src.funcimg_analysis.shared_signal import detrended_z, window_frames_from_s
from src.funcimg_analysis.transients import (
    DETREND_PERCENTILE,
    channel_comparison,
    transient_statistics,
)

CHANNELS = ("green", "red")
MARGIN_S = 10.0  # added after the last artefact frame, so its tail is left out too


class MotionUnavailableError(Exception):
    """``ops`` lacks an entry the motion check needs; the message says which."""


def ops_trace(ops: dict, key: str, n_frames: int) -> np.ndarray:
    """An ``ops`` entry with one value per frame, as floats.

    A non-rigid entry (one column per block, e.g. ``xoff1``) becomes its median block.

    Raises:
        MotionUnavailableError: If *ops* has no *key*.
        RunMismatchError: If the entry does not have *n_frames* values.
    """
    if key not in ops:
        raise MotionUnavailableError(f"ops has no {key!r}.")
    values = np.asarray(ops[key], dtype=float)
    if values.ndim > 1:
        values = np.nanmedian(values, axis=1)
    if len(values) != n_frames:
        raise RunMismatchError(f"ops[{key!r}] has {len(values)} values for {n_frames} frames.")
    return values


def optional_ops_trace(ops: dict, key: str, n_frames: int) -> np.ndarray | None:
    """:func:`ops_trace`, or None when *ops* has no *key* (for entries only some runs have)."""
    return ops_trace(ops, key, n_frames) if key in ops else None


def rigid_shift(xoff: np.ndarray, yoff: np.ndarray) -> np.ndarray:
    """Each frame's distance (px) from the run's median rigid position."""
    return np.hypot(xoff - np.nanmedian(xoff), yoff - np.nanmedian(yoff))


def frame_movement(xoff: np.ndarray, yoff: np.ndarray) -> np.ndarray:
    """How far (px) each frame moved from the previous one; NaN for the first frame."""
    return np.r_[np.nan, np.hypot(np.diff(xoff), np.diff(yoff))]


def shared_z(z: np.ndarray) -> np.ndarray:
    """Median across cells of ``(n_cells, n_frames)`` z traces: what the cells do together.

    The median, not the mean: one cell's large transient must not look shared.
    """
    with quiet_empty_means():
        return np.nanmedian(z, axis=0)


def suggest_artefact_end(
    median_red_z: np.ndarray,
    period_s: float,
    threshold: float,
    margin_s: float = MARGIN_S,
) -> dict:
    """Where an early artefact period ends: its last frame in the first half, plus a margin.

    A frame is artefact when ``|median red z|`` exceeds *threshold*. Only the first half is
    searched, so that one late artefact cannot cut away most of the run; artefact frames in the
    second half are counted instead.

    Returns:
        dict: ``end_s`` (0 when the first half has no artefact) and ``n_frames_over_late``.
    """
    over = np.abs(np.nan_to_num(median_red_z)) > threshold
    half = len(over) // 2
    early = np.flatnonzero(over[:half])
    end_s = float(early[-1] * period_s + margin_s) if len(early) else 0.0
    return {"end_s": end_s, "n_frames_over_late": int(over[half:].sum())}


def kept_frames(
    t_s: np.ndarray,
    cut_s: float,
    *,
    badframes: np.ndarray | None = None,
    shift_px: np.ndarray | None = None,
    max_shift_px: float | None = None,
) -> np.ndarray:
    """Frames at or after *cut_s*, not bad and not shifted more than *max_shift_px*.

    Raises:
        ValueError: If *max_shift_px* is given without *shift_px*.
    """
    keep = np.asarray(t_s) >= cut_s
    if badframes is not None:
        keep &= ~np.asarray(badframes, dtype=bool)
    if max_shift_px is not None:
        if shift_px is None:
            raise ValueError("max_shift_px needs the frames' shift_px.")
        keep &= ~(np.asarray(shift_px) > max_shift_px)
    return keep


def masked(traces: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """A float copy of ``(n_cells, n_frames)`` *traces* with the frames not in *keep* as NaN."""
    out = np.array(traces, dtype=float)
    out[:, ~keep] = np.nan
    return out


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return np.nan  # undefined, reported as such in the table
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def motion_correlations(
    median_z: dict[str, np.ndarray], motion: dict[str, np.ndarray], parts: dict[str, np.ndarray]
) -> list[dict]:
    """Correlation of each channel's shared signal with each motion trace, in each part of the run.

    Args:
        median_z (dict[str, np.ndarray]): Channel to :func:`shared_z`.
        motion (dict[str, np.ndarray]): Name to one value per frame (e.g. shift, corrXY).
        parts (dict[str, np.ndarray]): Label to a boolean frame mask, e.g. before and after the cut.

    Returns:
        list[dict]: One row per part and channel: ``r(median z, <name>)`` for every motion trace
        and ``SD of median z``.
    """
    return [
        {
            "frames": label,
            "channel": channel,
            **{
                f"r(median z, {name})": _correlation(median_z[channel][part], trace[part])
                for name, trace in motion.items()
            },
            "SD of median z": float(np.nanstd(median_z[channel][part])),
        }
        for label, part in parts.items()
        if part.sum() >= 3
        for channel in median_z
    ]


def motion_check(
    dff: dict[str, np.ndarray],
    ops: dict,
    period_s: float,
    *,
    detrend_window_s: float,
    z_threshold: float,
    min_gap_s: float,
    red_artefact_z: float,
    artefact_end_s: float | None = None,
    drop_badframes: bool = True,
    max_shift_px: float | None = None,
) -> dict:
    """The motion, the cells' shared signal, the cut, and the transient check with and without it.

    Args:
        dff (dict[str, np.ndarray]): ``green`` and ``red`` ``(n_cells, n_frames)`` ΔF/F0.
        ops (dict): Suite2p ops of the plane: ``xoff`` and ``yoff`` are needed; ``xoff1``,
            ``yoff1``, ``corrXY`` and ``badframes`` are used when present.
        period_s (float): Frame period (s).
        detrend_window_s (float): Running-median baseline removed before z-scoring (s).
        z_threshold (float): Transient excursion threshold (robust SD).
        min_gap_s (float): Smallest separation of two counted transients (s).
        red_artefact_z (float): Threshold of :func:`suggest_artefact_end`.
        artefact_end_s (float | None): The cut (s from the first frame); None uses the suggestion.
        drop_badframes (bool): Also leave out Suite2p's ``badframes``.
        max_shift_px (float | None): Also leave out frames shifted further than this.

    Returns:
        dict: ``t_s``, ``motion`` (``xoff``, ``yoff``, ``xoff1``, ``yoff1``, ``corrXY``,
        ``shift``; optional ones may be None), ``badframes``, ``median_z`` per channel,
        ``suggested_end_s``, ``n_frames_over_late``, ``cut_s``, ``keep``, ``correlations``
        (:func:`motion_correlations` before and after the cut), ``stats`` and ``comparison``
        (:func:`transient_statistics` and :func:`channel_comparison` for ``all frames`` and
        ``kept frames``), and ``z_kept`` (each channel's detrended robust z on the kept frames,
        NaN elsewhere).

    Raises:
        MotionUnavailableError: If ops has no ``xoff`` or ``yoff``.
        RunMismatchError: If an ops entry does not have one value per frame.
    """
    n_frames = dff["green"].shape[1]
    t_s = np.arange(n_frames) * period_s
    motion = {key: ops_trace(ops, key, n_frames) for key in ("xoff", "yoff")}
    motion |= {key: optional_ops_trace(ops, key, n_frames) for key in ("xoff1", "yoff1", "corrXY")}
    motion["shift"] = rigid_shift(motion["xoff"], motion["yoff"])
    badframes = (
        optional_ops_trace(ops, "badframes", n_frames).astype(bool)
        if "badframes" in ops
        else np.zeros(n_frames, bool)
    )

    window = window_frames_from_s(detrend_window_s, period_s)

    def z_of(traces):
        return detrended_z(traces, window, DETREND_PERCENTILE)[0]

    median_z = {channel: shared_z(z_of(dff[channel])) for channel in CHANNELS}
    suggestion = suggest_artefact_end(median_z["red"], period_s, red_artefact_z)
    cut_s = suggestion["end_s"] if artefact_end_s is None else float(artefact_end_s)
    keep = kept_frames(
        t_s,
        cut_s,
        badframes=badframes if drop_badframes else None,
        shift_px=motion["shift"],
        max_shift_px=max_shift_px,
    )

    correlation_motion = {
        "shift from median": motion["shift"],
        "frame-to-frame movement": frame_movement(motion["xoff"], motion["yoff"]),
    }
    if motion["corrXY"] is not None:
        correlation_motion["corrXY"] = motion["corrXY"]
    parts = {"before cut": t_s < cut_s, "after cut": t_s >= cut_s}

    settings = {
        "detrend_window_s": detrend_window_s,
        "z_threshold": z_threshold,
        "min_gap_s": min_gap_s,
    }
    frame_sets = {"all frames": np.ones(n_frames, bool), "kept frames": keep}
    stats = {
        label: {
            channel: transient_statistics(masked(dff[channel], frames), period_s, **settings)
            for channel in CHANNELS
        }
        for label, frames in frame_sets.items()
    }
    return {
        "t_s": t_s,
        "motion": motion,
        "badframes": badframes,
        "median_z": median_z,
        "suggested_end_s": suggestion["end_s"],
        "n_frames_over_late": suggestion["n_frames_over_late"],
        "cut_s": cut_s,
        "keep": keep,
        "correlations": motion_correlations(median_z, correlation_motion, parts),
        "stats": stats,
        "comparison": {
            label: channel_comparison(s["green"], s["red"]) for label, s in stats.items()
        },
        "z_kept": {channel: z_of(masked(dff[channel], keep)) for channel in CHANNELS},
    }
