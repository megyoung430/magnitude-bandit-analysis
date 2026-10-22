"""Do the most skewed cells flash in the movie? Peak images, peri-event traces and a clip (pure).

A cell's own calcium transient brightens the cell's pixels in green and stops at its edge, and
does nothing in red (no indicator). Motion shows as opposite-signed bands at edges in both
channels, and a field-wide signal brightens the whole crop. These functions take the registered
movies as ``(n_frames, Ly, Lx)`` arrays (a memory map works and is only read where indexed), so
reading the binaries stays in :mod:`collect_flash`.
"""

import math

import numpy as np
from scipy.ndimage import gaussian_filter

from src.funcimg_analysis.transients import most_skewed

CHANNELS = ("green", "red")


def offset_frames(range_s: tuple[float, float], period_s: float) -> np.ndarray:
    """Frame offsets from an event whose start falls in ``[range_s[0], range_s[1])``."""
    return np.arange(
        math.ceil(range_s[0] / period_s - 1e-9), math.ceil(range_s[1] / period_s - 1e-9)
    )


def peak_frames(
    z: np.ndarray,
    n_peaks: int,
    min_gap_frames: int,
    before_frames: int,
    after_frames: int,
    min_z: float = -np.inf,
) -> np.ndarray:
    """The frames of the *n_peaks* largest values of *z* above *min_z*, *min_gap_frames* apart.

    A peak needs every frame from ``before_frames`` before it to ``after_frames`` after it to be
    finite (inside the run and not left out, which is NaN in *z*).

    Returns:
        np.ndarray: Peak frames in time order; fewer than *n_peaks* (or none) if there are not
        enough usable ones.
    """
    finite = np.isfinite(z)
    order = np.argsort(-np.where(finite, z, -np.inf), kind="stable")
    n_candidates = int((finite & (np.where(finite, z, -np.inf) > min_z)).sum())
    chosen: list[int] = []
    for frame in order[:n_candidates]:  # sequential: each choice depends on the earlier ones
        if len(chosen) == n_peaks:
            break
        lo, hi = frame - before_frames, frame + after_frames
        if lo < 0 or hi >= len(z) or not finite[lo : hi + 1].all():
            continue
        if all(abs(frame - c) >= min_gap_frames for c in chosen):
            chosen.append(int(frame))
    return np.sort(np.array(chosen, dtype=int))


def crop_box(
    ypix: np.ndarray, xpix: np.ndarray, half_px: int, shape: tuple[int, int]
) -> tuple[slice, slice]:
    """Rows and columns of ±*half_px* around the ROI's median pixel, clipped to the image."""
    cy, cx = int(np.median(ypix)), int(np.median(xpix))
    return (
        slice(max(0, cy - half_px), min(shape[0], cy + half_px + 1)),
        slice(max(0, cx - half_px), min(shape[1], cx + half_px + 1)),
    )


def roi_mask(ypix: np.ndarray, xpix: np.ndarray, box: tuple[slice, slice]) -> np.ndarray:
    """Boolean image of the crop, True on the ROI's pixels that fall inside it."""
    rows, cols = box
    mask = np.zeros((rows.stop - rows.start, cols.stop - cols.start), dtype=bool)
    y, x = np.asarray(ypix) - rows.start, np.asarray(xpix) - cols.start
    inside = (y >= 0) & (y < mask.shape[0]) & (x >= 0) & (x < mask.shape[1])
    mask[y[inside], x[inside]] = True
    return mask


def _window_mean(movie, peak: int, offsets: np.ndarray, box: tuple[slice, slice]) -> np.ndarray:
    frames = movie[peak + offsets[0] : peak + offsets[-1] + 1, box[0], box[1]]
    return np.asarray(frames, dtype=float).mean(axis=0)


def event_dff_image(
    movie, peaks: np.ndarray, baseline: np.ndarray, response: np.ndarray, box: tuple[slice, slice]
) -> np.ndarray:
    """Mean over *peaks* of ``(response mean - baseline mean) / baseline mean`` in the crop.

    Pixels whose baseline is not positive have no ΔF/F0 and are NaN.
    """
    rows, cols = box
    shape = (rows.stop - rows.start, cols.stop - cols.start)
    if len(peaks) == 0:
        return np.full(shape, np.nan)

    def one(peak):
        base = _window_mean(movie, peak, baseline, box)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(
                base > 0, (_window_mean(movie, peak, response, box) - base) / base, np.nan
            )

    return np.mean([one(p) for p in peaks], axis=0)


def smoothed(image: np.ndarray, sigma_px: float) -> np.ndarray:
    """Gaussian-smoothed image that keeps NaN where the input is NaN; *sigma_px* 0 returns it."""
    if sigma_px <= 0:
        return image
    valid = np.isfinite(image)
    weight = gaussian_filter(valid.astype(float), sigma_px)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = gaussian_filter(np.where(valid, image, 0.0), sigma_px) / weight
    return np.where(valid, out, np.nan)


def peri_event(z: np.ndarray, peaks: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """``(n_peaks, n_offsets)``: *z* at each peak plus each offset."""
    return z[np.asarray(peaks, dtype=int)[:, None] + offsets[None, :]]


def binned_clip(
    movie, start: int, n_frames: int, bin_frames: int, box: tuple[slice, slice], bin_px: int = 1
) -> np.ndarray:
    """``n_frames`` frames from *start*, cropped to *box*, averaged in time and space bins.

    Raises:
        ValueError: If the clip runs past the movie's last frame.
    """
    if start < 0 or start + n_frames > len(movie):
        raise ValueError(
            f"A clip of {n_frames} frames from frame {start} does not fit the movie's "
            f"{len(movie)} frames."
        )
    n_out = n_frames // bin_frames
    clip = np.asarray(movie[start : start + n_out * bin_frames, box[0], box[1]], dtype=float)
    h, w = (clip.shape[1] // bin_px) * bin_px, (clip.shape[2] // bin_px) * bin_px
    clip = clip[:, :h, :w].reshape(n_out, bin_frames, h // bin_px, bin_px, w // bin_px, bin_px)
    return clip.mean(axis=(1, 3, 5))


def clip_dff(clip: np.ndarray) -> np.ndarray:
    """Each pixel of a ``(n_frames, h, w)`` clip over its mean over the clip, minus 1."""
    mean = clip.mean(axis=0, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(mean > 0, clip / mean - 1, np.nan)


def _flash_cell(movies, roi: int, roi_stat: dict, z, period_s, settings) -> dict:
    baseline = offset_frames(settings["baseline_s"], period_s)
    response = offset_frames(settings["response_s"], period_s)
    offsets = np.arange(baseline[0], response[-1] + 1)
    peaks = peak_frames(
        z["green"][roi],
        settings["n_peaks"],
        max(1, round(settings["min_peak_gap_s"] / period_s)),
        before_frames=-offsets[0],
        after_frames=offsets[-1],
        min_z=settings["min_peak_z"],
    )
    shape = movies["green"].shape[1:]
    box = crop_box(roi_stat["ypix"], roi_stat["xpix"], settings["crop_px"], shape)
    return {
        "roi": roi,
        "peaks": peaks,
        "offsets": offsets,
        "offsets_s": offsets * period_s,
        "box": box,
        "mask": roi_mask(roi_stat["ypix"], roi_stat["xpix"], box),
        "mean_image": np.asarray(
            movies["green"][:: max(1, len(movies["green"]) // 200), box[0], box[1]], dtype=float
        ).mean(axis=0),
        "image": {
            channel: smoothed(
                event_dff_image(movies[channel], peaks, baseline, response, box),
                settings["smooth_px"],
            )
            for channel in CHANNELS
        },
        "peri": {channel: peri_event(z[channel][roi], peaks, offsets) for channel in CHANNELS},
    }


def flash_check(
    movies: dict,
    stat,
    z: dict[str, np.ndarray],
    skew_green: np.ndarray,
    period_s: float,
    *,
    n_cells: int,
    n_peaks: int,
    min_peak_z: float,
    min_peak_gap_s: float,
    baseline_s: tuple[float, float],
    response_s: tuple[float, float],
    crop_px: int,
    smooth_px: float,
) -> list[dict]:
    """Peak images and peri-event traces of the *n_cells* most skewed green cells.

    Args:
        movies (dict): ``green`` and ``red`` registered movies, ``(n_frames, Ly, Lx)``.
        stat (Sequence[dict]): Suite2p ROI statistics (``ypix``, ``xpix``), one per row of *z*.
        z (dict[str, np.ndarray]): Each channel's ``(n_cells, n_frames)`` z, NaN on frames left out.
        skew_green (np.ndarray): Each cell's green skewness; the highest are checked.
        period_s (float): Frame period (s).
        n_cells (int): Cells to check.
        n_peaks (int): Largest separate green events per cell.
        min_peak_z (float): An event's green z must exceed this (e.g. the transient threshold),
            so a cell without transients gets no events rather than its largest noise.
        min_peak_gap_s (float): Smallest separation of two events (s).
        baseline_s (tuple[float, float]): Baseline range around each event (s).
        response_s (tuple[float, float]): Response range around each event (s).
        crop_px (int): Half-width of the crop around each cell.
        smooth_px (float): Gaussian sigma of the event images (0: none).

    Returns:
        list[dict]: Per cell, highest skew first: ``roi``, ``peaks`` (frames), ``offsets`` and
        ``offsets_s``, ``box`` (rows, cols), ``mask`` (ROI in the crop), ``mean_image`` (green,
        about 200 frames across the run), ``image`` (event ΔF/F0 image per channel) and ``peri``
        (``(n_peaks, n_offsets)`` z per channel).
    """
    settings = {
        "n_peaks": n_peaks,
        "min_peak_z": min_peak_z,
        "min_peak_gap_s": min_peak_gap_s,
        "baseline_s": baseline_s,
        "response_s": response_s,
        "crop_px": crop_px,
        "smooth_px": smooth_px,
    }
    return [
        _flash_cell(movies, int(roi), stat[roi], z, period_s, settings)
        for roi in most_skewed(skew_green, n_cells)
    ]
