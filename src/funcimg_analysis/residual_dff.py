"""ΔF/F0 of cells whose fluctuation shared with the field-wide background is removed (pure).

Per cell, ``F_corr = F - b * (background - mean(background))`` with ``b`` the OLS slope of F on
the background. Only the background's *fluctuation* is subtracted, so the cell keeps its own
mean brightness and F0 stays on the scale of the cell. ΔF/F0 is then taken of ``F_corr`` with the
baseline method and settings the cells were extracted with. This replaces Suite2p's neuropil
subtraction (a local ring, a fixed coefficient) by a global, per-cell scaled correction; the two
are never applied together, so the neuropil is not subtracted here.
"""

import numpy as np

from src.funcimg_analysis.event_frames import quiet_empty_means
from src.funcimg_analysis.shared_signal import fit_background, row_correlation

MIN_F0_FRACTION = 0.1  # F0 below this fraction of the cell's median F makes ΔF/F0 unreliable


def subtract_background_fluctuation(
    f: np.ndarray, background: np.ndarray, fit_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Each cell minus its fitted multiple of the background's deviation from its mean.

    Args:
        f (np.ndarray): ``(n_cells, n_frames)`` raw F.
        background (np.ndarray): ``(n_frames,)`` background F of the same channel.
        fit_mask (np.ndarray): Boolean ``(n_frames,)``; True for the frames the slope is fitted
            on (outside the outcome windows). The slope is applied to every frame.

    Returns:
        tuple[np.ndarray, np.ndarray]: The corrected traces and the slope ``b`` per cell (NaN,
        with a NaN trace, for a cell without variance over the fit frames).

    Raises:
        ValueError: If the background is constant over the fit frames, so no slope exists.
    """
    f, background = np.asarray(f, dtype=float), np.asarray(background, dtype=float)
    slope = fit_background(f, background, fit_mask)["slope"]
    if np.isnan(slope).all():
        raise ValueError(
            "the background is constant (or has fewer than 3 finite frames) over the frames used "
            "for the fit, so no slope can be fitted."
        )
    centred = background - np.nanmean(background)
    return f - slope[:, None] * centred[None, :], slope


def flag_cells(
    f: np.ndarray, f0: np.ndarray, slope: np.ndarray, min_f0_fraction: float
) -> dict[int, str]:
    """Why each unusable cell is excluded, ``{cell: reason}``; usable cells are absent.

    A cell is unusable if it has no slope, or if its F0 is not positive or falls below
    *min_f0_fraction* of its median F in any frame (a ratio to a near-zero baseline explodes).
    """
    with quiet_empty_means():
        median_f = np.nanmedian(np.asarray(f, dtype=float), axis=1)
    reasons = {}
    for cell in range(len(f)):
        found = []
        if not np.isfinite(slope[cell]):
            found.append("no slope (the cell's F has no variance)")
        else:
            non_positive = int(np.sum(f0[cell] <= 0))
            small = int(np.sum(f0[cell] < min_f0_fraction * median_f[cell]))
            if non_positive:
                found.append(f"F0 <= 0 in {non_positive} frames")
            if small:
                found.append(f"F0 below {min_f0_fraction:g} x median F in {small} frames")
        if found:
            reasons[cell] = "; ".join(found)
    return reasons


def residual_dff_channel(
    f: np.ndarray,
    background: np.ndarray,
    fit_mask: np.ndarray,
    fs_hz: float,
    settings,
    min_f0_fraction: float = MIN_F0_FRACTION,
) -> dict:
    """Background-corrected ΔF/F0 of one channel's cells, with the unusable cells excluded.

    Args:
        f (np.ndarray): ``(n_cells, n_frames)`` raw F.
        background (np.ndarray): ``(n_frames,)`` background F.
        fit_mask (np.ndarray): True for the frames the slope is fitted on.
        fs_hz (float): Imaging rate (Hz), for the baseline.
        settings (mini2p_preprocessing.config.FluorescenceSettings): Baseline method and
            parameters as recorded at extraction (the neuropil coefficient is not used).
        min_f0_fraction (float): Cells whose F0 is not positive or is below this fraction of
            their median F in any frame are excluded.

    Returns:
        dict: ``dff`` (``(n_cells, n_frames)``, all NaN for an excluded cell), ``slope`` and
        ``r_after`` (r of the corrected trace with the background over all frames, near 0 when
        the correction worked) per cell, ``kept`` (boolean per cell) and ``excluded``
        (``{cell: reason}``).

    Raises:
        ValueError: If the background is constant over the fit frames.
    """
    from mini2p_preprocessing.fluorescence import dff_from_corrected

    corrected, slope = subtract_background_fluctuation(f, background, fit_mask)
    fitted = np.isfinite(slope)
    dff = np.full(corrected.shape, np.nan)
    f0 = np.full(corrected.shape, np.nan)
    if fitted.any():
        result = dff_from_corrected(corrected[fitted], fs_hz, settings)
        dff[fitted], f0[fitted] = result.dff, result.f0
    excluded = flag_cells(f, f0, slope, min_f0_fraction)
    kept = np.array([cell not in excluded for cell in range(len(f))], dtype=bool)
    dff[~kept] = np.nan
    return {
        "dff": dff,
        "slope": slope,
        "r_after": row_correlation(corrected, np.asarray(background, dtype=float)),
        "kept": kept,
        "excluded": excluded,
    }
