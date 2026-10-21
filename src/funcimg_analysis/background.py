"""The background of a registered imaging field of view: every pixel that is not part of an ROI.

All the non-ROI pixels are treated as one large ROI whose F is the plain mean of its pixels in
each frame (what Suite2p computes for an ROI with uniform pixel weights), with no neuropil
subtraction. This module is pure array code; reading the binaries and caching live in
``background_io``.
"""

import numpy as np
from scipy.ndimage import binary_dilation


def _crop_slice(name: str, bounds: tuple[int, int], size: int) -> slice:
    start, stop = int(bounds[0]), int(bounds[1])
    if not 0 <= start < stop <= size:
        raise ValueError(f"{name} {tuple(bounds)} is not a non-empty range inside 0-{size}.")
    return slice(start, stop)


def background_mask(
    stat,
    shape: tuple[int, int],
    yrange: tuple[int, int],
    xrange: tuple[int, int],
    roi_margin_px: int = 0,
) -> np.ndarray:
    """True for pixels in no ROI (cell or not) and inside the registration's valid crop.

    A ROI's ``overlap`` flags are ignored on purpose: an overlapping pixel still belongs to the
    ROI, so it is not background. Pixels outside the crop hold no registered data.

    Args:
        stat (Sequence[dict]): Suite2p ROI statistics; each has ``ypix`` and ``xpix``.
        shape (tuple[int, int]): ``(Ly, Lx)`` of the field of view.
        yrange (tuple[int, int]): Valid rows ``[start, stop)`` after registration
            (``ops["yrange"]``).
        xrange (tuple[int, int]): Valid columns ``[start, stop)`` (``ops["xrange"]``).
        roi_margin_px (int): Also exclude this many pixels around every ROI (a square
            neighbourhood). Default 0, meaning only the ROIs' own pixels.

    Returns:
        np.ndarray: Boolean ``(Ly, Lx)`` mask.

    Raises:
        ValueError: If the margin is negative, a crop range is empty or outside the image, a ROI
            pixel lies outside the image, or no background pixel is left.
    """
    if roi_margin_px < 0:
        raise ValueError(f"roi_margin_px must be 0 or more, not {roi_margin_px}.")
    rows = _crop_slice("yrange", yrange, shape[0])
    cols = _crop_slice("xrange", xrange, shape[1])
    in_roi = np.zeros(shape, dtype=bool)
    for roi in stat:
        ypix, xpix = np.asarray(roi["ypix"], dtype=int), np.asarray(roi["xpix"], dtype=int)
        if (
            ypix.min(initial=0) < 0
            or xpix.min(initial=0) < 0
            or ypix.max(initial=0) >= shape[0]
            or xpix.max(initial=0) >= shape[1]
        ):
            raise ValueError(f"A ROI has pixels outside the {shape[0]} x {shape[1]} image.")
        in_roi[ypix, xpix] = True
    if roi_margin_px:
        in_roi = binary_dilation(in_roi, structure=np.ones((3, 3), bool), iterations=roi_margin_px)
    valid = np.zeros(shape, dtype=bool)
    valid[rows, cols] = True
    mask = valid & ~in_roi
    if not mask.any():
        raise ValueError("There are no background pixels: ROIs and margin cover the valid area.")
    return mask


def chunked_mask_mean(movie, mask: np.ndarray, chunk_frames: int) -> np.ndarray:
    """Mean over the masked pixels in each frame, reading *chunk_frames* frames at a time.

    Chunking keeps memory at one chunk even for a multi-GB memory-mapped movie; the sum is in
    float64, so the result does not depend on the chunk size.

    Args:
        movie (np.ndarray | np.memmap): ``(n_frames, Ly, Lx)`` integer movie.
        mask (np.ndarray): Boolean ``(Ly, Lx)``.
        chunk_frames (int): Frames per chunk, at least 1.

    Returns:
        np.ndarray: ``(n_frames,)`` float64 mean F.

    Raises:
        ValueError: If *chunk_frames* is below 1 or the mask does not fit the movie.
    """
    if chunk_frames < 1:
        raise ValueError(f"chunk_frames must be at least 1, not {chunk_frames}.")
    if mask.shape != movie.shape[1:]:
        raise ValueError(f"mask shape {mask.shape} does not match the movie's {movie.shape[1:]}.")
    n_frames = movie.shape[0]
    out = np.empty(n_frames, dtype=np.float64)
    for start in range(0, n_frames, chunk_frames):
        chunk = np.asarray(movie[start : start + chunk_frames])
        out[start : start + len(chunk)] = chunk[:, mask].mean(axis=1, dtype=np.float64)
    return out
