"""Motion-correction and z-drift quality control after Zong et al. 2022 (pure).

The paper uses Suite2p's "Metrics for registration": spatial principal components (SPCs) of the
registered movie, for each the mean of the frames with the highest scores (Itop) and the lowest
(Ibottom), the residual xy shift between the two (what motion correction left behind), and
Ibottom - Itop inspected by eye for z drift. The computations here are Suite2p's own
(``pclowhigh`` and ``pc_register``); only the number of frames averaged differs from Suite2p's
GUI (500 as in the paper instead of 300).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

BLOCK_CHUNK_FRAMES = 4000  # frames per chunk of the pairwise block-distance computation


@dataclass(frozen=True)
class MotionSettings:
    """Registration-metrics parameters; the defaults are the paper's.

    Attributes:
        n_pcs (int): Spatial principal components. Paper: 30.
        n_extremes (int): Frames averaged for each of Itop and Ibottom. Paper: 500.
        max_frames (int): Frames sampled evenly through the recording for the PCA. Paper: 5000,
            but the PCA peaks near 4 GB at 256 x 256 and 16 GB at 512 x 512 for 5000 frames, so
            the default is 2000 (about 40 % of that); raise it on a large machine.
    """

    n_pcs: int = 30
    n_extremes: int = 500
    max_frames: int = 2000

    def __post_init__(self):
        if min(self.n_pcs, self.n_extremes, self.max_frames) < 1:
            raise ValueError(f"Invalid MotionSettings: {self}.")


def sample_frame_indices(n_frames: int, max_frames: int) -> np.ndarray:
    """Frames equally spaced through the recording, at most *max_frames* (all if fewer)."""
    if n_frames <= max_frames:
        return np.arange(n_frames)
    return np.linspace(0, n_frames - 1, max_frames).astype(int)


def registration_settings(ops: dict) -> dict:
    """Suite2p's registration settings, with those recorded in ``ops["registration"]`` applied.

    Suite2p's defaults are used for anything ``ops`` does not record; the metrics only use the
    smoothing, block, shift-limit and SNR settings, so the defaults are a deliberate fallback.
    """
    import suite2p

    settings = dict(suite2p.default_settings()["registration"])
    recorded = ops.get("registration")
    if isinstance(recorded, dict):
        settings.update({k: v for k, v in recorded.items() if k in settings})
    return settings


def pc_images(movie: np.ndarray, n_pcs: int, n_extremes: int, random_state: int | None = 0) -> dict:
    """Spatial PCs of a registered movie and their top and bottom images.

    Args:
        movie (np.ndarray): ``(n_frames, Ly, Lx)`` sampled registered frames.
        n_pcs (int): Number of spatial principal components (paper: 30).
        n_extremes (int): Frames averaged for each of Itop and Ibottom (paper: 500); reduced to
            half the frames when the movie is shorter than twice this.
        random_state (int | None): Seed of the PCA.

    Returns:
        dict: ``i_top`` and ``i_bottom`` (``(n_pcs, Ly, Lx)``: means of the frames with the
        highest and lowest score on each PC), ``pc_time`` (``(n_frames, n_pcs)`` score of every
        frame), and ``n_extremes`` (the number of frames actually averaged).

    Raises:
        ValueError: If there are fewer frames than components.
    """
    from suite2p.registration.metrics import pclowhigh

    n_frames = movie.shape[0]
    if n_pcs > n_frames:
        raise ValueError(f"{n_pcs} components requested from only {n_frames} frames.")
    used = min(n_extremes, n_frames // 2)
    bottom, top, _, pc_time = pclowhigh(movie, nlowhigh=used, nPC=n_pcs, random_state=random_state)
    return {"i_top": top, "i_bottom": bottom, "pc_time": pc_time, "n_extremes": used}


def residual_drift(i_top: np.ndarray, i_bottom: np.ndarray, settings: dict) -> pd.DataFrame:
    """Residual xy drift per PC: Itop registered to Ibottom with Suite2p's rigid + non-rigid step.

    The paper registers both images to the stack mean; Suite2p's own metric registers Itop to
    Ibottom, which measures the same thing (a shift between the extreme frames of a PC that
    motion correction should have removed) and is what its GUI reports.

    Returns:
        pd.DataFrame: One row per PC (``pc`` from 1) with ``rigid_px`` (the whole-image shift),
        ``mean_nonrigid_px`` and ``max_nonrigid_px`` (mean and maximum over blocks of the
        block-shift vector length) and ``mean_total_px`` (rigid plus non-rigid, mean over blocks).
    """
    import torch
    from suite2p.registration.metrics import pc_register

    shifts = pc_register(
        torch.from_numpy(np.ascontiguousarray(i_bottom)).float(),
        torch.from_numpy(np.ascontiguousarray(i_top)).float(),
        smooth_sigma=settings["smooth_sigma"],
        block_size=settings["block_size"],
        maxregshift=settings["maxregshift"],
        maxregshiftNR=settings["maxregshiftNR"],
        snr_thresh=settings["snr_thresh"],
        spatial_taper=settings["spatial_taper"],
    )
    return pd.DataFrame(
        {
            "pc": np.arange(1, len(shifts) + 1),
            "rigid_px": shifts[:, 0],
            "mean_nonrigid_px": shifts[:, 1],
            "max_nonrigid_px": shifts[:, 2],
            "mean_total_px": shifts[:, 3],
        }
    )


def registration_metrics(
    movie: np.ndarray, n_pcs: int, n_extremes: int, settings: dict | None = None
) -> dict:
    """PC images (see :func:`pc_images`) and the residual drift of each PC (``drift``)."""
    pcs = pc_images(movie, n_pcs, n_extremes)
    settings = registration_settings({}) if settings is None else settings
    return {**pcs, "drift": residual_drift(pcs["i_top"], pcs["i_bottom"], settings)}


def nonrigid_motion(xoff1: np.ndarray, yoff1: np.ndarray) -> np.ndarray:
    """Per frame, the largest distance between the shifts of any two registration blocks.

    This is the paper's non-rigid motion: the maximum over block pairs of
    ``sqrt((Mx_ij - Mx_i'j')^2 + (My_ij - My_i'j')^2)``.

    Args:
        xoff1 (np.ndarray): ``(n_frames, n_blocks)`` x shift of each block (Suite2p ``xoff1``).
        yoff1 (np.ndarray): Same shape, y shifts.

    Raises:
        ValueError: If the shapes differ.
    """
    xoff1, yoff1 = np.asarray(xoff1, dtype=float), np.asarray(yoff1, dtype=float)
    if xoff1.shape != yoff1.shape or xoff1.ndim != 2:
        raise ValueError(
            f"xoff1 and yoff1 must share a 2-D shape, got {xoff1.shape}, {yoff1.shape}."
        )
    out = np.empty(len(xoff1))
    for start in range(0, len(xoff1), BLOCK_CHUNK_FRAMES):
        x = xoff1[start : start + BLOCK_CHUNK_FRAMES]
        y = yoff1[start : start + BLOCK_CHUNK_FRAMES]
        distance = np.hypot(x[:, :, None] - x[:, None, :], y[:, :, None] - y[:, None, :])
        out[start : start + BLOCK_CHUNK_FRAMES] = distance.max(axis=(1, 2))
    return out
