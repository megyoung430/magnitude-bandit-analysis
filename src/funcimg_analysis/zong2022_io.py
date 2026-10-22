"""Reading and writing for the Zong et al. 2022 pipeline: masks, extraction, frames, outputs.

Everything computed here is a function of files that already exist (the registered binaries,
``ops.npy`` and the Cellpose segmentation). Nothing is written to a Suite2p plane folder:
extracted traces go to a disposable cache in the repo and, only on request, to a ``zong2022``
folder beside ``suite2p``.
"""

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from src.funcimg_analysis.shared_cache import cached_result

CACHE_ROOT = "zong2022"
DERIVATIVES_FOLDER = "zong2022"
OFFSET_KEYS = ("xoff", "yoff", "xoff1", "yoff1")
_CODE_FILES = ("zong2022_io.py", "zong2022_motion.py", "zong2022_signal.py", "zong2022_run.py")
CODE_VERSION = hashlib.sha256(
    b"".join(
        (Path(__file__).parent / name).read_bytes()
        for name in _CODE_FILES
        if (Path(__file__).parent / name).is_file()
    )
).hexdigest()


# ---- Cellpose masks ----------------------------------------------------------------------------


def seg_paths(run_dir) -> tuple[Path, Path]:
    """The Cellpose segmentation of a run and the image it was drawn on.

    The batch extraction (``mini2p_preprocessing.cellpose_batch``) reads ``run-XX/<seg>`` and
    ``run-XX/<image>``, drawn on the run's own registration; the locations are the batch
    settings' defaults, so this finds exactly what it finds.
    """
    from mini2p_preprocessing.batch import BatchSettings

    fields = BatchSettings.model_fields
    run_dir = Path(run_dir)
    return run_dir / fields["cellpose_seg"].default, run_dir / fields["cellpose_image"].default


def alignment_settings(ops: dict, diameter_px: int):
    """The mini2p settings object ``align_to_plane`` reads (fluorescence and ROI diameter)."""
    from mini2p_preprocessing.config import (
        DistortionSettings,
        PipelineSettings,
        RegistrationSettings,
    )

    # only the fluorescence (mask alignment limits) and registration (diameter) parts are used;
    # the distortion mode is required by the model but plays no part here.
    return PipelineSettings(
        distortion=DistortionSettings(mode="off"),
        registration=RegistrationSettings(fs_hz=float(ops["fs"]), diameter_px=diameter_px),
    )


def aligned_labels(seg_file, mask_image, ops: dict, diameter_px: int) -> tuple[np.ndarray, dict]:
    """Cellpose labels shifted onto this plane's own mean image.

    Args:
        seg_file: Cellpose ``*_seg.npy``.
        mask_image: The image the masks were drawn on (grey PNG or TIFF).
        ops (dict): The plane's ``ops.npy`` contents.
        diameter_px (int): ROI diameter used for the background filter of the alignment.

    Returns:
        tuple[np.ndarray, dict]: The label image (0 = background) and mini2p's alignment report
        (``dy``, ``dx``, rank correlations, dropped labels).

    Raises:
        ValueError: If the masks do not have the plane's shape, or mini2p refuses the alignment.
    """
    from mini2p_preprocessing.extraction import align_to_plane, load_cellpose_labels

    labels = load_cellpose_labels(Path(seg_file))
    shape = (int(ops["Ly"]), int(ops["Lx"]))
    if labels.shape != shape:
        raise ValueError(
            f"Mask shape {labels.shape} differs from the registered movie shape {shape}; "
            "segment the mean image of this plane."
        )
    return align_to_plane(labels, Path(mask_image), ops, alignment_settings(ops, diameter_px))


# ---- extraction --------------------------------------------------------------------------------


def stat_from_labels(labels: np.ndarray, ops: dict, diameter_px: int) -> np.ndarray:
    """Suite2p ``stat`` for every Cellpose ROI (mini2p's ``build_stat``), none removed."""
    from mini2p_preprocessing.extraction import build_stat

    return build_stat(labels, int(ops["Ly"]), int(ops["Lx"]), diameter_px)


def _check_binary_length(path: Path, ops: dict) -> None:
    expected = int(ops["nframes"]) * int(ops["Ly"]) * int(ops["Lx"]) * 2  # int16
    if path.stat().st_size != expected:
        raise ValueError(
            f"{path} holds {path.stat().st_size / (int(ops['Ly']) * int(ops['Lx']) * 2):.2f} "
            f"frames of {ops['Ly']}x{ops['Lx']} but ops.npy says nframes={ops['nframes']}."
        )


def extract_channels(
    stat: np.ndarray, ops: dict, plane_dir, batch_size: int = 500, device: str = "cpu"
) -> dict:
    """Fcell and Fnp of every ROI in both channels with Suite2p's extraction.

    Cell masks are Suite2p's ``create_masks`` defaults: pixel weights ``lam`` normalised to sum
    to one (a weighted average) with the pixels flagged as overlapping another ROI removed; the
    neuropil of an ROI is the surrounding ring minus every ROI's pixels. Nothing is written.

    Returns:
        dict: ``f``, ``fneu`` (green) and ``f_chan2``, ``fneu_chan2`` (red; None for a
        single-channel plane), each ``(n_rois, n_frames)``.

    Raises:
        ValueError: If a registered binary's length does not match ``ops["nframes"]``.
    """
    import torch
    from mini2p_preprocessing.extraction import registered_binaries
    from suite2p.extraction.extract import extraction_wrapper
    from suite2p.extraction.masks import create_masks
    from suite2p.io import BinaryFile

    ly, lx = int(ops["Ly"]), int(ops["Lx"])
    green, red = registered_binaries(ops, Path(plane_dir))
    for binary in (green, red):
        if binary is not None:
            _check_binary_length(binary, ops)
    cell_masks, neuropil_masks = create_masks(stat, Ly=ly, Lx=lx)
    f, fneu, f_chan2, fneu_chan2 = extraction_wrapper(
        stat,
        BinaryFile(ly, lx, green),
        f_reg_chan2=BinaryFile(ly, lx, red) if red else None,
        cell_masks=cell_masks,
        neuropil_masks=neuropil_masks,
        settings={"batch_size": batch_size},
        device=torch.device(device),
    )
    return {"f": f, "fneu": fneu, "f_chan2": f_chan2, "fneu_chan2": fneu_chan2}


# ---- caching -----------------------------------------------------------------------------------


def file_signature(path) -> dict:
    """Size and modification time of a file: cheap, and changes when the file is rewritten."""
    stat = Path(path).stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def cached(cache_dir, kind: str, parts: dict, compute: Callable[[], dict]) -> dict:
    """``compute()`` saved under ``<cache_dir>/<kind>/``, reused while *parts* and the code match.

    The key is a hash of *parts* (anything JSON-serialisable or printable: file signatures,
    settings, an array's digest) and of this pipeline's source, so an edited setting, input file
    or piece of code never finds a stale result. Deleting the cache only costs time.
    """
    text = json.dumps({"code": CODE_VERSION, **parts}, sort_keys=True, default=str)
    key = hashlib.sha256(text.encode()).hexdigest()[:32]
    return cached_result(Path(cache_dir) / kind, key, compute)


def array_digest(array: np.ndarray) -> str:
    """A short hash of an array's dtype, shape and content, for use in a cache key."""
    array = np.ascontiguousarray(array)
    return hashlib.sha256(f"{array.dtype}{array.shape}".encode() + array.tobytes()).hexdigest()[:16]


def extract_cached(
    labels: np.ndarray, plane_dir, diameter_px: int, cache_dir, batch_size: int = 500
) -> dict:
    """:func:`extract_channels` for *labels*, cached on the labels and the binaries' signatures."""
    from mini2p_preprocessing.extraction import registered_binaries

    plane_dir = Path(plane_dir)
    ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
    green, red = registered_binaries(ops, plane_dir)
    parts = {
        "labels": array_digest(labels),
        "diameter_px": diameter_px,
        "ops": file_signature(plane_dir / "ops.npy"),
        "green": file_signature(green),
        "red": file_signature(red) if red else None,
    }

    def compute():
        stat = stat_from_labels(labels, ops, diameter_px)
        return extract_channels(stat, ops, plane_dir, batch_size)

    return cached(cache_dir, "extraction", parts, compute)


# ---- frames and registration offsets -----------------------------------------------------------


def read_frames(
    path,
    shape: tuple[int, int, int],
    indices: np.ndarray,
    crop: tuple[int, int, int, int],
    chunk_frames: int = 200,
) -> np.ndarray:
    """Chosen frames of an int16 binary movie, cropped, read a chunk at a time.

    Args:
        path: The registered ``data.bin`` (or ``data_chan2.bin``).
        shape (tuple[int, int, int]): ``(n_frames, Ly, Lx)``.
        indices (np.ndarray): Increasing frame numbers to read.
        crop (tuple[int, int, int, int]): ``(y0, y1, x0, x1)`` pixels to keep.
        chunk_frames (int): Frames read per memory-mapped slice, so a multi-GB movie is never
            loaded whole.

    Returns:
        np.ndarray: ``(len(indices), y1 - y0, x1 - x0)`` float32.

    Raises:
        ValueError: If the file's size does not match *shape*.
    """
    path = Path(path)
    if path.stat().st_size != int(np.prod(shape)) * 2:
        raise ValueError(f"{path} has size {path.stat().st_size} bytes, not {shape} int16 frames.")
    movie = np.memmap(path, dtype=np.int16, mode="r", shape=shape)
    y0, y1, x0, x1 = crop
    out = np.empty((len(indices), y1 - y0, x1 - x0), dtype=np.float32)
    for start in range(0, len(indices), chunk_frames):
        chosen = np.asarray(indices[start : start + chunk_frames])
        out[start : start + len(chosen)] = movie[chosen, y0:y1, x0:x1]
    return out


def registration_offsets(ops: dict) -> dict[str, np.ndarray]:
    """The rigid (``xoff``, ``yoff``) and non-rigid (``xoff1``, ``yoff1``) shifts ops records."""
    return {key: np.asarray(ops[key]) for key in OFFSET_KEYS if key in ops}


# ---- outputs -----------------------------------------------------------------------------------


def write_derivatives(variant_dir, arrays: dict, settings: dict) -> Path:
    """Save *arrays* and *settings* in ``<variant_dir>/zong2022/``, never inside ``suite2p/``.

    Raises:
        ValueError: If *variant_dir* is or lies inside a ``suite2p`` folder.
    """
    variant_dir = Path(variant_dir)
    if "suite2p" in variant_dir.parts:
        raise ValueError(f"Refusing to write into a suite2p folder: {variant_dir}.")
    out = variant_dir / DERIVATIVES_FOLDER
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "traces.npz", **arrays)
    (out / "settings.json").write_text(json.dumps(settings, indent=2, default=str))
    return out


def write_table(table: pd.DataFrame, path) -> Path:
    """Save a per-cell table as CSV, creating the folder."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return path
