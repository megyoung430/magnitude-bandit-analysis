"""Read a run's registered binaries into background F traces, cached on disk.

The two ~GB binaries (green ``data.bin``, red ``data_chan2.bin``) are read once in chunks; the
traces are kept in a disposable cache keyed on the plane's path, the files' sizes and
modification times, the margin and the code, so deleting the cache only costs time.
"""

import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np

from src.funcimg_analysis.background import background_mask, chunked_mask_mean
from src.funcimg_analysis.frame_times import RunMismatchError

CHUNK_FRAMES = 200
CACHE_SUBDIR = "funcimg_background"
# channel: (ops key of the binary's path, default file name, ops key of the mean image)
CHANNEL_FILES = {
    "green": ("reg_file", "data.bin", "meanImg"),
    "red": ("reg_file_chan2", "data_chan2.bin", "meanImg_chan2"),
}
# The source hashed into the cache key, so editing the computation invalidates old caches.
CODE_VERSION = hashlib.sha256(
    b"".join(
        (Path(__file__).parent / name).read_bytes()
        for name in ("background.py", "background_io.py")
    )
).hexdigest()


class BackgroundUnavailableError(Exception):
    """A plane lacks what the background needs (an ``ops`` entry); the message says what."""


def _binary_path(ops: dict, plane_dir: Path, channel: str) -> Path:
    from mini2p_preprocessing.registration import resolve_registered_binary

    key, filename, _ = CHANNEL_FILES[channel]
    return resolve_registered_binary(ops, plane_dir, key, filename)


def _ops_value(ops: dict, key: str, plane_dir: Path):
    if key not in ops:
        raise BackgroundUnavailableError(f"{plane_dir / 'ops.npy'} has no {key!r}.")
    return ops[key]


def _mean_images(ops: dict, plane_dir: Path, shape: tuple[int, int]) -> dict[str, np.ndarray]:
    images = {}
    for channel, (_, _, key) in CHANNEL_FILES.items():
        image = np.asarray(_ops_value(ops, key, plane_dir), dtype=float)
        if image.shape != shape:
            raise BackgroundUnavailableError(
                f"{plane_dir / 'ops.npy'}: {key} has shape {image.shape}, not the image's {shape}."
            )
        images[channel] = image
    return images


def _open_movie(path: Path, n_frames: int, shape: tuple[int, int]) -> np.memmap:
    expected = n_frames * shape[0] * shape[1] * np.dtype(np.int16).itemsize
    if path.stat().st_size != expected:
        raise RunMismatchError(
            f"{path.name} is {path.stat().st_size} bytes but ops say {n_frames} frames of "
            f"{shape[0]} x {shape[1]} int16 ({expected} bytes)."
        )
    return np.memmap(path, dtype=np.int16, mode="r", shape=(n_frames, *shape))


def _cache_path(cache_dir: Path, plane_dir: Path, files: list[Path], margin: int) -> Path:
    parts = [str(plane_dir.resolve()), str(margin), CODE_VERSION]
    for path in files:
        info = path.stat()
        parts += [str(path.resolve()), str(info.st_size), str(info.st_mtime_ns)]
    key = hashlib.sha256(json.dumps(parts).encode()).hexdigest()[:24]
    return cache_dir / f"{key}.npz"


def _cached(cache_file: Path, compute: Callable[[], dict]) -> dict:
    """The arrays saved in *cache_file*, or ``compute()`` saved there first on a miss."""
    if cache_file.is_file():
        with np.load(cache_file) as cached:
            return {name: cached[name] for name in cached.files}
    arrays = compute()
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_file.with_suffix(f".{os.getpid()}.tmp.npz")  # only complete files are read
    np.savez(tmp, **arrays)
    tmp.replace(cache_file)
    return arrays


def _read_traces(
    paths: dict[str, Path], n_frames: int, shape: tuple[int, int], mask: np.ndarray, chunk: int
) -> dict[str, np.ndarray]:
    # both binaries are opened (and their sizes checked) before either is read
    movies = {channel: _open_movie(path, n_frames, shape) for channel, path in paths.items()}
    return {channel: chunked_mask_mean(movie, mask, chunk) for channel, movie in movies.items()}


def load_background(
    plane_dir, roi_margin_px: int = 0, cache_dir=None, chunk_frames: int = CHUNK_FRAMES
) -> dict:
    """The background F of both channels of a plane, and what a sanity figure needs.

    Args:
        plane_dir (str | Path): ``<run>/<variant>/suite2p/plane0``.
        roi_margin_px (int): Pixels to exclude around every ROI (see ``background_mask``).
        cache_dir (str | Path | None): Where traces are cached. Default None, meaning
            ``<settings cache_dir>/funcimg_background`` (git-ignored). Never beside the data.
        chunk_frames (int): Frames read at a time.

    Returns:
        dict: ``F`` (``{("green", "F"): (n_frames,), ("red", "F"): ...}``), ``mask``,
        ``n_pixels``, ``labels`` (ROI label image) and ``mean_images`` (``green``, ``red``).

    Raises:
        BackgroundUnavailableError: If ``ops`` lacks ``yrange``, ``xrange`` or a mean image, or
            a mean image is not the image's shape.
        FileNotFoundError: If a registered binary is missing.
        RunMismatchError: If a binary's size does not match ``ops``.
    """
    from mini2p_preprocessing.plane import labels_from_stat

    plane_dir = Path(plane_dir)
    ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
    stat = np.load(plane_dir / "stat.npy", allow_pickle=True)
    shape = (int(ops["Ly"]), int(ops["Lx"]))
    yrange, xrange = (_ops_value(ops, key, plane_dir) for key in ("yrange", "xrange"))
    mask = background_mask(stat, shape, yrange, xrange, roi_margin_px)
    mean_images = _mean_images(ops, plane_dir, shape)
    paths = {channel: _binary_path(ops, plane_dir, channel) for channel in CHANNEL_FILES}
    if cache_dir is None:
        from src.settings import get_settings

        cache_dir = get_settings().cache_dir / CACHE_SUBDIR
    cache_file = _cache_path(
        Path(cache_dir),
        plane_dir,
        [*paths.values(), plane_dir / "stat.npy", plane_dir / "ops.npy"],
        roi_margin_px,
    )
    traces = _cached(
        cache_file,
        lambda: _read_traces(paths, int(ops["nframes"]), shape, mask, chunk_frames),
    )
    return {
        "F": {(channel, "F"): traces[channel] for channel in CHANNEL_FILES},
        "mask": mask,
        "n_pixels": int(mask.sum()),
        "labels": labels_from_stat(stat, *shape),
        "mean_images": mean_images,
    }
