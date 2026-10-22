"""Read the registered movies of a run for the flash check and the clip, with results cached.

The binaries are memory-mapped read-only and only the frames the check needs are read. Results
are cached under the repo's git-ignored cache, keyed on the binaries' paths, sizes and
modification times, the z traces, the settings and this code, so a rerun does not read the
movie over the network again and a changed input never finds a stale result.
"""

import hashlib
from pathlib import Path

import numpy as np

from src.funcimg_analysis.collect_motion import load_ops
from src.funcimg_analysis.flash_check import binned_clip, flash_check
from src.funcimg_analysis.shared_cache import cached_result, content_key

CACHE_SUBDIR = "funcimg_flash"
FLASH_CODE = hashlib.sha256(
    b"".join(
        (Path(__file__).parent / name).read_bytes()
        for name in ("flash_check.py", "collect_flash.py")
    )
).hexdigest()


def open_movies(plane_dir) -> tuple[dict, list[Path]]:
    """Memory maps of the green and red registered movies, and the binaries' paths.

    Raises:
        FileNotFoundError: If a registered binary is missing.
    """
    from mini2p_preprocessing.registration import load_registered_outputs

    outputs = load_registered_outputs(Path(plane_dir))
    movies = {"green": outputs.functional, "red": outputs.structural}
    return movies, [Path(movie.filename) for movie in movies.values()]


def _file_identity(paths: list[Path]) -> list:
    return [(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in paths]


def _cache_dir(cache_dir):
    if cache_dir is not None:
        return Path(cache_dir)
    from src.settings import get_settings

    return get_settings().cache_dir / CACHE_SUBDIR


def collect_flash_check(motion_run: dict, cache_dir=None, **settings) -> list[dict]:
    """:func:`flash_check` of one section-4 run, reading its plane's movies and ``stat.npy``.

    Args:
        motion_run (dict): A ``collect_motion_check`` result (``plane_dir``, ``period_s``,
            ``z_kept``, ``stats``).
        cache_dir (str | Path | None): Default None, ``<settings cache_dir>/funcimg_flash``.
        **settings: Passed on to :func:`flash_check`.
    """
    plane_dir = Path(motion_run["plane_dir"])
    movies, paths = open_movies(plane_dir)
    stat = np.load(plane_dir / "stat.npy", allow_pickle=True)
    skew = motion_run["stats"]["kept frames"]["green"]["skew"]
    key = content_key(
        {
            "kind": "flash_check",
            "flash_code": FLASH_CODE,
            "files": _file_identity([*paths, plane_dir / "stat.npy"]),
            "z": motion_run["z_kept"],
            "skew": skew,
            "period_s": motion_run["period_s"],
            "settings": settings,
        }
    )
    return cached_result(
        _cache_dir(cache_dir),
        key,
        lambda: {
            "cells": flash_check(
                movies, stat, motion_run["z_kept"], skew, motion_run["period_s"], **settings
            )
        },
    )["cells"]


def valid_box(ops: dict) -> tuple[slice, slice]:
    """The registration's valid crop (``yrange``, ``xrange``) as rows and columns."""
    return slice(*map(int, ops["yrange"])), slice(*map(int, ops["xrange"]))


def collect_clip(
    plane_dir, start_frame: int, n_frames: int, bin_frames: int, bin_px: int, cache_dir=None
) -> dict:
    """Green and red :func:`binned_clip` of the valid crop, and where the crop starts.

    Returns:
        dict: ``green`` and ``red`` ``(n_out, h, w)`` clips and ``origin`` (row, column of the
        crop's first pixel, before spatial binning).
    """
    plane_dir = Path(plane_dir)
    movies, paths = open_movies(plane_dir)
    box = valid_box(load_ops(plane_dir))
    key = content_key(
        {
            "kind": "flash_clip",
            "flash_code": FLASH_CODE,
            "files": _file_identity(paths),
            "clip": (
                start_frame,
                n_frames,
                bin_frames,
                bin_px,
                box[0].start,
                box[0].stop,
                box[1].start,
                box[1].stop,
            ),
        }
    )

    def compute():
        clips = {
            channel: binned_clip(movie, start_frame, n_frames, bin_frames, box, bin_px)
            for channel, movie in movies.items()
        }
        return {**clips, "origin": (box[0].start, box[1].start)}

    return cached_result(_cache_dir(cache_dir), key, compute)
