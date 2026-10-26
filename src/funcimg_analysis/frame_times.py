"""pyControl time of every imaging frame, saved beside each run as ``frame_times.npz``.

The file is written once per run (``derivatives/.../funcimg/run-XX/frame_times.npz``) and serves
both destriping variants, which share their frames. Frame ``i`` of the registered movie (and of
``F.npy``) has pyControl time ``time_s[i]`` (s; NaN outside the sync pulses), so analyses can
work in frames or in seconds.

The run's behaviour video gets the same treatment in ``video_frame_times.npz``: the pyControl
time of each of its frames, so imaging and video frames can be paired by time. The camera
records a pulse at the first frame it takes after the pulse starts, so a video frame's time
is on average half a frame period early (8 ms at 60 Hz), and up to one period.
"""

import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.behavior_import.import_data import read_tsv_file
from src.funcimg_analysis.event_frames import frame_period_s, limited_frame_times
from src.funcimg_import.daq_sync import load_daq_sync
from src.funcimg_import.runs import (
    RunFiles,
    VideoFiles,
    camera_frame_period_ms,
    video_match_accepted,
)
from src.sync.pycontrol_time import (
    align_to_pycontrol,
    pycontrol_sync_times,
    sync_quality,
    to_pycontrol_time,
    video_frame_times,
)
from src.sync.rsync import RsyncError

FRAME_TIMES_FILE = "frame_times.npz"
VIDEO_FRAME_TIMES_FILE = "video_frame_times.npz"
# The batch preprocessing's record of which raw TIFF each variant of a run was made from.
BATCH_MANIFEST = "source_tiff.json"
VARIANTS = ("without_destriping", "with_destriping")
TIFF_SUFFIXES = (".tif", ".tiff")
# The batch's processed-TIFF names, ``<stem>_dynamic_bidiphase[_destriped][_undistorted].tif``,
# are built in mini2p_preprocessing.config (corrected_tiff_path, undistorted_tiff_path); a test
# pins this pattern to those functions' output.
PROCESSED_SUFFIX = r"_dynamic_bidiphase(_destriped)?(_undistorted)?"


class RunMismatchError(ValueError):
    """A run's traces, frame clock and behaviour cannot be shown to be one acquisition."""


def same_tiff_name(name: str, other: str) -> bool:
    """Whether two TIFF file names agree: stems exactly, extensions ignoring case."""
    first, second = Path(name), Path(other)
    return first.stem == second.stem and first.suffix.lower() == second.suffix.lower()


def batch_source_tiffs(run_dir, variants=VARIANTS) -> dict[str, str]:
    """File name of the raw TIFF each of *variants* of *run_dir* was made from, per the batch."""
    names = {}
    for variant in variants:
        manifest = Path(run_dir) / variant / BATCH_MANIFEST
        if manifest.is_file():
            names[variant] = Path(json.loads(manifest.read_text())["source_tiff"]).name
    return names


def processed_tiff_names(variant_dir) -> list[str]:
    """Sorted names of the ``.tif`` / ``.tiff`` files directly in *variant_dir*.

    macOS AppleDouble files (``._*``) are metadata copies, not images, so they are skipped.
    """
    folder = Path(variant_dir)
    if not folder.is_dir():
        return []
    return sorted(
        f.name
        for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in TIFF_SUFFIXES and not f.name.startswith("._")
    )


def names_match_source(tiff_name: str, names) -> bool:
    """Whether *names* is non-empty and each is the source stem plus a suffix the batch writes.

    The batch (``mini2p_preprocessing.config``) names its outputs
    ``<stem>_dynamic_bidiphase[_destriped][_undistorted].tif``; anything else after the stem
    (``_retry_...``, a further acquisition number) is not accepted, and the exact stem keeps
    ``71rtask_00039`` from claiming ``71rtask_000390_...``. Only the extension ignores case
    (as in :func:`same_tiff_name`); the stem and the suffix are compared exactly.
    """
    pattern = re.compile(re.escape(Path(tiff_name).stem) + PROCESSED_SUFFIX + r"(?i:\.tiff?)")
    return len(names) > 0 and all(pattern.fullmatch(name) for name in names)


def _check_variant_by_names(tiff_name: str, run_dir: Path, variant: str) -> None:
    names = processed_tiff_names(run_dir / variant)
    if not names_match_source(tiff_name, names):
        raise RunMismatchError(
            f"{run_dir / variant} has no {BATCH_MANIFEST}, and its TIFFs {names} are not all "
            f"named {Path(tiff_name).stem}_dynamic_bidiphase[...].tif (or there are none); "
            "cannot confirm its TIFF."
        )
    warnings.warn(
        f"{run_dir.name}, {variant}: no {BATCH_MANIFEST}; the source {tiff_name} was inferred "
        f"from file names {names}.",
        UserWarning,
        stacklevel=3,
    )


def check_run_source(tiff_name: str, run_dir, variants=VARIANTS) -> None:
    """Refuse unless every listed variant of *run_dir* was made from *tiff_name*.

    The run numbering here and in the batch are worked out separately; a stray file in
    either can shift it, and runs of equal length would not reveal that. The batch's own
    record settles which acquisition a variant holds. A variant without a record is accepted
    only if it has processed TIFFs and all are named after *tiff_name*, with a warning.
    Listed variants with no folder were never processed and are not checked, but at least
    one listed variant must have a folder.

    Raises:
        RunMismatchError: If no listed variant has a folder, a record names another TIFF, or a
            variant without a record has no TIFFs or one with another name.
    """
    run_dir = Path(run_dir)
    recorded = batch_source_tiffs(run_dir, variants)
    wrong = {v: name for v, name in recorded.items() if not same_tiff_name(name, tiff_name)}
    if wrong:
        raise RunMismatchError(
            f"{run_dir}: the batch made {wrong} from another TIFF than {tiff_name}; the run "
            "numbering disagrees, so these frames would be paired with the wrong acquisition."
        )
    present = [v for v in variants if (run_dir / v).is_dir()]
    if not present:
        raise RunMismatchError(
            f"{run_dir} has no variant folder among {list(variants)}, so no {BATCH_MANIFEST} "
            "or TIFF names to confirm its TIFF."
        )
    for variant in present:
        if variant not in recorded:
            _check_variant_by_names(tiff_name, run_dir, variant)


@dataclass(frozen=True)
class FrameTimes:
    time_s: np.ndarray  # pyControl time of each frame's start; NaN outside the sync pulses
    frame_index: np.ndarray
    daq_frame_s: np.ndarray  # frame starts on the acquisition DAQ's clock
    period_s: float
    tiff: Path
    h5: Path
    tsv: Path
    quality: dict


def frame_times_from_files(run: RunFiles) -> dict:
    """Arrays and provenance for :func:`write_frame_times`, read from the run's raw files."""
    daq = load_daq_sync(run.h5)
    aligner = align_to_pycontrol(
        daq.rsync_s, pycontrol_sync_times(read_tsv_file(run.tsv)), units_ms=1000
    )
    time_s = limited_frame_times(
        to_pycontrol_time(daq.frame_s, aligner),
        to_pycontrol_time(daq.frame_s, aligner, extrapolate=True),
        daq.frame_s,
        daq.rsync_s,
    )
    return {
        "time_s": time_s,
        "frame_index": np.arange(len(time_s)),
        "daq_frame_s": daq.frame_s,
        "tiff": str(run.tiff),
        "h5": str(run.h5),
        "tsv": str(run.tsv),
        "quality": json.dumps(sync_quality(aligner)),
    }


def write_frame_times(run: RunFiles, run_dir, overwrite: bool = False) -> Path:
    """Write ``run_dir/frame_times.npz`` for *run*.

    Raises:
        FileNotFoundError: If *run_dir* (the derivatives run folder) does not exist.
        RunMismatchError: If the batch did not make this run folder from ``run.tiff``.
        FileExistsError: If the file exists and *overwrite* is False.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"No derivatives run folder {run_dir}.")
    check_run_source(run.tiff.name, run_dir)
    path = run_dir / FRAME_TIMES_FILE
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass overwrite=True to replace it.")
    np.savez(path, **frame_times_from_files(run))
    return path


def load_frame_times(run_dir) -> FrameTimes:
    """Read ``run_dir/frame_times.npz``."""
    with np.load(Path(run_dir) / FRAME_TIMES_FILE) as saved:
        time_s = saved["time_s"]
        return FrameTimes(
            time_s=time_s,
            frame_index=saved["frame_index"],
            daq_frame_s=saved["daq_frame_s"],
            period_s=frame_period_s(time_s),
            tiff=Path(str(saved["tiff"])),
            h5=Path(str(saved["h5"])),
            tsv=Path(str(saved["tsv"])),
            quality=json.loads(str(saved["quality"])),
        )


@dataclass(frozen=True)
class VideoFrameTimes:
    time_s: np.ndarray  # pyControl time of each video frame (~half a frame early); NaN outside
    video: Path
    gpio: Path
    tsv: Path
    quality: dict


def video_frame_times_from_files(video: VideoFiles, tsv: Path) -> dict:
    """Arrays and provenance for :func:`write_video_frame_times`, from the camera table.

    Raises:
        RunMismatchError: If the camera's pulses do not match the behaviour file's well
            enough to be one train (:func:`video_match_accepted`).
    """
    table = pd.read_csv(video.gpio)
    try:
        time_s, aligner = video_frame_times(table, pycontrol_sync_times(read_tsv_file(tsv)))
    except RsyncError as error:
        raise RunMismatchError(
            f"{video.gpio.name} shares no pulses with {Path(tsv).name}: {error}"
        ) from error
    quality = sync_quality(aligner)
    if not video_match_accepted(quality, camera_frame_period_ms(table)):
        raise RunMismatchError(
            f"{video.gpio.name} does not match {Path(tsv).name} well enough to be its camera: "
            f"{quality}"
        )
    return {
        "time_s": time_s,
        "video": str(video.video),
        "gpio": str(video.gpio),
        "tsv": str(tsv),
        "quality": json.dumps(quality),
    }


def write_video_frame_times(
    run: RunFiles, video: VideoFiles, run_dir, overwrite: bool = False
) -> Path:
    """Write ``run_dir/video_frame_times.npz``: the pyControl time of every frame of the
    run's behaviour video, on the same clock as ``frame_times.npz``.

    Raises:
        FileNotFoundError: If *run_dir* (the derivatives run folder) does not exist.
        RunMismatchError: If the batch did not make this run folder from ``run.tiff``, or
            the camera does not match the run's behaviour file.
        FileExistsError: If the file exists and *overwrite* is False.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"No derivatives run folder {run_dir}.")
    check_run_source(run.tiff.name, run_dir)
    path = run_dir / VIDEO_FRAME_TIMES_FILE
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass overwrite=True to replace it.")
    np.savez(path, **video_frame_times_from_files(video, run.tsv))
    return path


def load_video_frame_times(run_dir) -> VideoFrameTimes:
    """Read ``run_dir/video_frame_times.npz``."""
    with np.load(Path(run_dir) / VIDEO_FRAME_TIMES_FILE) as saved:
        return VideoFrameTimes(
            time_s=saved["time_s"],
            video=Path(str(saved["video"])),
            gpio=Path(str(saved["gpio"])),
            tsv=Path(str(saved["tsv"])),
            quality=json.loads(str(saved["quality"])),
        )
