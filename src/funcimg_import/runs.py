"""Which DAQ file and behaviour file belong to each imaging run.

The batch preprocessing numbers a session's raw TIFFs ``run-01``, ``run-02``, ... in order of
their acquisition counter (``71rtask_00039.tif``). The DAQ files (``file_01940.h5``) are numbered
the same way, so the n-th TIFF and the n-th DAQ file are one acquisition. Behaviour files are
matched by their sync pulses instead: a DAQ file belongs to the one ``.tsv`` whose rsync pulses
align with its own.
"""

import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.behavior_import.import_data import read_tsv_file
from src.sync.pycontrol_time import (
    NoSyncPulsesError,
    align_to_pycontrol,
    pycontrol_sync_times,
    sync_quality,
    video_frame_times,
)
from src.sync.rsync import RsyncError, RsyncWarning

COUNTER = re.compile(r"_(\d+)$")
# Two trains are one when this many pulses match and their intervals agree this closely; an
# unrelated train can still pass the aligner's own check by chance.
MIN_FRACTION_MATCHED = 0.9
MAX_INTERVAL_DISCREPANCY_MS = 10.0
# A camera sees a pulse at its next frame, so each interval is off by up to a frame period:
# its match allows a median error of 50 ms and a largest error of three frame periods.
MAX_VIDEO_MEDIAN_DISCREPANCY_MS = 50.0
MAX_VIDEO_DISCREPANCY_FRAMES = 3.0
GPIO_SUFFIX = "_GPIO_data.csv"
VIDEO_SUFFIXES = (".avi", ".mp4", ".mov", ".mkv")


def acquisition_counter(path: Path) -> int:
    """The trailing number of a file name, e.g. 40 for ``71rtask_00040.tif``.

    Raises:
        ValueError: If the name does not end in ``_<digits>``.
    """
    match = COUNTER.search(Path(path).stem)
    if not match:
        raise ValueError(f"{Path(path).name} has no acquisition counter (_<digits>).")
    return int(match.group(1))


def pair_by_acquisition(paths: list[Path]) -> list[Path]:
    """*paths* in acquisition order; see :func:`acquisition_counter`."""
    return sorted(paths, key=acquisition_counter)


def _aligns(daq_rsync_s, pycontrol_rsync_s) -> bool:
    """Whether two pulse trains are one train: nearly every pulse matched, intervals agree."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            aligner = align_to_pycontrol(daq_rsync_s, pycontrol_rsync_s, units_ms=1000)
    except (RsyncError, ValueError):
        return False
    quality = sync_quality(aligner)
    return (
        quality["fraction_matched"] >= MIN_FRACTION_MATCHED
        and quality["max_abs_interval_discrepancy_ms"] <= MAX_INTERVAL_DISCREPANCY_MS
    )


def match_by_sync(daq_rsync_s: list, pycontrol_rsync_s: list) -> list[int]:
    """For each DAQ file, the index of the behaviour file whose pulses align with its own.

    Args:
        daq_rsync_s (list): Rsync times (s, DAQ clock) of each DAQ file.
        pycontrol_rsync_s (list): Rsync times (s, pyControl clock) of each behaviour file.

    Returns:
        list[int]: Index into *pycontrol_rsync_s* for each DAQ file.

    Raises:
        ValueError: If a DAQ file aligns with no behaviour file or with more than one, or two
            DAQ files align with the same one.
    """
    matches = []
    for number, daq in enumerate(daq_rsync_s):
        hits = [i for i, pycontrol in enumerate(pycontrol_rsync_s) if _aligns(daq, pycontrol)]
        if len(hits) != 1:
            found = "no behaviour file" if not hits else f"behaviour files {hits}"
            raise ValueError(f"DAQ file {number} aligns with {found}.")
        matches.append(hits[0])
    if len(set(matches)) != len(matches):
        raise ValueError(f"Two DAQ files align with the same behaviour file: {matches}.")
    return matches


@dataclass(frozen=True)
class RunFiles:
    """The raw files of one imaging run."""

    run_name: str  # run-01, run-02, ... as the batch preprocessing numbers them
    tiff: Path
    h5: Path
    tsv: Path


def _pycontrol_pulses(tsv: Path):
    try:
        return pycontrol_sync_times(read_tsv_file(tsv))
    except NoSyncPulsesError:
        return None


def session_run_files(session_dir, source_substring: str) -> list[RunFiles]:
    """The TIFF, DAQ file and behaviour file of each run of a raw session.

    Args:
        session_dir (str | Path): ``.../ses-XXX_date-YYYYMMDD`` with ``funcimg/`` and ``behav/``.
        source_substring (str): Picks the imaging TIFFs, as in the batch (e.g. ``"71rtask"``).

    Returns:
        list[RunFiles]: One per TIFF, in acquisition order.

    Raises:
        ValueError: If the numbers of TIFFs and DAQ files differ, or a DAQ file cannot be
            matched to exactly one behaviour file (see :func:`match_by_sync`).
    """
    from src.funcimg_import.daq_sync import load_daq_sync

    session_dir = Path(session_dir)
    tiffs = pair_by_acquisition(
        [
            p
            for p in (session_dir / "funcimg").iterdir()
            if p.suffix.lower() == ".tif"
            and source_substring.lower() in p.name.lower()
            and not p.name.startswith("._")
        ]
    )
    daq_files = pair_by_acquisition(
        [p for p in (session_dir / "funcimg").glob("*.h5") if not p.name.startswith("._")]
    )
    if len(tiffs) != len(daq_files):
        raise ValueError(
            f"{session_dir.name}: {len(tiffs)} TIFFs but {len(daq_files)} DAQ files; cannot "
            "tell which acquisition is which."
        )
    tsvs = sorted(p for p in (session_dir / "behav").glob("*.tsv") if not p.name.startswith("._"))
    pulses = [(tsv, _pycontrol_pulses(tsv)) for tsv in tsvs]
    pulses = [(tsv, times) for tsv, times in pulses if times is not None]
    daq_rsync = [load_daq_sync(path).rsync_s for path in daq_files]
    matches = match_by_sync(daq_rsync, [times for _, times in pulses])
    return [
        RunFiles(f"run-{number:02d}", tiff, daq, pulses[match][0])
        for number, (tiff, daq, match) in enumerate(zip(tiffs, daq_files, matches, strict=True), 1)
    ]


@dataclass(frozen=True)
class VideoFiles:
    """The behaviour camera's files for one run."""

    video: Path
    gpio: Path  # ``*_GPIO_data.csv``: one row per frame, rsync pulses on GPIO1


def video_beside(gpio: Path) -> Path:
    """The video the camera wrote with *gpio*: same name without ``_GPIO_data.csv``, with a
    video extension (any case).

    Raises:
        ValueError: If *gpio* is not a ``*_GPIO_data.csv``, or several videos share its name.
        FileNotFoundError: If there is no such video.
    """
    gpio = Path(gpio)
    if not gpio.name.endswith(GPIO_SUFFIX):
        raise ValueError(f"{gpio.name} is not a camera table (*{GPIO_SUFFIX}).")
    stem = gpio.name[: -len(GPIO_SUFFIX)]
    videos = sorted(p for p in gpio.parent.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES)
    found = [p for p in videos if p.stem == stem]
    if len(found) > 1:
        raise ValueError(f"{len(found)} videos for {gpio.name}: {[p.name for p in found]}.")
    if not found:
        raise FileNotFoundError(
            f"No video {stem}{{{','.join(VIDEO_SUFFIXES)}}} beside {gpio.name}; videos there: "
            f"{[p.name for p in videos]}."
        )
    return found[0]


def camera_frame_period_ms(gpio_df: pd.DataFrame) -> float:
    """Median time between the camera's frames (ms), from its ``Timestamp`` column (ns)."""
    return float(np.median(np.diff(gpio_df["Timestamp"].to_numpy(dtype=float)))) / 1e6


def video_match_accepted(quality: dict, frame_period_ms: float) -> bool:
    """Whether a camera's pulses are one train with a behaviour file's: nearly all pulses of
    *both* trains paired (a camera that ran through two runs covers neither run alone), a
    median interval error under ``MAX_VIDEO_MEDIAN_DISCREPANCY_MS`` and none over
    ``MAX_VIDEO_DISCREPANCY_FRAMES`` frame periods.

    Args:
        quality (dict): From :func:`sync_quality`, with the camera as the other system.
        frame_period_ms (float): The camera's frame period.
    """
    longer = max(quality["n_pulses_other"], quality["n_pulses_pycontrol"])
    return (
        quality["n_matched"] / longer >= MIN_FRACTION_MATCHED
        and quality["median_abs_interval_discrepancy_ms"] <= MAX_VIDEO_MEDIAN_DISCREPANCY_MS
        and quality["max_abs_interval_discrepancy_ms"]
        <= MAX_VIDEO_DISCREPANCY_FRAMES * frame_period_ms
    )


def video_match_quality(gpio_df: pd.DataFrame, pycontrol_rsync_s) -> dict | None:
    """Sync quality of a camera table against a behaviour file's pulses; None when the
    aligner finds no common stretch (another session's train).

    Raises:
        ValueError: If the table is malformed (see :func:`video_frame_times`).
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RsyncWarning)
            _, aligner = video_frame_times(gpio_df, pycontrol_rsync_s)
    except RsyncError:
        return None
    return sync_quality(aligner)


def match_videos(tables: dict[str, pd.DataFrame], run_pulses: list) -> list[str]:
    """For each run's pyControl pulses, the one camera table (by name) that matches it.

    Raises:
        ValueError: If a table is malformed (named), a run matches no table or several, or
            one table matches several runs (each run has its own video).
    """
    accepted = {}
    for name, table in tables.items():
        try:
            period = camera_frame_period_ms(table)
            qualities = [video_match_quality(table, pulses) for pulses in run_pulses]
        except (KeyError, ValueError) as error:
            raise ValueError(f"Camera table {name} cannot be read: {error}") from error
        accepted[name] = [q is not None and video_match_accepted(q, period) for q in qualities]
    matches = []
    for run, _ in enumerate(run_pulses):
        hits = [name for name, ok in accepted.items() if ok[run]]
        if len(hits) != 1:
            found = "no camera table" if not hits else f"camera tables {hits}"
            raise ValueError(f"Run {run + 1} matches {found} of {sorted(tables)}.")
        matches.append(hits[0])
    if len(set(matches)) != len(matches):
        raise ValueError(f"One camera table matches several runs: {matches}.")
    return matches


def session_video_files(session_dir, runs: list[RunFiles]) -> list[VideoFiles]:
    """The camera files of each run of *session_dir*, matched to the run's ``.tsv`` by
    rsync pulses (see :func:`match_videos`), with the video beside each table.

    Raises:
        ValueError: As :func:`match_videos`.
        FileNotFoundError: If a matched table has no video beside it.
    """
    behav = Path(session_dir) / "behav"
    paths = sorted(p for p in behav.glob(f"*{GPIO_SUFFIX}") if not p.name.startswith("._"))
    tables = {p.name: pd.read_csv(p) for p in paths}
    pulses = [pycontrol_sync_times(read_tsv_file(run.tsv)) for run in runs]
    names = match_videos(tables, pulses)
    return [VideoFiles(video_beside(behav / name), behav / name) for name in names]
