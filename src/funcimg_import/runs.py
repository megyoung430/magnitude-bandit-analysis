"""Which DAQ file and behaviour file belong to each imaging run.

:func:`session_run_alignments` pairs every TIFF of a session with its DAQ file
(``file_01940.h5``) by acquisition order and number of frames, whatever the TIFFs are named
and even if a file is missing, then each DAQ file with the one ``.tsv`` whose rsync pulses
align with its own. The TIFFs that reach a behaviour file are the session's runs,
``run-01``, ``run-02``, ... in acquisition order; mini2p's batch takes those numbers from
the listing ``scripts/funcimg/list_aligned_runs.py`` writes.

:func:`session_run_files` is the older pairing, for runs numbered by the batch itself: the
n-th TIFF named with a source substring (by its acquisition counter, ``71rtask_00039.tif``)
with the n-th DAQ file.
"""

import re
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
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
SESSION_NUMBER = re.compile(r"^ses-(\d+)_")
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
    """*paths* in acquisition order (see :func:`acquisition_counter`), files with the same
    counter by name ignoring case, as mini2p's batch numbers them."""
    return sorted(paths, key=lambda p: (acquisition_counter(p), Path(p).name.lower()))


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


def _visible(paths) -> list[Path]:
    """*paths* without macOS AppleDouble files (``._*``), which are metadata, not data."""
    return [p for p in paths if not p.name.startswith("._")]


def session_tiffs(session_dir, source_substring: str) -> list[Path]:
    """The session's imaging TIFFs (``funcimg/*.tif`` whose name holds *source_substring*,
    any case), in acquisition order: the order the batch preprocessing numbers its runs in."""
    folder = Path(session_dir) / "funcimg"
    return pair_by_acquisition(
        _visible(
            p
            for p in folder.iterdir()
            if p.is_file()
            and p.suffix.lower() == ".tif"
            and source_substring.lower() in p.name.lower()
        )
    )


def session_daq_files(session_dir) -> list[Path]:
    """The session's DAQ files (``funcimg/*.h5``), in acquisition order."""
    return pair_by_acquisition(_visible((Path(session_dir) / "funcimg").glob("*.h5")))


def behavior_pulses(session_dir) -> list[tuple[Path, np.ndarray]]:
    """Each behavior file of the session that has rsync pulses, with their times (pyControl
    clock); files without pulses are left out."""
    behav = Path(session_dir) / "behav"
    tsvs = sorted(_visible(behav.glob("*.tsv"))) if behav.is_dir() else []
    pulses = [(tsv, _pycontrol_pulses(tsv)) for tsv in tsvs]
    return [(tsv, times) for tsv, times in pulses if times is not None]


def _count_mismatch(session_dir: Path, tiffs: list, daq_files: list) -> str:
    """Why a session whose TIFFs and DAQ files differ in number cannot be paired."""
    return (
        f"{Path(session_dir).name}: {len(tiffs)} TIFFs but {len(daq_files)} DAQ files; cannot "
        "tell which acquisition is which."
    )


# ---- pairing each TIFF with its DAQ file, whatever the file names ----------------------

EPOCH = re.compile(r"^epoch\s*=\s*\[([^\]]*)\]", re.MULTILINE)
# A TIFF and a DAQ file are one recording when their numbers of frames differ by at most
# this: a recording stopped mid-frame, and the TIFF's count estimated from its size.
FRAME_COUNT_TOLERANCE = 3


@dataclass(frozen=True)
class TiffInfo:
    """What a ScanImage TIFF's first pages say about its recording."""

    path: Path
    start: datetime  # ScanImage's acquisition start (``epoch`` in the first frame's header)
    timepoints: int  # frames per channel, estimated from the file size and page spacing


def tiff_info(path, n_channels: int) -> TiffInfo:
    """The acquisition start and number of timepoints of a ScanImage TIFF, read from its
    first pages only, so even a 12 GB file over a share takes a moment.

    ScanImage writes every frame with the same size, so the number of pages is the file's
    size past the first page divided by the spacing of consecutive pages.

    Raises:
        ValueError: If the first frame's header has no ``epoch``, or the file has fewer than
            three pages.
    """
    import tifffile

    path = Path(path)
    with tifffile.TiffFile(path) as tiff:
        match = EPOCH.search(tiff.pages[0].description or "")
        if match is None:
            raise ValueError(f"{path.name}: no ScanImage epoch in the first frame's header")
        if len(tiff.pages) < 3:
            raise ValueError(f"{path.name}: fewer than three pages, so not a recording")
        offsets = [tiff.pages[k].offset for k in range(3)]
    year, month, day, hour, minute, second = (float(v) for v in match.group(1).split())
    start = datetime(int(year), int(month), int(day), int(hour), int(minute)) + timedelta(
        seconds=second
    )
    stride = offsets[2] - offsets[1]
    n_pages = round((path.stat().st_size - offsets[0]) / stride)
    return TiffInfo(path, start, round(n_pages / n_channels))


def pair_by_frame_count(
    tiff_frames: list[int], daq_frames: list[int], tolerance: int = FRAME_COUNT_TOLERANCE
) -> list[int | None]:
    """For each TIFF, the index of its DAQ file, or None: both lists in acquisition order,
    pairs keeping that order, as many pairs as possible, and a pair only where the numbers
    of frames agree within *tolerance*.

    A missing file on either side then costs only its own recording. Equal-length
    recordings (fixed-length pre-task acquisitions) are told apart by their order; among
    equally good pairings the earliest DAQ file is taken.
    """
    n_tiff, n_daq = len(tiff_frames), len(daq_frames)

    def agree(i, j):
        return abs(tiff_frames[i] - daq_frames[j]) <= tolerance

    # best[i][j]: most pairs among TIFFs i.. and DAQ files j..
    best = [[0] * (n_daq + 1) for _ in range(n_tiff + 1)]
    for i in range(n_tiff - 1, -1, -1):
        for j in range(n_daq - 1, -1, -1):
            best[i][j] = max(
                best[i + 1][j],
                best[i][j + 1],
                best[i + 1][j + 1] + 1 if agree(i, j) else 0,
            )
    pairs: list[int | None] = [None] * n_tiff
    i = j = 0
    while i < n_tiff and j < n_daq:
        if agree(i, j) and best[i][j] == best[i + 1][j + 1] + 1:
            pairs[i] = j
            i, j = i + 1, j + 1
        elif best[i][j] == best[i][j + 1]:
            j += 1
        else:
            i += 1
    return pairs


@dataclass(frozen=True)
class RunAlignment:
    """Whether one imaging TIFF can be put on pyControl time, and with which files."""

    session_dir: Path
    run_name: str  # run-01, run-02, ... over the session's aligned TIFFs; "" when not aligned
    tiff: Path
    start: datetime | None  # the TIFF's acquisition start; None if its header has none
    timepoints: int | None  # the TIFF's estimated number of frames
    h5: Path | None  # None when the TIFF cannot be paired with a DAQ file
    daq_frames: int | None  # frame-clock pulses in that DAQ file
    tsv: Path | None  # None when the DAQ file matches no single behavior file
    reason: str  # why it cannot be aligned; empty when it can

    @property
    def aligned(self) -> bool:
        """Whether the TIFF has a DAQ file and a behavior file (no reason against it)."""
        return not self.reason


def sync_matches(daq_rsync_s: list, pycontrol_rsync_s: list, names: list[str]) -> list[str | int]:
    """For each DAQ file, the index of the one behavior file whose pulses align with its
    own (see :func:`_aligns`), or the reason there is none, naming behavior files by
    *names* (one per entry of *pycontrol_rsync_s*).

    Unlike :func:`match_by_sync`, one DAQ file that does not match leaves the others
    matched. A behavior file claimed by two DAQ files is refused for both.
    """
    hits = [
        [i for i, pycontrol in enumerate(pycontrol_rsync_s) if _aligns(daq, pycontrol)]
        for daq in daq_rsync_s
    ]
    claimed = [found[0] for found in hits if len(found) == 1]
    matches = []
    for found in hits:
        if not found:
            matches.append("its rsync pulses align with no behavior file")
        elif len(found) > 1:
            several = [names[i] for i in found]
            matches.append(f"its rsync pulses align with several behavior files {several}")
        elif claimed.count(found[0]) > 1:
            matches.append(f"another DAQ file aligns with the same behavior file {names[found[0]]}")
        else:
            matches.append(found[0])
    return matches


def _daq_sync_or_reason(h5: Path):
    """The DAQ file's pulse times, or why it holds none (a line missing or flat). A file
    that cannot be opened raises: that is a storage problem, not a fact about the
    recording to write into the listing."""
    from src.funcimg_import.daq_sync import NoPulsesError, load_daq_sync

    try:
        return load_daq_sync(h5)
    except (KeyError, NoPulsesError) as error:
        return f"{h5.name} has no usable sync lines: {error}"


def session_recordings(session_dir) -> list[Path]:
    """Every TIFF in the session's ``funcimg/`` (task, pre-task or test, whatever its name)."""
    folder = Path(session_dir) / "funcimg"
    return sorted(
        _visible(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".tif")
    )


def session_run_alignments(session_dir, n_channels: int = 2) -> list[RunAlignment]:
    """Whether each TIFF of a raw session can be put on pyControl time, and its run number.

    1. TIFFs are put in order of their acquisition start (ScanImage's header), DAQ files
       (``file_<counter>.h5``, numbered across all recordings) by their counter.
    2. Each TIFF is paired with a DAQ file in that order where their numbers of frames agree
       (:func:`pair_by_frame_count`), so neither file names nor a missing file on either
       side can pair a TIFF with another recording's DAQ file.
    3. A paired DAQ file whose rsync pulses match exactly one behavior file's
       (:func:`sync_matches`) makes its TIFF aligned.
    4. The aligned TIFFs are numbered ``run-01``, ``run-02``, ... in acquisition order;
       others (no DAQ file, no behavior: pre-task and test recordings) get no number.

    Args:
        n_channels (int): Channels interleaved in each TIFF, to count its timepoints.

    Returns:
        list[RunAlignment]: One per TIFF, in acquisition order, each unaligned one with the
        reason.

    Raises:
        ValueError: If a TIFF has no ScanImage header or a DAQ file name no counter.
    """
    session_dir = Path(session_dir)
    tiffs = sorted(
        (tiff_info(p, n_channels) for p in session_recordings(session_dir)),
        key=lambda t: (t.start, t.path.name.lower()),
    )
    daq_files = session_daq_files(session_dir)
    daq = [_daq_sync_or_reason(h5) for h5 in daq_files]
    readable = [k for k, d in enumerate(daq) if not isinstance(d, str)]
    pairs = pair_by_frame_count(
        [t.timepoints for t in tiffs], [len(daq[k].frame_s) for k in readable]
    )
    pulses = behavior_pulses(session_dir)
    paired = [readable[p] for p in pairs if p is not None]
    behavior = dict(
        zip(
            paired,
            sync_matches(
                [daq[k].rsync_s for k in paired],
                [t for _, t in pulses],
                [p.name for p, _ in pulses],
            ),
            strict=True,
        )
    )
    unreadable = "; ".join(d for d in daq if isinstance(d, str))
    rows, number = [], 0
    for tiff, pair in zip(tiffs, pairs, strict=True):
        common = {"session_dir": session_dir, "tiff": tiff.path, "start": tiff.start}
        common["timepoints"] = tiff.timepoints
        if pair is None:
            reason = f"no DAQ file with {tiff.timepoints} frames (+-{FRAME_COUNT_TOLERANCE}) "
            reason += "in acquisition order" + (f" ({unreadable})" if unreadable else "")
            rows.append(RunAlignment(run_name="", h5=None, daq_frames=None, tsv=None,
                                     reason=reason, **common))  # fmt: skip
            continue
        k = readable[pair]
        match = behavior[k]
        h5, frames = daq_files[k], len(daq[k].frame_s)
        if isinstance(match, str):
            rows.append(RunAlignment(run_name="", h5=h5, daq_frames=frames, tsv=None,
                                     reason=match, **common))  # fmt: skip
            continue
        number += 1
        rows.append(RunAlignment(run_name=f"run-{number:02d}", h5=h5, daq_frames=frames,
                                 tsv=pulses[match][0], reason="", **common))  # fmt: skip
    return rows


def numbered_sessions(raw_subject_dir) -> list[Path]:
    """The subject's numbered raw sessions (``ses-<number>_...``), in number order; others
    (``ses-test``, ``ses-baseplating``) are not task recordings and are left out."""
    sessions = [
        (int(m.group(1)), d)
        for d in Path(raw_subject_dir).iterdir()
        if d.is_dir() and (m := SESSION_NUMBER.match(d.name))
    ]
    return [d for _, d in sorted(sessions)]


def subject_run_alignments(raw_subject_dir, n_channels: int = 2) -> list[RunAlignment]:
    """:func:`session_run_alignments` for every numbered session with a ``funcimg/`` folder.

    Raises:
        ValueError: Any a session raises (e.g. a file name without ``_<digits>``, so no
            acquisition order, or an unreadable behavior file), naming the session.
    """
    alignments = []
    for session_dir in numbered_sessions(raw_subject_dir):
        if not (session_dir / "funcimg").is_dir():
            continue
        try:
            alignments += session_run_alignments(session_dir, n_channels)
        except ValueError as error:
            raise ValueError(f"{session_dir.name}: {error}") from error
    return alignments


def aligned_run_files(session_dir, n_channels: int = 2) -> tuple[list[RunFiles], list]:
    """The runs of a session that can be put on pyControl time, as :class:`RunFiles`, and
    the :class:`RunAlignment` of each that cannot, with its reason: the same choice as
    ``scripts/funcimg/list_aligned_runs.py``, so mini2p's batch and the frame-times step
    agree on which runs have behavior."""
    alignments = session_run_alignments(session_dir, n_channels)
    aligned = [RunFiles(a.run_name, a.tiff, a.h5, a.tsv) for a in alignments if a.aligned]
    return aligned, [a for a in alignments if not a.aligned]


def alignment_table(alignments: list[RunAlignment]) -> pd.DataFrame:
    """One row per TIFF, in acquisition order: ``session``, ``run`` (empty when not aligned),
    ``tiff``, ``h5``, ``tsv`` (absolute paths, empty when missing), ``aligned``, ``reason``,
    and the evidence for the pairing: ``tiff_start``, ``tiff_timepoints``, ``daq_frames``."""

    def path(p):
        return str(p.resolve()) if p else ""

    return pd.DataFrame(
        {
            "session": [a.session_dir.name for a in alignments],
            "run": [a.run_name for a in alignments],
            "tiff": [path(a.tiff) for a in alignments],
            "h5": [path(a.h5) for a in alignments],
            "tsv": [path(a.tsv) for a in alignments],
            "aligned": [a.aligned for a in alignments],
            "reason": [a.reason for a in alignments],
            "tiff_start": [a.start.isoformat() if a.start else "" for a in alignments],
            "tiff_timepoints": [a.timepoints for a in alignments],
            "daq_frames": [a.daq_frames if a.daq_frames is not None else "" for a in alignments],
        },
        columns=[
            "session", "run", "tiff", "h5", "tsv", "aligned", "reason",
            "tiff_start", "tiff_timepoints", "daq_frames",
        ],
    )  # fmt: skip


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
    tiffs = session_tiffs(session_dir, source_substring)
    daq_files = session_daq_files(session_dir)
    if len(tiffs) != len(daq_files):
        raise ValueError(_count_mismatch(session_dir, tiffs, daq_files))
    pulses = behavior_pulses(session_dir)
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


def match_videos(
    tables: dict[str, pd.DataFrame], run_pulses: list, run_names: list[str] | None = None
) -> list[str]:
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
            name = run_names[run] if run_names is not None else f"Run {run + 1}"
            raise ValueError(f"{name} matches {found} of {sorted(tables)}.")
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
    names = match_videos(tables, pulses, [run.run_name for run in runs])
    return [VideoFiles(video_beside(behav / name), behav / name) for name in names]
