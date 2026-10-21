"""pyControl time of every imaging frame, saved beside each run as ``frame_times.npz``.

The file is written once per run (``derivatives/.../funcimg/run-XX/frame_times.npz``) and serves
both destriping variants, which share their frames. Frame ``i`` of the registered movie (and of
``F.npy``) has pyControl time ``time_s[i]`` (s; NaN outside the sync pulses), so analyses can
work in frames or in seconds.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.behavior_import.import_data import read_tsv_file
from src.funcimg_analysis.event_frames import frame_period_s, limited_frame_times
from src.funcimg_import.daq_sync import load_daq_sync
from src.funcimg_import.runs import RunFiles
from src.sync.pycontrol_time import (
    align_to_pycontrol,
    pycontrol_sync_times,
    sync_quality,
    to_pycontrol_time,
)

FRAME_TIMES_FILE = "frame_times.npz"
# The batch preprocessing's record of which raw TIFF each variant of a run was made from.
BATCH_MANIFEST = "source_tiff.json"
VARIANTS = ("without_destriping", "with_destriping")


class RunMismatchError(ValueError):
    """A run's traces, frame clock and behaviour cannot be shown to be one acquisition."""


def batch_source_tiffs(run_dir) -> dict[str, str]:
    """File name of the raw TIFF each variant of *run_dir* was made from, per the batch."""
    names = {}
    for variant in VARIANTS:
        manifest = Path(run_dir) / variant / BATCH_MANIFEST
        if manifest.is_file():
            names[variant] = Path(json.loads(manifest.read_text())["source_tiff"]).name
    return names


def check_run_source(tiff_name: str, run_dir, variants=VARIANTS) -> None:
    """Refuse unless the batch made every listed variant of *run_dir* from *tiff_name*.

    The run numbering here and in the batch are worked out separately; a stray file in
    either can shift it, and runs of equal length would not reveal that. The batch's own
    record settles which acquisition a run folder holds.

    Raises:
        RunMismatchError: If no variant has a batch record, or one names another TIFF.
    """
    recorded = {v: name for v, name in batch_source_tiffs(run_dir).items() if v in variants}
    if not recorded:
        raise RunMismatchError(
            f"{run_dir} has no {BATCH_MANIFEST} for {list(variants)}; cannot confirm its TIFF."
        )
    wrong = {v: name for v, name in recorded.items() if name != tiff_name}
    if wrong:
        raise RunMismatchError(
            f"{run_dir}: the batch made {wrong} from another TIFF than {tiff_name}; the run "
            "numbering disagrees, so these frames would be paired with the wrong acquisition."
        )


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
