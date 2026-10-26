"""Behaviour video of each imaging run: which camera files, and the pyControl time of each frame."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("h5py")  # the optional funcimg extra

from src.funcimg_analysis.frame_times import (
    VIDEO_FRAME_TIMES_FILE,
    RunMismatchError,
    load_video_frame_times,
    write_video_frame_times,
)
from src.funcimg_import.runs import run_video_files, session_run_files, video_beside
from tests.funcimg_fixtures import daq_session
from tests.test_funcimg_runs import write_daq, write_manifest, write_tsv

VIDEO_RATE_HZ = 30.0
PULSE_WIDTH_S = 0.05


def camera_table(
    pycontrol_rsync_s: np.ndarray, duration_s: float
) -> tuple[pd.DataFrame, np.ndarray]:
    """A camera's ``*_GPIO_data.csv`` (one row per frame, rsync on GPIO1) and the true
    pyControl time of each frame. The camera starts 0.5 s into the DAQ recording, on its own
    clock (ns), and GPIO1 is high on frames taken while a pulse is high."""
    daq_rsync_s = (pycontrol_rsync_s - 12.3) / 1.0001  # back onto the DAQ clock (see daq_session)
    frame_daq_s = 0.5 + np.arange(int((duration_s - 0.5) * VIDEO_RATE_HZ)) / VIDEO_RATE_HZ
    since_pulse = frame_daq_s[:, None] - daq_rsync_s[None, :]
    high = ((since_pulse >= 0) & (since_pulse < PULSE_WIDTH_S)).any(axis=1)
    table = pd.DataFrame(
        {
            "GPIO1": high.astype(int),
            "GPIO2": 0,
            "Timestamp": np.round((frame_daq_s + 1000.0) * 1e9).astype(np.int64),
        }
    )
    return table, 12.3 + frame_daq_s * 1.0001


@pytest.fixture
def raw_session_with_video(tmp_path):
    """Two runs, each with a behaviour file, a camera table and a video; the behaviour and
    camera files are named so their order is the reverse of the runs'."""
    session_dir = tmp_path / "raw" / "ses-001_date-20260101"
    (session_dir / "funcimg").mkdir(parents=True)
    (session_dir / "behav").mkdir()
    truths = []
    for number, seed in enumerate((5, 6)):
        session = daq_session(np.random.default_rng(seed), n_frames=1500)
        (session_dir / "funcimg" / f"71rtask_{39 + number:05d}.tif").write_bytes(b"")
        write_daq(session_dir / "funcimg" / f"file_{1940 + number:05d}.h5", session)
        stamp = f"2026-01-01T1{2 - number}_00_00"
        write_tsv(session_dir / "behav" / f"M-2026-01-01-1{2 - number}0000.tsv",
                  session["pycontrol_rsync_s"])  # fmt: skip
        table, truth = camera_table(session["pycontrol_rsync_s"], session["frame_s"][-1] + 2.0)
        table.to_csv(session_dir / "behav" / f"camera_{stamp}_GPIO_data.csv", index=False)
        (session_dir / "behav" / f"camera_{stamp}.avi").write_bytes(b"")
        truths.append(truth)
    return session_dir, truths


def test_each_run_gets_the_camera_files_whose_pulses_match_its_behaviour(raw_session_with_video):
    session_dir, _ = raw_session_with_video
    runs = session_run_files(session_dir, "71rtask")
    videos = [run_video_files(session_dir, run.tsv) for run in runs]
    assert [v.gpio.name for v in videos] == [
        "camera_2026-01-01T12_00_00_GPIO_data.csv",
        "camera_2026-01-01T11_00_00_GPIO_data.csv",
    ]
    assert [v.video.name for v in videos] == [
        "camera_2026-01-01T12_00_00.avi",
        "camera_2026-01-01T11_00_00.avi",
    ]


def test_a_run_without_matching_camera_files_is_refused(raw_session_with_video):
    session_dir, _ = raw_session_with_video
    run = session_run_files(session_dir, "71rtask")[0]
    for gpio in (session_dir / "behav").glob("*_GPIO_data.csv"):
        if "T12" in gpio.name:
            gpio.unlink()
    with pytest.raises(ValueError, match="No camera"):
        run_video_files(session_dir, run.tsv)


def test_video_beside_its_camera_table(tmp_path):
    gpio = tmp_path / "camera_A_GPIO_data.csv"
    gpio.write_text("")
    with pytest.raises(FileNotFoundError, match="camera_A"):
        video_beside(gpio)
    (tmp_path / "camera_A.avi").write_bytes(b"")
    (tmp_path / "camera_B.avi").write_bytes(b"")
    assert video_beside(gpio).name == "camera_A.avi"
    (tmp_path / "camera_A.mp4").write_bytes(b"")
    with pytest.raises(ValueError, match="2 videos"):
        video_beside(gpio)


def test_video_beside_refuses_a_file_that_is_not_a_camera_table(tmp_path):
    with pytest.raises(ValueError, match="_GPIO_data.csv"):
        video_beside(tmp_path / "camera_A.csv")


def test_video_frame_times_are_written_beside_the_run_and_read_back(
    raw_session_with_video, tmp_path
):
    session_dir, truths = raw_session_with_video
    run = session_run_files(session_dir, "71rtask")[1]
    run_dir = tmp_path / "der" / "run-02"
    write_manifest(run_dir / "without_destriping", run.tiff)
    video = run_video_files(session_dir, run.tsv)
    path = write_video_frame_times(run, video, run_dir)
    assert path == run_dir / VIDEO_FRAME_TIMES_FILE
    loaded = load_video_frame_times(run_dir)
    known = np.isfinite(loaded.time_s)
    assert len(loaded.time_s) == len(truths[1]) and known.mean() > 0.9
    # an edge is seen at the first frame after the pulse, up to one frame late
    assert np.median(np.abs(loaded.time_s[known] - truths[1][known])) < 1 / VIDEO_RATE_HZ
    assert loaded.video == video.video and loaded.gpio == video.gpio and loaded.tsv == run.tsv
    assert loaded.quality["fraction_matched"] > 0.9
    with pytest.raises(FileExistsError):
        write_video_frame_times(run, video, run_dir)


def test_video_frame_times_refuse_a_run_the_batch_made_from_another_tiff(
    raw_session_with_video, tmp_path
):
    session_dir, _ = raw_session_with_video
    first, second = session_run_files(session_dir, "71rtask")
    run_dir = tmp_path / "der" / "run-01"
    write_manifest(run_dir / "without_destriping", second.tiff)
    with pytest.raises(RunMismatchError):
        write_video_frame_times(first, run_video_files(session_dir, first.tsv), run_dir)
    assert not (run_dir / VIDEO_FRAME_TIMES_FILE).exists()
