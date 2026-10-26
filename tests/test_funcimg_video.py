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
from src.funcimg_import.runs import (
    match_videos,
    session_run_files,
    session_video_files,
    video_beside,
    video_match_accepted,
)
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
    videos = session_video_files(session_dir, session_run_files(session_dir, "71rtask"))
    assert [v.gpio.name for v in videos] == [
        "camera_2026-01-01T12_00_00_GPIO_data.csv",
        "camera_2026-01-01T11_00_00_GPIO_data.csv",
    ]
    assert [v.video.name for v in videos] == [
        "camera_2026-01-01T12_00_00.avi",
        "camera_2026-01-01T11_00_00.avi",
    ]


def test_a_run_without_its_camera_table_is_refused(raw_session_with_video):
    session_dir, _ = raw_session_with_video
    (session_dir / "behav" / "camera_2026-01-01T12_00_00_GPIO_data.csv").unlink()
    with pytest.raises(ValueError, match="Run 1 matches no camera table"):
        session_video_files(session_dir, session_run_files(session_dir, "71rtask"))


def test_a_camera_that_ran_through_two_runs_belongs_to_neither(raw_session_with_video):
    session_dir, _ = raw_session_with_video
    behav = session_dir / "behav"
    first, second = (pd.read_csv(behav / f"camera_2026-01-01T1{h}_00_00_GPIO_data.csv")
                     for h in (2, 1))  # fmt: skip
    second["Timestamp"] += first["Timestamp"].iloc[-1] + 10**9
    for table in behav.glob("*_GPIO_data.csv"):
        table.unlink()
    pd.concat([first, second]).to_csv(behav / "camera_all_GPIO_data.csv", index=False)
    (behav / "camera_all.avi").write_bytes(b"")
    with pytest.raises(ValueError, match="no camera table"):
        session_video_files(session_dir, session_run_files(session_dir, "71rtask"))


def test_two_camera_tables_for_one_run_are_refused(raw_session_with_video):
    session_dir, _ = raw_session_with_video
    behav = session_dir / "behav"
    copy = behav / "camera_copy_GPIO_data.csv"
    copy.write_text((behav / "camera_2026-01-01T12_00_00_GPIO_data.csv").read_text())
    (behav / "camera_copy.avi").write_bytes(b"")
    with pytest.raises(ValueError, match="Run 1 matches camera tables"):
        session_video_files(session_dir, session_run_files(session_dir, "71rtask"))


def test_one_camera_table_for_two_runs_is_refused(raw_session_with_video):
    session_dir, _ = raw_session_with_video
    session = daq_session(np.random.default_rng(5), n_frames=1500)
    table, _ = camera_table(session["pycontrol_rsync_s"], session["frame_s"][-1] + 2.0)
    pulses = session["pycontrol_rsync_s"]
    with pytest.raises(ValueError, match="matches several runs"):
        match_videos({"camera_GPIO_data.csv": table}, [pulses, pulses])


def test_a_malformed_camera_table_is_named_not_skipped(raw_session_with_video):
    session_dir, _ = raw_session_with_video
    path = session_dir / "behav" / "camera_2026-01-01T12_00_00_GPIO_data.csv"
    pd.read_csv(path).rename(columns={"GPIO1": "GPIO 1"}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="camera_2026-01-01T12_00_00_GPIO_data.csv cannot be read"):
        session_video_files(session_dir, session_run_files(session_dir, "71rtask"))


def test_a_matched_table_without_its_video_is_refused(raw_session_with_video):
    session_dir, _ = raw_session_with_video
    (session_dir / "behav" / "camera_2026-01-01T11_00_00.avi").unlink()
    with pytest.raises(FileNotFoundError, match="camera_2026-01-01T11_00_00"):
        session_video_files(session_dir, session_run_files(session_dir, "71rtask"))


@pytest.mark.parametrize(
    ("change", "accepted"),
    [
        ({}, True),
        ({"n_matched": 89}, False),  # under 90 % of the longer train
        ({"n_pulses_other": 200}, False),  # a camera with twice the run's pulses
        ({"median_abs_interval_discrepancy_ms": 50.1}, False),
        ({"max_abs_interval_discrepancy_ms": 3 * 1000 / 30 + 0.1}, False),
        ({"max_abs_interval_discrepancy_ms": 3 * 1000 / 30 - 0.1}, True),
    ],
)
def test_video_match_rule(change, accepted):
    quality = {
        "n_pulses_other": 100,
        "n_pulses_pycontrol": 100,
        "n_matched": 95,
        "median_abs_interval_discrepancy_ms": 10.0,
        "max_abs_interval_discrepancy_ms": 30.0,
    }
    assert video_match_accepted({**quality, **change}, 1000 / 30) is accepted


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
    video = session_video_files(session_dir, session_run_files(session_dir, "71rtask"))[1]
    path = write_video_frame_times(run, video, run_dir)
    assert path == run_dir / VIDEO_FRAME_TIMES_FILE
    loaded = load_video_frame_times(run_dir)
    known = np.isfinite(loaded.time_s)
    assert len(loaded.time_s) == len(truths[1]) and known.mean() > 0.9
    # an edge is seen at the first frame after the pulse, so frame times are early by up to
    # one frame period, half a period on average
    error = loaded.time_s[known] - truths[1][known]
    assert -1 / VIDEO_RATE_HZ <= np.median(error) <= 0 and np.abs(error).max() < 2 / VIDEO_RATE_HZ
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
    video = session_video_files(session_dir, [first, second])[0]
    with pytest.raises(RunMismatchError):
        write_video_frame_times(first, video, run_dir)
    assert not (run_dir / VIDEO_FRAME_TIMES_FILE).exists()


def test_video_frame_times_refuse_a_camera_that_is_not_the_runs(raw_session_with_video, tmp_path):
    session_dir, _ = raw_session_with_video
    first, second = session_run_files(session_dir, "71rtask")
    run_dir = tmp_path / "der" / "run-01"
    write_manifest(run_dir / "without_destriping", first.tiff)
    others = session_video_files(session_dir, [first, second])[1]
    with pytest.raises(RunMismatchError):
        write_video_frame_times(first, others, run_dir)
    assert not (run_dir / VIDEO_FRAME_TIMES_FILE).exists()
