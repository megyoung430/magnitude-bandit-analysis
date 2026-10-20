from pathlib import Path

import numpy as np
import pytest

from src.funcimg_import.daq_sync import edge_times, rising_edges
from src.funcimg_import.runs import acquisition_counter, match_by_sync, pair_by_acquisition
from src.sync.pycontrol_time import imaging_frame_times
from tests.funcimg_fixtures import DAQ_RATE, daq_session, square_wave


def test_rising_edges_find_each_pulse_start():
    signal = square_wave(5000, np.array([100, 2000, 4000]), 50)
    assert rising_edges(signal).tolist() == [100, 2000, 4000]
    assert edge_times(signal, DAQ_RATE).tolist() == [0.02, 0.4, 0.8]


def test_a_signal_already_high_at_the_start_does_not_count_as_an_edge():
    signal = square_wave(1000, np.array([0, 500]), 100)
    assert rising_edges(signal).tolist() == [500]


def test_a_flat_line_has_no_edges_rather_than_noise_edges():
    with pytest.raises(ValueError, match="no pulses"):
        rising_edges(np.random.default_rng(1).normal(0, 0.01, 5000))


def test_imaging_frames_land_on_pycontrol_time():
    session = daq_session(np.random.default_rng(0))
    frame_s = edge_times(session["frame_clock"], DAQ_RATE)
    rsync_s = edge_times(session["rsync"], DAQ_RATE)
    times, aligner = imaging_frame_times(frame_s, rsync_s, session["pycontrol_rsync_s"])
    inside = np.isfinite(times)
    assert inside.sum() > 0.95 * len(times)
    error_ms = 1000 * np.abs(times[inside] - session["frame_pycontrol_s"][inside])
    assert error_ms.max() < 1.0
    # frames before the first or after the last matched pulse have no pyControl time
    assert not inside[0] and not inside[-1]


def test_files_are_ordered_by_acquisition_counter():
    names = ["71rtask_00010.tif", "71rtask_00002.tif", "file_01942.h5", "file_01940.h5"]
    assert acquisition_counter(Path("71rtask_00040.tif")) == 40
    assert [p.name for p in pair_by_acquisition([Path(n) for n in names[:2]])] == [
        "71rtask_00002.tif",
        "71rtask_00010.tif",
    ]
    with pytest.raises(ValueError, match="counter"):
        acquisition_counter(Path("notes.tif"))


def test_each_daq_file_is_matched_to_the_behaviour_file_its_pulses_align_with():
    sessions = [daq_session(np.random.default_rng(seed)) for seed in (1, 2, 3)]
    daq = [edge_times(s["rsync"], DAQ_RATE) for s in sessions]
    pycontrol = [s["pycontrol_rsync_s"] for s in sessions]
    assert match_by_sync(daq, [pycontrol[2], pycontrol[0], pycontrol[1]]) == [1, 2, 0]
    with pytest.raises(ValueError, match="no behaviour file"):
        match_by_sync(daq, pycontrol[:2])
