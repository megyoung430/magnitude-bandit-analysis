import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("h5py")  # the optional funcimg extra

from src.funcimg_analysis.frame_times import (
    FRAME_TIMES_FILE,
    RunMismatchError,
    check_run_source,
    load_frame_times,
    names_match_source,
    write_frame_times,
)
from src.funcimg_analysis.run_responses import RAW_VIEWS, Z_VIEWS, pulse_responses
from src.funcimg_import.runs import session_run_files
from tests.funcimg_fixtures import DAQ_RATE, daq_session


def write_daq(path, session):
    import h5py

    with h5py.File(path, "w") as daq:
        daq.attrs["samplerate"] = DAQ_RATE
        daq["FrameClock"] = session["frame_clock"]
        daq["PyControlPulse"] = session["rsync"]


def write_tsv(path, rsync_s):
    rows = [(0.0, "info", "subject_id", "M")] + [(t, "event", "sync", "rsync") for t in rsync_s]
    pd.DataFrame(rows, columns=["time", "type", "subtype", "content"]).to_csv(
        path, sep="\t", index=False, float_format="%.3f"
    )


@pytest.fixture
def raw_session(tmp_path):
    """Two acquisitions; the behaviour files are named so their order is reversed."""
    session_dir = tmp_path / "raw" / "ses-001_date-20260101"
    (session_dir / "funcimg").mkdir(parents=True)
    (session_dir / "behav").mkdir()
    sessions = [daq_session(np.random.default_rng(seed), n_frames=1500) for seed in (5, 6)]
    for number, session in enumerate(sessions):
        (session_dir / "funcimg" / f"71rtask_{39 + number:05d}.tif").write_bytes(b"")
        write_daq(session_dir / "funcimg" / f"file_{1940 + number:05d}.h5", session)
        write_tsv(
            session_dir / "behav" / f"M-2026-01-01-1{2 - number}0000.tsv",
            session["pycontrol_rsync_s"],
        )
    return session_dir, sessions


def test_each_run_gets_its_tiff_daq_and_behaviour_files(raw_session):
    session_dir, _ = raw_session
    runs = session_run_files(session_dir, "71rtask")
    assert [r.run_name for r in runs] == ["run-01", "run-02"]
    assert [r.tiff.name for r in runs] == ["71rtask_00039.tif", "71rtask_00040.tif"]
    assert [r.h5.name for r in runs] == ["file_01940.h5", "file_01941.h5"]
    assert [r.tsv.name for r in runs] == ["M-2026-01-01-120000.tsv", "M-2026-01-01-110000.tsv"]


def test_unequal_numbers_of_tiffs_and_daq_files_are_refused(raw_session):
    session_dir, _ = raw_session
    (session_dir / "funcimg" / "71rtask_00041.tif").write_bytes(b"")
    with pytest.raises(ValueError, match="3 TIFFs but 2 DAQ"):
        session_run_files(session_dir, "71rtask")


def test_frame_times_are_written_beside_the_run_and_read_back(raw_session, tmp_path):
    session_dir, sessions = raw_session
    run = session_run_files(session_dir, "71rtask")[0]
    run_dir = tmp_path / "der" / "run-01"
    write_manifest(run_dir / "without_destriping", run.tiff)
    path = write_frame_times(run, run_dir)
    assert path == run_dir / FRAME_TIMES_FILE
    loaded = load_frame_times(run_dir)
    truth = sessions[0]["frame_pycontrol_s"]
    known = np.isfinite(loaded.time_s)
    assert known.mean() > 0.95
    assert np.abs(loaded.time_s[known] - truth[known]).max() < 1e-3
    assert loaded.frame_index.tolist() == list(range(len(truth)))
    assert loaded.tsv.name == run.tsv.name and loaded.quality["fraction_matched"] > 0.9
    with pytest.raises(FileExistsError):
        write_frame_times(run, run_dir)


def write_manifest(folder, tiff):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "source_tiff.json").write_text(json.dumps({"source_tiff": str(tiff)}))


def test_frame_times_refuse_a_run_the_batch_made_from_another_tiff(raw_session, tmp_path):
    session_dir, _ = raw_session
    first, second = session_run_files(session_dir, "71rtask")
    run_dir = tmp_path / "der" / "run-01"
    write_manifest(run_dir / "without_destriping", first.tiff)
    write_manifest(run_dir / "with_destriping", second.tiff)  # numbering shifted by a stray file
    with pytest.raises(RunMismatchError, match="71rtask_00040.tif"):
        write_frame_times(first, run_dir)
    assert not (run_dir / FRAME_TIMES_FILE).exists()


def test_frame_times_need_a_batch_record_of_the_run(raw_session, tmp_path):
    session_dir, _ = raw_session
    run_dir = tmp_path / "der" / "run-01"
    run_dir.mkdir(parents=True)
    with pytest.raises(RunMismatchError, match="source_tiff.json"):
        write_frame_times(session_run_files(session_dir, "71rtask")[0], run_dir)


def test_tiffs_are_found_whatever_the_case_of_their_extension(raw_session):
    session_dir, _ = raw_session
    tif = session_dir / "funcimg" / "71rtask_00040.tif"
    tif.rename(tif.with_suffix(".TIF"))
    assert [r.tiff.name for r in session_run_files(session_dir, "71rtask")][
        1
    ] == "71rtask_00040.TIF"


def touch_tiffs(folder, *names):
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        (folder / name).write_bytes(b"")


DESTRIPED = (
    "71rtask_00039_dynamic_bidiphase_destriped.tif",
    "71rtask_00039_dynamic_bidiphase_destriped_undistorted.tif",
)


def test_names_match_a_source_that_is_their_stem_plus_underscore():
    assert names_match_source("71rtask_00039.tif", ["71rtask_00039_a.tif", "71rtask_00039_b.tif"])
    assert not names_match_source("71rtask_00039.tif", [])
    assert not names_match_source(
        "71rtask_00039.tif", ["71rtask_00039_a.tif", "71rtask_00040_a.tif"]
    )
    assert not names_match_source("71rtask_00039.tif", ["71rtask_000390_a.tif"])  # needs the "_"


def test_a_variant_with_a_record_is_checked_by_it(tmp_path):
    write_manifest(tmp_path / "without_destriping", "/raw/71rtask_00039.tif")
    check_run_source("71rtask_00039.tif", tmp_path, variants=("without_destriping",))


def test_a_variant_without_a_record_is_accepted_by_its_tiff_names_with_a_warning(tmp_path):
    touch_tiffs(tmp_path / "with_destriping", *DESTRIPED)
    with pytest.warns(UserWarning, match="inferred from file names") as caught:
        check_run_source("71rtask_00039.tif", tmp_path, variants=("with_destriping",))
    message = str(caught[0].message)
    assert tmp_path.name in message and "with_destriping" in message and DESTRIPED[0] in message


def test_a_variant_without_a_record_and_with_another_stem_is_refused(tmp_path):
    touch_tiffs(tmp_path / "with_destriping", DESTRIPED[0], "71rtask_00040_destriped.tif")
    with pytest.raises(RunMismatchError, match="71rtask_00040_destriped.tif"):
        check_run_source("71rtask_00039.tif", tmp_path, variants=("with_destriping",))


def test_a_variant_without_a_record_or_a_tiff_is_refused(tmp_path):
    (tmp_path / "with_destriping").mkdir()
    with pytest.raises(RunMismatchError, match="source_tiff.json"):
        check_run_source("71rtask_00039.tif", tmp_path, variants=("with_destriping",))


def test_apple_double_files_are_not_processed_tiffs(tmp_path):
    touch_tiffs(tmp_path / "with_destriping", *DESTRIPED, "._" + DESTRIPED[0])
    with pytest.warns(UserWarning, match="inferred"):
        check_run_source("71rtask_00039.tif", tmp_path, variants=("with_destriping",))


def test_names_do_not_rescue_a_record_naming_another_tiff(tmp_path):
    write_manifest(tmp_path / "with_destriping", "/raw/71rtask_00040.tif")
    touch_tiffs(tmp_path / "with_destriping", *DESTRIPED)
    with pytest.raises(RunMismatchError, match="71rtask_00040.tif"):
        check_run_source("71rtask_00039.tif", tmp_path)


def test_one_variant_by_record_and_the_other_by_names(tmp_path, recwarn):
    write_manifest(tmp_path / "without_destriping", "/raw/71rtask_00039.tif")
    touch_tiffs(tmp_path / "without_destriping", "71rtask_00039_dynamic_bidiphase.tif")
    touch_tiffs(tmp_path / "with_destriping", *DESTRIPED)
    write_manifest(tmp_path, "/raw/71rtask_00038.tif")  # an older run's record at the top level
    check_run_source("71rtask_00039.tif", tmp_path)
    inferred = [w for w in recwarn if "inferred" in str(w.message)]
    assert len(inferred) == 1 and "with_destriping" in str(inferred[0].message)


def outcomes(times, pulses, phase="task"):
    return pd.DataFrame(
        {"outcome_time": times, "num_rew_pulses": pulses, "phase": [phase] * len(times)}
    )


def test_pulse_responses_average_each_cell_over_its_trials():
    n_frames, period = 400, 0.133
    frame_times = 5.0 + period * np.arange(n_frames)
    event_frames = np.array([50, 100, 150, 200, 250])
    pulses = np.array([0, 0, 2, 2, 4])
    green = np.zeros((3, n_frames))
    for frame, count in zip(event_frames, pulses, strict=True):
        green[:, frame : frame + 3] += count + 1  # a response that grows with reward size
    views = {view: green for view in RAW_VIEWS}
    table = outcomes(frame_times[event_frames] + 0.01, pulses)
    table.loc[len(table)] = [1.0, 1, "task"]  # before imaging started: dropped
    table.loc[len(table)] = [frame_times[300], 3, "port_finding"]  # not a task trial
    result = pulse_responses(views, frame_times, table, (-0.5, 1.0), phases=("task",))
    assert result["offsets"].tolist() == list(range(-4, 9))  # 0.5 s and 1.0 s at 133 ms
    assert result["n_trials"] == {0: 2, 2: 2, 4: 1}
    assert result["n_events_dropped"] == 1
    means = result["means"][("green", "dF/F0")]
    assert means[2].shape == (3, 13)
    assert means[2][0, 4] == 3.0 and means[4][0, 4] == 5.0 and means[0][0, 0] == 0.0


def test_traces_and_frame_times_of_different_lengths_are_refused():
    views = {view: np.zeros((2, 100)) for view in RAW_VIEWS}
    with pytest.raises(ValueError, match="100 frames.*120"):
        pulse_responses(views, np.arange(120.0), outcomes([10.0], [2]), (-1, 1), ("task",))


def test_z_responses_do_not_depend_on_a_cell_s_scale_and_start_at_zero():
    n_frames, period = 600, 0.133
    frame_times = 5.0 + period * np.arange(n_frames)
    event_frames = np.arange(60, 540, 60)
    rng = np.random.default_rng(0)
    shape = rng.normal(0, 1, n_frames)
    for frame in event_frames:
        shape[frame : frame + 4] += 6
    cells = np.vstack([100 + shape, 1000 + 10 * shape, np.full(n_frames, 3.0)])  # last: flat
    views = {view: cells for view in RAW_VIEWS}
    table = outcomes(frame_times[event_frames] + 0.01, np.full(len(event_frames), 2))
    result = pulse_responses(views, frame_times, table, (-0.5, 1.0), phases=("task",))
    z = result["means"][("green", "dF/F0 z")][2]
    assert np.allclose(z[0], z[1])
    assert np.isnan(z[2]).all()
    assert result["flat_cells"][("green", "dF/F0 z")] == [2]
    before = result["offsets"] < 0
    assert np.allclose(z[:2, before].mean(axis=1), 0)  # the mean of per-trial zeroed baselines
    assert set(Z_VIEWS) <= set(result["means"])
    raw = result["means"][("green", "dF/F0")][2]
    assert not np.allclose(raw[0], raw[1])
