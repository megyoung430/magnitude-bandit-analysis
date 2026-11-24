import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("h5py")  # the optional funcimg extra

from src.funcimg_analysis.frame_times import (
    FRAME_TIMES_FILE,
    RunMismatchError,
    batch_source_tiffs,
    check_frame_times_belong,
    check_run_source,
    load_frame_times,
    names_match_source,
    processed_tiff_names,
    same_tiff_name,
    write_frame_times,
)
from src.funcimg_analysis.run_responses import (
    BACKGROUND_MEASURES,
    MEASURES,
    RAW_VIEWS,
    Z_VIEWS,
    check_measures,
    pulse_responses,
)
from src.funcimg_import.runs import (
    aligned_run_files,
    alignment_table,
    numbered_sessions,
    pair_by_acquisition,
    pair_by_frame_count,
    session_run_alignments,
    session_run_files,
    subject_run_alignments,
    sync_matches,
    tiff_info,
)
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


def test_names_match_a_source_followed_by_a_known_processing_suffix():
    stem = "71rtask_00039"
    good = [
        f"{stem}_dynamic_bidiphase.tif",
        f"{stem}_dynamic_bidiphase_undistorted.tif",
        f"{stem}_dynamic_bidiphase_destriped.tif",
        f"{stem}_dynamic_bidiphase_destriped_undistorted.TIF",
        f"{stem}_dynamic_bidiphase.tiff",
    ]
    assert names_match_source("71rtask_00039.tif", good)
    assert not names_match_source("71rtask_00039.tif", [])
    assert not names_match_source(
        "71rtask_00039.tif", [good[0], "71rtask_00040_dynamic_bidiphase.tif"]
    )
    assert not names_match_source("71rtask_00039.tif", ["71rtask_000390_dynamic_bidiphase.tif"])


@pytest.mark.parametrize(
    "name",
    [
        "71rtask_00039_retry_dynamic_bidiphase.tif",
        "71rtask_00039_00002_dynamic_bidiphase.tif",
        "71rtask_00039_a.tif",
        "71rtask_00039_dynamic_bidiphase_other.tif",
    ],
)
def test_names_with_something_else_after_the_stem_are_not_accepted(name):
    assert not names_match_source("71rtask_00039.tif", [name])


def test_tiff_and_capitalised_extensions_count_as_processed_tiffs(tmp_path):
    touch_tiffs(tmp_path, "a.tiff", "b.TIF", "c.TIFF", "d.tif.json", "._e.tif", "f.png")
    assert processed_tiff_names(tmp_path) == ["a.tiff", "b.TIF", "c.TIFF"]


def test_a_stray_tiff_extension_variant_is_not_overlooked(tmp_path):
    touch_tiffs(tmp_path / "with_destriping", *DESTRIPED, "71rtask_00040_dynamic_bidiphase.tiff")
    with pytest.raises(RunMismatchError, match="71rtask_00040"):
        check_run_source("71rtask_00039.tif", tmp_path, variants=("with_destriping",))


def test_a_record_in_any_checked_variant_folder_is_read(tmp_path):
    write_manifest(tmp_path / "other_variant", "/raw/71rtask_00040.tif")
    touch_tiffs(tmp_path / "other_variant", "71rtask_00039_dynamic_bidiphase.tif")
    with pytest.raises(RunMismatchError, match="71rtask_00040.tif"):
        check_run_source("71rtask_00039.tif", tmp_path, variants=("other_variant",))
    assert batch_source_tiffs(tmp_path, variants=("other_variant",)) == {
        "other_variant": "71rtask_00040.tif"
    }


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
    touch_tiffs(
        tmp_path / "with_destriping", DESTRIPED[0], "71rtask_00040_dynamic_bidiphase_destriped.tif"
    )
    with pytest.raises(RunMismatchError, match="71rtask_00040_dynamic_bidiphase_destriped.tif"):
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


def z_test_session(event_shift_s):
    n_frames, period = 600, 0.133
    frame_times = 5.0 + period * np.arange(n_frames)
    event_frames = np.arange(60, 540, 60)
    trace = np.random.default_rng(0).normal(0, 1, n_frames)
    trace[event_frames - 1] += 50  # the frame before the event's nearest frame
    table = outcomes(frame_times[event_frames] + event_shift_s, np.full(len(event_frames), 2))
    return {view: trace[None, :] for view in RAW_VIEWS}, frame_times, table


def test_the_frame_holding_an_early_event_is_not_part_of_the_baseline():
    views, times, table = z_test_session(-0.05)  # the event falls inside frame -1
    result = pulse_responses(views, times, table, (-0.5, 1.0), ("task",))
    z = result["means"][("green", "F z")][2][0]
    far = (result["offsets"] >= -3) & (result["offsets"] <= -2)
    assert abs(z[far].mean()) < 1
    assert z[result["offsets"] == -1][0] > 3  # the bump is visible against the clean baseline


def test_a_window_starting_at_zero_gives_raw_views_and_refuses_z_views():
    views, times, table = z_test_session(0.01)
    result = pulse_responses(views, times, table, (0.0, 1.0), ("task",), measures=("F", "dF/F0"))
    assert set(result["means"]) == set(RAW_VIEWS) and result["flat_cells"] == {}
    with pytest.raises(ValueError, match="pre-event"):
        pulse_responses(views, times, table, (0.0, 1.0), ("task",))
    with pytest.raises(ValueError, match="pre-event"):
        pulse_responses(views, times, table, (0.0, 1.0), ("task",), measures=("F z",))


def test_only_the_requested_measures_are_computed():
    views, times, table = z_test_session(0.01)
    result = pulse_responses(views, times, table, (-0.5, 1.0), ("task",), measures=("dF/F0 z",))
    assert set(result["means"]) == {("green", "dF/F0 z"), ("red", "dF/F0 z")}
    assert list(result["flat_cells"]) == [("green", "dF/F0 z"), ("red", "dF/F0 z")]


@pytest.mark.parametrize("measures", [("dF/Fo",), ("f z", "x"), ("F z", "dF/Fo"), ("",)])
def test_unknown_measures_are_refused_listing_the_valid_names(measures):
    views, times, table = z_test_session(0.01)
    with pytest.raises(ValueError, match="unknown measure") as caught:
        pulse_responses(views, times, table, (-0.5, 1.0), ("task",), measures=measures)
    message = str(caught.value)
    assert all(f"'{name}'" in message.split("valid")[1] for name in MEASURES)
    assert all(repr(m) in message.split("valid")[0] for m in measures if m not in MEASURES)
    assert all(repr(m) not in message.split("valid")[0] for m in measures if m in MEASURES)


def test_a_partly_wrong_measure_set_names_only_the_wrong_one():
    with pytest.raises(ValueError, match=r"unknown measure\(s\) \['dF/Fo'\]"):
        check_measures(("F z", "dF/Fo"))


def test_no_measures_and_a_bare_string_are_refused():
    with pytest.raises(ValueError, match="at least one"):
        check_measures(())
    with pytest.raises(TypeError, match="tuple"):
        check_measures("F z")


def test_valid_measures_pass_and_the_background_set_is_narrower():
    check_measures(MEASURES)
    check_measures(("F", "F z"), BACKGROUND_MEASURES)
    with pytest.raises(ValueError, match="unknown measure"):
        check_measures(("dF/F0",), BACKGROUND_MEASURES)


@pytest.mark.parametrize("window", [(0.0, 1.0), (-0.13, 1.0)])  # 0 and 1 pre-event frames
def test_a_z_measure_needs_at_least_two_pre_event_frames(window):
    views, times, table = z_test_session(-0.05)
    with pytest.raises(ValueError, match="at least 2 pre-event frames"):
        pulse_responses(views, times, table, window, ("task",), measures=("F z",))
    raw = pulse_responses(views, times, table, window, ("task",), measures=("F",))
    assert raw["n_trials"] == {2: 8} and raw["n_trials_no_baseline"] == 0


def test_the_per_run_result_reports_trials_without_a_baseline_and_counts_the_rest():
    views, times, table = z_test_session(-0.05)
    result = pulse_responses(views, times, table, (-0.27, 1.0), ("task",))  # 2 pre-event frames
    assert result["n_trials_no_baseline"] == 0 and result["n_trials"] == {2: 8}
    assert result["n_events_dropped"] == 0


def mini2p_processed_names(stem):
    """The four names mini2p's own naming functions give a run's processed TIFFs."""
    from mini2p_preprocessing.config import (
        DistortionSettings,
        PipelineSettings,
        corrected_tiff_path,
        undistorted_tiff_path,
        with_destriping,
    )

    base = PipelineSettings(distortion=DistortionSettings(mode="off"))
    names = []
    for destripe in (False, True):
        settings = with_destriping(base, destripe)
        for path_for in (corrected_tiff_path, undistorted_tiff_path):
            names.append(path_for(Path("out"), Path(f"/raw/{stem}.tif"), settings).name)
    return names


def test_the_name_pattern_accepts_exactly_the_names_mini2p_writes():
    pytest.importorskip("mini2p_preprocessing")
    names = mini2p_processed_names("71rtask_00039")
    assert len(set(names)) == 4
    for name in names:
        assert names_match_source("71rtask_00039.tif", [name]), name
        assert not names_match_source("71rtask_00040.tif", [name]), name


def test_stems_are_compared_exactly_but_extensions_ignore_case():
    assert names_match_source("71rtask_00039.tif", ["71rtask_00039_dynamic_bidiphase.TIF"])
    assert names_match_source("71rtask_00039.TIF", ["71rtask_00039_dynamic_bidiphase.tiff"])
    assert not names_match_source("71rtask_00039.tif", ["71RTASK_00039_dynamic_bidiphase.tif"])
    assert not names_match_source("71rtask_00039.tif", ["71rtask_00039_Dynamic_Bidiphase.tif"])


def test_the_batch_record_and_the_file_names_agree_on_case():
    assert same_tiff_name("71rtask_00039.TIF", "71rtask_00039.tif")
    assert not same_tiff_name("71RTASK_00039.tif", "71rtask_00039.tif")
    assert not same_tiff_name("71rtask_00040.tif", "71rtask_00039.tif")


def test_a_record_that_differs_only_in_the_extension_case_is_accepted(tmp_path):
    write_manifest(tmp_path / "without_destriping", "/raw/71rtask_00039.TIF")
    check_run_source("71rtask_00039.tif", tmp_path, variants=("without_destriping",))


def test_a_record_that_differs_in_the_stem_case_is_refused(tmp_path):
    write_manifest(tmp_path / "without_destriping", "/raw/71RTASK_00039.tif")
    with pytest.raises(RunMismatchError, match="71RTASK_00039.tif"):
        check_run_source("71rtask_00039.tif", tmp_path, variants=("without_destriping",))


# ------------------------------------------------- which TIFFs can be aligned (batch listing)


def write_scanimage_tiff(path, timepoints, start, n_channels=2):
    """A tiny TIFF laid out as ScanImage writes it: one page per channel and timepoint,
    each with its header, the first holding the acquisition start (``epoch``)."""
    import tifffile

    epoch = f"epoch = [{start.year} {start.month:2d} {start.day:2d} {start.hour} {start.minute} "
    epoch += f"{start.second + start.microsecond / 1e6:.3f}]"
    with tifffile.TiffWriter(path, bigtiff=True) as tiff:
        for page in range(timepoints * n_channels):
            header = f"frameNumbers = {page // n_channels + 1}\n{epoch}\n"
            header += f"frameTimestamps_sec = {page // n_channels * 0.133:.9f}"
            tiff.write(np.zeros((4, 4), np.int16), description=header, contiguous=False)


@pytest.fixture
def recorded_session(tmp_path):
    """ses-220-like: pre-task and task recordings interleaved, each with its DAQ file; the
    pre-task ones have equal lengths and no behavior. Names do not give the order (the pre
    counter is higher), and the behavior files are named in reverse order."""
    session_dir = tmp_path / "raw" / "ses-220_date-20260803"
    (session_dir / "funcimg").mkdir(parents=True)
    (session_dir / "behav").mkdir()
    task = [
        daq_session(np.random.default_rng(seed), n_frames=n) for seed, n in ((5, 900), (6, 700))
    ]
    pre = daq_session(np.random.default_rng(9), n_frames=300)
    recordings = [("71rpre_00029", pre), ("71rtask_00022", task[0]),
                  ("71rpre_00030", pre), ("71rtask_00023", task[1])]  # fmt: skip
    for k, (stem, session) in enumerate(recordings):
        start = datetime(2026, 8, 3, 11, 10 + 10 * k, 21, 750000)
        write_scanimage_tiff(
            session_dir / "funcimg" / f"{stem}.tif", len(session["frame_s"]), start
        )
        write_daq(session_dir / "funcimg" / f"file_{1799 + k:05d}.h5", session)
    for k, session in enumerate(task):
        write_tsv(
            session_dir / "behav" / f"M-2026-08-03-1{2 - k}0000.tsv", session["pycontrol_rsync_s"]
        )
    return session_dir


def test_task_recordings_are_numbered_and_pre_task_ones_are_not(recorded_session):
    rows = session_run_alignments(recorded_session, 2)
    assert [(r.tiff.stem, r.run_name, r.aligned) for r in rows] == [
        ("71rpre_00029", "", False),
        ("71rtask_00022", "run-01", True),
        ("71rpre_00030", "", False),
        ("71rtask_00023", "run-02", True),
    ]
    assert [r.h5.name for r in rows] == [f"file_{1799 + k:05d}.h5" for k in range(4)]
    assert rows[1].tsv.name == "M-2026-08-03-120000.tsv"
    assert rows[3].tsv.name == "M-2026-08-03-110000.tsv"
    assert "no behavior file" in rows[0].reason
    assert (rows[1].timepoints, rows[1].daq_frames) == (900, 900)


def test_a_missing_daq_file_costs_only_its_own_recording(recorded_session):
    (recorded_session / "funcimg" / "file_01799.h5").unlink()  # the first pre-task one
    rows = session_run_alignments(recorded_session, 2)
    assert [(r.tiff.stem, r.run_name) for r in rows if r.aligned] == [
        ("71rtask_00022", "run-01"),
        ("71rtask_00023", "run-02"),
    ]
    assert rows[0].h5 is None and "no DAQ file with 300 frames" in rows[0].reason
    assert rows[2].h5.name == "file_01801.h5"  # not the first pre-task recording's place


def test_a_missing_tiff_leaves_the_other_pairs_right(recorded_session):
    (recorded_session / "funcimg" / "71rtask_00022.tif").unlink()
    rows = session_run_alignments(recorded_session, 2)
    assert [(r.tiff.stem, r.h5.name, r.run_name) for r in rows] == [
        ("71rpre_00029", "file_01799.h5", ""),
        ("71rpre_00030", "file_01801.h5", ""),
        ("71rtask_00023", "file_01802.h5", "run-01"),
    ]


def test_a_missing_behavior_file_renumbers_the_later_runs(recorded_session):
    (recorded_session / "behav" / "M-2026-08-03-120000.tsv").unlink()  # run-01's behavior
    rows = session_run_alignments(recorded_session, 2)
    assert [(r.tiff.stem, r.run_name) for r in rows if r.aligned] == [("71rtask_00023", "run-01")]
    assert "no behavior file" in rows[1].reason


def test_a_daq_file_without_usable_lines_is_reported_and_the_others_kept(recorded_session):
    import h5py

    with h5py.File(recorded_session / "funcimg" / "file_01800.h5", "w") as broken:
        broken.attrs["samplerate"] = 5000.0  # no FrameClock or PyControlPulse lines
    rows = session_run_alignments(recorded_session, 2)
    assert not rows[1].aligned and "without a frame count: file_01800.h5 has no" in rows[1].reason
    assert [(r.tiff.stem, r.run_name) for r in rows if r.aligned] == [("71rtask_00023", "run-01")]


def test_a_daq_file_that_cannot_be_opened_stops_the_listing(recorded_session):
    (recorded_session / "funcimg" / "file_01800.h5").write_bytes(b"not hdf5")
    with pytest.raises(OSError):
        session_run_alignments(recorded_session, 2)


def test_a_tiff_without_a_scanimage_header_costs_only_itself(recorded_session):
    import tifffile

    tifffile.imwrite(
        recorded_session / "funcimg" / "stray.tif",
        np.zeros((3, 4, 4), np.int16),
        photometric="minisblack",
    )
    rows = session_run_alignments(recorded_session, 2)
    assert [r.run_name for r in rows if r.aligned] == ["run-01", "run-02"]
    assert rows[-1].tiff.name == "stray.tif"
    assert "stray.tif: no ScanImage epoch" in rows[-1].reason


def test_without_a_channel_count_it_is_read_from_scanimage_s_header(recorded_session):
    rows = session_run_alignments(recorded_session)  # the test TIFFs have no SI metadata
    assert not any(r.aligned for r in rows)
    assert all("no SI.hChannels.channelSave" in r.reason for r in rows)


def test_a_flat_rsync_line_still_pairs_the_recording_by_its_frames(recorded_session):
    import h5py

    with h5py.File(recorded_session / "funcimg" / "file_01799.h5", "r+") as daq:
        daq["PyControlPulse"][:] = 0.0  # pyControl idle during the pre-task recording
    rows = session_run_alignments(recorded_session, 2)
    assert (rows[0].h5.name, rows[0].daq_frames) == ("file_01799.h5", 300)
    assert "PyControlPulse" in rows[0].reason
    assert [r.run_name for r in rows if r.aligned] == ["run-01", "run-02"]


def test_equal_length_recordings_beside_a_missing_daq_file_are_not_used(tmp_path):
    session_dir = tmp_path / "ses-001_date-x"
    (session_dir / "funcimg").mkdir(parents=True)
    (session_dir / "behav").mkdir()
    sessions = [daq_session(np.random.default_rng(seed), n_frames=400) for seed in (3, 4)]
    for k, session in enumerate(sessions):
        start = datetime(2026, 8, 3, 11, 10 + 10 * k)
        write_scanimage_tiff(session_dir / "funcimg" / f"t_{k:05d}.tif", 400, start)
        write_tsv(session_dir / "behav" / f"M-{k}.tsv", session["pycontrol_rsync_s"])
    write_daq(session_dir / "funcimg" / "file_00002.h5", sessions[1])  # the first is missing
    rows = session_run_alignments(session_dir, 2)
    assert not any(r.aligned for r in rows)
    assert "ambiguous" in rows[0].reason


def test_tiff_info_reads_the_start_and_counts_timepoints(tmp_path):
    start = datetime(2026, 8, 3, 11, 35, 21, 750000)
    write_scanimage_tiff(tmp_path / "a.tif", 123, start)
    info = tiff_info(tmp_path / "a.tif", n_channels=2)
    assert (info.start, info.timepoints) == (start, 123)


@pytest.mark.parametrize(
    "tiffs, daqs, expected, ambiguous",
    [
        ([300, 900, 300, 700], [300, 900, 300, 700], [0, 1, 2, 3], set()),
        ([300, 900, 300, 700], [900, 300, 700], [None, 0, 1, 2], set()),  # first DAQ missing
        ([300, 300, 700], [300, 900, 300, 700], [0, 2, 3], set()),  # a TIFF missing
        ([901, 699], [900, 702], [0, 1], set()),  # within the tolerance
        ([900], [910], [None], set()),
        ([], [300], [], set()),
        ([300, 700], [None, 300, 700], [1, 2], set()),  # a DAQ file without a frame count
        ([15000, 15000], [15000], [0, None], {0}),  # capped recordings, one DAQ file missing
        ([4907], [4905, 4907], [0], {0}),  # two DAQ files could be its own
    ],
)
def test_pairing_keeps_order_needs_agreeing_counts_and_flags_ambiguity(
    tiffs, daqs, expected, ambiguous
):
    assert pair_by_frame_count(tiffs, daqs) == (expected, ambiguous)


def test_a_session_without_a_behav_folder_has_no_aligned_runs(recorded_session):
    for tsv in (recorded_session / "behav").iterdir():
        tsv.unlink()
    (recorded_session / "behav").rmdir()
    assert not any(r.aligned for r in session_run_alignments(recorded_session, 2))


def test_two_daq_files_claiming_one_behavior_file_are_both_refused():
    pulses = np.cumsum(np.random.default_rng(1).uniform(0.6, 4.5, 200))
    matches = sync_matches([pulses, pulses + 5.0], [pulses], ["M-120000.tsv"])
    assert all(isinstance(m, str) and "same behavior file M-120000.tsv" in m for m in matches)


def test_tiffs_with_the_same_counter_are_ordered_by_name_as_in_mini2p(tmp_path):
    names = ["B_00001.tif", "a_00001.tif", "c_00000.tif"]
    assert [p.name for p in pair_by_acquisition([tmp_path / n for n in names])] == [
        "c_00000.tif",
        "a_00001.tif",
        "B_00001.tif",
    ]


def test_the_frame_time_step_gets_the_aligned_runs_and_the_reasons(recorded_session):
    runs, unaligned = aligned_run_files(recorded_session, 2)
    assert [(r.run_name, r.tiff.stem, r.h5.name) for r in runs] == [
        ("run-01", "71rtask_00022", "file_01800.h5"),
        ("run-02", "71rtask_00023", "file_01802.h5"),
    ]
    assert [a.tiff.stem for a in unaligned] == ["71rpre_00029", "71rpre_00030"]


def test_saved_frame_times_of_another_recording_are_refused(recorded_session, tmp_path):
    first, second = aligned_run_files(recorded_session, 2)[0]
    run_dir = tmp_path / "der" / "run-01"
    write_manifest(run_dir / "with_distortion_correction" / "with_destriping", second.tiff)
    write_frame_times(second, run_dir)  # e.g. left from an older numbering
    with pytest.raises(RunMismatchError, match="71rtask_00023.tif"):
        check_frame_times_belong(load_frame_times(run_dir), first)
    check_frame_times_belong(load_frame_times(run_dir), second)


def test_frame_times_of_an_aligned_run_match_its_recording(recorded_session, tmp_path):
    run = aligned_run_files(recorded_session, 2)[0][1]  # run-02, the 700-frame recording
    run_dir = tmp_path / "der" / "run-02"
    write_manifest(run_dir / "with_distortion_correction" / "with_destriping", run.tiff)
    write_frame_times(run, run_dir)
    assert len(load_frame_times(run_dir).time_s) == 700


def test_an_empty_qc_folder_beside_a_recorded_variant_is_passed_over(raw_session, tmp_path):
    session_dir, _ = raw_session
    run = session_run_files(session_dir, "71rtask")[0]
    run_dir = tmp_path / "der" / "run-01"
    write_manifest(run_dir / "without_destriping", run.tiff)
    (run_dir / "with_distortion_correction" / "with_destriping").mkdir(parents=True)  # empty
    assert write_frame_times(run, run_dir) == run_dir / FRAME_TIMES_FILE


def test_only_numbered_sessions_with_imaging_are_listed(recorded_session):
    subject = recorded_session.parent
    (subject / "ses-test_date-20260101" / "funcimg").mkdir(parents=True)
    (subject / "ses-test_date-20260101" / "funcimg" / "stray.tif").write_bytes(b"")
    (subject / "ses-002_date-20260102" / "behav").mkdir(parents=True)  # no imaging
    rows = subject_run_alignments(subject, 2)
    assert {r.session_dir.name for r in rows} == {recorded_session.name}


def test_sessions_are_listed_in_number_order(tmp_path):
    for name in ("ses-010_date-2", "ses-009_date-1", "ses-100_date-3", "ses-pre_date-0"):
        (tmp_path / name).mkdir()
    assert [d.name for d in numbered_sessions(tmp_path)] == [
        "ses-009_date-1",
        "ses-010_date-2",
        "ses-100_date-3",
    ]


def test_the_table_has_one_row_per_tiff_with_its_evidence(recorded_session):
    table = alignment_table(session_run_alignments(recorded_session, 2))
    assert list(table.columns) == [
        "session", "run", "tiff", "h5", "tsv", "aligned", "reason",
        "tiff_start", "tiff_timepoints", "daq_frames",
    ]  # fmt: skip
    assert table["run"].tolist() == ["", "run-01", "", "run-02"]
    assert table["tiff_timepoints"].tolist() == [300, 900, 300, 700]
    assert table.loc[0, "tsv"] == "" and Path(table.loc[1, "tsv"]).is_absolute()
    assert table.loc[1, "tiff_start"] == "2026-08-03T11:20:21.750000"


def test_frame_times_accept_a_run_in_the_distortion_folder_layout(raw_session, tmp_path):
    session_dir, _ = raw_session
    run = session_run_files(session_dir, "71rtask")[0]
    run_dir = tmp_path / "der" / "run-01"
    write_manifest(run_dir / "with_distortion_correction" / "with_destriping", run.tiff)
    assert write_frame_times(run, run_dir) == run_dir / FRAME_TIMES_FILE


def test_the_distortion_folder_layout_still_refuses_another_tiff(raw_session, tmp_path):
    session_dir, _ = raw_session
    first, second = session_run_files(session_dir, "71rtask")
    run_dir = tmp_path / "der" / "run-01"
    write_manifest(run_dir / "with_distortion_correction" / "with_destriping", second.tiff)
    with pytest.raises(RunMismatchError, match="71rtask_00040.tif"):
        write_frame_times(first, run_dir)


def test_the_listing_script_writes_the_table_and_refuses_to_replace_it(recorded_session, tmp_path):
    from scripts.funcimg.list_aligned_runs import main

    out = tmp_path / "aligned_runs.csv"
    main([str(recorded_session.parent), str(out), "--channels", "2"])
    table = pd.read_csv(out, keep_default_na=False)
    assert table["run"].tolist() == ["", "run-01", "", "run-02"]
    with pytest.raises(SystemExit, match="exists"):
        main([str(recorded_session.parent), str(out), "--channels", "2"])
    main([str(recorded_session.parent), str(out), "--overwrite", "--channels", "2"])


def test_the_listing_script_refuses_a_subject_without_tiffs(tmp_path):
    from scripts.funcimg.list_aligned_runs import main

    (tmp_path / "ses-001_date-x" / "funcimg").mkdir(parents=True)
    with pytest.raises(SystemExit, match="No TIFF"):
        main([str(tmp_path), str(tmp_path / "out.csv")])


def test_a_video_mismatch_names_the_run_not_its_position():
    from src.funcimg_import.runs import match_videos

    with pytest.raises(ValueError, match="run-02 matches no camera table"):
        match_videos({}, [np.arange(10.0)], ["run-02"])
