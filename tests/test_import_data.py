import json
import os

import pandas as pd
import pytest

from src.behavior_import.import_data import import_data

TOWERS = ["A1", "B1", "C1"]


def write_tsv(folder, name, run_start=None, marker=None, start_time=None):
    """A minimal pyControl recording: optional start_time, run_start and a marker variable."""
    rows = [("0.000", "info", "subject_id", "M")]
    if start_time:
        rows.append(("0.000", "info", "start_time", start_time))
    rows.append(("0.001", "state", "", "wait_init"))  # plain-text content is not JSON
    run_start = (
        {"curr_choice_tows": TOWERS, "curr_initiation_tow": "B2"}
        if run_start is None
        else run_start
    )
    rows.append(("0.002", "variable", "run_start", json.dumps(run_start)))
    if marker is not None:
        rows.append(("1.000", "variable", "marker", json.dumps({"marker": marker})))
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["time", "type", "subtype", "content"]).to_csv(
        folder / name, sep="\t", index=False
    )
    return folder / name


def session_dir(root, subject="sub-01_id-M1", session="ses-1_date-20260101"):
    return root / subject / session


def test_sessions_are_grouped_by_subject_and_keyed_by_session_folder(tmp_path):
    write_tsv(session_dir(tmp_path), "M1-2026-01-01-100000.tsv")
    write_tsv(
        session_dir(tmp_path, "sub-02_id-M2", "ses-4_date-20260105"), "M2-2026-01-05-100000.tsv"
    )
    data = import_data(tmp_path)
    assert set(data) == {"M1", "M2"}
    assert list(data["M1"]) == ["ses-1_date-20260101"]
    assert list(data["M2"]) == ["ses-4_date-20260105"]


def test_sessions_are_ordered_by_date_then_number(tmp_path):
    for ses in ("ses-3_date-20260102", "ses-2_date-20260102", "ses-9_date-20260101"):
        write_tsv(session_dir(tmp_path, session=ses), "M1-2026-01-01-100000.tsv")
    assert list(import_data(tmp_path)["M1"]) == [
        "ses-9_date-20260101",
        "ses-2_date-20260102",
        "ses-3_date-20260102",
    ]


def test_tower_identifiers_are_read_from_the_run_start_row(tmp_path):
    write_tsv(session_dir(tmp_path), "M1-2026-01-01-100000.tsv")
    session = import_data(tmp_path)["M1"]["ses-1_date-20260101"]
    assert session["choice_towers"] == set(TOWERS)
    assert session["initiation_tower"] == "B2"
    assert session["problem"] == 1


def test_towers_fall_back_to_the_problems_table(tmp_path):
    run_start = {"num_problem": 2, "problems": {"2": ["C3", "A1", "B1", "C1"]}}
    write_tsv(session_dir(tmp_path), "M1-2026-01-01-100000.tsv", run_start=run_start)
    session = import_data(tmp_path)["M1"]["ses-1_date-20260101"]
    assert session["initiation_tower"] == "C3"
    assert session["choice_towers"] == {"A1", "B1", "C1"}

    run_start = {"num_problem": 1, "problems": {"1": ["A1", "B1", "C1"]}}
    write_tsv(
        session_dir(tmp_path, session="ses-2_date-20260102"),
        "M1-2026-01-02-100000.tsv",
        run_start=run_start,
    )
    session = import_data(tmp_path)["M1"]["ses-2_date-20260102"]
    assert session["initiation_tower"] == "B2"  # default when three towers are listed
    assert session["choice_towers"] == {"A1", "B1", "C1"}


def test_data_has_json_expanded_columns_and_df_keeps_the_raw_four(tmp_path):
    write_tsv(session_dir(tmp_path), "M1-2026-01-01-100000.tsv", marker=7)
    session = import_data(tmp_path)["M1"]["ses-1_date-20260101"]
    assert list(session["df"].columns) == ["time", "type", "subtype", "content"]
    assert {"marker", "curr_initiation_tow"} <= set(session["data"].columns)
    assert session["data"]["marker"].dropna().tolist() == [7]
    assert session["data"] is not session["df"]


def test_one_recording_is_a_dataframe_and_several_are_a_chronological_list(tmp_path):
    folder = session_dir(tmp_path)
    write_tsv(folder, "M1-2026-01-01-120000.tsv", marker=2)
    assert isinstance(import_data(tmp_path)["M1"]["ses-1_date-20260101"]["data"], pd.DataFrame)

    write_tsv(folder, "M1-2026-01-01-100000.tsv", marker=1)
    session = import_data(tmp_path)["M1"]["ses-1_date-20260101"]
    assert [d["marker"].dropna().iloc[0] for d in session["data"]] == [1, 2]
    assert len(session["df"]) == 2 and all(isinstance(d, pd.DataFrame) for d in session["df"])


def test_problem_and_towers_come_from_the_earliest_recording(tmp_path):
    folder = session_dir(tmp_path)
    write_tsv(folder, "M1-2026-01-01-100000.tsv", run_start={"curr_choice_tows": ["X1"]})
    write_tsv(folder, "M1-2026-01-01-120000.tsv", run_start={"curr_choice_tows": ["Y1"]})
    assert import_data(tmp_path)["M1"]["ses-1_date-20260101"]["choice_towers"] == {"X1"}


def test_recording_without_a_filename_timestamp_is_placed_by_its_start_time(tmp_path):
    folder = session_dir(tmp_path)
    write_tsv(folder, "M1-2026-01-01-120000.tsv", marker=2)
    write_tsv(folder, "recording.tsv", marker=1, start_time="2026-01-01T11:00:00.000")
    session = import_data(tmp_path)["M1"]["ses-1_date-20260101"]
    assert [d["marker"].dropna().iloc[0] for d in session["data"]] == [1, 2]


def test_recording_without_any_timestamp_is_placed_by_file_modification_time(tmp_path):
    folder = session_dir(tmp_path)
    late = write_tsv(folder, "M1-2026-01-01-120000.tsv", marker=2)
    early = write_tsv(folder, "recording.tsv", marker=1)
    os.utime(early, (1, 1))  # 1970: before the 2026 timestamp in the other file name
    os.utime(late, (2, 2))
    session = import_data(tmp_path)["M1"]["ses-1_date-20260101"]
    assert [d["marker"].dropna().iloc[0] for d in session["data"]] == [1, 2]


def test_unreadable_file_and_files_outside_a_subject_folder_are_skipped_with_a_warning(
    tmp_path, capsys
):
    write_tsv(session_dir(tmp_path), "M1-2026-01-01-100000.tsv")
    (session_dir(tmp_path) / "empty.tsv").write_text("")
    write_tsv(tmp_path / "loose" / "ses-1_date-20260101", "x-2026-01-01-100000.tsv")
    data = import_data(tmp_path)
    out = capsys.readouterr().out
    assert list(data) == ["M1"] and len(data["M1"]["ses-1_date-20260101"]["df"]) > 1
    assert "[WARN] Failed to read" in out and "empty.tsv" in out
    assert "[WARN] Skipping" in out and "no subject_id parsed" in out


def test_prints_how_many_subjects_and_sessions_were_processed(tmp_path, capsys):
    write_tsv(session_dir(tmp_path), "M1-2026-01-01-100000.tsv")
    write_tsv(session_dir(tmp_path, session="ses-2_date-20260102"), "M1-2026-01-02-100000.tsv")
    write_tsv(session_dir(tmp_path, "sub-02_id-M2"), "M2-2026-01-01-100000.tsv")
    import_data(tmp_path)
    assert "[INFO] Processed 2 subjects(s), 3 session(s)." in capsys.readouterr().out


def test_empty_folder_gives_no_subjects(tmp_path):
    assert import_data(tmp_path) == {}


def test_accepts_str_and_path_roots(tmp_path):
    write_tsv(session_dir(tmp_path), "M1-2026-01-01-100000.tsv")
    assert list(import_data(str(tmp_path))) == list(import_data(tmp_path)) == ["M1"]


def test_recording_without_a_run_start_row_raises(tmp_path):
    path = write_tsv(session_dir(tmp_path), "M1-2026-01-01-100000.tsv")
    path.write_text("time\ttype\tsubtype\tcontent\n0.0\tinfo\tsubject_id\tM1\n")
    with pytest.raises(IndexError):
        import_data(tmp_path)
