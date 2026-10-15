import json
import os
import warnings

import pandas as pd
import pytest

from src.behavior_import.extract_trials import group_sessions_by_problem
from src.behavior_import.import_data import import_data

TOWERS = ["A1", "B1", "C1"]
# A first session of a layout: has num_t_found and starts with found_ports false.
RUN_START = {
    "curr_choice_tows": TOWERS,
    "curr_initiation_tow": "B2",
    "num_t_found": 0,
    "found_ports": False,
}


def run_start_for(towers, initiation="B2", **extra):
    return {
        "curr_choice_tows": towers,
        "curr_initiation_tow": initiation,
        "num_t_found": 0,
        "found_ports": False,
        **extra,
    }


def write_tsv(
    folder, name, run_start=None, marker=None, start_time=None, task_name=None, port_finding=False
):
    """A minimal pyControl recording: optional task_name, start_time, run_start, marker."""
    rows = [("0.000", "info", "subject_id", "M")]
    if task_name:
        rows.append(("0.000", "info", "task_name", task_name))
    if start_time:
        rows.append(("0.000", "info", "start_time", start_time))
    rows.append(("0.001", "state", "", "wait_init"))  # plain-text content is not JSON
    run_start = RUN_START if run_start is None else run_start
    content = run_start if isinstance(run_start, str) else json.dumps(run_start)
    rows.append(("0.002", "variable", "run_start", content))
    if port_finding:
        rows.append(("0.500", "variable", "print", json.dumps({"num_t_found": 1})))
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
    with pytest.raises(ValueError, match=r"M1 ses-1_date-20260101.*M1-2026-01-01.*no run_start"):
        import_data(tmp_path)


@pytest.mark.parametrize("content", ["{not json", ""])
def test_corrupt_or_empty_first_run_start_raises_naming_subject_session_and_file(tmp_path, content):
    write_tsv(session_dir(tmp_path), "M1-2026-01-01-100000.tsv", run_start=content)
    with pytest.raises(ValueError, match=r"M1 ses-1_date-20260101 .*M1-2026-01-01-100000.tsv"):
        import_data(tmp_path)


def import_sessions(root, run_starts):
    """Write one session per run_start dict (ses-1, ses-2, ...) and import them.

    An item may also be ``(run_start, task_name, port_finding)``.
    """
    for n, item in enumerate(run_starts, start=1):
        run_start, task_name, port_finding = (
            item if isinstance(item, tuple) else (item, None, False)
        )
        write_tsv(
            session_dir(root, session=f"ses-{n}_date-202601{n:02d}"),
            f"M1-2026-01-{n:02d}-100000.tsv",
            run_start=run_start,
            task_name=task_name,
            port_finding=port_finding,
        )
    return import_data(root)["M1"]


A = run_start_for(["A1", "B1", "C1"])
B = run_start_for(["A2", "B2", "C2"])
C = run_start_for(["A3", "B3", "C3"])


def problems(sessions):
    return [s["problem"] for s in sessions.values()]


def test_problem_number_increments_when_the_choice_towers_change(tmp_path):
    sessions = import_sessions(tmp_path, [A, B, B])
    assert sessions["ses-1_date-20260101"]["choice_towers"] == {"A1", "B1", "C1"}
    assert sessions["ses-2_date-20260102"]["choice_towers"] == {"A2", "B2", "C2"}
    assert problems(sessions) == [1, 2, 2]


def test_problem_number_ignores_tower_order(tmp_path):
    reordered = run_start_for(["C1", "A1", "B1"])
    assert problems(import_sessions(tmp_path, [A, reordered])) == [1, 1]


def test_changing_only_the_initiation_tower_starts_a_new_problem(tmp_path):
    other = run_start_for(["A1", "B1", "C1"], initiation="A2")
    assert problems(import_sessions(tmp_path, [A, other, other])) == [1, 2, 2]


def test_a_return_to_an_earlier_layout_is_a_new_problem(tmp_path):
    assert problems(import_sessions(tmp_path, [A, B, B, A])) == [1, 2, 2, 3]


def test_a_layout_lasting_one_session_is_flagged_not_numbered(tmp_path):
    sessions = import_sessions(tmp_path, [A, B, C, C])
    assert problems(sessions) == [1, None, 2, 2]
    assert [s["problem_flag"] for s in sessions.values()] == [None, "suspected_settings_error"] + [
        None,
        None,
    ]


def test_flagged_session_between_equal_layouts_keeps_the_original_problem(tmp_path):
    with pytest.warns(UserWarning, match=r"M1 ses-2_date-20260102.*suspected settings error"):
        sessions = import_sessions(tmp_path, [A, B, A])
    assert problems(sessions) == [1, None, 1]
    assert "ses-2_date-20260102" in sessions  # kept for single-session analyses


def test_changed_session_without_num_t_found_is_flagged(tmp_path):
    no_num_t_found = {k: v for k, v in B.items() if k != "num_t_found"}
    sessions = import_sessions(tmp_path, [A, no_num_t_found, B, B])
    assert problems(sessions) == [1, None, 2, 2]
    assert sessions["ses-2_date-20260102"]["settings_evidence"]["has_num_t_found"] is False


def test_num_t_found_in_a_printed_trial_counts(tmp_path):
    no_key = {k: v for k, v in B.items() if k != "num_t_found"}
    write_tsv(session_dir(tmp_path), "M1-2026-01-01-100000.tsv", run_start=A)
    path = write_tsv(
        session_dir(tmp_path, session="ses-2_date-20260102"),
        "M1-2026-01-02-100000.tsv",
        run_start=no_key,
    )
    with path.open("a") as f:
        f.write('1.5\tvariable\tprint\t{"num_t_found": 3}\n')
    write_tsv(
        session_dir(tmp_path, session="ses-3_date-20260103"),
        "M1-2026-01-03-100000.tsv",
        run_start=B,
    )
    assert problems(import_data(tmp_path)["M1"]) == [1, 2, 2]


@pytest.mark.parametrize("found_ports", [True, None, "missing"])
def test_changed_session_not_starting_with_found_ports_false_is_flagged(tmp_path, found_ports):
    changed = {**B, "found_ports": found_ports}
    if found_ports == "missing":
        del changed["found_ports"]
    assert problems(import_sessions(tmp_path, [A, changed, B, B])) == [1, None, 2, 2]


def test_evidence_is_not_checked_for_sessions_that_keep_the_layout(tmp_path):
    later = {**A, "found_ports": True}
    assert problems(import_sessions(tmp_path, [A, later, later])) == [1, 1, 1]


NO_TOWERS = {"num_t_found": 0, "found_ports": False}


def flags_of(sessions):
    return [s["problem_flag"] for s in sessions.values()]


def test_task_name_is_stored_on_each_session(tmp_path):
    sessions = import_sessions(tmp_path, [(A, "Meg\\3x3_field_drifting_value_bandit", False), A])
    first, second = sessions.values()
    assert first["task_name"] == "Meg\\3x3_field_drifting_value_bandit"
    assert first["settings_evidence"]["task_name"] == first["task_name"]
    assert second["task_name"] is None


@pytest.mark.parametrize("task", ["mag", "Meg\\initiation_pretraining"])
def test_pretraining_sessions_are_skipped_silently_and_do_not_break_pooling(tmp_path, task):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sessions = import_sessions(tmp_path, [A, (NO_TOWERS, task, False), A, A])
    assert problems(sessions) == [1, None, 1, 1]
    assert flags_of(sessions) == [None, "pretraining", None, None]
    assert len(sessions) == 4  # kept for single-session analyses


def test_port_finding_sessions_are_skipped_flagged_and_warned(tmp_path):
    with pytest.warns(UserWarning, match=r"M1 ses-2_date-20260102.*port finding"):
        sessions = import_sessions(tmp_path, [A, (NO_TOWERS, None, True), A])
    assert problems(sessions) == [1, None, 1]
    assert flags_of(sessions) == [None, "port_finding", None]


def test_towerless_session_of_unknown_kind_is_flagged_and_warned(tmp_path):
    with pytest.warns(UserWarning, match=r"M1 ses-2_date-20260102.*no choice towers"):
        sessions = import_sessions(tmp_path, [A, NO_TOWERS, A])
    assert problems(sessions) == [1, None, 1]
    assert flags_of(sessions) == [None, "no_choice_towers", None]


def test_problem_numbers_agree_with_group_sessions_by_problem(tmp_path):
    other_init = run_start_for(["A1", "B1", "C1"], initiation="A2")
    scenario = [
        (NO_TOWERS, "mag", False),
        A,
        (NO_TOWERS, None, True),
        A,
        B,
        B,
        C,
        A,
        other_init,
        other_init,
        NO_TOWERS,
        A,
    ]
    with pytest.warns(UserWarning):  # port finding, no towers, and the one-session C and A
        imported = import_sessions(tmp_path, scenario)
    # group_sessions_by_problem reads trial-level fields that extract_trials would add.
    trials = {
        "M1": {
            key: {
                "trial_info": [1],
                "choice_towers": [sorted(s["choice_towers"])],
                "initiation_tower": s["initiation_tower"],
                "settings_evidence": s["settings_evidence"],
                "task_name": s["task_name"],
            }
            for key, s in imported.items()
        }
    }
    with pytest.warns(UserWarning):
        by_problem = group_sessions_by_problem(trials)
    expected = {key: p for p, subjs in by_problem.items() for key in subjs["M1"]}
    numbered = {k: s["problem"] for k, s in imported.items() if s["problem"] is not None}
    assert numbered == expected
    assert [s["problem"] for s in imported.values()] == [
        None,
        1,
        None,
        1,
        2,
        2,
        None,
        None,
        3,
        3,
        None,
        4,
    ]


def test_subjects_come_out_sorted(tmp_path):
    names = ["M9", "M3", "M12", "M1", "M7", "M5", "M2", "M11", "M10", "M4", "M8", "M6"]
    for n, name in enumerate(names, start=1):
        write_tsv(session_dir(tmp_path, f"sub-{n:02d}_id-{name}"), f"{name}-2026-01-01-100000.tsv")
    assert list(import_data(tmp_path)) == sorted(names)


def test_recording_outside_any_session_folder_raises_naming_the_path(tmp_path):
    path = write_tsv(tmp_path / "sub-01_id-M1", "M1-2026-01-01-100000.tsv")
    with pytest.raises(ValueError, match="ses-<N>_date-<YYYYMMDD>") as excinfo:
        import_data(tmp_path)
    assert str(path) in str(excinfo.value)


@pytest.mark.parametrize("folder", ["ses-1_date-2026", "ses-1_date-20261340"])
def test_session_folder_with_an_invalid_date_raises_naming_the_path(tmp_path, folder):
    path = write_tsv(session_dir(tmp_path, session=folder), "M1-2026-01-01-100000.tsv")
    with pytest.raises(ValueError) as excinfo:
        import_data(tmp_path)
    assert str(path) in str(excinfo.value)
