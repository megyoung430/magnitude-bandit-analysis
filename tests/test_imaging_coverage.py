"""Imaged sessions per problem and their LED setting."""

import json

import pandas as pd
import pytest

from src.behavior_analysis.review import imaging_coverage as ic


def frame(*variables):
    """A raw pyControl frame with one variable row per ``(subtype, dict)``."""
    rows = [dict(type="info", subtype="task_name", content="Meg\\task")]
    rows += [dict(type="variable", subtype=st, content=json.dumps(c)) for st, c in variables]
    return pd.DataFrame(rows)


def session(*variables, task_type="drifting"):
    return {"df": frame(*variables), "task_type": task_type}


def test_the_led_setting_is_read_from_every_variable_row():
    assert ic.led_setting(frame(("run_start", {"led": False}))) == ic.OFF
    assert ic.led_setting(frame(("run_start", {"with_LED": True}))) == ic.ON
    changed = frame(("run_start", {"led": True}), ("user_set", {"led": False}))
    assert ic.led_setting(changed) == ic.MIXED
    assert ic.led_setting(frame(("run_start", {"num_t": 0}))) == ic.UNSET


def test_a_multi_file_session_reads_all_its_files():
    files = [frame(("run_start", {"led": False})), frame(("run_start", {"led": False}))]
    assert ic.led_setting(files) == ic.OFF
    assert ic.led_setting([frame(("run_start", {"led": False})), frame()]) == ic.OFF
    assert ic.led_setting(None) == ic.UNSET


def test_rows_that_are_not_json_dicts_are_skipped():
    rows = [("print", "not json"), ("print", "[1, 2]"), ("run_start", '{"led": true}')]
    df = pd.DataFrame([dict(type="variable", subtype=st, content=c) for st, c in rows])
    assert ic.led_setting(df) == ic.ON


def imported():
    return {
        4: {"m": {"ses-9_date-20260101": session(("run_start", {"num_t": 0})),
                  "ses-10_date-20260102": session(("run_start", {"num_t": 0}))}},
        5: {"m": {"ses-11_date-20260103": session(("run_start", {"led": False})),
                  "ses-12_date-20260104": session(("run_start", {"led": True})),
                  "ses-13_date-20260105": session(("run_start", {"led": False}))}},
    }  # fmt: skip


def modality():
    sessions = [
        ("ses-9_date-20260101", True, False),
        ("ses-10_date-20260102", True, True),
        ("ses-11_date-20260103", True, True),
        ("ses-12_date-20260104", True, True),
        ("ses-13_date-20260105", True, False),
        ("ses-14_date-20260106", True, True),
    ]
    return pd.DataFrame([dict(subject_id="m", ses_id=s, date=pd.Timestamp(s[-8:]), has_behav=b,
                              has_funcimg=f) for s, b, f in sessions])  # fmt: skip


def test_the_session_table_has_one_row_per_imported_session():
    table = ic.session_table(imported())
    assert len(table) == 5
    assert table.set_index("ses_id")["led_setting"].to_dict() == {
        "ses-9_date-20260101": ic.UNSET, "ses-10_date-20260102": ic.UNSET,
        "ses-11_date-20260103": ic.OFF, "ses-12_date-20260104": ic.ON,
        "ses-13_date-20260105": ic.OFF,
    }  # fmt: skip


def test_the_parts_of_a_split_session_are_one_folder_per_problem():
    parts = {
        1: {"m": {"ses-3_date-20260101_part-01": session(("run_start", {"led": False}))}},
        2: {"m": {"ses-3_date-20260101_part-02": session(("run_start", {"led": False})),
                  "ses-4_date-20260102": session(("run_start", {"led": True}))}},
    }  # fmt: skip
    table = ic.session_table(parts)
    assert list(zip(table["ses_id"], table["problem"], strict=True)) == [
        ("ses-3_date-20260101", 1), ("ses-3_date-20260101", 2), ("ses-4_date-20260102", 2),
    ]  # fmt: skip
    off, on = session(("run_start", {"led": False})), session(("run_start", {"led": True}))
    same = {1: {"m": {"ses-3_date-20260101_part-01": off, "ses-3_date-20260101_part-02": on}}}
    one = ic.session_table(same)
    assert len(one) == 1 and one.loc[0, "led_setting"] == ic.MIXED


def test_imaged_sessions_need_both_folders_and_keep_sessions_without_a_problem():
    imaged = ic.imaged_sessions(ic.session_table(imported()), modality())
    assert list(imaged["ses_id"]) == ["ses-10_date-20260102", "ses-11_date-20260103",
                                      "ses-12_date-20260104", "ses-14_date-20260106"]  # fmt: skip
    no_problem = imaged.set_index("ses_id").loc["ses-14_date-20260106"]
    assert pd.isna(no_problem["problem"]) and no_problem["led_setting"] == ic.UNSET


def test_an_unset_led_counts_as_on_unless_told_otherwise():
    sessions = ic.session_table(imported())
    on = ic.imaged_sessions(sessions, modality()).set_index("ses_id")["led_on"]
    assert on["ses-10_date-20260102"] and not on["ses-11_date-20260103"]
    strict = ic.imaged_sessions(sessions, modality(), unset_means_on=False)
    assert not strict.set_index("ses_id").loc["ses-10_date-20260102", "led_on"]


def test_coverage_counts_imaged_and_led_on_sessions_out_of_the_problem_total():
    sessions = ic.session_table(imported())
    coverage = ic.coverage_by_problem(sessions, ic.imaged_sessions(sessions, modality()))
    rows = coverage.set_index("problem")
    assert rows.loc[4, ["n_sessions", "n_imaged", "n_led_on"]].tolist() == [2, 1, 1]
    assert rows.loc[5, ["n_sessions", "n_imaged", "n_led_on"]].tolist() == [3, 2, 1]
    assert rows.loc[5, "imaged_fraction"] == pytest.approx(2 / 3)
    assert rows.loc[5, "led_on_sessions"] == "ses-12_date-20260104"
    assert rows.loc[5, "led_off_sessions"] == "ses-11_date-20260103"
    assert rows.loc[4, "led_on_sessions"] == "ses-10_date-20260102 (not set)"
    assert rows.loc[4, "n_led_not_set"] == 1 and rows.loc[5, "n_led_not_set"] == 0


def test_sessions_are_listed_in_session_number_order():
    sessions = pd.DataFrame(dict(mouse="m", problem=1, task_type="drifting",
                                 ses_id=["ses-10_date-20260102", "ses-9_date-20260101"],
                                 led_setting=ic.ON))  # fmt: skip
    imaged = sessions.assign(date=pd.Timestamp("2026-01-01"), led_on=True)
    coverage = ic.coverage_by_problem(sessions, imaged)
    assert coverage.loc[0, "imaged_sessions"] == "ses-9_date-20260101, ses-10_date-20260102"


def test_no_imaged_session_gives_an_empty_table():
    sessions = ic.session_table(imported())
    empty = ic.imaged_sessions(sessions, modality().assign(has_funcimg=False))
    assert ic.coverage_by_problem(sessions, empty).empty
