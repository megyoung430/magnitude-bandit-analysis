"""Sessions that hold only port finding are kept in the import log for the poke analysis.

They are still left out of ``trials_by_problem`` (every task analysis assumes bandit trials), but
their raw recordings stay available so the port-finding poke maps can count their trials.
"""

import pandas as pd
import pytest

from src.behavior_import import import_log
from src.behavior_import.load_trials import load_trials_with_log
from tests.test_import_data import run_start_for, session_dir, write_tsv
from tests.test_import_log import LAYOUT_A, LAYOUT_B, TASK, load, write_prints


def port_finding_print(n, towers=LAYOUT_A):
    return {
        "num_t_found": n,
        "found_ports": False,
        "chosen_tow": towers[0],
        "c_tows": towers,
    }


def write_bandit_session(root, number, towers=LAYOUT_A):
    write_tsv(
        session_dir(root, session=f"ses-{number}_date-2026010{number}"),
        f"M1-2026-01-0{number}-100000.tsv",
        run_start=run_start_for(towers),
        task_name=TASK,
    )


def write_port_finding_session(root, number, prints, towers=LAYOUT_A):
    return write_prints(
        session_dir(root, session=f"ses-{number}_date-2026010{number}"),
        f"M1-2026-01-0{number}-090000.tsv",
        prints,
        run_start=run_start_for(towers),
    )


def test_a_port_finding_only_session_is_kept_by_mouse_and_session(tmp_path):
    write_port_finding_session(tmp_path, 1, [port_finding_print(1), port_finding_print(2)])
    write_bandit_session(tmp_path, 2)
    problems, log = load(tmp_path)
    assert list(problems[1]["M1"]) == ["ses-2_date-20260102"]  # trials_by_problem is unchanged
    assert list(log.port_finding_sessions) == ["M1"]
    ((key, record),) = log.port_finding_sessions["M1"].items()
    assert key == "ses-1_date-20260101"
    assert isinstance(record["df"], pd.DataFrame)
    assert record["choice_towers"] == frozenset(LAYOUT_A)
    assert record["initiation_tower"] == "B2"


def test_the_towers_are_the_c_tows_of_the_port_finding_prints(tmp_path):
    # run_start names layout A, the prints name layout B: the prints are what the mouse faced
    write_port_finding_session(tmp_path, 1, [port_finding_print(1, LAYOUT_B)], towers=LAYOUT_A)
    write_bandit_session(tmp_path, 2, LAYOUT_B)
    _, log = load(tmp_path)
    assert log.port_finding_sessions["M1"]["ses-1_date-20260101"]["choice_towers"] == frozenset(
        LAYOUT_B
    )


def test_without_c_tows_in_the_prints_the_towers_come_from_run_start(tmp_path):
    write_port_finding_session(tmp_path, 1, [{"num_t_found": 1, "found_ports": False}])
    write_bandit_session(tmp_path, 2)
    _, log = load(tmp_path)
    assert log.port_finding_sessions["M1"]["ses-1_date-20260101"]["choice_towers"] == frozenset(
        LAYOUT_A
    )


@pytest.mark.parametrize(
    "prints",
    [
        [],  # no prints at all
        [{"num_t": 1, "chosen_tow": "A1"}],  # prints without bandit fields: not port finding
        ["plain text print"],
    ],
)
def test_other_empty_sessions_are_not_kept(tmp_path, prints):
    write_bandit_session(tmp_path, 1)
    write_prints(session_dir(tmp_path, session="ses-2_date-20260102"), "M1.tsv", prints)
    _, log = load(tmp_path)
    assert log.port_finding_sessions == {}


def test_a_session_with_bandit_trials_is_not_kept_even_if_it_also_has_port_finding(tmp_path):
    folder = session_dir(tmp_path)
    write_prints(folder, "M1-2026-01-01-090000.tsv", [port_finding_print(1)])
    write_tsv(folder, "M1-2026-01-01-100000.tsv", task_name=TASK)
    problems, log = load(tmp_path)
    assert list(problems[1]["M1"]) == ["ses-1_date-20260101"]
    assert log.port_finding_sessions == {}


def test_all_recordings_of_a_port_finding_only_session_are_kept(tmp_path):
    folder = session_dir(tmp_path)
    write_prints(folder, "M1-2026-01-01-090000.tsv", [port_finding_print(1)])
    write_prints(folder, "M1-2026-01-01-100000.tsv", [port_finding_print(2)])
    write_bandit_session(tmp_path, 2)
    _, log = load(tmp_path)
    frames = log.port_finding_sessions["M1"]["ses-1_date-20260101"]["df"]
    assert isinstance(frames, list) and len(frames) == 2


def test_the_skip_says_the_trials_are_used_in_the_port_finding_poke_maps(tmp_path):
    write_port_finding_session(tmp_path, 1, [port_finding_print(1)])
    write_bandit_session(tmp_path, 2)
    _, log = load(tmp_path)
    (skip,) = log.skips
    assert (skip.kind, skip.level) == (import_log.ONLY_PORT_FINDING, import_log.SESSION)
    assert "port-finding poke maps" in skip.reason
    assert "no bandit trials" in skip.reason


def test_the_kept_sessions_survive_a_cached_load(tmp_path):
    root = tmp_path / "raw"
    write_port_finding_session(root, 1, [port_finding_print(1), port_finding_print(2)])
    write_bandit_session(root, 2)
    cache = tmp_path / "cache"
    fresh = load_trials_with_log(root, cache_dir=cache, classify_task_types=True)
    cached = load_trials_with_log(root, cache_dir=cache, classify_task_types=True)
    for imported in (fresh, cached):
        record = imported.log.port_finding_sessions["M1"]["ses-1_date-20260101"]
        assert record["choice_towers"] == frozenset(LAYOUT_A)
        assert record["initiation_tower"] == "B2"
        assert not record["df"].empty
    fresh_record = fresh.log.port_finding_sessions["M1"]["ses-1_date-20260101"]
    cached_record = cached.log.port_finding_sessions["M1"]["ses-1_date-20260101"]
    pd.testing.assert_frame_equal(cached_record["df"], fresh_record["df"])
    assert cached.log == fresh.log  # the log's equality leaves the recordings out


def test_a_pretraining_session_is_not_kept_even_if_it_prints_port_finding(tmp_path):
    write_prints(
        session_dir(tmp_path),
        "M1-2026-01-01-090000.tsv",
        [port_finding_print(1)],
        run_start=run_start_for(LAYOUT_A),
        task_name="Meg\\pretraining_1",
    )
    write_bandit_session(tmp_path, 2)
    _, log = load(tmp_path)
    assert log.port_finding_sessions == {}


def test_recordings_with_other_towers_are_kept_as_separate_segments(tmp_path):
    folder = session_dir(tmp_path)
    for name, towers in (
        ("M1-2026-01-01-090000.tsv", LAYOUT_A),
        ("M1-2026-01-01-100000.tsv", LAYOUT_B),
    ):
        write_prints(folder, name, [port_finding_print(1, towers)], run_start=run_start_for(towers))
    write_bandit_session(tmp_path, 2, LAYOUT_B)
    _, log = load(tmp_path)
    kept = log.port_finding_sessions["M1"]
    assert {key: record["choice_towers"] for key, record in kept.items()} == {
        "ses-1_date-20260101_part-01": frozenset(LAYOUT_A),
        "ses-1_date-20260101_part-02": frozenset(LAYOUT_B),
    }
