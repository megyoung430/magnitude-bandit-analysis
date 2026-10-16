"""The importer records what it skips and flags, without changing what it loads."""

import json
import warnings

import pandas as pd
import pytest

from src.behavior_import import import_log
from src.behavior_import.extract_trials import extract_trials_grouped_by_problem
from src.behavior_import.import_data import import_data
from src.behavior_import.import_log import ImportLog
from src.behavior_import.load_trials import load_trials_with_log
from src.behavior_import.problem_numbering import ProblemNumberingWarning
from src.behavior_import.warning_records import record_warnings
from tests.test_import_data import run_start_for, session_dir, write_tsv

TASK = "Meg\\3x3_field_drifting_value_bandit"
LAYOUT_A = ["A1", "B1", "C1"]
LAYOUT_B = ["A3", "B3", "C3"]


def write_prints(folder, name, prints, run_start=None, task_name=TASK):
    """A recording whose only content is *prints* (a list of print dicts or plain text)."""
    rows = [("0.000", "info", "task_name", task_name)]
    run_start = run_start_for(LAYOUT_A) if run_start is None else run_start
    rows.append(("0.002", "variable", "run_start", json.dumps(run_start)))
    for i, printed in enumerate(prints):
        content = printed if isinstance(printed, str) else json.dumps(printed)
        rows.append((f"{1 + i}.000", "variable", "print", content))
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["time", "type", "subtype", "content"]).to_csv(
        folder / name, sep="\t", index=False
    )
    return folder / name


def load(root, **kwargs):
    log = ImportLog()
    data = import_data(root, classify_task_types=True, log=log, **kwargs)
    return extract_trials_grouped_by_problem(data, log=log), log


def kinds(log):
    return sorted((s.kind, s.level) for s in log.skips)


def test_unreadable_file_is_recorded_with_its_mouse_and_session(tmp_path):
    folder = session_dir(tmp_path)
    write_tsv(folder, "M1-2026-01-01-100000.tsv", task_name=TASK)
    (folder / "M1-2026-01-01-110000.tsv").write_text("")  # empty: pandas cannot read it
    _, log = load(tmp_path)
    (skip,) = log.skips
    assert (skip.mouse, skip.session, skip.file) == (
        "M1",
        "ses-1_date-20260101",
        str(folder / "M1-2026-01-01-110000.tsv"),
    )
    assert (skip.kind, skip.level) == (import_log.UNREADABLE_FILE, import_log.FILE)
    assert "read" in skip.reason


def test_file_without_a_subject_is_recorded(tmp_path):
    write_tsv(tmp_path / "loose" / "ses-1_date-20260101", "x.tsv", task_name=TASK)
    _, log = load(tmp_path)
    (skip,) = log.skips
    assert skip.kind == import_log.NO_SUBJECT
    assert skip.mouse is None
    assert skip.session == "ses-1_date-20260101"


def test_non_numbered_session_folder_is_recorded_and_still_warns(tmp_path):
    write_tsv(session_dir(tmp_path), "M1-2026-01-01-100000.tsv", task_name=TASK)
    write_tsv(
        session_dir(tmp_path, session="ses-test_date-20260102"),
        "M1-2026-01-02-100000.tsv",
        task_name=TASK,
    )
    with pytest.warns(UserWarning, match="non-task session"):
        _, log = load(tmp_path)
    (skip,) = log.skips
    assert (skip.mouse, skip.session, skip.kind) == (
        "M1",
        "ses-test_date-20260102",
        import_log.UNNUMBERED_SESSION,
    )


def test_recording_without_bandit_fields_is_a_recording_skip_when_others_have_trials(tmp_path):
    folder = session_dir(tmp_path)
    write_tsv(folder, "M1-2026-01-01-100000.tsv", task_name=TASK)
    write_prints(folder, "M1-2026-01-01-110000.tsv", [{"num_t": 1, "chosen_tow": "A1"}])
    problems, log = load(tmp_path)
    (skip,) = log.skips
    assert (skip.kind, skip.level) == (import_log.NO_BANDIT_FIELDS, import_log.RECORDING)
    assert skip.file.endswith("M1-2026-01-01-110000.tsv")
    assert skip.task_name == TASK
    assert list(problems[1]["M1"]) == ["ses-1_date-20260101"]


@pytest.mark.parametrize(
    ("prints", "kind"),
    [
        ([{"num_t": 1, "chosen_tow": "A1"}], import_log.NO_BANDIT_FIELDS),
        ([{"num_t_found": 1, "found_ports": False}], import_log.ONLY_PORT_FINDING),
        ([], import_log.NO_TRIAL_PRINTS),
        (["plain text print"], import_log.NO_BANDIT_FIELDS),
    ],
)
def test_session_without_any_trial_is_a_session_skip_with_its_kind(tmp_path, prints, kind):
    write_tsv(session_dir(tmp_path), "M1-2026-01-01-100000.tsv", task_name=TASK)
    write_prints(session_dir(tmp_path, session="ses-2_date-20260102"), "M1.tsv", prints)
    _, log = load(tmp_path)
    (skip,) = log.skips
    assert (skip.session, skip.kind, skip.level) == (
        "ses-2_date-20260102",
        kind,
        import_log.SESSION,
    )


def test_every_recording_of_an_empty_session_gets_a_row(tmp_path):
    folder = session_dir(tmp_path, session="ses-2_date-20260102")
    write_tsv(session_dir(tmp_path), "M1-2026-01-01-100000.tsv", task_name=TASK)
    write_prints(folder, "M1-2026-01-02-100000.tsv", [])
    write_prints(folder, "M1-2026-01-02-110000.tsv", [{"num_t_found": 1}])
    _, log = load(tmp_path)
    assert kinds(log) == [
        (import_log.NO_TRIAL_PRINTS, import_log.SESSION),
        (import_log.ONLY_PORT_FINDING, import_log.SESSION),
    ]


def write_layouts(root, layouts):
    for number, towers in enumerate(layouts, 1):
        write_tsv(
            session_dir(root, session=f"ses-{number}_date-2026010{number}"),
            f"M1-2026-01-0{number}-100000.tsv",
            run_start=run_start_for(towers),
            task_name=TASK,
        )


def test_flagged_sessions_are_recorded_with_a_plain_reason(tmp_path):
    write_layouts(tmp_path, [LAYOUT_A, LAYOUT_A, LAYOUT_B, LAYOUT_A])
    with pytest.warns(ProblemNumberingWarning):
        problems, log = load(tmp_path)
    (error,) = log.flags
    assert (error.mouse, error.session, error.flag, error.problem) == (
        "M1",
        "ses-3_date-20260103",
        "suspected_settings_error",
        None,
    )
    assert "one session" in error.reason
    assert error.files[0].endswith("M1-2026-01-03-100000.tsv")
    assert list(problems) == [1]


def test_review_settings_and_provisional_numbers_are_recorded(tmp_path):
    write_layouts(tmp_path, [LAYOUT_A, LAYOUT_A, LAYOUT_B])
    with pytest.warns(ProblemNumberingWarning):
        _, log = load(tmp_path)
    (flag,) = log.flags
    assert (flag.session, flag.flag, flag.problem, flag.provisional) == (
        "ses-3_date-20260103",
        "review_settings",
        2,
        True,
    )
    assert "no port-finding trials" in flag.reason


def test_a_provisional_number_without_a_flag_is_still_recorded(tmp_path):
    write_layouts(tmp_path, [LAYOUT_A, LAYOUT_A])
    # a new layout that starts with port finding is a fine problem start, but one session long
    write_tsv(
        session_dir(tmp_path, session="ses-3_date-20260103"),
        "M1-2026-01-03-100000.tsv",
        run_start=run_start_for(LAYOUT_B),
        task_name=TASK,
        port_finding=True,
    )
    problems, log = load(tmp_path)
    (flag,) = log.flags
    assert (flag.session, flag.flag, flag.provisional, flag.problem) == (
        "ses-3_date-20260103",
        None,
        True,
        2,
    )
    assert "ses-3_date-20260103" in problems[2]["M1"]


def test_pretraining_session_without_trials_still_gets_a_flag(tmp_path):
    write_tsv(session_dir(tmp_path), "M1-2026-01-01-100000.tsv", task_name=TASK)
    write_prints(
        session_dir(tmp_path, session="ses-2_date-20260102"),
        "M1.tsv",
        [{"num_t": 1}],
        task_name="Meg\\pretraining_1",
    )
    _, log = load(tmp_path)
    (pretraining,) = [f for f in log.flags if f.flag]
    assert (pretraining.session, pretraining.flag) == ("ses-2_date-20260102", "pretraining")
    assert "Meg\\pretraining_1" in pretraining.reason


def test_logging_does_not_change_what_is_loaded(tmp_path):
    folder = session_dir(tmp_path)
    write_tsv(folder, "M1-2026-01-01-100000.tsv", task_name=TASK)
    write_prints(folder, "M1-2026-01-01-110000.tsv", [{"num_t": 1}])
    write_tsv(session_dir(tmp_path, session="ses-2_date-20260102"), "M1.tsv", task_name=TASK)
    plain = extract_trials_grouped_by_problem(import_data(tmp_path, classify_task_types=True))
    logged, _ = load(tmp_path)

    def summary(problems):
        return {
            (p, mouse, key): (sess["trial"], sess["blocks"], sess["problem_flag"])
            for p, mice in problems.items()
            for mouse, sessions in mice.items()
            for key, sess in sessions.items()
        }

    assert summary(plain) == summary(logged)


@pytest.fixture
def raw_root(tmp_path):
    root = tmp_path / "raw"
    write_tsv(session_dir(root), "M1-2026-01-01-100000.tsv", task_name=TASK)
    (session_dir(root) / "M1-2026-01-01-110000.tsv").write_text("")
    write_tsv(session_dir(root, session="ses-test_date-20260105"), "M1.tsv", task_name=TASK)
    return root


def test_skip_records_survive_a_cached_load(raw_root, tmp_path):
    cache = tmp_path / "cache"
    with pytest.warns(UserWarning, match="non-task session"):
        fresh = load_trials_with_log(raw_root, cache_dir=cache, classify_task_types=True)
    assert len(fresh.log.skips) == 2
    cached = load_trials_with_log(raw_root, cache_dir=cache, classify_task_types=True)
    assert cached.log == fresh.log
    assert {s.kind for s in cached.log.skips} == {
        import_log.UNREADABLE_FILE,
        import_log.UNNUMBERED_SESSION,
    }


def test_import_warnings_are_kept_in_the_log_and_survive_the_cache(tmp_path):
    root = tmp_path / "raw"
    write_tsv(session_dir(root, session="ses-test_date-20260105"), "M1.tsv", task_name=TASK)
    write_tsv(session_dir(root), "M1-2026-01-01-100000.tsv", task_name=TASK)
    cache = tmp_path / "cache"
    with pytest.warns(UserWarning):
        fresh = load_trials_with_log(root, cache_dir=cache, classify_task_types=True)
    cached = load_trials_with_log(root, cache_dir=cache, classify_task_types=True)
    assert [w.category for w in fresh.log.warnings] == ["UserWarning"]
    assert cached.log.warnings == fresh.log.warnings
    assert "non-task session" in cached.log.warnings[0].message


def test_record_warnings_records_and_still_shows_them():
    with warnings.catch_warnings(record=True) as shown:
        warnings.simplefilter("always")
        with record_warnings() as recorded:
            warnings.warn("careful", ProblemNumberingWarning, stacklevel=1)
            warnings.warn("other", UserWarning, stacklevel=1)
    assert [(w.category, w.message) for w in recorded] == [
        ("ProblemNumberingWarning", "careful"),
        ("UserWarning", "other"),
    ]
    assert [str(w.message) for w in shown] == ["careful", "other"]


def test_record_warnings_stops_recording_on_exit():
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        with record_warnings() as recorded:
            pass
        warnings.warn("after", UserWarning, stacklevel=1)
    assert recorded == []
