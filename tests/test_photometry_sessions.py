"""Tests for finding photometry sessions in a cohort folder."""

import dataclasses
from datetime import date

import pytest

from src.photometry_import.sessions import PhotometrySession, find_photometry_sessions


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def _session(cohort, subject_dir, session_dir, ppds=(), tsvs=(), gpios=(), extra=()):
    folder = cohort / subject_dir / session_dir
    (folder / "photometry").mkdir(parents=True, exist_ok=True)
    for name in ppds:
        _touch(folder / "photometry" / name)
    for name in tsvs:
        _touch(folder / "behav" / name)
    for name in gpios:
        _touch(folder / "behav" / name)
    for name in extra:
        _touch(folder / "behav" / name)
    return folder


def test_finds_a_session_with_its_files(tmp_path):
    folder = _session(
        tmp_path,
        "sub-06_id-RT_64_R",
        "ses-44_date-20261014",
        ppds=["b.ppd", "a.ppd"],
        tsvs=["y.tsv", "x.tsv"],
        gpios=["v_GPIO_data.csv"],
        extra=["v_metadata.json"],
    )
    (session,) = find_photometry_sessions(tmp_path)

    assert session.subject == "RT_64_R"
    assert session.session == "ses-44"
    assert session.date == date(2026, 10, 14)
    assert session.folder == folder
    assert [p.name for p in session.ppd_paths] == ["a.ppd", "b.ppd"]
    assert [p.name for p in session.tsv_paths] == ["x.tsv", "y.tsv"]
    assert [p.name for p in session.gpio_paths] == ["v_GPIO_data.csv"]


def test_a_session_without_a_ppd_is_not_a_photometry_session(tmp_path):
    _session(tmp_path, "sub-01_id-RT_1", "ses-01_date-20261001", tsvs=["x.tsv"])
    (tmp_path / "sub-01_id-RT_2" / "ses-02_date-20261001" / "behav").mkdir(parents=True)
    _touch(tmp_path / "sub-01_id-RT_2" / "ses-02_date-20261001" / "behav" / "x.tsv")
    assert find_photometry_sessions(tmp_path) == []


def test_test_and_other_non_numeric_session_labels_are_skipped_even_with_files(tmp_path):
    _session(tmp_path, "sub-01_id-RT_1", "ses-test_date-20261001", ppds=["a.ppd"], tsvs=["x.tsv"])
    _session(tmp_path, "sub-01_id-RT_1", "ses-pilot_date-20261001", ppds=["a.ppd"])
    _session(tmp_path, "sub-01_id-RT_1", "ses-07_date-20261002", ppds=["a.ppd"])
    sessions = find_photometry_sessions(tmp_path)
    assert [s.session for s in sessions] == ["ses-07"]


def test_sessions_are_ordered_by_subject_then_session_number(tmp_path):
    for subject, ses in [
        ("sub-02_id-B", "ses-10"),
        ("sub-01_id-A", "ses-9"),
        ("sub-01_id-A", "ses-10"),
    ]:
        _session(tmp_path, subject, f"{ses}_date-20261001", ppds=["a.ppd"])
    sessions = find_photometry_sessions(tmp_path)
    assert [(s.subject, s.session) for s in sessions] == [
        ("A", "ses-9"),
        ("A", "ses-10"),
        ("B", "ses-10"),
    ]


def test_ignores_files_and_unrelated_folders_in_the_cohort_folder(tmp_path):
    _session(tmp_path, "sub-01_id-RT_1", "ses-01_date-20261001", ppds=["a.ppd"])
    _touch(tmp_path / ".DS_Store")
    (tmp_path / "notes").mkdir()
    assert len(find_photometry_sessions(tmp_path)) == 1


def test_session_is_frozen(tmp_path):
    _session(tmp_path, "sub-01_id-RT_1", "ses-01_date-20261001", ppds=["a.ppd"])
    (session,) = find_photometry_sessions(tmp_path)
    assert isinstance(session, PhotometrySession)
    with pytest.raises(dataclasses.FrozenInstanceError):
        session.subject = "other"


def test_missing_cohort_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="nope"):
        find_photometry_sessions(tmp_path / "nope")


@pytest.mark.parametrize(
    ("subject_dir", "session_dir", "message"),
    [
        ("sub-01_RT_1", "ses-01_date-20261001", "subject folder"),
        ("sub-01_id-RT_1", "ses-01", "session folder"),
        ("sub-01_id-RT_1", "ses-01_date-20261340", "date"),
    ],
)
def test_malformed_folder_names_raise(tmp_path, subject_dir, session_dir, message):
    _session(tmp_path, subject_dir, session_dir, ppds=["a.ppd"])
    with pytest.raises(ValueError, match=message):
        find_photometry_sessions(tmp_path)
