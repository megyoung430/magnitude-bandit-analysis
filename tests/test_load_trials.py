import os

import pytest

import src.behavior_import.load_trials as load_trials
from src.behavior_import.import_log import ImportLog
from src.behavior_import.load_trials import (
    ImportedTrials,
    load_trials_by_problem,
    raw_data_fingerprint,
)


@pytest.fixture
def raw_root(tmp_path):
    session = tmp_path / "raw" / "sub-01_id-M1" / "ses-01_date-20260101" / "behav"
    session.mkdir(parents=True)
    (session / "M1-2026-01-01-120000.tsv").write_text("type\tsubtype\tcontent\n")
    return tmp_path / "raw"


@pytest.fixture
def computed(monkeypatch):
    """Replace the slow import with a counter so tests need no real sessions."""
    calls = []

    def fake_import(root, classify_task_types=False):
        calls.append(root)
        return ImportedTrials({1: {"M1": {"n_calls": len(calls)}}}, ImportLog())

    monkeypatch.setattr(load_trials, "_import_and_extract", fake_import)
    return calls


def test_second_load_comes_from_cache(raw_root, tmp_path, computed):
    first = load_trials_by_problem(raw_root, cache_dir=tmp_path / "cache")
    second = load_trials_by_problem(raw_root, cache_dir=tmp_path / "cache")
    assert first == second == {1: {"M1": {"n_calls": 1}}}
    assert len(computed) == 1


def test_new_or_changed_raw_file_invalidates_cache(raw_root, tmp_path, computed):
    load_trials_by_problem(raw_root, cache_dir=tmp_path / "cache")
    tsv = next(raw_root.rglob("*.tsv"))
    tsv.write_text("type\tsubtype\tcontent\nvariable\tprint\t{}\n")
    os.utime(tsv, ns=(1, 1))
    load_trials_by_problem(raw_root, cache_dir=tmp_path / "cache")
    assert len(computed) == 2


def test_import_code_change_invalidates_cache(raw_root, tmp_path, computed, monkeypatch):
    load_trials_by_problem(raw_root, cache_dir=tmp_path / "cache")
    monkeypatch.setattr(load_trials, "import_code_version", lambda: "edited")
    load_trials_by_problem(raw_root, cache_dir=tmp_path / "cache")
    assert len(computed) == 2


def test_use_cache_false_always_recomputes(raw_root, tmp_path, computed):
    for _ in range(2):
        load_trials_by_problem(raw_root, cache_dir=tmp_path / "cache", use_cache=False)
    assert len(computed) == 2
    assert not (tmp_path / "cache").exists()


def test_fingerprint_lists_tsv_files_relative_to_root(raw_root):
    (raw_root / "notes.txt").write_text("ignored")
    ((path, size, _),) = raw_data_fingerprint(raw_root)
    assert path == "sub-01_id-M1/ses-01_date-20260101/behav/M1-2026-01-01-120000.tsv"
    assert size > 0


def test_missing_root_fails_loudly(tmp_path, computed):
    with pytest.raises(FileNotFoundError):
        load_trials_by_problem(tmp_path / "unmounted", cache_dir=tmp_path / "cache")


def test_classifying_task_types_is_part_of_the_cache_key(raw_root, tmp_path, computed):
    load_trials_by_problem(raw_root, cache_dir=tmp_path)
    load_trials_by_problem(raw_root, cache_dir=tmp_path, classify_task_types=True)
    assert len(computed) == 2
