"""``load_trials_with_log(..., from_cache=True)``: the newest cached import, without the share."""

import json
import warnings
from datetime import datetime
from pathlib import Path

import pytest

import src.behavior_import.load_trials as load_trials
from src.behavior_import.import_log import ImportLog
from src.behavior_import.load_trials import (
    CacheEntry,
    ImportedTrials,
    StaleCacheWarning,
    cached_import_entries,
    choose_cache_entry,
    load_trials_with_log,
)
from src.behavior_import.warning_records import WarningRecord

ROOT = "/Volumes/behrens/meg/3x3_field_magnitude_bandit/rawdata/cohort-03"


def entry(name, *, root=ROOT, classify=True, code="current", time=0.0, n_files=3):
    return CacheEntry(Path(name), root, classify, code, time, n_files)


# --- choosing an entry (pure) ---------------------------------------------------------------


def test_an_entry_made_with_the_current_import_code_wins_over_a_newer_one():
    older_same_code = entry("a", code="current", time=1.0)
    newer_old_code = entry("b", code="old", time=2.0)
    chosen = choose_cache_entry([newer_old_code, older_same_code], ROOT, True, "current")
    assert chosen == older_same_code


def test_the_newest_entry_is_taken_when_none_has_the_current_code():
    entries = [entry("a", code="x", time=1.0), entry("b", code="y", time=3.0), entry("c", time=2)]
    assert choose_cache_entry(entries, ROOT, True, "current").path == Path("c")
    assert choose_cache_entry(entries, ROOT, True, "other").path == Path("b")


def test_the_newest_of_several_current_code_entries_is_taken():
    entries = [entry("a", time=1.0), entry("b", time=2.0)]
    assert choose_cache_entry(entries, ROOT, True, "current").path == Path("b")


def test_entries_of_other_roots_and_of_the_other_classify_flag_are_ignored():
    entries = [
        entry("other-root", root=ROOT.replace("03", "04"), time=9.0),
        entry("unclassified", classify=False, time=8.0),
        entry("match", code="old", time=1.0),
    ]
    assert choose_cache_entry(entries, ROOT, True, "current").path == Path("match")


def test_no_matching_entry_names_the_root_and_lists_what_is_cached():
    entries = [
        entry("a", root="/elsewhere/cohort-01", time=datetime(2026, 9, 1, 12).timestamp()),
        entry("b", classify=False),
    ]
    with pytest.raises(FileNotFoundError) as err:
        choose_cache_entry(entries, ROOT, True, "current")
    text = str(err.value)
    assert ROOT in text
    assert "/elsewhere/cohort-01" in text
    assert "2026-09-01 12:00" in text


def test_an_empty_cache_says_so():
    with pytest.raises(FileNotFoundError, match="no cached imports"):
        choose_cache_entry([], ROOT, True, "current")


# --- the cache on disk ----------------------------------------------------------------------


@pytest.fixture
def raw_root(tmp_path):
    session = tmp_path / "raw" / "sub-01_id-M1" / "ses-01_date-20260101" / "behav"
    session.mkdir(parents=True)
    for name in ("M1-2026-01-01-120000.tsv", "M1-2026-01-01-130000.tsv"):
        (session / name).write_text("type\tsubtype\tcontent\n")
    return tmp_path / "raw"


@pytest.fixture
def fake_import(monkeypatch):
    """The slow import replaced by one that returns a fixed result and counts its calls."""
    calls = []

    def fake(root, classify_task_types=False):
        calls.append(root)
        log = ImportLog(warnings=[WarningRecord("UserWarning", "from the import", "x.py", 1)])
        return ImportedTrials({1: {"M1": {"call": len(calls)}}}, log)

    monkeypatch.setattr(load_trials, "_import_and_extract", fake)
    return calls


def fill(raw_root, cache_dir, **kwargs):
    return load_trials_with_log(raw_root, cache_dir=cache_dir, **kwargs)


def test_the_cached_entries_are_listed_with_their_inputs(raw_root, tmp_path, fake_import):
    fill(raw_root, tmp_path / "cache", classify_task_types=True)
    (found,) = cached_import_entries(tmp_path / "cache")
    assert found.root == str(raw_root.resolve())
    assert found.classify_task_types is True
    assert found.code_version == load_trials.import_code_version()
    assert found.n_files == 2
    assert abs(found.time - datetime.now().timestamp()) < 60


def test_an_empty_or_missing_cache_lists_nothing(tmp_path):
    assert cached_import_entries(tmp_path / "nothing-here") == []


def test_an_entry_without_its_output_is_not_listed(raw_root, tmp_path, fake_import):
    fill(raw_root, tmp_path / "cache")
    (found,) = cached_import_entries(tmp_path / "cache")
    (found.path / "output.pkl").unlink()
    assert cached_import_entries(tmp_path / "cache") == []


def test_a_from_cache_load_never_touches_the_raw_folder(
    raw_root, tmp_path, fake_import, monkeypatch
):
    expected = fill(raw_root, tmp_path / "cache")
    root = raw_root.resolve()
    touched = []
    for method in ("is_dir", "exists", "stat", "rglob", "resolve", "iterdir", "glob"):
        original = getattr(Path, method)

        def spy(self, *args, _original=original, _method=method, **kwargs):
            if str(self).startswith(str(root)):
                touched.append((_method, self))
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(Path, method, spy)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", StaleCacheWarning)
        loaded = fill(root, tmp_path / "cache", from_cache=True)
    assert touched == []
    assert loaded.trials_by_problem == expected.trials_by_problem
    assert len(fake_import) == 1


def test_a_from_cache_load_works_with_the_raw_folder_gone(raw_root, tmp_path, fake_import):
    root = raw_root.resolve()
    expected = fill(root, tmp_path / "cache")
    root.rename(tmp_path / "unmounted")
    with pytest.warns(StaleCacheWarning):
        loaded = fill(root, tmp_path / "cache", from_cache=True)
    assert loaded.trials_by_problem == expected.trials_by_problem
    assert len(fake_import) == 1


def test_the_stale_warning_says_what_may_be_missing(raw_root, tmp_path, fake_import):
    fill(raw_root, tmp_path / "cache")
    with pytest.warns(StaleCacheWarning) as caught:
        fill(raw_root.resolve(), tmp_path / "cache", from_cache=True)
    (message,) = [str(w.message) for w in caught if w.category is StaleCacheWarning]
    assert datetime.now().strftime("%Y-%m-%d") in message
    assert "2 .tsv files" in message
    assert "made with the current import code" in message
    assert "sessions added on the server since then are missing" in message


def test_older_import_code_is_used_with_a_warning_saying_so(
    raw_root, tmp_path, fake_import, monkeypatch
):
    expected = fill(raw_root, tmp_path / "cache")
    monkeypatch.setattr(load_trials, "import_code_version", lambda: "edited")
    with pytest.warns(StaleCacheWarning, match="made with older import code") as caught:
        loaded = fill(raw_root.resolve(), tmp_path / "cache", from_cache=True)
    message = next(str(w.message) for w in caught if w.category is StaleCacheWarning)
    assert "changes to src/behavior_import since then are not applied" in message
    assert loaded.trials_by_problem == expected.trials_by_problem
    assert len(fake_import) == 1


def test_the_stale_warning_is_kept_in_the_import_log_after_the_imports_own(
    raw_root, tmp_path, fake_import
):
    fill(raw_root, tmp_path / "cache")
    for _ in range(2):  # a second load in the same process still records it
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            log = fill(raw_root.resolve(), tmp_path / "cache", from_cache=True).log
        assert [w.category for w in log.warnings] == ["UserWarning", "StaleCacheWarning"]
        assert "sessions added on the server" in log.warnings[-1].message


def test_no_cached_import_of_the_root_raises_naming_what_is_cached(raw_root, tmp_path, fake_import):
    fill(raw_root, tmp_path / "cache")
    with pytest.raises(FileNotFoundError, match="cohort-09") as err:
        fill(Path("/Volumes/behrens/x/cohort-09"), tmp_path / "cache", from_cache=True)
    assert str(raw_root.resolve()) in str(err.value)


def test_a_cache_that_cannot_be_unpickled_fails_with_the_entry_named(
    raw_root, tmp_path, fake_import
):
    fill(raw_root, tmp_path / "cache")
    (found,) = cached_import_entries(tmp_path / "cache")
    (found.path / "output.pkl").write_bytes(b"not a pickle")
    with pytest.raises(load_trials.CachedImportError) as err:
        fill(raw_root.resolve(), tmp_path / "cache", from_cache=True)
    assert str(found.path) in str(err.value)
    assert "without --from-cache" in str(err.value)
    assert len(fake_import) == 1  # no fall-through to an import from the share


def test_a_metadata_file_that_cannot_be_read_fails_loudly(raw_root, tmp_path, fake_import):
    fill(raw_root, tmp_path / "cache")
    (found,) = cached_import_entries(tmp_path / "cache")
    (found.path / "metadata.json").write_text(json.dumps({"time": 1.0, "input_args": {}}))
    with pytest.raises(load_trials.CachedImportError, match="metadata"):
        cached_import_entries(tmp_path / "cache")


def test_from_cache_needs_the_cache(raw_root, tmp_path):
    with pytest.raises(ValueError, match="use_cache"):
        fill(raw_root, tmp_path / "cache", from_cache=True, use_cache=False)


def test_the_cache_dir_default_does_not_need_the_share(
    raw_root, tmp_path, fake_import, monkeypatch
):
    fill(raw_root, tmp_path / "cache")
    monkeypatch.setenv("BANDIT_DATA_ROOT", str(tmp_path / "unmounted"))
    monkeypatch.setenv("BANDIT_CACHE_DIR", str(tmp_path / "cache"))
    load_trials.get_unchecked_settings.cache_clear()
    try:
        with pytest.warns(StaleCacheWarning):
            loaded = load_trials_with_log(raw_root.resolve(), from_cache=True)
    finally:
        load_trials.get_unchecked_settings.cache_clear()
    assert loaded.trials_by_problem == {1: {"M1": {"call": 1}}}


def test_a_cached_log_missing_a_field_of_the_current_code_fails_with_the_entry_named(
    raw_root, tmp_path, monkeypatch
):
    def old_import(root, classify_task_types=False):
        log = ImportLog()
        del log.port_finding_sessions  # as pickled by code that predates the field
        return ImportedTrials({1: {}}, log)

    monkeypatch.setattr(load_trials, "_import_and_extract", old_import)
    fill(raw_root, tmp_path / "cache")
    with pytest.raises(load_trials.CachedImportError, match="port_finding_sessions"):
        fill(raw_root.resolve(), tmp_path / "cache", from_cache=True)
