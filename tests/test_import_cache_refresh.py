"""Refreshing the import cache incrementally, and cache-first loads of its snapshots.

The raw folders are real (tiny) pyControl recordings, so the per-file parses, the assembly
and the comparison with a fresh import all run the real import code.
"""

import builtins
import hashlib
import os
import warnings
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import pytest

import src.behavior_import.cache_store as cache_store
import src.behavior_import.load_trials as load_trials
from src.behavior_import.cache_check import check_import_cache, describe_check
from src.behavior_import.cache_prune import list_file_entries
from src.behavior_import.cache_store import ShareReadError, list_snapshots
from src.behavior_import.load_trials import (
    StaleCacheWarning,
    load_trials_with_log,
    refresh_import_cache,
)
from tests.test_import_data import run_start_for, session_dir, write_tsv

pytestmark = pytest.mark.filterwarnings("ignore")

TASK = "Meg\\3x3_field_drifting_value_bandit"
LAYOUT_A = ["A1", "B1", "C1"]
LAYOUT_B = ["A3", "B3", "C3"]


def recording(root, subject, session, name, layout=LAYOUT_A):
    return write_tsv(
        session_dir(root, subject, session), name, run_start=run_start_for(layout), task_name=TASK
    )


def with_traceback(path):
    """Append the text pyControl writes after an uncaught task error (a non-numeric time)."""
    with open(path, "a") as f:
        f.write("AttributeError: boom\t\t\t\n")
    return path


def first_cohort(root):
    """Mouse M1 with three sessions (layout A then B), an unreadable file and a test session."""
    recording(root, "sub-01_id-M1", "ses-1_date-20260101", "M1-2026-01-01-100000.tsv")
    recording(root, "sub-01_id-M1", "ses-1_date-20260101", "M1-2026-01-01-110000.tsv")
    recording(root, "sub-01_id-M1", "ses-2_date-20260102", "M1-2026-01-02-100000.tsv")
    with_traceback(
        recording(root, "sub-01_id-M1", "ses-3_date-20260103", "M1-2026-01-03-100000.tsv", LAYOUT_B)
    )
    (session_dir(root, "sub-01_id-M1", "ses-2_date-20260102") / "empty.tsv").write_text("")
    recording(root, "sub-01_id-M1", "ses-test_date-20260104", "M1.tsv")
    return root


def grow(root):
    """New subject, new session, a changed file and a removed file."""
    recording(root, "sub-02_id-M2", "ses-1_date-20260101", "M2-2026-01-01-100000.tsv")
    recording(root, "sub-02_id-M2", "ses-2_date-20260102", "M2-2026-01-02-100000.tsv")
    recording(root, "sub-01_id-M1", "ses-4_date-20260104", "M1-2026-01-04-100000.tsv", LAYOUT_B)
    changed = recording(
        root, "sub-01_id-M1", "ses-2_date-20260102", "M1-2026-01-02-100000.tsv", LAYOUT_A
    )
    with_traceback(changed)
    os.utime(changed, ns=(2_000_000_000_000_000_000, 2_000_000_000_000_000_000))
    (session_dir(root, "sub-01_id-M1", "ses-1_date-20260101") / "M1-2026-01-01-110000.tsv").unlink()
    return root


@pytest.fixture
def raw(tmp_path):
    return first_cohort(tmp_path / "exp" / "rawdata" / "cohort-01")


@pytest.fixture
def cache(tmp_path):
    return tmp_path / "cache"


@pytest.fixture
def parse_calls(monkeypatch):
    calls = []
    original = cache_store.parse_tsv_file

    def counting(path):
        calls.append(Path(path).name)
        return original(path)

    monkeypatch.setattr(cache_store, "parse_tsv_file", counting)
    return calls


def assert_same(a, b, where="trials"):
    """Deep equality of nested dicts, lists, sets and DataFrames (with their source paths)."""
    if isinstance(a, pd.DataFrame):
        pd.testing.assert_frame_equal(a, b, obj=where)
        assert a.attrs == b.attrs, where
    elif isinstance(a, dict):
        assert list(a) == list(b), where
        for k in a:
            assert_same(a[k], b[k], f"{where}[{k!r}]")
    elif isinstance(a, (list, tuple)) and not hasattr(a, "_fields"):
        assert len(a) == len(b), where
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            assert_same(x, y, f"{where}[{i}]")
    elif isinstance(a, float) and pd.isna(a):
        assert pd.isna(b), where
    else:
        assert a == b, where


def assert_same_import(cached, fresh):
    assert_same(cached.trials_by_problem, fresh.trials_by_problem)
    assert cached.log == fresh.log
    assert_same(cached.log.port_finding_sessions, fresh.log.port_finding_sessions, "port finding")


# --- incremental refresh --------------------------------------------------------------------


def test_a_first_refresh_parses_every_readable_file_and_a_second_none(raw, cache, parse_calls):
    first = refresh_import_cache(raw, cache_dir=cache, classify_task_types=True)
    assert len(first.parsed) == 5 and first.reused == () and not first.up_to_date
    assert sorted(parse_calls) == sorted(p.name for p in raw.rglob("*.tsv"))
    parse_calls.clear()
    again = refresh_import_cache(raw, cache_dir=cache, classify_task_types=True)
    assert again.up_to_date and again.parsed == () and parse_calls == []
    assert again.snapshot == first.snapshot


def test_only_new_and_changed_files_are_parsed(raw, cache, parse_calls):
    refresh_import_cache(raw, cache_dir=cache, classify_task_types=True)
    parse_calls.clear()
    result = refresh_import_cache(grow(raw), cache_dir=cache, classify_task_types=True)
    expected = {
        "M2-2026-01-01-100000.tsv",
        "M2-2026-01-02-100000.tsv",
        "M1-2026-01-04-100000.tsv",
        "M1-2026-01-02-100000.tsv",
        "empty.tsv",  # unreadable files are not cached, so they are tried again
    }
    assert set(parse_calls) == expected
    assert {Path(p).name for p in result.parsed} == expected - {"empty.tsv"}
    assert {Path(p).name for p in result.reused} == {
        "M1-2026-01-01-100000.tsv",
        "M1-2026-01-03-100000.tsv",
        "M1.tsv",
    }


@pytest.mark.parametrize("action", ["default", "always"])
def test_the_incremental_result_equals_a_fresh_import(raw, cache, action):
    warnings.simplefilter(action)  # the warnings are part of the log, so they must be shown
    refresh_import_cache(raw, cache_dir=cache, classify_task_types=True)
    incremental = refresh_import_cache(grow(raw), cache_dir=cache, classify_task_types=True)
    fresh = load_trials_with_log(raw, use_cache=False, classify_task_types=True)
    assert_same_import(incremental.imported, fresh)
    assert {w.category for w in fresh.log.warnings} >= {"UserWarning", "ImportSkipWarning"}
    assert any("non-numeric" in w.message for w in fresh.log.warnings)
    # and the stored snapshot holds the same
    with pytest.warns(StaleCacheWarning):
        loaded = load_trials_with_log(
            raw, cache_dir=cache, classify_task_types=True, from_cache=True
        )
    assert_same(loaded.trials_by_problem, fresh.trials_by_problem)


def test_a_changed_import_code_reassembles_from_the_cached_parses(
    raw, cache, parse_calls, monkeypatch
):
    first = refresh_import_cache(raw, cache_dir=cache, classify_task_types=True)
    parse_calls.clear()
    monkeypatch.setattr(load_trials, "import_code_version", lambda: "edited")
    result = refresh_import_cache(raw, cache_dir=cache, classify_task_types=True)
    assert not result.up_to_date and result.parsed == ()
    assert [p.name for p in raw.rglob("*.tsv") if p.name != "empty.tsv"] != []
    assert parse_calls == ["empty.tsv"]
    assert result.snapshot != first.snapshot


def test_a_refresh_prunes_the_old_snapshot_and_the_parses_only_it_used(raw, cache):
    first = refresh_import_cache(raw, cache_dir=cache, classify_task_types=True)
    n_parses = len(list_file_entries(cache))
    second = refresh_import_cache(grow(raw), cache_dir=cache, classify_task_types=True)
    assert list_snapshots(cache).snapshots == (second.snapshot,)
    assert first.snapshot.path in {i.path for i in second.pruned}
    # the changed and the removed file's old parses go; four new parses arrive
    assert len(list_file_entries(cache)) == n_parses - 2 + 4


def test_keep_old_skips_the_prune(raw, cache):
    first = refresh_import_cache(raw, cache_dir=cache, classify_task_types=True)
    second = refresh_import_cache(
        grow(raw), cache_dir=cache, classify_task_types=True, keep_old=True
    )
    assert second.pruned == ()
    assert set(list_snapshots(cache).snapshots) == {first.snapshot, second.snapshot}


# --- robustness -----------------------------------------------------------------------------


def test_a_share_failing_mid_refresh_leaves_the_old_snapshot_in_use(raw, cache, monkeypatch):
    first = refresh_import_cache(raw, cache_dir=cache, classify_task_types=True)
    grow(raw)
    original = cache_store.parse_tsv_file

    def flaky(path):
        if "M2" in Path(path).name:
            raise TimeoutError("the share stopped answering")
        return original(path)

    monkeypatch.setattr(cache_store, "parse_tsv_file", flaky)
    with pytest.raises(ShareReadError, match="stopped answering.*cache is unchanged"):
        refresh_import_cache(raw, cache_dir=cache, classify_task_types=True)
    assert list_snapshots(cache).snapshots == (first.snapshot,)
    with pytest.warns(StaleCacheWarning):
        loaded = load_trials_with_log(
            raw, cache_dir=cache, classify_task_types=True, from_cache=True
        )
    assert_same(loaded.trials_by_problem, first.imported.trials_by_problem)


def test_a_snapshot_write_failing_leaves_the_old_snapshot_in_use(raw, cache, monkeypatch):
    first = refresh_import_cache(raw, cache_dir=cache, classify_task_types=True)

    def crash(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(load_trials, "write_snapshot", crash)
    with pytest.raises(OSError, match="disk full"):
        refresh_import_cache(grow(raw), cache_dir=cache, classify_task_types=True)
    assert list_snapshots(cache).snapshots == (first.snapshot,)


def tree_digest(root: Path) -> dict:
    return {
        str(p.relative_to(root)): (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest())
        for p in sorted(root.rglob("*"))
        if p.is_file()
    } | {str(p.relative_to(root)): "dir" for p in root.rglob("*") if p.is_dir()}


@contextmanager
def writes_under(prefix: str):
    """Record every write-mode open, rename, mkdir and removal under *prefix*."""
    seen = []

    def spy(name, original, is_write=lambda *a, **k: True):
        def wrapper(path, *args, **kwargs):
            if str(os.fspath(path)).startswith(prefix) and is_write(*args, **kwargs):
                seen.append((name, str(path)))
            return original(path, *args, **kwargs)

        return wrapper

    def write_mode(mode="r", *args, **kwargs):
        return any(c in mode for c in "wax+")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(builtins, "open", spy("open", builtins.open, write_mode))
        for name in ("rename", "replace", "mkdir", "makedirs", "remove", "unlink", "rmdir"):
            patch.setattr(os, name, spy(name, getattr(os, name)))
        yield seen


def test_refresh_check_and_prune_never_write_under_the_raw_folder(raw, cache):
    before = tree_digest(raw)
    with writes_under(str(raw)) as seen:
        refresh_import_cache(raw, cache_dir=cache, classify_task_types=True)
        check_import_cache(raw, cache_dir=cache, classify_task_types=True)
        load_trials.prune_import_cache(cache, dry_run=False)
    assert seen == []
    assert tree_digest(raw) == before


# --- cache-first loads ----------------------------------------------------------------------


@contextmanager
def file_system_spy(prefix: str):
    """Record every os-level access to a path under *prefix* made inside the block."""
    touched = []

    def spy(name, original):
        def wrapper(p, *args, **kwargs):
            try:
                s = os.fspath(p)
            except TypeError:
                s = ""
            if str(s).startswith(prefix):
                touched.append((name, s))
            return original(p, *args, **kwargs)

        return wrapper

    with pytest.MonkeyPatch.context() as patch:
        for name in ("stat", "lstat", "listdir", "scandir", "access"):
            patch.setattr(os, name, spy(name, getattr(os, name)))
        patch.setattr(builtins, "open", spy("open", builtins.open))
        yield touched


def test_a_cache_first_load_uses_the_newest_snapshot_without_touching_the_folder(raw, cache):
    refresh_import_cache(raw, cache_dir=cache, classify_task_types=True)
    newest = refresh_import_cache(
        grow(raw), cache_dir=cache, classify_task_types=True, keep_old=True
    )
    with file_system_spy(str(raw)) as touched, pytest.warns(StaleCacheWarning) as caught:
        loaded = load_trials_with_log(
            raw, cache_dir=cache, classify_task_types=True, from_cache=True
        )
    assert touched == []
    assert_same(loaded.trials_by_problem, newest.imported.trials_by_problem)
    source = loaded.log.cache_source
    assert source.time == newest.snapshot.manifest.created
    assert source.n_files == len(newest.snapshot.manifest.files)
    assert "--refresh-cache" in str(caught[0].message)


def test_a_cache_first_load_prints_the_date_the_files_and_how_to_refresh(raw, cache, capsys):
    refresh_import_cache(raw, cache_dir=cache, classify_task_types=True)
    capsys.readouterr()
    load_trials_with_log(raw, cache_dir=cache, classify_task_types=True, from_cache=True)
    out = capsys.readouterr().out
    assert "import cache of" in out and "6 .tsv files" in out and "--refresh-cache" in out


def test_a_snapshot_is_preferred_to_a_newer_legacy_import(raw, cache, monkeypatch):
    refresh_import_cache(raw, cache_dir=cache, classify_task_types=True)
    root = raw.resolve()
    monkeypatch.setattr(
        load_trials,
        "_import_and_extract",
        lambda r, c=False, **_: load_trials.ImportedTrials({"legacy": 1}, load_trials.ImportLog()),
    )
    load_trials._cached_load(cache)(
        str(root), load_trials.raw_data_fingerprint(root), "older code", True
    )
    loaded = load_trials_with_log(raw, cache_dir=cache, classify_task_types=True, from_cache=True)
    assert "legacy" not in loaded.trials_by_problem


def test_a_moved_root_matches_its_snapshot_on_the_path_tail(raw, cache, tmp_path):
    refresh_import_cache(raw, cache_dir=cache, classify_task_types=True)
    elsewhere = tmp_path / "other-mount" / "exp" / "rawdata" / "cohort-01"
    with pytest.warns(StaleCacheWarning, match="matched on its last 3 path parts"):
        loaded = load_trials_with_log(
            elsewhere, cache_dir=cache, classify_task_types=True, from_cache=True
        )
    assert loaded.log.cache_source.matched_on_tail


def test_with_no_cache_at_all_a_cache_first_load_imports_and_caches(raw, cache, capsys):
    loaded = load_trials_with_log(raw, cache_dir=cache, classify_task_types=True, from_cache=True)
    assert "no import cache yet; importing from the server" in capsys.readouterr().out
    assert loaded.log.cache_source is None
    assert len(list_snapshots(cache).snapshots) == 1
    fresh = load_trials_with_log(raw, use_cache=False, classify_task_types=True)
    assert_same_import(loaded, fresh)


def test_with_no_cache_and_no_share_the_error_says_both(cache, tmp_path):
    with pytest.raises(FileNotFoundError, match="no cached imports|not reachable"):
        load_trials_with_log(tmp_path / "gone", cache_dir=cache, from_cache=True)


# --- check ----------------------------------------------------------------------------------


def test_check_reports_new_changed_and_removed_files_by_subject_and_session(raw, cache):
    refresh_import_cache(raw, cache_dir=cache, classify_task_types=True)
    up_to_date = check_import_cache(raw, cache_dir=cache, classify_task_types=True)
    assert up_to_date.up_to_date
    assert describe_check(up_to_date, "cohort-01")[-1] == "  up to date"
    check = check_import_cache(grow(raw), cache_dir=cache, classify_task_types=True)
    assert not check.up_to_date
    assert (check.diff.count("new"), check.diff.count("changed"), check.diff.count("removed")) == (
        3,
        1,
        1,
    )
    text = "\n".join(describe_check(check, "cohort-01"))
    assert "cohort-01: cache of" in text and "snapshot, 6 .tsv files, current import code" in text
    assert "3 new file(s) (1 new subject(s), 3 new session(s)), 1 changed, 1 removed" in text
    assert "M2 (new subject)" in text
    assert "ses-4_date-20260104 (new session)" in text
    assert "removed  M1-2026-01-01-110000.tsv" in text
    assert "changed  M1-2026-01-02-100000.tsv (size" in text


def test_check_without_a_cache_says_so(raw, cache):
    check = check_import_cache(raw, cache_dir=cache, classify_task_types=True)
    assert check.cached is None and not check.up_to_date
    lines = describe_check(check, "cohort-01")
    assert lines[0].startswith("cohort-01: no import cache of")
    assert "6 .tsv files, 1 subject(s), 4 session(s); none are cached" in lines[1]


def test_check_compares_a_legacy_import_by_its_recorded_files(raw, cache, monkeypatch):
    root = raw.resolve()
    monkeypatch.setattr(
        load_trials,
        "_import_and_extract",
        lambda r, c=False, **_: load_trials.ImportedTrials({}, load_trials.ImportLog()),
    )
    load_trials._cached_load(cache)(
        str(root), load_trials.raw_data_fingerprint(root), "older code", True
    )
    check = check_import_cache(grow(raw), cache_dir=cache, classify_task_types=True)
    assert check.cached.kind == "legacy" and not check.cached.code_matches
    assert check.diff.count("new") == 3
    assert "legacy, 6 .tsv files, import code changed since" in describe_check(check, "c")[0]


def test_check_fails_clearly_when_the_share_is_missing(cache, tmp_path):
    with pytest.raises(FileNotFoundError, match="is the share mounted"):
        check_import_cache(tmp_path / "gone", cache_dir=cache)
