"""The import cache on disk: server listing, per-file parses, cohort snapshots and manifests."""

import os
import pickle
import warnings
from pathlib import Path

import pandas as pd
import pytest

from src.behavior_import import cache_store
from src.behavior_import.cache_store import (
    Manifest,
    ServerFile,
    ShareReadError,
    file_key,
    list_server_files,
    list_snapshots,
    load_snapshot_data,
    parse_and_record,
    replay,
    write_snapshot,
)
from src.behavior_import.import_data import ImportAborted
from tests.test_import_data import session_dir, write_tsv

ROOT = "/Volumes/behrens/meg/3x3_field_magnitude_bandit/rawdata/cohort-03"


def manifest(*, root=ROOT, classify=True, created=1_000.0, files=()):
    return Manifest(root, classify, "code", "file-code", created, tuple(files))


# --- listing the server ---------------------------------------------------------------------


def test_the_listing_gives_relative_paths_sizes_and_mtimes_in_rglob_order(tmp_path):
    a = write_tsv(session_dir(tmp_path), "M1-2026-01-01-100000.tsv")
    write_tsv(session_dir(tmp_path, session="ses-2_date-20260102"), "M1-2026-01-02-100000.tsv")
    (tmp_path / "notes.txt").write_text("not a recording")
    listed = list_server_files(tmp_path)
    assert [f.path for f in listed] == [
        p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*.tsv")
    ]
    first = next(f for f in listed if f.path.endswith(a.name))
    assert (first.size, first.mtime_ns) == (a.stat().st_size, a.stat().st_mtime_ns)


def test_the_listing_reports_progress(tmp_path):
    for i in range(5):
        write_tsv(session_dir(tmp_path), f"M1-2026-01-01-10000{i}.tsv")
    seen = []
    list_server_files(tmp_path, progress=seen.append, every=2)
    assert seen == [2, 4]


def test_a_failing_listing_raises_a_clear_error(tmp_path, monkeypatch):
    def broken(self, pattern):
        raise TimeoutError("share went away")

    monkeypatch.setattr(Path, "rglob", broken)
    with pytest.raises(ShareReadError, match="share went away.*cache is unchanged"):
        list_server_files(tmp_path)
    assert issubclass(ShareReadError, ImportAborted)


# --- manifests and snapshots ----------------------------------------------------------------


def test_a_manifest_round_trips_through_json():
    m = manifest(files=[ServerFile("sub-01_id-M1/ses-1/x.tsv", 10, 123)])
    assert Manifest.from_json(m.to_json()) == m


def test_a_manifest_keeps_its_files_sorted():
    files = [ServerFile("b.tsv", 1, 1), ServerFile("a.tsv", 1, 1)]
    assert [f.path for f in manifest(files=files).files] == ["a.tsv", "b.tsv"]


def test_a_written_snapshot_is_listed_and_loads(tmp_path):
    written = write_snapshot(tmp_path, {"trials": 1}, manifest())
    (listed,) = list_snapshots(tmp_path).snapshots
    assert listed == written
    assert listed.manifest == manifest()
    assert load_snapshot_data(listed) == {"trials": 1}


def test_a_manifest_is_read_without_unpickling_the_data(tmp_path, monkeypatch):
    write_snapshot(tmp_path, {"trials": 1}, manifest())
    monkeypatch.setattr(pickle, "load", lambda *a, **k: pytest.fail("unpickled"))
    assert list_snapshots(tmp_path).snapshots[0].manifest.root == ROOT


def test_a_write_that_fails_part_way_leaves_the_old_snapshot_the_only_one(tmp_path, monkeypatch):
    old = write_snapshot(tmp_path, {"trials": "old"}, manifest(created=1.0))

    def crash(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(pickle, "dump", crash)
    with pytest.raises(OSError, match="disk full"):
        write_snapshot(tmp_path, {"trials": "new"}, manifest(created=2.0))
    monkeypatch.undo()
    assert list_snapshots(tmp_path).snapshots == (old,)
    assert load_snapshot_data(old) == {"trials": "old"}
    assert not any(p.name.startswith(".tmp") for p in old.path.parent.iterdir())


def test_a_snapshot_folder_without_a_manifest_is_not_listed(tmp_path):
    written = write_snapshot(tmp_path, {"trials": 1}, manifest())
    (written.path / cache_store.MANIFEST_FILE).unlink()
    assert list_snapshots(tmp_path).snapshots == ()


def test_an_unreadable_manifest_is_listed_as_unreadable_with_a_warning(tmp_path):
    written = write_snapshot(tmp_path, {"trials": 1}, manifest())
    (written.path / cache_store.MANIFEST_FILE).write_text("{not json")
    with pytest.warns(cache_store.ImportCacheWarning, match="cannot be read"):
        listing = list_snapshots(tmp_path)
    assert listing.unreadable == (written.path,)


def test_no_cache_folder_lists_nothing(tmp_path):
    assert list_snapshots(tmp_path / "nothing").snapshots == ()


# --- per-file parses ------------------------------------------------------------------------


def test_the_file_key_changes_with_path_size_mtime_and_code():
    f = ServerFile("a.tsv", 10, 5)
    keys = {
        file_key(f, "v1"),
        file_key(f._replace(path="b.tsv"), "v1"),
        file_key(f._replace(size=11), "v1"),
        file_key(f._replace(mtime_ns=6), "v1"),
        file_key(f, "v2"),
    }
    assert len(keys) == 5
    assert file_key(f, "v1") == file_key(ServerFile("a.tsv", 10, 5), "v1")


def test_the_file_code_version_is_a_stable_hash():
    assert cache_store.file_code_version() == cache_store.file_code_version()
    assert len(cache_store.file_code_version()) == 64


def write_traceback_recording(folder):
    """A recording with a non-numeric time row, which the reader drops with a warning."""
    folder.mkdir(parents=True, exist_ok=True)
    rows = [("0.0", "info", "task_name", "t"), ("AttributeError: ", "x", "y", "z")]
    path = folder / "M1-2026-01-01-100000.tsv"
    pd.DataFrame(rows, columns=["time", "type", "subtype", "content"]).to_csv(
        path, sep="\t", index=False
    )
    return path


def test_a_parse_records_its_warnings_and_a_replay_shows_them_for_the_new_path(tmp_path):
    path = write_traceback_recording(tmp_path / "a")
    with warnings.catch_warnings(record=True) as shown:
        warnings.simplefilter("always")
        cached = parse_and_record(path)
    assert shown == []
    (caught,) = cached.warnings
    assert str(path) in caught.message
    moved = tmp_path / "b" / path.name
    with warnings.catch_warnings(record=True) as shown:
        warnings.simplefilter("always")
        parsed = replay(cached, moved, {})
    (w,) = shown
    assert str(moved) in str(w.message) and str(path) not in str(w.message)
    assert (w.filename, w.lineno) == (caught.filename, caught.lineno)
    assert parsed.df.attrs["source_path"] == parsed.data.attrs["source_path"] == str(moved)


def test_a_failing_parse_still_shows_the_warnings_it_raised(tmp_path, monkeypatch):
    def warn_then_fail(path):
        warnings.warn("first", UserWarning, stacklevel=1)
        raise ValueError("bad file")

    monkeypatch.setattr(cache_store, "parse_tsv_file", warn_then_fail)
    with warnings.catch_warnings(record=True) as shown:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="bad file"):
            parse_and_record(tmp_path / "x.tsv")
    assert [str(w.message) for w in shown] == ["first"]


def test_a_saved_parse_loads_back(tmp_path):
    path = write_tsv(session_dir(tmp_path / "raw"), "M1-2026-01-01-100000.tsv")
    cached = parse_and_record(path)
    location = tmp_path / "store" / "k.pkl"
    cache_store.save_parsed(location, cached)
    loaded = cache_store.load_parsed(location)
    pd.testing.assert_frame_equal(loaded.parsed.data, cached.parsed.data)
    assert cache_store.load_parsed(tmp_path / "store" / "missing.pkl") is None
    assert [p.name for p in location.parent.iterdir()] == ["k.pkl"]


def test_a_damaged_parse_entry_is_a_miss_with_a_warning(tmp_path):
    location = tmp_path / "k.pkl"
    location.write_bytes(b"not a pickle")
    with pytest.warns(cache_store.ImportCacheWarning, match="re-parsed"):
        assert cache_store.load_parsed(location) is None


def test_per_file_entries_live_in_a_folder_per_cohort(tmp_path):
    folder = cache_store.files_dir(tmp_path, ROOT)
    assert folder.parent == tmp_path / cache_store.FILES_DIR
    assert folder.name == "3x3_field_magnitude_bandit_rawdata_cohort-03"
    assert os.path.commonpath([folder, tmp_path]) == str(tmp_path)
