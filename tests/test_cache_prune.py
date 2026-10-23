"""Pruning the import cache: the plan (pure) and applying it (only inside the cache)."""

from pathlib import Path
from typing import NamedTuple

import pytest

from src.behavior_import.cache_prune import (
    FileEntry,
    PruneItem,
    apply_prune,
    list_file_entries,
    prune_plan,
)
from src.behavior_import.cache_store import (
    Manifest,
    ServerFile,
    Snapshot,
    file_key,
    files_dir,
)

C3 = "/Volumes/behrens/meg/3x3_field_magnitude_bandit/rawdata/cohort-03"
C4 = C3.replace("03", "04")
SLUG3 = "3x3_field_magnitude_bandit_rawdata_cohort-03"
DAY = 24 * 3600


class Legacy(NamedTuple):
    path: Path
    root: str
    classify_task_types: bool
    time: float


def snap(name, *, root=C3, classify=True, created=0.0, files=(), version="fv"):
    return Snapshot(Path(name), Manifest(root, classify, "c", version, created, tuple(files)))


def plan(snapshots=(), legacy=(), files=(), incomplete=(), now=10 * DAY, cohorts=None, roots=None):
    return prune_plan(snapshots, legacy, files, incomplete, now=now, cohorts=cohorts, roots=roots)


def removed(items):
    return sorted(str(i.path) for i in items)


def test_older_snapshots_of_a_cohort_and_flag_are_removed_and_the_newest_kept():
    snaps = [
        snap("old", created=1.0),
        snap("new", created=2.0),
        snap("other-flag", classify=False, created=0.5),
        snap("other-cohort", root=C4, created=0.1),
    ]
    items = plan(snaps)
    assert removed(items) == ["old"]
    assert items[0].kind == "snapshot"


def test_a_snapshot_of_the_same_cohort_at_another_root_keeps_its_own_newest():
    copy = "/tmp/partial-copy" + C3[16:]
    snaps = [snap("share-old", created=1.0), snap("share", created=2.0)]
    snaps.append(snap("copy", root=copy, created=3.0))
    assert removed(plan(snaps)) == ["share-old"]


def test_roots_limit_the_plan_to_those_exact_roots():
    copy = "/tmp/partial-copy" + C3[16:]
    snaps = [snap("share-old", created=1.0), snap("share", created=2.0)]
    snaps += [snap("copy-old", root=copy, created=3.0), snap("copy", root=copy, created=4.0)]
    assert removed(plan(snaps, roots=[copy])) == ["copy-old"]


def test_a_legacy_import_goes_only_for_a_newer_snapshot_of_its_exact_root():
    legacy = [Legacy(Path("legacy"), C3, True, 1.0)]
    assert plan([snap("copy", root="/tmp/partial-copy" + C3[16:], created=3.0)], legacy) == []


def test_parse_temporaries_older_than_a_day_are_removed():
    temps = [
        (Path("import-files/c/.tmp-k.pkl.1"), 0.0),
        (Path("import-files/c/.tmp-j.pkl.2"), 10 * DAY),
    ]
    assert removed(plan(incomplete=temps)) == ["import-files/c/.tmp-k.pkl.1"]


def test_legacy_entries_go_only_when_a_newer_snapshot_of_the_cohort_exists():
    legacy = [
        Legacy(Path("legacy-old"), C3, True, 1.0),
        Legacy(Path("legacy-newer"), C3, True, 5.0),
        Legacy(Path("legacy-other-flag"), C3, False, 1.0),
        Legacy(Path("legacy-only-cache"), C4, True, 1.0),
    ]
    items = plan([snap("s", created=3.0)], legacy)
    assert removed(items) == ["legacy-old"]
    assert items[0].kind == "legacy"


def test_the_only_cache_of_a_cohort_is_never_removed():
    assert plan([snap("only")]) == []
    assert plan([], [Legacy(Path("only"), C3, True, 1.0)]) == []


def test_per_file_entries_not_referenced_by_a_kept_snapshot_are_removed():
    kept_file, old_file = ServerFile("a.tsv", 1, 1), ServerFile("a.tsv", 1, 0)
    snaps = [
        snap("old", created=1.0, files=[old_file]),
        snap("new", created=2.0, files=[kept_file]),
    ]
    files = [
        FileEntry(Path("kept"), SLUG3, file_key(kept_file, "fv")),
        FileEntry(Path("only-in-old"), SLUG3, file_key(old_file, "fv")),
        FileEntry(Path("old-code"), SLUG3, file_key(kept_file, "older code")),
    ]
    assert removed(plan(snaps, files=files)) == ["old", "old-code", "only-in-old"]


def test_per_file_entries_of_either_flag_are_kept():
    f = ServerFile("a.tsv", 1, 1)
    snaps = [snap("unclassified", classify=False, files=[f])]
    assert plan(snaps, files=[FileEntry(Path("e"), SLUG3, file_key(f, "fv"))]) == []


def test_temporary_folders_older_than_a_day_are_removed():
    incomplete = [(Path(".tmp-old"), 10 * DAY - 2 * DAY), (Path(".tmp-running"), 10 * DAY - 60)]
    assert removed(plan(incomplete=incomplete)) == [".tmp-old"]


def test_cohorts_limit_the_plan_to_those_cohorts():
    snaps = [snap("c3-old", created=1.0), snap("c3", created=2.0)]
    snaps += [snap("c4-old", root=C4, created=1.0), snap("c4", root=C4, created=2.0)]
    files = [FileEntry(Path("c4-file"), SLUG3.replace("03", "04"), "unreferenced")]
    assert removed(plan(snaps, files=files, cohorts=[C3])) == ["c3-old"]
    assert removed(plan(snaps, files=files)) == ["c3-old", "c4-file", "c4-old"]


# --- applying it ----------------------------------------------------------------------------


def test_a_dry_run_deletes_nothing_and_a_real_run_frees_the_space(tmp_path, capsys):
    folder = tmp_path / "import-snapshots" / "old"
    folder.mkdir(parents=True)
    (folder / "trials.pkl").write_bytes(b"x" * 1000)
    items = [PruneItem(folder, "snapshot", "older than the one of 2026-10-20 12:00")]
    assert apply_prune(tmp_path, items, dry_run=True) == 1000
    assert folder.exists()
    assert "would remove" in capsys.readouterr().out
    assert apply_prune(tmp_path, items, dry_run=False) == 1000
    assert not folder.exists()
    assert "removed" in capsys.readouterr().out


def test_nothing_outside_the_cache_is_ever_deleted(tmp_path):
    outside = tmp_path / "share" / "raw.tsv"
    outside.parent.mkdir()
    outside.write_text("data")
    with pytest.raises(ValueError, match="outside the cache"):
        apply_prune(tmp_path / "cache", [PruneItem(outside, "file", "x")], dry_run=False)
    assert outside.exists()


def test_the_per_file_entries_are_listed_by_cohort_and_key(tmp_path):
    folder = files_dir(tmp_path, C3)
    folder.mkdir(parents=True)
    (folder / "abc.pkl").write_bytes(b"")
    (folder / ".tmp-abc.pkl.1234").write_bytes(b"")
    assert list_file_entries(tmp_path) == (FileEntry(folder / "abc.pkl", SLUG3, "abc"),)
    assert list_file_entries(tmp_path / "none") == ()
