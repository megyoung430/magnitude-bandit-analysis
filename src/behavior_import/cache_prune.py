"""Removing import-cache entries that no load would use any more.

:func:`prune_plan` decides, from listings alone, what goes:

- a cohort snapshot older than the newest one of the same raw folder (its exact root) and
  ``classify_task_types``; the same cohort reached through another path (a partial copy, another
  mount point) keeps its own newest snapshot, since matching on the path's tail is only a
  fallback for loading;
- a legacy joblib import of a root that has a newer snapshot with the same flag;
- a per-file parse that no kept snapshot of its cohort folder refers to;
- a temporary snapshot folder or per-file parse temporary older than a day (a write that was
  cut off).

The newest snapshot of a root is always kept, and a legacy import only goes once a snapshot
replaces it, so a cohort's only cache is never removed. :func:`apply_prune` deletes the plan,
refusing any path outside the cache directory.
"""

import shutil
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import NamedTuple, Protocol

from src.behavior_import.cache_store import (
    FILES_DIR,
    TMP_PREFIX,
    Snapshot,
    cohort_slug,
    file_key,
)

SNAPSHOT, LEGACY, FILE, INCOMPLETE = "snapshot", "legacy", "file", "incomplete"
# A temporary snapshot folder this old belongs to a write that was cut off, not one running.
INCOMPLETE_AFTER_SECONDS = 24 * 3600


class LegacyEntry(Protocol):
    """What the plan needs of a legacy import (a ``load_trials.CacheEntry``)."""

    path: Path
    root: str
    classify_task_types: bool
    time: float


class FileEntry(NamedTuple):
    """One stored per-file parse.

    Attributes:
        path (Path): Its file.
        cohort (str): Its cohort folder name (:func:`~src.behavior_import.cache_store.cohort_slug`).
        key (str): Its :func:`~src.behavior_import.cache_store.file_key`.
    """

    path: Path
    cohort: str
    key: str


class PruneItem(NamedTuple):
    """Something the plan removes.

    Attributes:
        path (Path): The file or folder.
        kind (str): ``"snapshot"``, ``"legacy"``, ``"file"`` or ``"incomplete"``.
        reason (str): Why, in plain words.
    """

    path: Path
    kind: str
    reason: str


def _time(epoch_seconds: float) -> str:
    return datetime.fromtimestamp(epoch_seconds).strftime("%Y-%m-%d %H:%M")


def _newest_snapshots(snapshots: Iterable[Snapshot]) -> dict[tuple[str, bool], Snapshot]:
    """``{(root, classify_task_types): newest snapshot}``."""
    newest: dict[tuple[str, bool], Snapshot] = {}
    for s in snapshots:
        key = (s.manifest.root, s.manifest.classify_task_types)
        if key not in newest or s.manifest.created > newest[key].manifest.created:
            newest[key] = s
    return newest


def _referenced_files(kept: Iterable[Snapshot]) -> set[tuple[str, str]]:
    return {
        (cohort_slug(s.manifest.root), file_key(f, s.manifest.file_code_version))
        for s in kept
        for f in s.manifest.files
    }


class _Scope(NamedTuple):
    """Which entries a plan may touch: exact roots and/or cohort folders; ``None`` is all."""

    roots: frozenset[str] | None
    slugs: frozenset[str] | None

    def has_root(self, root: str) -> bool:
        in_roots = self.roots is None or root in self.roots
        return in_roots and (self.slugs is None or cohort_slug(root) in self.slugs)

    def has_slug(self, slug: str) -> bool:
        if self.roots is not None and slug not in {cohort_slug(r) for r in self.roots}:
            return False
        return self.slugs is None or slug in self.slugs

    @property
    def whole_cache(self) -> bool:
        return self.roots is None and self.slugs is None


def prune_plan(
    snapshots: Iterable[Snapshot],
    legacy: Iterable[LegacyEntry],
    files: Iterable[FileEntry],
    incomplete: Iterable[tuple[Path, float]],
    *,
    now: float,
    roots: Iterable[str] | None = None,
    cohorts: Iterable[str] | None = None,
) -> list[PruneItem]:
    """What to remove from the cache (see the module docstring for the rules).

    Args:
        snapshots (Iterable[Snapshot]): The listed snapshots.
        legacy (Iterable[LegacyEntry]): The legacy joblib imports.
        files (Iterable[FileEntry]): The per-file parses.
        incomplete (Iterable[tuple[Path, float]]): ``(path, mtime)`` of temporary snapshot
            folders and per-file parse temporaries.
        now (float): The current time, in epoch seconds. Keyword-only.
        roots (Iterable[str] | None): Exact raw-data folders to limit the plan to (a refresh
            prunes only the root it refreshed); ``None`` for no such limit. Keyword-only.
        cohorts (Iterable[str] | None): Raw-data folders whose cohort (path tail) limits the plan,
            whatever root it was cached from; ``None`` for no such limit. Temporaries are only
            removed when neither limit is given. Keyword-only.

    Returns:
        list[PruneItem]: Snapshots, then legacy imports, per-file parses and temporaries.
    """
    snapshots = list(snapshots)
    newest = _newest_snapshots(snapshots)
    scope = _Scope(
        None if roots is None else frozenset(str(r) for r in roots),
        None if cohorts is None else frozenset(cohort_slug(str(c)) for c in cohorts),
    )

    def newest_of(s: Snapshot) -> Snapshot:
        return newest[(s.manifest.root, s.manifest.classify_task_types)]

    items = [
        PruneItem(s.path, SNAPSHOT, f"older than the snapshot of {_time(keep.manifest.created)}")
        for s in snapshots
        if scope.has_root(s.manifest.root) and (keep := newest_of(s)) != s
    ]
    for entry in legacy:
        replacement = newest.get((entry.root, entry.classify_task_types))
        if scope.has_root(entry.root) and replacement and replacement.manifest.created > entry.time:
            reason = f"replaced by the snapshot of {_time(replacement.manifest.created)}"
            items.append(PruneItem(entry.path, LEGACY, reason))
    referenced = _referenced_files(newest.values())
    items += [
        PruneItem(f.path, FILE, "no kept snapshot uses this parse")
        for f in files
        if scope.has_slug(f.cohort) and (f.cohort, f.key) not in referenced
    ]
    if scope.whole_cache:
        items += [
            PruneItem(path, INCOMPLETE, f"an interrupted write of {_time(mtime)}")
            for path, mtime in incomplete
            if now - mtime > INCOMPLETE_AFTER_SECONDS
        ]
    return items


# --- I/O ------------------------------------------------------------------------------------


def list_file_entries(cache_dir: Path) -> tuple[FileEntry, ...]:
    """Every stored per-file parse under *cache_dir*.

    Args:
        cache_dir (Path): The cache directory.

    Returns:
        tuple[FileEntry, ...]: Sorted by cohort and key; temporary files of a write are left out.
    """
    base = Path(cache_dir) / FILES_DIR
    if not base.is_dir():
        return ()
    return tuple(
        FileEntry(p, p.parent.name, p.stem)
        for folder in sorted(base.iterdir())
        if folder.is_dir()
        for p in sorted(folder.glob("*.pkl"))
        if not p.name.startswith(TMP_PREFIX)
    )


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _megabytes(n_bytes: int) -> str:
    return f"{n_bytes / 1e6:,.1f} MB"


def _check_inside(cache_dir: Path, path: Path) -> None:
    if not Path(path).resolve().is_relative_to(Path(cache_dir).resolve()):
        raise ValueError(f"Refusing to remove {path}: it is outside the cache {cache_dir}")


def apply_prune(cache_dir: Path, items: Iterable[PruneItem], *, dry_run: bool) -> int:
    """Delete the planned items (or, for a dry run, only list them) and print each.

    Per-file parses are summed in one line per cohort folder rather than listed one by one.

    Args:
        cache_dir (Path): The cache directory; every item must be inside it.
        items (Iterable[PruneItem]): From :func:`prune_plan`.
        dry_run (bool): List only. Keyword-only.

    Returns:
        int: Bytes freed (or that would be).

    Raises:
        ValueError: If an item is outside *cache_dir*; checked before anything is deleted.
    """
    items = list(items)
    for item in items:
        _check_inside(cache_dir, item.path)
    verb = "would remove" if dry_run else "removed"
    freed, file_totals = 0, {}
    for item in items:
        size = _size(item.path)
        freed += size
        if not dry_run and item.path.is_dir():
            shutil.rmtree(item.path)
        elif not dry_run:
            item.path.unlink()
        if item.kind == FILE:
            count, total = file_totals.get(item.path.parent, (0, 0))
            file_totals[item.path.parent] = (count + 1, total + size)
        else:
            print(f"[PRUNE] {verb} {item.kind} {item.path} ({_megabytes(size)}): {item.reason}")
    for folder, (count, total) in sorted(file_totals.items()):
        print(
            f"[PRUNE] {verb} {count} per-file parse(s) in {folder} ({_megabytes(total)}): no "
            "kept snapshot uses them"
        )
    print(f"[PRUNE] {verb} {len(items)} item(s), {_megabytes(freed)}")
    return freed
