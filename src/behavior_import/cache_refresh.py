"""Bringing a cohort's import cache up to date with the server, and pruning the cache.

:func:`refresh_import_cache` lists a raw-data folder, reuses the cached parse of every file whose
path, size and modification time are unchanged, parses the new and changed ones, assembles the
cohort from all of them and stores it as a snapshot (see :mod:`~src.behavior_import.cache_store`);
then it prunes that root's older snapshots, the legacy imports they replace and the parses no
kept snapshot uses (:func:`prune_import_cache`). :mod:`~src.behavior_import.load_trials` calls it
for every load that reads the server, so a notebook's ``load_trials_with_log(root)`` refreshes
and prunes too: its first call deletes that root's legacy joblib imports once the snapshot
exists.

Nothing is written when the cache directory lies inside the data root or the folder being
refreshed (:func:`~src.behavior_import.cache_store.refuse_cache_inside`).
"""

import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import NamedTuple

from pydantic import ValidationError

from src.behavior_import import load_trials
from src.behavior_import.cache_prune import PruneItem, apply_prune, list_file_entries, prune_plan
from src.behavior_import.cache_store import (
    CachedParser,
    Manifest,
    ServerFile,
    Snapshot,
    file_code_version,
    files_dir,
    incomplete_snapshots,
    list_server_files,
    list_snapshots,
    refuse_cache_inside,
    stray_parse_temps,
    write_snapshot,
)
from src.behavior_import.load_trials import ImportedTrials
from src.settings import get_settings, get_unchecked_settings


class RefreshResult(NamedTuple):
    """What :func:`refresh_import_cache` did.

    Attributes:
        imported (ImportedTrials): The cohort as now cached.
        snapshot (Snapshot): The snapshot it is stored in (new, or the current one).
        parsed (tuple[str, ...]): Files parsed in this refresh (new or changed), relative paths.
        reused (tuple[str, ...]): Files whose earlier parse was reused.
        skipped (tuple[str, ...]): Files the import skipped because they could not be read; the
            next refresh tries them again.
        up_to_date (bool): Whether the newest snapshot already matched the server and the
            import code, so nothing was parsed or assembled.
        pruned (tuple[PruneItem, ...]): What the prune after the refresh removed.
    """

    imported: ImportedTrials
    snapshot: Snapshot
    parsed: tuple[str, ...]
    reused: tuple[str, ...]
    skipped: tuple[str, ...]
    up_to_date: bool
    pruned: tuple[PruneItem, ...]


def configured_data_root() -> list[str]:
    """``[BANDIT_DATA_ROOT]`` as set (not checked, so the share is not touched), or ``[]``."""
    try:
        return [str(get_unchecked_settings().data_root)]
    except ValidationError:
        return []


def current_snapshot(snapshots: Iterable[Snapshot], manifest: Manifest) -> Snapshot | None:
    """The newest snapshot of *manifest*'s root and flag, if it was made from the same files
    with the same code; ``None`` otherwise.

    A snapshot that skipped unreadable files does not cover them, so it never matches a
    listing that has them and the next refresh tries them again.

    Args:
        snapshots (Iterable[Snapshot]): The listed snapshots.
        manifest (Manifest): The manifest a refresh would write now, all listed files in
            ``files``.

    Returns:
        Snapshot | None: A snapshot that a refresh can load instead of assembling.
    """
    same = [
        s
        for s in snapshots
        if (s.manifest.root, s.manifest.classify_task_types)
        == (manifest.root, manifest.classify_task_types)
    ]
    newest = max(same, key=lambda s: s.manifest.created, default=None)
    fields = ("files", "code_version", "file_code_version")
    if newest and all(getattr(newest.manifest, f) == getattr(manifest, f) for f in fields):
        return newest
    return None


def _assemble_and_store(
    root: Path, cache_dir: Path, manifest: Manifest, listing: tuple[ServerFile, ...]
) -> tuple[ImportedTrials, Snapshot, CachedParser]:
    """Import *root* reusing the per-file parses, and store the result as a snapshot."""
    parser = CachedParser(
        root,
        files_dir(cache_dir, manifest.root),
        {f.path: f for f in listing},
        manifest.file_code_version,
    )
    imported = load_trials._import_and_extract(
        root,
        manifest.classify_task_types,
        parse=parser,
        paths=[root / f.path for f in listing],
    )
    skipped = parser.skipped()
    covered = tuple(f for f in listing if f not in set(skipped))
    stored = Manifest(
        manifest.root,
        manifest.classify_task_types,
        manifest.code_version,
        manifest.file_code_version,
        manifest.created,
        covered,
        skipped,
    )
    return imported, write_snapshot(cache_dir, imported, stored), parser


def refresh_import_cache(
    root: Path,
    *,
    cache_dir: Path | None = None,
    classify_task_types: bool = False,
    keep_old: bool = False,
    progress: Callable[[int], None] | None = None,
) -> RefreshResult:
    """Bring *root*'s cache snapshot up to date with the server, parsing only what changed.

    Lists the folder; if the newest snapshot was made from exactly these files with the current
    import code it is loaded and nothing else is done. Otherwise each file's cached parse is
    reused when its path, size and modification time are unchanged, new and changed files are
    parsed (and cached), the whole cohort is assembled from them, and a new snapshot is written.
    Then this root's old snapshots, the legacy imports they replace and unused parses are pruned.
    If the share fails part-way, the error says so and the previous snapshot stays the one
    loaded; a single unreadable file is skipped, as a fresh import skips it.

    Args:
        root (Path): The raw-data folder.
        cache_dir (Path | None): Cache location. Default ``None``: ``get_settings().cache_dir``.
            Keyword-only.
        classify_task_types (bool): As for
            :func:`~src.behavior_import.load_trials.load_trials_with_log`. Default ``False``.
            Keyword-only.
        keep_old (bool): Skip the prune. Default ``False``. Keyword-only.
        progress (Callable[[int], None] | None): Called while listing, as for
            :func:`~src.behavior_import.cache_store.list_server_files`. Keyword-only.

    Returns:
        RefreshResult: The cohort, its snapshot, which files were parsed, reused or skipped, and
        what was pruned.

    Raises:
        FileNotFoundError: If *root* is not an existing directory.
        CacheLocationError: If the cache directory is inside *root* or the data root.
        ShareReadError: If the folder cannot be listed or the share stops answering.
    """
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Raw data folder not found (is the share mounted?): {root}")
    cache_dir = Path(cache_dir or get_settings().cache_dir)
    refuse_cache_inside(cache_dir, [root, *configured_data_root()])
    listing = list_server_files(root, progress)
    manifest = Manifest(
        str(root),
        classify_task_types,
        load_trials.import_code_version(),
        file_code_version(),
        time.time(),
        listing,
    )
    current = current_snapshot(list_snapshots(cache_dir).snapshots, manifest)
    if current is not None:
        print(f"[INFO] Loading trials for {root} from cache (up to date with the server)")
        imported, _ = load_trials.load_snapshot_import(current)
        snapshot, parsed, reused, skipped = current, (), tuple(f.path for f in listing), ()
    else:
        imported, snapshot, parser = _assemble_and_store(root, cache_dir, manifest, listing)
        parsed, reused = tuple(parser.parsed), tuple(parser.reused)
        skipped = tuple(f.path for f in snapshot.manifest.skipped)
        print(
            f"[INFO] {root}: parsed {len(parsed)} new or changed file(s), reused {len(reused)}"
            + (f", skipped {len(skipped)} unreadable" if skipped else "")
            + f"; new cache snapshot {snapshot.path.name}"
        )
    pruned = ()
    if not keep_old:
        pruned = tuple(prune_import_cache(cache_dir, roots=[str(root)], report_empty=False))
    return RefreshResult(imported, snapshot, parsed, reused, skipped, current is not None, pruned)


def prune_import_cache(
    cache_dir: Path | None = None,
    *,
    roots: Iterable[str] | None = None,
    cohorts: Iterable[str] | None = None,
    dry_run: bool = False,
    report_empty: bool = True,
) -> list[PruneItem]:
    """Remove the cache entries no load would use (see :mod:`~src.behavior_import.cache_prune`).

    Args:
        cache_dir (Path | None): Cache location. Default ``None``: the ``BANDIT_CACHE_DIR``
            setting (the share is not touched).
        roots (Iterable[str] | None): Exact raw-data folders to limit the prune to. Default
            ``None``. Keyword-only.
        cohorts (Iterable[str] | None): Raw-data folders whose cohort (path tail) limits the
            prune, whatever root it was cached from. Default ``None``. With neither, the whole
            cache is pruned. Keyword-only.
        dry_run (bool): Print what would go and delete nothing. Default ``False``. Keyword-only.
        report_empty (bool): Print a line when there is nothing to remove. Default ``True``.
            Keyword-only.

    Returns:
        list[PruneItem]: What was (or would be) removed.

    Raises:
        CacheLocationError: If the cache directory is inside the data root.
    """
    cache_dir = Path(cache_dir or get_unchecked_settings().cache_dir)
    refuse_cache_inside(cache_dir, configured_data_root())
    items = prune_plan(
        list_snapshots(cache_dir).snapshots,
        load_trials.cached_import_entries(cache_dir).entries,
        list_file_entries(cache_dir),
        (*incomplete_snapshots(cache_dir), *stray_parse_temps(cache_dir)),
        now=time.time(),
        roots=roots,
        cohorts=cohorts,
    )
    if items:
        apply_prune(cache_dir, items, dry_run=dry_run)
    elif report_empty:
        print("[PRUNE] nothing to remove")
    return items
