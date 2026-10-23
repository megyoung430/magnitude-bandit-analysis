"""Import and extract trials for a raw-data folder, through the import cache.

Parsing every pyControl ``.tsv`` in a cohort is the slowest step of any analysis, so imports go
through a cache in ``get_settings().cache_dir`` (see :mod:`~src.behavior_import.cache_store`):

- each file's parse is kept, keyed on its relative path, size, modification time and the
  parsing code, so :func:`refresh_import_cache` parses only new or changed files and reuses
  the rest, then assembles the cohort (sessions, problem numbers, trials) from all of them;
- each refresh stores the assembled :class:`ImportedTrials` as a cohort *snapshot*, with a
  JSON manifest of the files it was made from, and prunes what is no longer needed
  (:func:`prune_import_cache`);
- :func:`~src.behavior_import.cache_check.check_import_cache` compares the newest snapshot's
  manifest with the server and says which subjects, sessions and files are new, changed or
  removed.

:func:`load_trials_with_log` refreshes by default (always current; quick when little changed).
With ``from_cache=True`` it is cache-first: the newest snapshot of the folder is loaded without
reading the folder at all, falling back on a legacy joblib import made by older versions of
this module (:func:`choose_cache_entry`), and on importing from the server only when the
folder has no cache at all. A cache-first load raises a :class:`StaleCacheWarning`, also kept
in the import log, saying when the cache was made, since sessions added since are missing.

The cache holds the trials together with the :class:`~src.behavior_import.import_log.ImportLog`
of the import (what was skipped or flagged, and the warnings it raised), so a cached load has
the same log as the import that filled the cache. Deleting the cache directory only costs time.
"""

import ast
import dataclasses
import hashlib
import os
import time
import warnings
from collections.abc import Callable, Iterable
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

from joblib import Memory

from src.behavior_import.cache_prune import (
    PruneItem,
    apply_prune,
    list_file_entries,
    prune_plan,
)
from src.behavior_import.cache_store import (
    TAIL_PARTS,
    CachedParser,
    Manifest,
    ServerFile,
    Snapshot,
    file_code_version,
    files_dir,
    incomplete_snapshots,
    list_server_files,
    list_snapshots,
    load_snapshot_data,
    path_tail,
    write_snapshot,
)
from src.behavior_import.extract_trials import extract_trials_grouped_by_problem
from src.behavior_import.import_data import import_data
from src.behavior_import.import_log import CacheSource, ImportLog
from src.behavior_import.warning_records import record_warnings
from src.settings import get_settings, get_unchecked_settings

# The source hashed into the cache key. It includes this module, so a change to what is cached
# invalidates the caches written before it.
IMPORT_CODE_DIR = Path(__file__).resolve().parent
# How to bring the cache up to date, quoted wherever a load used the cache without the server.
REFRESH_HINT = (
    "run with --refresh-cache (or scripts/full_pipelines/import_cache.py refresh) to include "
    "new data from the server"
)
# What to do when the share is back, quoted in the errors of a cache-first load.
WHEN_THE_SHARE_IS_BACK = "run with --refresh-cache once the server is reachable"


class StaleCacheWarning(UserWarning):
    """A cached import was loaded without checking it against the raw files."""


class CachedImportError(RuntimeError):
    """A cached import could not be read."""


class CacheMetadataWarning(UserWarning):
    """A cached import was skipped because its metadata could not be read."""


class CacheEntry(NamedTuple):
    """One legacy cached import, as recorded in its joblib ``metadata.json``.

    Older versions of this module cached whole imports with joblib; they are still found and
    loaded (cache-first, when a cohort has no snapshot) and pruned once a snapshot replaces them.

    Attributes:
        path (Path): The entry's folder, holding ``metadata.json`` and ``output.pkl``.
        root (str): The raw-data folder that was imported, as an absolute path.
        classify_task_types (bool): Whether problems were numbered by task type too.
        code_version (str): :func:`import_code_version` when the import ran.
        time (float): When the import was cached, in epoch seconds.
        n_files (int): How many ``.tsv`` files the raw folder held then.
        files (tuple[ServerFile, ...]): Those files, from the cache key. Default empty.
    """

    path: Path
    root: str
    classify_task_types: bool
    code_version: str
    time: float
    n_files: int
    files: tuple[ServerFile, ...] = ()


class ImportedTrials(NamedTuple):
    """What an import produces.

    Attributes:
        trials_by_problem (dict): ``{problem_number: {subject: {session_key: session_dict}}}``.
        log (ImportLog): What the import skipped, flagged and warned about.
    """

    trials_by_problem: dict
    log: ImportLog


def raw_data_fingerprint(root: Path) -> tuple[tuple[str, int, int], ...]:
    """``(relative path, size, mtime_ns)`` of every ``.tsv`` under *root*, sorted.

    Args:
        root (Path): Raw-data folder to scan recursively.

    Returns:
        tuple[tuple[str, int, int], ...]: One ``(posix path relative to root, size in bytes,
        modification time in ns)`` entry per file; it changes whenever a file is added, edited
        or removed.
    """
    root = Path(root)
    return tuple(
        sorted(
            (p.relative_to(root).as_posix(), p.stat().st_size, p.stat().st_mtime_ns)
            for p in root.rglob("*.tsv")
        )
    )


def import_code_version() -> str:
    """Hash of the import code, so editing it invalidates cached results.

    Returns:
        str: Hex SHA-256 of the names and contents of the ``*.py`` files directly in
        ``src/behavior_import``.
    """
    digest = hashlib.sha256()
    for path in sorted(IMPORT_CODE_DIR.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _import_and_extract(
    root: Path, classify_task_types: bool = False, **import_options
) -> ImportedTrials:
    """Import and extract a cohort; *import_options* (``parse``, ``paths``) go to import_data."""
    log = ImportLog()
    # The warnings are shown as usual and also kept in the log, because a cached load raises none.
    with record_warnings() as recorded:
        data = import_data(root, classify_task_types=classify_task_types, log=log, **import_options)
        trials = extract_trials_grouped_by_problem(data, log=log)
    log.warnings = recorded
    return ImportedTrials(trials, log)


def _load(root: str, fingerprint, code_version: str, classify_task_types: bool) -> ImportedTrials:
    # The legacy cache entries are joblib's stored outputs of this function, found by its name.
    # fingerprint and code_version are not used here: they are arguments only
    # so that they become part of the cache key
    return _import_and_extract(Path(root), classify_task_types)


@lru_cache
def _cached_load(cache_dir: Path):
    return Memory(location=cache_dir, verbose=0).cache(_load)


def format_time(epoch_seconds: float) -> str:
    """*epoch_seconds* as local ``YYYY-MM-DD HH:MM``, as the cache messages show times."""
    return datetime.fromtimestamp(epoch_seconds).strftime("%Y-%m-%d %H:%M")


def _describe_cached(entries: Iterable[CacheEntry]) -> str:
    lines = sorted(
        f"  {e.root} (classify_task_types={e.classify_task_types}, {format_time(e.time)})"
        for e in entries
    )
    return "\n".join(lines) if lines else "  (no cached imports)"


def matching_root(items: Iterable, root: str, root_of: Callable[[object], str]) -> list:
    """The cached *items* of *root*: those with that root, else those ending in its tail.

    Args:
        items (Iterable): Cached imports or snapshots.
        root (str): The raw-data folder, as an absolute path.
        root_of (Callable[[object], str]): The root an item was made from.

    Returns:
        list: The items whose root is *root*; if there are none, those whose root ends in the
        same :func:`~src.behavior_import.cache_store.path_tail`, provided they all have one
        root. Empty if none match.

    Raises:
        FileNotFoundError: If several roots match only on the tail (the message lists them).
    """
    items = list(items)
    exact = [i for i in items if root_of(i) == root]
    if exact:
        return exact
    tail = [i for i in items if path_tail(root_of(i)) == path_tail(root)]
    tail_roots = sorted({root_of(i) for i in tail})
    if len(tail_roots) > 1:
        raise FileNotFoundError(
            f"No cached import of {root}, and several cached roots end in "
            f"{'/'.join(path_tail(root))}; set BANDIT_DATA_ROOT to the one to use:\n"
            + "\n".join(f"  {r}" for r in tail_roots)
        )
    return tail


def choose_cache_entry(
    entries: Iterable[CacheEntry],
    root: str,
    classify_task_types: bool,
    code_version: str,
    unreadable: Iterable[Path] = (),
) -> CacheEntry:
    """The cached import to load for *root* when the raw files cannot be checked.

    Only entries with the same *classify_task_types* qualify, and of those the ones whose root
    is *root*. If none is, the ones whose root ends in the same :func:`path_tail`, provided
    they all have one root (the caller can tell from ``entry.root != root``). Among the
    qualifying entries, those made with *code_version* come first, because only they reflect
    the current import code; within that preference the newest wins.

    Args:
        entries (Iterable[CacheEntry]): The cached imports, e.g. from
            :func:`cached_import_entries`.
        root (str): The raw-data folder, as an absolute path.
        classify_task_types (bool): The import option the caller needs.
        code_version (str): The current :func:`import_code_version`.
        unreadable (Iterable[Path]): Entries whose metadata could not be read, named in the
            error if nothing qualifies. Default none.

    Returns:
        CacheEntry: The chosen entry.

    Raises:
        FileNotFoundError: If no entry qualifies (the message lists what is cached), or several
            roots match only on the tail (the message lists them).
    """
    entries = list(entries)
    same_flag = [e for e in entries if e.classify_task_types == classify_task_types]
    matching = matching_root(same_flag, root, lambda e: e.root)
    if not matching:
        skipped = "".join(f"\n  {p}" for p in unreadable)
        raise FileNotFoundError(
            f"No cached import of {root} (classify_task_types={classify_task_types}). "
            f"Cached imports:\n{_describe_cached(entries)}"
            + (f"\nEntries with unreadable metadata:{skipped}" if skipped else "")
        )
    return max(matching, key=lambda e: (e.code_version == code_version, e.time))


def _entry_from_metadata(path: Path, metadata: dict) -> CacheEntry | None:
    """The :class:`CacheEntry` of joblib *metadata* (the arguments are stored as reprs), or
    ``None`` if the metadata lacks what is needed or cannot be parsed."""
    try:
        args = {k: ast.literal_eval(v) for k, v in metadata["input_args"].items()}
        files = tuple(ServerFile(*f) for f in args["fingerprint"])
        return CacheEntry(
            path,
            str(args["root"]),
            bool(args["classify_task_types"]),
            str(args["code_version"]),
            float(metadata["time"]),
            len(files),
            files,
        )
    except (KeyError, ValueError, SyntaxError, TypeError, AttributeError):
        return None


class CacheListing(NamedTuple):
    """The cached imports of a cache directory.

    Attributes:
        entries (tuple[CacheEntry, ...]): Those whose metadata could be read.
        unreadable (tuple[Path, ...]): Folders of complete entries whose metadata could not.
    """

    entries: tuple[CacheEntry, ...]
    unreadable: tuple[Path, ...]


def cached_import_entries(cache_dir: Path) -> CacheListing:
    """Every complete cached import under *cache_dir*, read from joblib's metadata files.

    An entry folder without both ``metadata.json`` and ``output.pkl`` is an import that was
    interrupted or is still being written, and is left out. A complete entry whose metadata
    cannot be read is listed as unreadable with a :class:`CacheMetadataWarning`, so one damaged
    file does not hide the others.

    Args:
        cache_dir (Path): The cache location (as for :func:`load_trials_with_log`).

    Returns:
        CacheListing: Sorted by folder name; empty if nothing is cached.
    """
    if not Path(cache_dir).is_dir():
        return CacheListing((), ())
    load = _cached_load(Path(cache_dir))
    function_dir = Path(load.store_backend.location) / load.func_id
    if not function_dir.is_dir():
        return CacheListing((), ())
    complete = [
        d
        for d in sorted(function_dir.iterdir())
        if (d / "metadata.json").is_file() and (d / "output.pkl").is_file()
    ]
    # joblib's get_metadata returns {} for a file it cannot read
    parsed = [
        (d, _entry_from_metadata(d, load.store_backend.get_metadata([load.func_id, d.name])))
        for d in complete
    ]
    unreadable = tuple(d for d, e in parsed if e is None)
    for folder in unreadable:
        warnings.warn(
            f"Skipping the cached import {folder}: its metadata.json cannot be read",
            CacheMetadataWarning,
            stacklevel=2,
        )
    return CacheListing(tuple(e for _, e in parsed if e is not None), unreadable)


def fill_missing_fields(log) -> tuple[str, ...]:
    """Give an unpickled dataclass the fields its class gained since it was pickled.

    Unpickling restores an object's attributes as they were saved, so a log written before a
    field was added loads without error and fails later, far from the cause. A missing field
    with a default is set to it, in place.

    Args:
        log: An unpickled dataclass instance (an :class:`ImportLog`); changed in place.

    Returns:
        tuple[str, ...]: The fields that were missing and are now set, in class order.

    Raises:
        CachedImportError: If a missing field has no default.
    """
    missing = [f for f in dataclasses.fields(log) if f.name not in vars(log)]
    no_default = [
        f.name
        for f in missing
        if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
    ]
    if no_default:
        raise CachedImportError(
            f"The cached {type(log).__name__} lacks {', '.join(no_default)}, which has no "
            "default: the cache is too old for the current code"
        )
    for f in missing:
        default = f.default if f.default is not dataclasses.MISSING else f.default_factory()
        setattr(log, f.name, default)
    return tuple(f.name for f in missing)


def _checked_import(loaded) -> tuple[ImportedTrials, tuple]:
    """*loaded* as trials with a complete log, and the log fields that were filled.

    Raises:
        CachedImportError: If it is not trials with an ImportLog, or its log is too old.
    """
    if not isinstance(loaded, ImportedTrials) or not isinstance(loaded.log, ImportLog):
        raise CachedImportError(f"it holds a {type(loaded).__name__}, not trials with an ImportLog")
    return loaded, fill_missing_fields(loaded.log)


def _unusable(err: Exception, what: str) -> CachedImportError:
    return CachedImportError(
        f"Cannot use the {what}: {type(err).__name__}: {err}. "
        f"{WHEN_THE_SHARE_IS_BACK[0].upper()}{WHEN_THE_SHARE_IS_BACK[1:]}."
    )


def _load_cache_entry(cache_dir: Path, entry: CacheEntry) -> tuple[ImportedTrials, tuple]:
    """Unpickle the output of *entry* and fill its log's missing fields; ``(trials, filled)``.

    Raises:
        CachedImportError: Naming the entry, if it cannot be unpickled or is too old.
    """
    load = _cached_load(Path(cache_dir))
    try:
        return _checked_import(
            load.store_backend.load_item([load.func_id, entry.path.name], verbose=0)
        )
    except MemoryError:
        raise
    except Exception as err:  # re-raised: any failure to unpickle makes the entry unusable
        what = f"cached import {entry.path} of {entry.root} ({format_time(entry.time)})"
        raise _unusable(err, what) from err


def _load_snapshot(snapshot: Snapshot) -> tuple[ImportedTrials, tuple]:
    """Unpickle a snapshot and fill its log's missing fields; ``(trials, filled)``.

    Raises:
        CachedImportError: Naming the snapshot, if it cannot be unpickled or is too old.
    """
    try:
        return _checked_import(load_snapshot_data(snapshot))
    except MemoryError:
        raise
    except Exception as err:  # re-raised: any failure to unpickle makes the snapshot unusable
        m = snapshot.manifest
        what = f"import snapshot {snapshot.path} of {m.root} ({format_time(m.created)})"
        raise _unusable(err, what) from err


def stale_cache_message(source: CacheSource, root: str) -> str:
    """The warning text for a cached import loaded without checking the raw files.

    Args:
        source (CacheSource): Where the loaded import came from.
        root (str): The raw-data folder that was asked for.

    Returns:
        str: When the import was cached, how many files it covered, whether the import code
        has changed since, how the root matched, which log fields were filled, that newer
        sessions are missing, and how to include them.
    """
    code = (
        "made with the current import code"
        if source.code_matches
        else "made with older import code: changes to src/behavior_import since then are not "
        "applied"
    )
    parts = [
        f"Using the cached import of {source.root} from {format_time(source.time)} "
        f"({source.n_files} .tsv files, {code}) without reading the server; sessions added on "
        f"the server since then are missing ({REFRESH_HINT})."
    ]
    if source.matched_on_tail:
        parts.append(
            f"{root} has no cached import of its own; the cached root matched on its last "
            f"{TAIL_PARTS} path parts."
        )
    if source.filled_fields:
        parts.append(
            "The cached import log predates "
            f"{', '.join(source.filled_fields)}, filled with defaults."
        )
    return " ".join(parts)


def absolute_path_string(path: Path) -> str:
    """*path* as an absolute path string, without touching the file system.

    ``Path.resolve`` would query each folder, which hangs or fails on an unresponsive share,
    so symbolic links are not followed; :func:`matching_root` falls back on the path's tail
    instead.

    Args:
        path (Path): Any path.

    Returns:
        str: ``os.path.abspath(path)`` (normalised: no trailing slash, no ``..``).
    """
    return os.path.abspath(path)


def _with_stale_warning(
    imported: ImportedTrials, source: CacheSource, wanted: str
) -> ImportedTrials:
    """*imported* with *source* and a :class:`StaleCacheWarning` (raised and logged) added."""
    code = "matches" if source.code_matches else "is older"
    print(
        f"[INFO] {source.root}: import cache of {format_time(source.time)}, "
        f"{source.n_files} .tsv files, import code {code}; {REFRESH_HINT}"
    )
    with record_warnings() as recorded:
        # "always": a second load in the same process must warn, and log the warning, again
        warnings.simplefilter("always", StaleCacheWarning)
        warnings.warn(stale_cache_message(source, wanted), StaleCacheWarning, stacklevel=5)
    log = dataclasses.replace(
        imported.log, warnings=[*imported.log.warnings, *recorded], cache_source=source
    )
    return ImportedTrials(imported.trials_by_problem, log)


def choose_snapshot(
    snapshots: Iterable[Snapshot], root: str, classify_task_types: bool
) -> Snapshot | None:
    """The newest snapshot of *root* (or of its tail, see :func:`matching_root`), or ``None``.

    Args:
        snapshots (Iterable[Snapshot]): The listed snapshots.
        root (str): The raw-data folder, as an absolute path.
        classify_task_types (bool): The import option the caller needs.

    Returns:
        Snapshot | None: The newest qualifying snapshot; ``None`` if none qualifies.

    Raises:
        FileNotFoundError: If several roots match only on the tail.
    """
    same_flag = [s for s in snapshots if s.manifest.classify_task_types == classify_task_types]
    matching = matching_root(same_flag, root, lambda s: s.manifest.root)
    return max(matching, key=lambda s: s.manifest.created, default=None)


def _snapshot_source(snapshot: Snapshot, wanted: str, filled: tuple) -> CacheSource:
    m = snapshot.manifest
    code_matches = m.code_version == import_code_version()
    return CacheSource(m.root, m.created, code_matches, len(m.files), m.root != wanted, filled)


def _load_legacy(cache_dir: Path, entry: CacheEntry, wanted: str) -> ImportedTrials:
    imported, filled = _load_cache_entry(cache_dir, entry)
    code_matches = entry.code_version == import_code_version()
    source = CacheSource(
        entry.root, entry.time, code_matches, entry.n_files, entry.root != wanted, filled
    )
    return _with_stale_warning(imported, source, wanted)


def has_legacy_import(entries: Iterable[CacheEntry], root: str, classify_task_types: bool) -> bool:
    """Whether :func:`choose_cache_entry` has an entry of *root* to choose.

    Args:
        entries (Iterable[CacheEntry]): The legacy imports.
        root (str): The raw-data folder, as an absolute path.
        classify_task_types (bool): The import option the caller needs.

    Returns:
        bool: True if an entry with that flag has *root*, or its tail.

    Raises:
        FileNotFoundError: If several roots match only on the tail.
    """
    same_flag = [e for e in entries if e.classify_task_types == classify_task_types]
    return bool(matching_root(same_flag, root, lambda e: e.root))


def _no_cache_error(root: str, classify_task_types: bool, listing: CacheListing):
    skipped = "".join(f"\n  {p}" for p in listing.unreadable)
    return FileNotFoundError(
        f"No cached import of {root} (classify_task_types={classify_task_types}), and the raw "
        f"folder is not reachable (is the share mounted?). Cached imports:\n"
        f"{_describe_cached(listing.entries)}"
        + (f"\nEntries with unreadable metadata:{skipped}" if skipped else "")
    )


def _load_cache_first(root: Path, cache_dir: Path, classify_task_types: bool) -> ImportedTrials:
    """The newest snapshot of *root*, else its newest legacy import, else a refresh.

    Only the last case reads *root*: it is for a folder the cache has never seen.

    Raises:
        FileNotFoundError: If nothing is cached for *root* and *root* is not reachable (the
            message lists what is cached).
    """
    wanted = absolute_path_string(root)
    snapshot = choose_snapshot(list_snapshots(cache_dir).snapshots, wanted, classify_task_types)
    if snapshot is not None:
        imported, filled = _load_snapshot(snapshot)
        return _with_stale_warning(imported, _snapshot_source(snapshot, wanted, filled), wanted)
    listing = cached_import_entries(cache_dir)
    if has_legacy_import(listing.entries, wanted, classify_task_types):
        entry = choose_cache_entry(
            listing.entries, wanted, classify_task_types, import_code_version()
        )
        return _load_legacy(cache_dir, entry, wanted)
    if not Path(wanted).is_dir():
        raise _no_cache_error(wanted, classify_task_types, listing)
    print(f"[INFO] {wanted}: no import cache yet; importing from the server")
    return refresh_import_cache(
        root, cache_dir=cache_dir, classify_task_types=classify_task_types
    ).imported


# --- refreshing, pruning and checking -------------------------------------------------------


class RefreshResult(NamedTuple):
    """What :func:`refresh_import_cache` did.

    Attributes:
        imported (ImportedTrials): The cohort as now cached.
        snapshot (Snapshot): The snapshot it is stored in (new, or the current one).
        parsed (tuple[str, ...]): Files parsed in this refresh (new or changed), relative paths.
        reused (tuple[str, ...]): Files whose earlier parse was reused.
        up_to_date (bool): Whether the newest snapshot already matched the server and the
            import code, so nothing was parsed or assembled.
        pruned (tuple[PruneItem, ...]): What the prune after the refresh removed.
    """

    imported: ImportedTrials
    snapshot: Snapshot
    parsed: tuple[str, ...]
    reused: tuple[str, ...]
    up_to_date: bool
    pruned: tuple[PruneItem, ...]


def current_snapshot(snapshots: Iterable[Snapshot], manifest: Manifest) -> Snapshot | None:
    """The newest snapshot of *manifest*'s root and flag, if it was made from the same files
    with the same code; ``None`` otherwise.

    Args:
        snapshots (Iterable[Snapshot]): The listed snapshots.
        manifest (Manifest): The manifest a refresh would write now.

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
    imported = _import_and_extract(
        root,
        manifest.classify_task_types,
        parse=parser,
        paths=[root / f.path for f in listing],
    )
    return imported, write_snapshot(cache_dir, imported, manifest), parser


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
    Then the cohort's old snapshots, replaced legacy imports and unused parses are pruned. If the
    share fails part-way, the error says so and the previous snapshot stays the one loaded.

    Args:
        root (Path): The raw-data folder.
        cache_dir (Path | None): Cache location. Default ``None``: ``get_settings().cache_dir``.
            Keyword-only.
        classify_task_types (bool): As for :func:`load_trials_with_log`. Default ``False``.
            Keyword-only.
        keep_old (bool): Skip the prune. Default ``False``. Keyword-only.
        progress (Callable[[int], None] | None): Called while listing, as for
            :func:`~src.behavior_import.cache_store.list_server_files`. Keyword-only.

    Returns:
        RefreshResult: The cohort, its snapshot, which files were parsed or reused, and what was
        pruned.

    Raises:
        FileNotFoundError: If *root* is not an existing directory.
        ShareReadError: If the folder cannot be listed or a file cannot be read.
    """
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Raw data folder not found (is the share mounted?): {root}")
    cache_dir = Path(cache_dir or get_settings().cache_dir)
    listing = list_server_files(root, progress)
    manifest = Manifest(
        str(root),
        classify_task_types,
        import_code_version(),
        file_code_version(),
        time.time(),
        listing,
    )
    current = current_snapshot(list_snapshots(cache_dir).snapshots, manifest)
    if current is not None:
        print(f"[INFO] Loading trials for {root} from cache (up to date with the server)")
        imported, _ = _load_snapshot(current)
        snapshot, parsed, reused = current, (), tuple(f.path for f in listing)
    else:
        imported, snapshot, parser = _assemble_and_store(root, cache_dir, manifest, listing)
        parsed, reused = tuple(parser.parsed), tuple(parser.reused)
        print(
            f"[INFO] {root}: parsed {len(parsed)} new or changed file(s), reused {len(reused)}; "
            f"new cache snapshot {snapshot.path.name}"
        )
    pruned = () if keep_old else tuple(prune_import_cache(cache_dir, roots=[str(root)]))
    return RefreshResult(imported, snapshot, parsed, reused, current is not None, pruned)


def prune_import_cache(
    cache_dir: Path | None = None,
    *,
    roots: Iterable[str] | None = None,
    dry_run: bool = False,
) -> list[PruneItem]:
    """Remove the cache entries no load would use (see :mod:`~src.behavior_import.cache_prune`).

    Args:
        cache_dir (Path | None): Cache location. Default ``None``: the ``BANDIT_CACHE_DIR``
            setting (the share is not touched).
        roots (Iterable[str] | None): Raw-data folders to limit the prune to; ``None`` for the
            whole cache. Keyword-only.
        dry_run (bool): Print what would go and delete nothing. Default ``False``. Keyword-only.

    Returns:
        list[PruneItem]: What was (or would be) removed.
    """
    cache_dir = Path(cache_dir or get_unchecked_settings().cache_dir)
    items = prune_plan(
        list_snapshots(cache_dir).snapshots,
        cached_import_entries(cache_dir).entries,
        list_file_entries(cache_dir),
        incomplete_snapshots(cache_dir),
        now=time.time(),
        cohorts=roots,
    )
    if items:
        apply_prune(cache_dir, items, dry_run=dry_run)
    else:
        print("[PRUNE] nothing to remove")
    return items


def load_trials_with_log(
    root: Path,
    *,
    cache_dir: Path | None = None,
    use_cache: bool = True,
    classify_task_types: bool = False,
    from_cache: bool = False,
) -> ImportedTrials:
    """``extract_trials_grouped_by_problem(import_data(root))`` and the import log, cached.

    Args:
        root (Path): Raw-data folder, e.g. ``<data_root>/<experiment>/rawdata/<cohort>``.
        cache_dir (Path | None): Cache location. Default ``None``, meaning
            ``get_settings().cache_dir``. Keyword-only.
        use_cache (bool): ``False`` always re-imports and leaves the cache untouched. Default
            ``True``: :func:`refresh_import_cache` (only new or changed files are parsed).
            Keyword-only.
        classify_task_types (bool): Number problems by task type as well as layout (see
            :func:`~src.behavior_import.import_data.import_data`). Default ``False``.
            Keyword-only.
        from_cache (bool): Cache-first: load the newest cache snapshot of *root* without
            reading *root* at all, or its newest legacy import if it has no snapshot (see
            :func:`choose_cache_entry`); only a folder with no cache at all is imported from
            the server (and cached). Sessions added since the cache was made are missing; a
            :class:`StaleCacheWarning` says so and is appended to the log's warnings. *root* is
            compared as given, made absolute, not resolved. Default ``False``. Keyword-only.

    Returns:
        ImportedTrials: The trials, ``{problem_number: {subject: {session_key:
        session_dict}}}``, and the :class:`~src.behavior_import.import_log.ImportLog` of the
        import that produced them (the one stored in the cache, on a cached load).

    Raises:
        FileNotFoundError: If *root* is not an existing directory or, with *from_cache*, has no
            cache and cannot be reached (the message lists the cached imports).
        ValueError: If *from_cache* is combined with ``use_cache=False``.
        CachedImportError: If a cached import cannot be read.
        ShareReadError: If the server fails while a refresh reads it.
    """
    if from_cache:
        if not use_cache:
            raise ValueError("from_cache loads from the cache, so it needs use_cache=True")
        return _load_cache_first(
            Path(root),
            Path(cache_dir or get_unchecked_settings().cache_dir),
            classify_task_types,
        )
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Raw data folder not found (is the share mounted?): {root}")
    if not use_cache:
        return _import_and_extract(root, classify_task_types)
    return refresh_import_cache(
        root, cache_dir=cache_dir, classify_task_types=classify_task_types
    ).imported


def load_trials_by_problem(
    root: Path,
    *,
    cache_dir: Path | None = None,
    use_cache: bool = True,
    classify_task_types: bool = False,
) -> dict:
    """``extract_trials_grouped_by_problem(import_data(root))``, cached on disk.

    The trials of :func:`load_trials_with_log`, without the log.

    Args:
        root (Path): Raw-data folder, e.g. ``<data_root>/<experiment>/<cohort>/rawdata``.
        cache_dir (Path | None): Cache location. Default ``None``, meaning
            ``get_settings().cache_dir``. Keyword-only.
        use_cache (bool): ``False`` always re-imports and leaves the cache untouched.
            Default ``True``. Keyword-only.
        classify_task_types (bool): Number problems by task type as well as layout (see
            :func:`~src.behavior_import.import_data.import_data`). Default ``False``.
            Keyword-only.

    Returns:
        dict: ``{problem_number: {subject: {session_key: session_dict}}}``.

    Raises:
        FileNotFoundError: If *root* is not an existing directory.
    """
    return load_trials_with_log(
        root, cache_dir=cache_dir, use_cache=use_cache, classify_task_types=classify_task_types
    ).trials_by_problem
