"""Import and extract trials for a raw-data folder, cached on disk.

Parsing every pyControl ``.tsv`` in a cohort is the slowest step of any
analysis. The result of :func:`extract_trials_grouped_by_problem` is cached
with joblib, keyed on the raw files (path, size, modification time), the
source of ``src/behavior_import``, and the options. A new session, an edited
file, or a change to the import code therefore re-runs the import
automatically; otherwise the result loads from disk in seconds. Deleting the
cache directory only costs time.

The cache holds the trials together with the :class:`~src.behavior_import.import_log.ImportLog`
of the import (what was skipped or flagged, and the warnings it raised), so a cached load has
the same log as the import that filled the cache. :func:`load_trials_with_log` returns both;
:func:`load_trials_by_problem` returns the trials alone.

With ``from_cache=True`` the raw folder is never read: the newest cached import of that folder
is loaded whatever its key, preferring one made with the current import code (see
:func:`choose_cache_entry`). Sessions added since that import are missing, and changes to the
import code since then are not applied, so the load raises a :class:`StaleCacheWarning` that is
also kept in the import log.
"""

import ast
import dataclasses
import hashlib
import os
import warnings
from collections.abc import Iterable
from datetime import datetime
from functools import lru_cache
from pathlib import Path, PurePath
from typing import NamedTuple

from joblib import Memory

from src.behavior_import.extract_trials import extract_trials_grouped_by_problem
from src.behavior_import.import_data import import_data
from src.behavior_import.import_log import CacheSource, ImportLog
from src.behavior_import.warning_records import record_warnings
from src.settings import get_settings, get_unchecked_settings

# The source hashed into the cache key. It includes this module, so a change to what is cached
# invalidates the caches written before it.
IMPORT_CODE_DIR = Path(__file__).resolve().parent
# What to do when the share is back, quoted in the errors of a from-cache load.
WHEN_THE_SHARE_IS_BACK = "run without --from-cache once the server is reachable"
# A cached root whose full path differs (another mount point, a symbolic link) is matched on
# this many trailing parts: <experiment>/rawdata/<cohort>.
TAIL_PARTS = 3


class StaleCacheWarning(UserWarning):
    """A cached import was loaded without checking it against the raw files."""


class CachedImportError(RuntimeError):
    """A cached import could not be read."""


class CacheMetadataWarning(UserWarning):
    """A cached import was skipped because its metadata could not be read."""


class CacheEntry(NamedTuple):
    """One cached import, as recorded in its joblib ``metadata.json``.

    Attributes:
        path (Path): The entry's folder, holding ``metadata.json`` and ``output.pkl``.
        root (str): The raw-data folder that was imported, as an absolute path.
        classify_task_types (bool): Whether problems were numbered by task type too.
        code_version (str): :func:`import_code_version` when the import ran.
        time (float): When the import was cached, in epoch seconds.
        n_files (int): How many ``.tsv`` files the raw folder held then.
    """

    path: Path
    root: str
    classify_task_types: bool
    code_version: str
    time: float
    n_files: int


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


def _import_and_extract(root: Path, classify_task_types: bool = False) -> ImportedTrials:
    log = ImportLog()
    # The warnings are shown as usual and also kept in the log, because a cached load raises none.
    with record_warnings() as recorded:
        data = import_data(root, classify_task_types=classify_task_types, log=log)
        trials = extract_trials_grouped_by_problem(data, log=log)
    log.warnings = recorded
    return ImportedTrials(trials, log)


def _load(root: str, fingerprint, code_version: str, classify_task_types: bool) -> ImportedTrials:
    # fingerprint and code_version are not used here: they are arguments only
    # so that they become part of the cache key
    return _import_and_extract(Path(root), classify_task_types)


@lru_cache
def _cached_load(cache_dir: Path):
    return Memory(location=cache_dir, verbose=0).cache(_load)


def _format_time(epoch_seconds: float) -> str:
    return datetime.fromtimestamp(epoch_seconds).strftime("%Y-%m-%d %H:%M")


def _describe_cached(entries: Iterable[CacheEntry]) -> str:
    lines = sorted(
        f"  {e.root} (classify_task_types={e.classify_task_types}, {_format_time(e.time)})"
        for e in entries
    )
    return "\n".join(lines) if lines else "  (no cached imports)"


def path_tail(path: str) -> tuple[str, ...]:
    """The last :data:`TAIL_PARTS` parts of *path*, e.g. ``(experiment, "rawdata", cohort)``.

    Args:
        path (str): An absolute path.

    Returns:
        tuple[str, ...]: The parts, compared when the full path of a cached root differs (the
        same folder reached through another mount point or a symbolic link).
    """
    return PurePath(path).parts[-TAIL_PARTS:]


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
    matching = [e for e in same_flag if e.root == root]
    if not matching:
        matching = [e for e in same_flag if path_tail(e.root) == path_tail(root)]
        tail_roots = sorted({e.root for e in matching})
        if len(tail_roots) > 1:
            raise FileNotFoundError(
                f"No cached import of {root}, and several cached roots end in "
                f"{'/'.join(path_tail(root))}; set BANDIT_DATA_ROOT to the one to use:\n"
                + "\n".join(f"  {r}" for r in tail_roots)
            )
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
        return CacheEntry(
            path,
            str(args["root"]),
            bool(args["classify_task_types"]),
            str(args["code_version"]),
            float(metadata["time"]),
            len(args["fingerprint"]),
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


def _load_cache_entry(cache_dir: Path, entry: CacheEntry) -> tuple[ImportedTrials, tuple]:
    """Unpickle the output of *entry* and fill its log's missing fields; ``(trials, filled)``.

    Raises:
        CachedImportError: Naming the entry, if it cannot be unpickled or is too old.
    """
    load = _cached_load(Path(cache_dir))
    try:
        loaded = load.store_backend.load_item([load.func_id, entry.path.name], verbose=0)
        if not isinstance(loaded, ImportedTrials) or not isinstance(loaded.log, ImportLog):
            raise CachedImportError(
                f"it holds a {type(loaded).__name__}, not trials with an ImportLog"
            )
        filled = fill_missing_fields(loaded.log)
    except MemoryError:
        raise
    except Exception as err:  # re-raised: any failure to unpickle makes the entry unusable
        raise CachedImportError(
            f"Cannot use the cached import {entry.path} of {entry.root} "
            f"({_format_time(entry.time)}): {type(err).__name__}: {err}. "
            f"{WHEN_THE_SHARE_IS_BACK[0].upper()}{WHEN_THE_SHARE_IS_BACK[1:]}."
        ) from err
    return loaded, filled


def stale_cache_message(source: CacheSource, root: str) -> str:
    """The warning text for a cached import loaded without checking the raw files.

    Args:
        source (CacheSource): Where the loaded import came from.
        root (str): The raw-data folder that was asked for.

    Returns:
        str: When the import was cached, how many files it covered, whether the import code
        has changed since, how the root matched, which log fields were filled, and that newer
        sessions are missing.
    """
    code = (
        "made with the current import code"
        if source.code_matches
        else "made with older import code: changes to src/behavior_import since then are not "
        "applied"
    )
    parts = [
        f"Using the cached import of {source.root} from {_format_time(source.time)} "
        f"({source.n_files} .tsv files, {code}) without reading the server; sessions added on "
        "the server since then are missing."
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
    so symbolic links are not followed; :func:`choose_cache_entry` falls back on the path's
    tail instead.

    Args:
        path (Path): Any path.

    Returns:
        str: ``os.path.abspath(path)`` (normalised: no trailing slash, no ``..``).
    """
    return os.path.abspath(path)


def _load_from_cache(root: Path, cache_dir: Path, classify_task_types: bool) -> ImportedTrials:
    """The newest cached import of *root*, with a :class:`StaleCacheWarning` in its log."""
    code_version = import_code_version()
    wanted = absolute_path_string(root)
    listing = cached_import_entries(cache_dir)
    entry = choose_cache_entry(
        listing.entries, wanted, classify_task_types, code_version, listing.unreadable
    )
    (trials, log), filled = _load_cache_entry(cache_dir, entry)
    source = CacheSource(
        entry.root,
        entry.time,
        entry.code_version == code_version,
        entry.n_files,
        entry.root != wanted,
        filled,
    )
    print(
        f"[INFO] {entry.root}: cached import of {_format_time(entry.time)}, import code "
        f"{'matches' if source.code_matches else 'is older'}"
    )
    with record_warnings() as recorded:
        # "always": a second load in the same process must warn, and log the warning, again
        warnings.simplefilter("always", StaleCacheWarning)
        warnings.warn(stale_cache_message(source, wanted), StaleCacheWarning, stacklevel=4)
    log = dataclasses.replace(log, warnings=[*log.warnings, *recorded], cache_source=source)
    return ImportedTrials(trials, log)


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
        root (Path): Raw-data folder, e.g. ``<data_root>/<experiment>/<cohort>/rawdata``.
        cache_dir (Path | None): Cache location. Default ``None``, meaning
            ``get_settings().cache_dir``. Keyword-only.
        use_cache (bool): ``False`` always re-imports and leaves the cache untouched.
            Default ``True``. Keyword-only.
        classify_task_types (bool): Number problems by task type as well as layout (see
            :func:`~src.behavior_import.import_data.import_data`). Default ``False``.
            Keyword-only.
        from_cache (bool): Load the newest cached import of *root* without reading *root* at
            all (see :func:`choose_cache_entry`), for when the share is unreachable. Sessions
            added since are missing; a :class:`StaleCacheWarning` says so and is appended to
            the log's warnings. *root* is compared as given, made absolute, not resolved.
            Default ``False``. Keyword-only.

    Returns:
        ImportedTrials: The trials, ``{problem_number: {subject: {session_key:
        session_dict}}}``, and the :class:`~src.behavior_import.import_log.ImportLog` of the
        import that produced them (the one stored in the cache, on a cached load).

    Raises:
        FileNotFoundError: If *root* is not an existing directory or, with *from_cache*, has no
            cached import (the message lists the cached ones).
        ValueError: If *from_cache* is combined with ``use_cache=False``.
        CachedImportError: With *from_cache*, if the cached import cannot be read.
    """
    if from_cache:
        if not use_cache:
            raise ValueError("from_cache loads from the cache, so it needs use_cache=True")
        return _load_from_cache(
            Path(root),
            Path(cache_dir or get_unchecked_settings().cache_dir),
            classify_task_types,
        )
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Raw data folder not found (is the share mounted?): {root}")
    if not use_cache:
        return _import_and_extract(root, classify_task_types)

    load = _cached_load(Path(cache_dir or get_settings().cache_dir))
    args = (
        str(root),
        raw_data_fingerprint(root),
        import_code_version(),
        classify_task_types,
    )
    if load.check_call_in_cache(*args):
        print(f"[INFO] Loading trials for {root} from cache")
    return load(*args)


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
