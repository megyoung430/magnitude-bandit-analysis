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
from pathlib import Path
from typing import NamedTuple

from joblib import Memory

from src.behavior_import.extract_trials import extract_trials_grouped_by_problem
from src.behavior_import.import_data import import_data
from src.behavior_import.import_log import ImportLog
from src.behavior_import.warning_records import record_warnings
from src.settings import get_settings, get_unchecked_settings

# The source hashed into the cache key. It includes this module, so a change to what is cached
# invalidates the caches written before it.
IMPORT_CODE_DIR = Path(__file__).resolve().parent
# What to do when the share is back, quoted in the errors of a from-cache load.
WHEN_THE_SHARE_IS_BACK = "run without --from-cache once the server is reachable"


class StaleCacheWarning(UserWarning):
    """A cached import was loaded without checking it against the raw files."""


class CachedImportError(RuntimeError):
    """A cached import could not be read."""


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


def choose_cache_entry(
    entries: Iterable[CacheEntry], root: str, classify_task_types: bool, code_version: str
) -> CacheEntry:
    """The cached import to load for *root* when the raw files cannot be checked.

    Only entries of *root* with the same *classify_task_types* qualify. Among them, those made
    with *code_version* come first, because only they reflect the current import code; within
    that preference the newest wins.

    Args:
        entries (Iterable[CacheEntry]): The cached imports, e.g. from
            :func:`cached_import_entries`.
        root (str): The raw-data folder, as an absolute path.
        classify_task_types (bool): The import option the caller needs.
        code_version (str): The current :func:`import_code_version`.

    Returns:
        CacheEntry: The chosen entry.

    Raises:
        FileNotFoundError: If no entry qualifies; the message lists what is cached.
    """
    entries = list(entries)
    matching = [
        e for e in entries if e.root == root and e.classify_task_types == classify_task_types
    ]
    if not matching:
        raise FileNotFoundError(
            f"No cached import of {root} (classify_task_types={classify_task_types}). "
            f"Cached imports:\n{_describe_cached(entries)}"
        )
    return max(matching, key=lambda e: (e.code_version == code_version, e.time))


def _entry_from_metadata(path: Path, metadata: dict) -> CacheEntry:
    """The :class:`CacheEntry` of joblib *metadata* (the arguments are stored as reprs)."""
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
    except (KeyError, ValueError, SyntaxError, TypeError) as err:
        raise CachedImportError(
            f"Cannot read the metadata of the cached import {path}: {err!r}"
        ) from err


def cached_import_entries(cache_dir: Path) -> list[CacheEntry]:
    """Every complete cached import under *cache_dir*, read from joblib's metadata files.

    An entry folder without both ``metadata.json`` and ``output.pkl`` is an import that was
    interrupted or is still being written, and is left out.

    Args:
        cache_dir (Path): The cache location (as for :func:`load_trials_with_log`).

    Returns:
        list[CacheEntry]: In no particular order; empty if nothing is cached.

    Raises:
        CachedImportError: If a complete entry's metadata cannot be parsed.
    """
    if not Path(cache_dir).is_dir():
        return []
    load = _cached_load(Path(cache_dir))
    function_dir = Path(load.store_backend.location) / load.func_id
    if not function_dir.is_dir():
        return []
    complete = [
        d
        for d in sorted(function_dir.iterdir())
        if (d / "metadata.json").is_file() and (d / "output.pkl").is_file()
    ]
    return [
        _entry_from_metadata(d, load.store_backend.get_metadata([load.func_id, d.name]))
        for d in complete
    ]


def outdated_cache_reason(loaded) -> str | None:
    """Why an unpickled cache output cannot be used by the current code, or ``None``.

    Unpickling restores an object's attributes as they were saved, so a log written before a
    field was added to :class:`~src.behavior_import.import_log.ImportLog` loads without error
    and fails later, far from the cause. This checks for that up front.

    Args:
        loaded: The unpickled output of :func:`_load`.

    Returns:
        str | None: What is wrong, or ``None`` if it has the current layout.
    """
    if not isinstance(loaded, ImportedTrials) or not isinstance(loaded.log, ImportLog):
        return f"it holds a {type(loaded).__name__}, not trials with an ImportLog"
    missing = [f.name for f in dataclasses.fields(ImportLog) if not hasattr(loaded.log, f.name)]
    if missing:
        return f"its ImportLog lacks {', '.join(missing)}"
    return None


def _load_cache_entry(cache_dir: Path, entry: CacheEntry) -> ImportedTrials:
    """Unpickle the output of *entry*, naming it if that fails or the layout is outdated."""
    load = _cached_load(Path(cache_dir))
    cause = None
    try:
        loaded = load.store_backend.load_item([load.func_id, entry.path.name], verbose=0)
        reason = outdated_cache_reason(loaded)
    except Exception as err:  # re-raised below: any failure to unpickle makes the entry unusable
        cause, reason = err, f"{type(err).__name__}: {err}"
    if reason is not None:
        raise CachedImportError(
            f"Cannot use the cached import {entry.path} of {entry.root} "
            f"({_format_time(entry.time)}): {reason}. It may predate the current code; "
            f"{WHEN_THE_SHARE_IS_BACK}."
        ) from cause
    return loaded


def stale_cache_message(entry: CacheEntry, code_version: str) -> str:
    """The warning text for a cached import loaded without checking the raw files.

    Args:
        entry (CacheEntry): The loaded entry.
        code_version (str): The current :func:`import_code_version`.

    Returns:
        str: When the import was cached, how many files it covered, whether the import code
        has changed since, and that newer sessions are missing.
    """
    code = (
        "made with the current import code"
        if entry.code_version == code_version
        else "made with older import code: changes to src/behavior_import since then are not "
        "applied"
    )
    return (
        f"Using the cached import of {entry.root} from {_format_time(entry.time)} "
        f"({entry.n_files} .tsv files, {code}) without reading the server; sessions added on "
        "the server since then are missing."
    )


def absolute_path_string(path: Path) -> str:
    """*path* as an absolute path string, without touching the file system.

    ``Path.resolve`` would query each folder, which hangs or fails on an unresponsive share,
    so symbolic links are not followed: give the path as it was when the cache was filled.

    Args:
        path (Path): Any path.

    Returns:
        str: ``os.path.abspath(path)``.
    """
    return os.path.abspath(path)


def _load_from_cache(root: Path, cache_dir: Path, classify_task_types: bool) -> ImportedTrials:
    """The newest cached import of *root*, with a :class:`StaleCacheWarning` in its log."""
    code_version = import_code_version()
    entry = choose_cache_entry(
        cached_import_entries(cache_dir),
        absolute_path_string(root),
        classify_task_types,
        code_version,
    )
    trials, log = _load_cache_entry(cache_dir, entry)
    print(
        f"[INFO] {entry.root}: cached import of {_format_time(entry.time)}, import code "
        f"{'matches' if entry.code_version == code_version else 'is older'}"
    )
    with record_warnings() as recorded:
        # "always": a second load in the same process must warn, and log the warning, again
        warnings.simplefilter("always", StaleCacheWarning)
        warnings.warn(stale_cache_message(entry, code_version), StaleCacheWarning, stacklevel=4)
    return ImportedTrials(trials, dataclasses.replace(log, warnings=[*log.warnings, *recorded]))


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
