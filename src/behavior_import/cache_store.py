"""The import cache on disk: the server listing, per-file parses and cohort snapshots.

An import has two parts. Reading and JSON-expanding each ``.tsv``
(:func:`~src.behavior_import.import_data.parse_tsv_file`) depends on that file alone and is
the slow part; it is cached per file, keyed on the file's relative path, size, modification
time and :func:`file_code_version`, so a refresh parses only new or changed files. Grouping the
files into sessions, numbering problems and extracting trials depend on every session of the
cohort and run on the whole cohort each time; their result is stored as a cohort *snapshot*,
which is what runs load.

Layout under the cache directory::

    import-files/<experiment>_rawdata_<cohort>/<key>.pkl      one CachedParse per file
    import-snapshots/<name>/trials.pkl                         the assembled import
    import-snapshots/<name>/manifest.json                      what it was made from

A snapshot is written into a hidden ``.tmp-`` folder that is renamed into place once both files
are complete, so an interrupted write never replaces or hides the previous snapshot. The
manifest is small JSON, so checking a snapshot against the server never unpickles the data.
Nothing here writes outside the cache directory, and :func:`refuse_cache_inside` stops any
write when the cache directory lies inside the raw data.

Both the raw and the expanded frame of each file are stored. On 50 real recordings (5.2 MB
of ``.tsv``) reading took 3-49 ms a file (warm and cold disk) and expanding 39 ms a file, while
unpickling the expanded frame took 5 ms; the expanded frames are about 25 times the size of
the raw ones (95 MB against 3.7 MB pickled). Expanding on reuse would cost every refresh about
35 ms a file in exchange for that space.
"""

from __future__ import annotations

import errno
import hashlib
import inspect
import json
import os
import pickle
import shutil
import sys
import uuid
import warnings
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path, PurePath
from typing import NamedTuple

import pandas as pd

from src.behavior_import import import_data
from src.behavior_import.import_data import ImportAborted, ParsedFile, parse_tsv_file
from src.settings import REPO_ROOT

FILES_DIR = "import-files"
SNAPSHOTS_DIR = "import-snapshots"
DATA_FILE = "trials.pkl"
MANIFEST_FILE = "manifest.json"
TMP_PREFIX = ".tmp-"
# A cached root whose full path differs (another mount point, a symbolic link) is matched on
# this many trailing parts: <experiment>/rawdata/<cohort>.
TAIL_PARTS = 3
MANIFEST_FORMAT = 1


class ShareReadError(ImportAborted):
    """The raw-data folder could not be listed or read; the cache was left as it was."""


class CacheLocationError(ValueError):
    """The cache directory lies inside the raw data, where nothing may be written."""


# The errors of a share that has stopped answering, as opposed to one file that cannot be read.
SHARE_GONE_ERRNOS = frozenset(
    {errno.ETIMEDOUT, errno.EIO, errno.ENOTCONN, errno.EHOSTDOWN, errno.ESTALE}
)


def share_is_gone(err: OSError, root: Path) -> bool:
    """Whether *err*, raised reading a file under *root*, means the share itself failed.

    Args:
        err (OSError): The error.
        root (Path): The raw-data folder.

    Returns:
        bool: True for a timeout, an errno in :data:`SHARE_GONE_ERRNOS`, or *root* no longer
        being a reachable directory; False for an error of the one file (e.g. ``EACCES``, or
        ``ENOENT`` for a file removed since it was listed), which an import skips.
    """
    if isinstance(err, TimeoutError) or err.errno in SHARE_GONE_ERRNOS:
        return True
    try:
        return not Path(root).is_dir()
    except OSError:
        return True


def refuse_cache_inside(cache_dir: Path, protected: Iterable[str | Path]) -> None:
    """Stop if *cache_dir* is inside one of the *protected* folders (the raw data).

    The cache directory is resolved (it is local); the protected folders are only made absolute,
    because resolving them would query the share.

    Args:
        cache_dir (Path): The cache directory.
        protected (Iterable[str | Path]): Folders nothing may be written in, e.g. the data root
            and the raw-data folder being refreshed.

    Raises:
        CacheLocationError: Naming both folders.
    """
    forms = {Path(os.path.abspath(cache_dir)), Path(cache_dir).resolve()}
    for folder in protected:
        guarded = Path(os.path.abspath(folder))
        if any(form == guarded or form.is_relative_to(guarded) for form in forms):
            raise CacheLocationError(
                f"The import cache {cache_dir} is inside {folder}, which holds raw data and is "
                "never written to; set BANDIT_CACHE_DIR to a local folder"
            )


class ImportCacheWarning(UserWarning):
    """Part of the import cache could not be read and is ignored."""


def path_tail(path: str) -> tuple[str, ...]:
    """The last :data:`TAIL_PARTS` parts of *path*, e.g. ``(experiment, "rawdata", cohort)``.

    Args:
        path (str): An absolute path.

    Returns:
        tuple[str, ...]: The parts, compared when the full path of a cached root differs (the
        same folder reached through another mount point or a symbolic link).
    """
    return PurePath(path).parts[-TAIL_PARTS:]


def cohort_slug(root: str) -> str:
    """The folder name for *root*'s cache entries, ``<experiment>_rawdata_<cohort>``.

    Args:
        root (str): The raw-data folder.

    Returns:
        str: Its :func:`path_tail` joined with ``_``.
    """
    return "_".join(path_tail(str(root)))


# --- the server listing ---------------------------------------------------------------------


class ServerFile(NamedTuple):
    """One ``.tsv`` of a raw-data folder, as listed.

    Attributes:
        path (str): Posix path relative to the raw-data folder.
        size (int): Size in bytes.
        mtime_ns (int): Modification time in nanoseconds.
    """

    path: str
    size: int
    mtime_ns: int


def list_server_files(
    root: Path, progress: Callable[[int], None] | None = None, every: int = 500
) -> tuple[ServerFile, ...]:
    """Every ``.tsv`` under *root* with its size and modification time; reads nothing else.

    Args:
        root (Path): The raw-data folder.
        progress (Callable[[int], None] | None): Called with the number of files listed so far
            every *every* files, for a progress line. Default ``None``.
        every (int): How often *progress* is called. Default 500.

    Returns:
        tuple[ServerFile, ...]: In ``rglob`` order, the order an import reads them in.

    Raises:
        ShareReadError: If listing or a ``stat`` fails (an unreachable share).
    """
    root = Path(root)
    files = []
    try:
        for path in root.rglob("*.tsv"):
            stat = path.stat()
            relative = path.relative_to(root).as_posix()
            files.append(ServerFile(relative, stat.st_size, stat.st_mtime_ns))
            if progress is not None and len(files) % every == 0:
                progress(len(files))
    except OSError as err:
        raise ShareReadError(
            f"Could not list {root}: {type(err).__name__}: {err}. The import cache is unchanged."
        ) from err
    return tuple(files)


# --- snapshots ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Manifest:
    """What a cohort snapshot was made from, stored as JSON beside it.

    Attributes:
        root (str): The raw-data folder, as an absolute path.
        classify_task_types (bool): Whether problems were numbered by task type too.
        code_version (str): The import code hash when the snapshot was made.
        file_code_version (str): :func:`file_code_version` then, which keys its per-file parses.
        created (float): When the snapshot was made, in epoch seconds.
        files (tuple[ServerFile, ...]): The listed ``.tsv`` files the snapshot holds, sorted.
        skipped (tuple[ServerFile, ...]): Listed files the import skipped because they could not
            be read, sorted. They are not covered, so the next refresh tries them again.
    """

    root: str
    classify_task_types: bool
    code_version: str
    file_code_version: str
    created: float
    files: tuple[ServerFile, ...] = field(default=())
    skipped: tuple[ServerFile, ...] = field(default=())

    def __post_init__(self):
        for name in ("files", "skipped"):
            listed = tuple(sorted(ServerFile(*f) for f in getattr(self, name)))
            object.__setattr__(self, name, listed)

    def to_json(self) -> str:
        """The manifest as JSON text."""
        return json.dumps({"format": MANIFEST_FORMAT, **asdict(self)}, indent=1)

    @classmethod
    def from_json(cls, text: str) -> Manifest:
        """Parse :meth:`to_json` output.

        Raises:
            ValueError: If *text* is not a manifest of this format.
        """
        data = json.loads(text)
        if data.pop("format", None) != MANIFEST_FORMAT:
            raise ValueError("unknown manifest format")
        return cls(**data)


class Snapshot(NamedTuple):
    """A complete snapshot folder and its manifest.

    Attributes:
        path (Path): The folder holding :data:`DATA_FILE` and :data:`MANIFEST_FILE`.
        manifest (Manifest): Its manifest.
    """

    path: Path
    manifest: Manifest


class SnapshotListing(NamedTuple):
    """The snapshots of a cache directory.

    Attributes:
        snapshots (tuple[Snapshot, ...]): Those whose manifest could be read, by folder name.
        unreadable (tuple[Path, ...]): Complete folders whose manifest could not be read.
    """

    snapshots: tuple[Snapshot, ...]
    unreadable: tuple[Path, ...]


def snapshots_dir(cache_dir: Path) -> Path:
    """Where the snapshots of *cache_dir* live."""
    return Path(cache_dir) / SNAPSHOTS_DIR


def _snapshot_name(manifest: Manifest) -> str:
    created = datetime.fromtimestamp(manifest.created).strftime("%Y%m%dT%H%M%S")
    flag = "classified" if manifest.classify_task_types else "unclassified"
    return f"{cohort_slug(manifest.root)}-{flag}-{created}-{uuid.uuid4().hex[:8]}"


def write_snapshot(cache_dir: Path, data, manifest: Manifest) -> Snapshot:
    """Store *data* with *manifest* as a new snapshot, atomically.

    Both files go into a hidden temporary folder that is renamed into place last, so a failure
    or interruption leaves no partial snapshot among the listed ones.

    Args:
        cache_dir (Path): The cache directory.
        data: The assembled import (pickled).
        manifest (Manifest): What it was made from.

    Returns:
        Snapshot: The new snapshot.

    Raises:
        OSError: If writing fails; the temporary folder is removed and nothing else changes.
    """
    base = snapshots_dir(cache_dir)
    base.mkdir(parents=True, exist_ok=True)
    final = base / _snapshot_name(manifest)
    tmp = base / f"{TMP_PREFIX}{final.name}"
    tmp.mkdir()
    try:
        with open(tmp / DATA_FILE, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        (tmp / MANIFEST_FILE).write_text(manifest.to_json())
        tmp.rename(final)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return Snapshot(final, manifest)


def _read_manifest(folder: Path) -> Manifest | None:
    try:
        return Manifest.from_json((folder / MANIFEST_FILE).read_text())
    except (OSError, ValueError, TypeError) as err:
        warnings.warn(
            f"Skipping the import snapshot {folder}: its manifest cannot be read ({err})",
            ImportCacheWarning,
            stacklevel=3,
        )
        return None


def list_snapshots(cache_dir: Path) -> SnapshotListing:
    """Every complete snapshot under *cache_dir*, from the manifests alone.

    Args:
        cache_dir (Path): The cache directory.

    Returns:
        SnapshotListing: Sorted by folder name; empty if there are none. Temporary folders of
        a write in progress (or interrupted) are left out.
    """
    base = snapshots_dir(cache_dir)
    if not base.is_dir():
        return SnapshotListing((), ())
    complete = [
        d
        for d in sorted(base.iterdir())
        if not d.name.startswith(TMP_PREFIX)
        and (d / MANIFEST_FILE).is_file()
        and (d / DATA_FILE).is_file()
    ]
    read = [(d, _read_manifest(d)) for d in complete]
    return SnapshotListing(
        tuple(Snapshot(d, m) for d, m in read if m is not None),
        tuple(d for d, m in read if m is None),
    )


def incomplete_snapshots(cache_dir: Path) -> tuple[tuple[Path, float], ...]:
    """``(folder, mtime)`` of the temporary folders of writes that are running or were cut off.

    Args:
        cache_dir (Path): The cache directory.

    Returns:
        tuple[tuple[Path, float], ...]: Sorted by folder name.
    """
    base = snapshots_dir(cache_dir)
    if not base.is_dir():
        return ()
    return tuple(
        (d, d.stat().st_mtime) for d in sorted(base.iterdir()) if d.name.startswith(TMP_PREFIX)
    )


def stray_parse_temps(cache_dir: Path) -> tuple[tuple[Path, float], ...]:
    """``(file, mtime)`` of per-file parse temporaries left by writes that were cut off.

    Args:
        cache_dir (Path): The cache directory.

    Returns:
        tuple[tuple[Path, float], ...]: Sorted by path; also lists writes still running, which
        the prune leaves alone by their age.
    """
    base = Path(cache_dir) / FILES_DIR
    if not base.is_dir():
        return ()
    return tuple((p, p.stat().st_mtime) for p in sorted(base.glob(f"*/{TMP_PREFIX}*")))


def load_snapshot_data(snapshot: Snapshot):
    """Unpickle a snapshot's data.

    Args:
        snapshot (Snapshot): A listed snapshot.

    Returns:
        object: What :func:`write_snapshot` stored.
    """
    with open(snapshot.path / DATA_FILE, "rb") as f:
        return pickle.load(f)


# --- per-file parses ------------------------------------------------------------------------


class CaughtWarning(NamedTuple):
    """A warning raised while a file was parsed, kept so it can be raised again.

    A warning raised in one of the parsing functions records its place as that function and the
    offset from its first line, so a replay names the line where it is now even after code
    above it has moved; *filename* and *lineno* are used for warnings raised elsewhere (pandas).

    Attributes:
        category (type[Warning]): The warning class.
        message (str): Its text, naming the file by the path it was parsed from.
        filename (str): The source file that raised it, when parsed.
        lineno (int): The line in *filename*, when parsed.
        module (str | None): The raising module's name, which warning filters match on.
        site (str): The parsing function it was raised in (see :func:`parse_functions`), or
            ``""``.
        offset (int): Its line relative to that function's first line.
    """

    category: type
    message: str
    filename: str
    lineno: int
    module: str | None = None
    site: str = ""
    offset: int = 0


class CachedParse(NamedTuple):
    """The cached per-file part of an import.

    Attributes:
        parsed (ParsedFile): The raw and the expanded recording.
        source_path (str): The path it was parsed from, replaced in messages on replay.
        warnings (tuple[CaughtWarning, ...]): The warnings the parse raised, in order.
    """

    parsed: ParsedFile
    source_path: str
    warnings: tuple[CaughtWarning, ...]


def parse_functions() -> dict[str, Callable]:
    """The functions whose output a :class:`CachedParse` holds, by name, as bound now.

    Returns:
        dict[str, Callable]: The parse this module calls and the import_data functions it uses.
    """
    return {
        "parse_tsv_file": parse_tsv_file,
        "read_tsv_file": import_data.read_tsv_file,
        "drop_non_numeric_times": import_data.drop_non_numeric_times,
        "expand_content_columns": import_data.expand_content_columns,
        "parse_content_json": import_data.parse_content_json,
    }


def _source_file(obj) -> str:
    """*obj*'s source file relative to the repository (absolute if outside it)."""
    path = Path(inspect.getsourcefile(obj)).resolve()
    return path.relative_to(REPO_ROOT).as_posix() if path.is_relative_to(REPO_ROOT) else str(path)


def file_code_version() -> str:
    """Hash of the code a per-file parse depends on, and of the pandas version.

    A change to any parsing function (or the file it lives in), to the stored classes, or to
    pandas (whose pickles are not portable across versions) gives every file a new key. Line
    numbers are not part of it: warnings are replayed at their current line (see
    :class:`CaughtWarning`).

    Returns:
        str: Hex SHA-256; part of every :func:`file_key`.
    """
    digest = hashlib.sha256(pd.__version__.encode())
    stored = (ParsedFile, CachedParse, CaughtWarning, parse_and_record, replay, warning_site)
    for obj in (*parse_functions().values(), *stored):
        digest.update(_source_file(obj).encode())
        digest.update(inspect.getsource(obj).encode())
    return digest.hexdigest()


def file_key(file: ServerFile, code_version: str) -> str:
    """The per-file cache key of a listed file.

    Args:
        file (ServerFile): The file as listed.
        code_version (str): :func:`file_code_version`.

    Returns:
        str: Hex SHA-256 of the relative path, size, modification time and code version.
    """
    text = "\0".join((file.path, str(file.size), str(file.mtime_ns), code_version))
    return hashlib.sha256(text.encode()).hexdigest()


def files_dir(cache_dir: Path, root: str) -> Path:
    """The folder of *root*'s per-file parses.

    Args:
        cache_dir (Path): The cache directory.
        root (str): The raw-data folder.

    Returns:
        Path: ``<cache_dir>/import-files/<cohort slug>``.
    """
    return Path(cache_dir) / FILES_DIR / cohort_slug(root)


def parse_and_record(path: Path) -> CachedParse:
    """Parse one file, keeping the warnings it raises instead of showing them.

    The caller shows them with :func:`replay`, so a cached file and a freshly parsed one warn
    alike. If the parse fails, its warnings are shown at once and the error is re-raised.

    Args:
        path (Path): The ``.tsv`` file.

    Returns:
        CachedParse: The parse, the path and the warnings.

    Raises:
        Exception: Whatever :func:`~src.behavior_import.import_data.parse_tsv_file` raises.
    """
    error = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            parsed = parse_tsv_file(path)
        except Exception as err:  # re-raised below, once the warnings before it are shown
            error = err
    functions = parse_functions()
    kept = tuple(
        CaughtWarning(w.category, str(w.message), w.filename, w.lineno, *warning_site(w, functions))
        for w in caught
    )
    if error is not None:
        _show(kept, {})
        raise error
    return CachedParse(parsed, str(path), kept)


def _line_range(function: Callable) -> tuple[str, int, int]:
    code = function.__code__
    lines = [line for _, _, line in code.co_lines() if line is not None]
    return code.co_filename, code.co_firstlineno, max(lines, default=code.co_firstlineno)


def _module_of(filename: str) -> str | None:
    """The name of the loaded module whose file is *filename*, as warning filters see it."""
    for name, module in list(sys.modules.items()):
        if getattr(module, "__file__", None) == filename:
            return name
    return None


def warning_site(w: warnings.WarningMessage, functions: dict[str, Callable]) -> tuple:
    """``(module, site, offset)`` of a caught warning, for :class:`CaughtWarning`.

    Args:
        w (warnings.WarningMessage): The caught warning.
        functions (dict[str, Callable]): :func:`parse_functions`.

    Returns:
        tuple: The raising module's name, the parsing function the line is in (``""`` if
        none) and the line's offset from that function's first line.
    """
    for name, function in functions.items():
        filename, first, last = _line_range(function)
        if filename == w.filename and first <= w.lineno <= last:
            return function.__module__, name, w.lineno - first
    return _module_of(w.filename), "", 0


def _where(w: CaughtWarning, functions: dict[str, Callable]) -> tuple[str, int]:
    """``(filename, lineno)`` to raise *w* at now."""
    if w.site and w.site in functions:
        filename, first, _ = _line_range(functions[w.site])
        return filename, first + w.offset
    return w.filename, w.lineno


def _show(caught, registry: dict, old: str = "", new: str = "") -> None:
    functions = parse_functions()
    for w in caught:
        message = w.message.replace(old, new) if old else w.message
        filename, lineno = _where(w, functions)
        warnings.warn_explicit(
            message, w.category, filename, lineno, module=w.module, registry=registry
        )


def replay(cached: CachedParse, path: Path, registry: dict) -> ParsedFile:
    """The parse of *cached* for the file now at *path*, raising its warnings again.

    Args:
        cached (CachedParse): A stored parse.
        path (Path): Where the file is now; messages and ``attrs["source_path"]`` name it.
        registry (dict): One dict per import, so a warning repeated across files is shown once
            per import as Python's default filter would.

    Returns:
        ParsedFile: The frames, their ``source_path`` set to *path*.
    """
    _show(cached.warnings, registry, cached.source_path, str(path))
    for frame in cached.parsed:
        frame.attrs["source_path"] = str(path)
    return cached.parsed


def save_parsed(location: Path, cached: CachedParse) -> None:
    """Store a parse atomically (temporary file, then rename).

    Args:
        location (Path): The entry's file.
        cached (CachedParse): The parse.

    Raises:
        OSError: If writing fails; no partial entry is left.
    """
    location.parent.mkdir(parents=True, exist_ok=True)
    tmp = location.with_name(f"{TMP_PREFIX}{location.name}.{uuid.uuid4().hex[:8]}")
    try:
        with open(tmp, "wb") as f:
            pickle.dump(cached, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, location)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def load_parsed(location: Path) -> CachedParse | None:
    """A stored parse, or ``None`` if there is none or it cannot be read.

    A damaged entry is treated as missing, with a warning: the file is then parsed again,
    which costs only time.

    Args:
        location (Path): The entry's file.

    Returns:
        CachedParse | None: The parse, or ``None``.
    """
    try:
        with open(location, "rb") as f:
            loaded = pickle.load(f)
        if not isinstance(loaded, CachedParse):
            raise TypeError(f"it holds a {type(loaded).__name__}")
    except FileNotFoundError:
        return None
    except MemoryError:
        raise
    except Exception as err:  # any unreadable entry is replaced by a fresh parse
        warnings.warn(
            f"The cached parse {location} cannot be read ({type(err).__name__}: {err}); the file "
            "is re-parsed",
            ImportCacheWarning,
            stacklevel=2,
        )
        return None
    return loaded


@dataclass
class CachedParser:
    """A ``parse`` for :func:`~src.behavior_import.import_data.import_data` that reuses parses.

    Each file is looked up by the :func:`file_key` of its listed state; on a miss it is parsed
    and stored. Either way its warnings are raised again (:func:`replay`), so the import looks
    the same as one that parsed every file.

    Attributes:
        root (Path): The raw-data folder the paths are under.
        store (Path): The cohort's per-file folder (:func:`files_dir`).
        listing (dict[str, ServerFile]): ``{relative path: listed file}``.
        code_version (str): :func:`file_code_version`.
        parsed (list[str]): Relative paths parsed in this import, in order.
        reused (list[str]): Relative paths loaded from the store, in order.
    """

    root: Path
    store: Path
    listing: dict[str, ServerFile]
    code_version: str
    parsed: list[str] = field(default_factory=list)
    reused: list[str] = field(default_factory=list)
    _registry: dict = field(default_factory=dict, repr=False)

    def __call__(self, path: Path) -> ParsedFile:
        """The parse of one listed file.

        Raises:
            ShareReadError: If the share has stopped answering (see :func:`share_is_gone`).
            OSError: If only this file cannot be read (no permission, removed since it was
                listed); like a file that cannot be parsed, the import skips it.
            ImportAborted: If the parse cannot be stored in the cache.
        """
        relative = Path(path).relative_to(self.root).as_posix()
        location = self.store / f"{file_key(self.listing[relative], self.code_version)}.pkl"
        cached = load_parsed(location)
        if cached is None:
            cached = self._parse(Path(path))
            try:
                save_parsed(location, cached)
            except OSError as err:
                raise ImportAborted(f"Cannot write the import cache {location}: {err}") from err
            self.parsed.append(relative)
        else:
            self.reused.append(relative)
        return replay(cached, Path(path), self._registry)

    def skipped(self) -> tuple[ServerFile, ...]:
        """The listed files that were neither parsed nor reused: the import skipped them."""
        covered = {*self.parsed, *self.reused}
        return tuple(f for path, f in self.listing.items() if path not in covered)

    def _parse(self, path: Path) -> CachedParse:
        try:
            return parse_and_record(path)
        except OSError as err:
            if not share_is_gone(err, self.root):
                raise  # one unreadable file: the import skips it, as a fresh import does
            raise ShareReadError(
                f"Could not read {path}: {type(err).__name__}: {err}. The import cache is "
                "unchanged."
            ) from err
