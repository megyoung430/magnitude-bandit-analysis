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
"""

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

from joblib import Memory

from src.behavior_import.extract_trials import extract_trials_grouped_by_problem
from src.behavior_import.import_data import import_data
from src.behavior_import.import_log import ImportLog
from src.behavior_import.warning_records import record_warnings
from src.settings import get_settings

# The source hashed into the cache key. It includes this module, so a change to what is cached
# invalidates the caches written before it.
IMPORT_CODE_DIR = Path(__file__).resolve().parent


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


def load_trials_with_log(
    root: Path,
    *,
    cache_dir: Path | None = None,
    use_cache: bool = True,
    classify_task_types: bool = False,
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

    Returns:
        ImportedTrials: The trials, ``{problem_number: {subject: {session_key:
        session_dict}}}``, and the :class:`~src.behavior_import.import_log.ImportLog` of the
        import that produced them (the one stored in the cache, on a cached load).

    Raises:
        FileNotFoundError: If *root* is not an existing directory.
    """
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
