"""Import and extract trials for a raw-data folder, cached on disk.

Parsing every pyControl ``.tsv`` in a cohort is the slowest step of any
analysis. The result of :func:`extract_trials_grouped_by_problem` is cached
with joblib, keyed on the raw files (path, size, modification time), the
source of ``src/behavior_import``, and the options. A new session, an edited
file, or a change to the import code therefore re-runs the import
automatically; otherwise the result loads from disk in seconds. Deleting the
cache directory only costs time.
"""

import hashlib
from functools import lru_cache
from pathlib import Path

from joblib import Memory

from src.behavior_import.extract_trials import extract_trials_grouped_by_problem
from src.behavior_import.import_data import import_data
from src.settings import get_settings

IMPORT_CODE_DIR = Path(__file__).resolve().parent


def raw_data_fingerprint(root: Path) -> tuple[tuple[str, int, int], ...]:
    """``(relative path, size, mtime_ns)`` of every ``.tsv`` under *root*, sorted."""
    root = Path(root)
    return tuple(
        sorted(
            (p.relative_to(root).as_posix(), p.stat().st_size, p.stat().st_mtime_ns)
            for p in root.rglob("*.tsv")
        )
    )


def import_code_version() -> str:
    """Hash of the import code, so editing it invalidates cached results."""
    digest = hashlib.sha256()
    for path in sorted(IMPORT_CODE_DIR.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _import_and_extract(root: Path) -> dict:
    return extract_trials_grouped_by_problem(import_data(root))


def _load(root: str, fingerprint, code_version: str) -> dict:
    # fingerprint and code_version are not used here: they are arguments only
    # so that they become part of the cache key
    return _import_and_extract(Path(root))


@lru_cache
def _cached_load(cache_dir: Path):
    return Memory(location=cache_dir, verbose=0).cache(_load)


def load_trials_by_problem(
    root: Path,
    *,
    cache_dir: Path | None = None,
    use_cache: bool = True,
) -> dict:
    """``extract_trials_grouped_by_problem(import_data(root))``, cached on disk.

    Args:
        root: Raw-data folder, e.g. ``<data_root>/<experiment>/<cohort>/rawdata``.
        cache_dir: Cache location (default: ``get_settings().cache_dir``).
        use_cache: ``False`` always re-imports and leaves the cache untouched.

    Returns:
        ``{problem_number: {subject: {session_key: session_dict}}}``.
    """
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Raw data folder not found (is the share mounted?): {root}")
    if not use_cache:
        return _import_and_extract(root)

    load = _cached_load(Path(cache_dir or get_settings().cache_dir))
    args = (str(root), raw_data_fingerprint(root), import_code_version())
    if load.check_call_in_cache(*args):
        print(f"[INFO] Loading trials for {root} from cache")
    return load(*args)
