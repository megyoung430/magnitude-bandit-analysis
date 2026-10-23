"""Whether a folder's import cache is up to date with the server, and what it lacks.

Read-only: :func:`check_import_cache` lists the server (sizes and modification times only) and
reads the cache's manifests and legacy metadata, never the cached data; :func:`describe_check`
turns the result into the report of ``scripts/full_pipelines/import_cache.py check``.
"""

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

from src.behavior_import import load_trials
from src.behavior_import.cache_diff import CacheDiff, cache_diff, describe_diff
from src.behavior_import.cache_store import ServerFile, list_server_files, list_snapshots
from src.behavior_import.load_trials import format_time
from src.settings import get_unchecked_settings


class CachedState(NamedTuple):
    """The cache of a folder that :func:`check_import_cache` compares with the server.

    Attributes:
        kind (str): ``"snapshot"`` or ``"legacy"`` (an older joblib import).
        root (str): The folder it was made from.
        time (float): When it was made, in epoch seconds.
        code_matches (bool): Whether it was made with the current import code.
        files (tuple[ServerFile, ...]): The files it was made from.
    """

    kind: str
    root: str
    time: float
    code_matches: bool
    files: tuple[ServerFile, ...]


class CacheCheck(NamedTuple):
    """A folder's cache against the server.

    Attributes:
        root (str): The raw-data folder.
        cached (CachedState | None): What cache-first runs load; ``None`` if nothing.
        server_files (tuple[ServerFile, ...]): The files on the server now.
        diff (CacheDiff): The cached files against the server's.
    """

    root: str
    cached: CachedState | None
    server_files: tuple[ServerFile, ...]
    diff: CacheDiff

    @property
    def up_to_date(self) -> bool:
        """A cache made from exactly the server's files, with the current import code."""
        return self.cached is not None and self.cached.code_matches and self.diff.up_to_date


def _cached_state(root: str, cache_dir: Path, classify_task_types: bool) -> CachedState | None:
    """What a cache-first load of *root* would use, from manifests and metadata alone."""
    code = load_trials.import_code_version()
    snapshot = load_trials.choose_snapshot(
        list_snapshots(cache_dir).snapshots, root, classify_task_types
    )
    if snapshot is not None:
        m = snapshot.manifest
        return CachedState("snapshot", m.root, m.created, m.code_version == code, m.files)
    entries = load_trials.cached_import_entries(cache_dir).entries
    if not load_trials.has_legacy_import(entries, root, classify_task_types):
        return None
    e = load_trials.choose_cache_entry(entries, root, classify_task_types, code)
    return CachedState("legacy", e.root, e.time, e.code_version == code, e.files)


def check_import_cache(
    root: Path,
    *,
    cache_dir: Path | None = None,
    classify_task_types: bool = False,
    progress: Callable[[int], None] | None = None,
) -> CacheCheck:
    """Compare what a cache-first load of *root* would use with the files on the server.

    Reads only the server listing (sizes and modification times) and the cache's manifests and
    metadata; writes nothing.

    Args:
        root (Path): The raw-data folder.
        cache_dir (Path | None): Cache location. Default ``None``: the ``BANDIT_CACHE_DIR``
            setting. Keyword-only.
        classify_task_types (bool): As for :func:`load_trials_with_log`. Keyword-only.
        progress (Callable[[int], None] | None): Called while listing. Keyword-only.

    Returns:
        CacheCheck: The cache, the server's files and the differences.

    Raises:
        FileNotFoundError: If *root* is not an existing directory.
        ShareReadError: If the folder cannot be listed.
    """
    wanted = load_trials.absolute_path_string(root)
    cache_dir = Path(cache_dir or get_unchecked_settings().cache_dir)
    cached = _cached_state(wanted, cache_dir, classify_task_types)
    if not Path(wanted).is_dir():
        raise FileNotFoundError(f"Raw data folder not found (is the share mounted?): {wanted}")
    server = list_server_files(Path(wanted), progress)
    return CacheCheck(wanted, cached, server, cache_diff(cached.files if cached else (), server))


def describe_check(check: CacheCheck, label: str) -> list[str]:
    """The report of :func:`check_import_cache` for one folder, as lines.

    Args:
        check (CacheCheck): The check.
        label (str): What to call the folder, e.g. ``"cohort-03"``.

    Returns:
        list[str]: A header with the cache's date, kind, file count and import code and the
        server's file count, then "up to date" or the counts of new, changed and removed files
        and those files grouped by subject and session.
    """
    n_server, d = len(check.server_files), check.diff
    if check.cached is None:
        return [
            f"{label}: no import cache of {check.root}",
            f"  server: {n_server} .tsv files, {len(d.new_subjects)} subject(s), "
            f"{len(d.new_sessions)} session(s); none are cached",
        ]
    c = check.cached
    code = "current import code" if c.code_matches else "import code changed since"
    lines = [
        f"{label}: cache of {format_time(c.time)} ({c.kind}, {len(c.files)} .tsv files, {code})",
        f"  server: {n_server} .tsv files in {check.root}",
    ]
    if d.up_to_date:
        state = "up to date" if c.code_matches else "files up to date, but the import code changed"
        return [*lines, f"  {state}"]
    lines.append(
        f"  not in the cache: {d.count('new')} new file(s) ({len(d.new_subjects)} new "
        f"subject(s), {len(d.new_sessions)} new session(s)), {d.count('changed')} changed, "
        f"{d.count('removed')} removed"
    )
    return [*lines, *(f"  {line}" for line in describe_diff(d))]


def check_to_dict(check: CacheCheck, label: str) -> dict:
    """The check as plain JSON-ready data, for ``import_cache.py check --json``.

    Args:
        check (CacheCheck): The check.
        label (str): What to call the folder, e.g. ``"cohort-03"``.

    Returns:
        dict: ``cohort``, ``root``, ``up_to_date``, ``cache`` (``None`` or its kind, creation
        time, whether the import code matches and its file count), ``server_files`` (a count),
        the ``new``, ``changed`` and ``removed`` relative paths, ``new_subjects`` and
        ``new_sessions``.
    """
    c = check.cached
    cache = None
    if c is not None:
        cache = {
            "kind": c.kind,
            "created": datetime.fromtimestamp(c.time).isoformat(timespec="seconds"),
            "code_matches": c.code_matches,
            "files": len(c.files),
        }
    by_status = {
        status: [x.path for x in check.diff.changes if x.status == status]
        for status in ("new", "changed", "removed")
    }
    return {
        "cohort": label,
        "root": check.root,
        "up_to_date": check.up_to_date,
        "cache": cache,
        "server_files": len(check.server_files),
        **by_status,
        "new_subjects": list(check.diff.new_subjects),
        "new_sessions": [list(s) for s in check.diff.new_sessions],
    }
