"""What the server has that a cache snapshot lacks: new, changed and removed files.

Pure functions on two listings (a snapshot's manifest and the server), grouped by subject and
session for :mod:`scripts.full_pipelines.import_cache`'s ``check``.
"""

from collections.abc import Iterable
from datetime import datetime
from pathlib import PurePosixPath
from typing import NamedTuple

from src.behavior_import.cache_store import ServerFile
from src.behavior_import.import_data import path_labels

NEW, CHANGED, REMOVED, UNREADABLE = "new", "changed", "removed", "unreadable"
NO_SUBJECT = "(no subject folder)"
NO_SESSION = "(no session folder)"


class FileChange(NamedTuple):
    """One file that differs between the cache and the server.

    Attributes:
        status (str): ``"new"`` (only on the server), ``"changed"`` (size or modification time
            differ), ``"removed"`` (only in the cache) or ``"unreadable"`` (skipped by the last
            refresh because it could not be read; the next refresh tries it again).
        path (str): Relative path.
        cached (ServerFile | None): The file as the cache recorded it; ``None`` if new.
        server (ServerFile | None): The file on the server now; ``None`` if removed.
    """

    status: str
    path: str
    cached: ServerFile | None
    server: ServerFile | None


class CacheDiff(NamedTuple):
    """The differences between a snapshot and the server.

    Attributes:
        changes (tuple[FileChange, ...]): Every differing file, sorted by path.
        new_subjects (tuple[str, ...]): Subjects with files on the server and none in the cache.
        new_sessions (tuple[tuple[str, str], ...]): ``(subject, session folder)`` on the server
            and not in the cache (including those of new subjects).
    """

    changes: tuple[FileChange, ...]
    new_subjects: tuple[str, ...]
    new_sessions: tuple[tuple[str, str], ...]

    @property
    def up_to_date(self) -> bool:
        """Whether the cache holds exactly the server's files."""
        return not self.changes

    def count(self, status: str) -> int:
        """How many changes have *status*."""
        return sum(c.status == status for c in self.changes)


def _labels(path: str) -> tuple[str, str]:
    subject, session = path_labels(path)
    return subject or NO_SUBJECT, session or NO_SESSION


def _change(path: str, cached: dict, server: dict, skipped: dict) -> FileChange | None:
    before, now = cached.get(path), server.get(path)
    if path in skipped and now is not None:
        return FileChange(UNREADABLE, path, skipped[path], now)
    if before is None:
        return FileChange(NEW, path, None, now)
    if now is None:
        return FileChange(REMOVED, path, before, None)
    return FileChange(CHANGED, path, before, now) if before != now else None


def cache_diff(
    cached_files: Iterable[ServerFile],
    server_files: Iterable[ServerFile],
    skipped_files: Iterable[ServerFile] = (),
) -> CacheDiff:
    """Compare a snapshot's files with the server's.

    Args:
        cached_files (Iterable[ServerFile]): The files of the snapshot's manifest.
        server_files (Iterable[ServerFile]): The files listed on the server now.
        skipped_files (Iterable[ServerFile]): Files the snapshot's import skipped as unreadable.
            Default none.

    Returns:
        CacheDiff: The new, changed, removed and unreadable files and the new subjects and
        sessions.
    """
    cached = {f.path: f for f in cached_files}
    server = {f.path: f for f in server_files}
    skipped = {f.path: f for f in skipped_files}
    paths = sorted(cached.keys() | server.keys())
    changes = tuple(c for path in paths if (c := _change(path, cached, server, skipped)))
    known = {_labels(p) for p in (*cached, *skipped)}
    known_subjects = {subject for subject, _ in known}
    on_server = {_labels(p) for p in server}
    return CacheDiff(
        changes,
        tuple(sorted({s for s, _ in on_server} - known_subjects)),
        tuple(sorted(on_server - known)),
    )


def grouped_changes(diff: CacheDiff) -> dict[str, dict[str, list[FileChange]]]:
    """``{subject: {session folder: [changes]}}``, all sorted.

    Args:
        diff (CacheDiff): From :func:`cache_diff`.

    Returns:
        dict[str, dict[str, list[FileChange]]]: Files without a subject or session folder are
        under :data:`NO_SUBJECT` / :data:`NO_SESSION`.
    """
    grouped: dict[str, dict[str, list[FileChange]]] = {}
    for change in diff.changes:
        subject, session = _labels(change.path)
        grouped.setdefault(subject, {}).setdefault(session, []).append(change)
    return {
        subject: {session: grouped[subject][session] for session in sorted(grouped[subject])}
        for subject in sorted(grouped)
    }


def _time(mtime_ns: int) -> str:
    return datetime.fromtimestamp(mtime_ns / 1e9).strftime("%Y-%m-%d %H:%M")


def _within_session(path: str) -> str:
    """*path* from below its session folder, or whole when it has none."""
    parts = PurePosixPath(path).parts
    starts = [i for i, part in enumerate(parts) if part.startswith("ses-")]
    return "/".join(parts[starts[0] + 1 :]) if starts else path


def _file_line(change: FileChange) -> str:
    name = _within_session(change.path)
    if change.status == UNREADABLE:
        return f"{'skipped':<8} {name} (could not be read at the last refresh; tried again next)"
    if change.status != CHANGED:
        return f"{change.status:<8} {name}"
    before, now = change.cached, change.server
    return (
        f"{change.status:<8} {name} (size {before.size} -> {now.size} bytes, modified "
        f"{_time(before.mtime_ns)} -> {_time(now.mtime_ns)})"
    )


def describe_diff(diff: CacheDiff) -> list[str]:
    """The changes as indented lines: subject, then session, then one line per file.

    Args:
        diff (CacheDiff): From :func:`cache_diff`.

    Returns:
        list[str]: Empty when up to date. New subjects and sessions are marked so.
    """
    new_subjects, new_sessions = set(diff.new_subjects), set(diff.new_sessions)
    lines = []
    for subject, sessions in grouped_changes(diff).items():
        lines.append(f"{subject} (new subject)" if subject in new_subjects else subject)
        for session, changes in sessions.items():
            new = (subject, session) in new_sessions and subject not in new_subjects
            lines.append(f"  {session} (new session)" if new else f"  {session}")
            lines.extend(f"    {_file_line(c)}" for c in changes)
    return lines
