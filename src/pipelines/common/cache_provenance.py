"""Notes that mark outputs made from the import cache without reading the server.

The cohort runs are cache-first (see :func:`~src.behavior_import.load_trials.load_trials_with_log`
with ``from_cache``): they may miss the newest sessions, and nothing in their figures would show
it. Each run therefore writes ``from-cache.txt`` into the folders it fills, listing per cohort
when its cached import was made. A run that read the server (``--refresh-cache``) removes the
file again, so a folder only carries it while its contents come from the cache.
"""

from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path

from src.behavior_import.import_log import CacheSource, ImportLog
from src.behavior_import.load_trials import TAIL_PARTS

FROM_CACHE_FILE = "from-cache.txt"
# The run scripts' help for their cache options.
REFRESH_HELP = (
    "first bring each cohort's import cache up to date with the server (only new or changed "
    "files are parsed), then run"
)
CACHE_FIRST_NOTE = (
    "runs read each cohort's newest import cache without the server by default; sessions added "
    "since then are missing until --refresh-cache"
)
FROM_CACHE_HELP = f"deprecated, changes nothing: {CACHE_FIRST_NOTE}"


def _date(epoch_seconds: float, fmt: str = "%Y-%m-%d") -> str:
    return datetime.fromtimestamp(epoch_seconds).strftime(fmt)


def cache_sources(logs: Mapping[str, ImportLog | None]) -> dict[str, CacheSource]:
    """``{cohort: CacheSource}`` of the logs that were loaded from the cache.

    Args:
        logs (Mapping[str, ImportLog | None]): ``{cohort name: import log}``; ``None`` for
            trials passed in without a log.

    Returns:
        dict[str, CacheSource]: Only the cohorts loaded cache-first; empty for a run that read
        the server.
    """
    return {
        cohort: log.cache_source
        for cohort, log in logs.items()
        if log is not None and log.cache_source is not None
    }


def _source_line(cohort: str, source: CacheSource) -> str:
    code = "current import code" if source.code_matches else "older import code"
    extras = []
    if source.matched_on_tail:
        extras.append(f"root matched on its last {TAIL_PARTS} path parts")
    if source.filled_fields:
        extras.append(f"log fields set to defaults: {', '.join(source.filled_fields)}")
    return (
        f"{cohort}: cached import of {_date(source.time, '%Y-%m-%d %H:%M')}, {code}, "
        f"{source.n_files} .tsv files, {source.root}" + "".join(f"; {e}" for e in extras)
    )


def from_cache_description(sources: Mapping[str, CacheSource]) -> str:
    """The text of ``from-cache.txt``.

    Args:
        sources (Mapping[str, CacheSource]): ``{cohort: source}``, non-empty.

    Returns:
        str: A header saying the server was not read, then one line per cohort with the date
        of its cached import, whether the import code matched, and how many ``.tsv`` files it
        covered.
    """
    lines = [
        "Made from the import cache, without reading the server: sessions added since are "
        "missing (run with --refresh-cache to include them).",
        *(_source_line(cohort, source) for cohort, source in sorted(sources.items())),
    ]
    return "\n".join(lines) + "\n"


def write_from_cache_notes(folders: Iterable[Path], sources: Mapping[str, CacheSource]) -> None:
    """Write ``from-cache.txt`` into each folder, or remove it when *sources* is empty.

    Args:
        folders (Iterable[Path]): The output folders of the run; created if needed when
            writing.
        sources (Mapping[str, CacheSource]): ``{cohort: source}``; empty for a run that read
            the server, whose folders must not keep a note from an earlier run.
    """
    text = from_cache_description(sources) if sources else None
    for folder in folders:
        path = Path(folder) / FROM_CACHE_FILE
        if text is None:
            path.unlink(missing_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
