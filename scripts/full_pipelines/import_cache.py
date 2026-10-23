"""Check, refresh and prune the import cache that the cohort runs read.

The runs (run_cohorts.py, run_all_drifting_problems.py) load each cohort from its newest cache
snapshot without touching the server. Bring the cache up to date when there is new data:

    uv run python scripts/full_pipelines/import_cache.py check --cohorts 1 3 4
    uv run python scripts/full_pipelines/import_cache.py refresh --cohorts 3
    uv run python scripts/full_pipelines/import_cache.py prune --dry-run

check lists the server (file sizes and dates only) and compares it with each cohort's cache:
when the cache was made, whether the import code changed since, and which subjects, sessions
and files are new, changed or removed. It exits 0 when every cohort is up to date, 1 when one
is not, 2 when the server cannot be read or the settings are missing; --json prints the same
as JSON.

refresh parses only the new and changed files (reusing the earlier parse of every other file),
assembles each cohort, stores it as a new snapshot and prunes what is no longer needed
(--keep-old to skip that). If the server fails part-way, the old snapshot stays in use.

prune removes snapshots older than the newest of their cohort, old joblib imports that a
snapshot has replaced, and per-file parses no kept snapshot uses; a cohort's only cache is
never removed. --dry-run lists what would go and the space it would free.

Raw data is read (never written) from <BANDIT_DATA_ROOT>/3x3_field_magnitude_bandit/rawdata/
cohort-0N; the cache lives in BANDIT_CACHE_DIR (default <repo>/.cache).
"""

import argparse
import json
import sys

from pydantic import ValidationError

from src.behavior_import.cache_check import check_import_cache, check_to_dict, describe_check
from src.behavior_import.cache_refresh import prune_import_cache, refresh_import_cache
from src.behavior_import.cache_store import CacheLocationError, ShareReadError
from src.pipelines.cohort_analysis import cohort_name, cohort_raw_dir
from src.settings import get_settings, get_unchecked_settings

# Problems are numbered by task type in every cohort run (see cohort_analysis.load_cohort).
CLASSIFY_TASK_TYPES = True
UP_TO_DATE, NOT_UP_TO_DATE, SERVER_ERROR = 0, 1, 2


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(__doc__.splitlines()[1:]),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="say whether each cohort's cache is up to date")
    check.add_argument("--cohorts", type=int, nargs="+", required=True, metavar="N")
    check.add_argument("--json", action="store_true", help="print the result as JSON")
    refresh = commands.add_parser("refresh", help="bring each cohort's cache up to date")
    refresh.add_argument("--cohorts", type=int, nargs="+", required=True, metavar="N")
    refresh.add_argument(
        "--keep-old", action="store_true", help="keep the older snapshots (no prune)"
    )
    prune = commands.add_parser("prune", help="remove cache entries no run would use")
    prune.add_argument(
        "--cohorts", type=int, nargs="+", metavar="N", help="only these cohorts (default all)"
    )
    prune.add_argument("--dry-run", action="store_true", help="list only; delete nothing")
    return parser.parse_args(argv)


class ListingProgress:
    """Keeps one line on stderr up to date while a folder is listed; :meth:`end` closes it."""

    def __init__(self, label: str):
        self.label, self.shown = label, False

    def __call__(self, n_files: int) -> None:
        print(f"\r  listing {self.label}: {n_files} files...", end="", file=sys.stderr, flush=True)
        self.shown = True

    def end(self) -> None:
        if self.shown:
            print(file=sys.stderr, flush=True)


def run_check(cohort_ids, as_json: bool) -> int:
    data_root = get_unchecked_settings().data_root
    checks = {}
    for cohort_id in cohort_ids:
        label = cohort_name(cohort_id)
        progress = ListingProgress(label)
        try:
            checks[label] = check_import_cache(
                cohort_raw_dir(data_root, cohort_id),
                classify_task_types=CLASSIFY_TASK_TYPES,
                progress=progress,
            )
        finally:
            progress.end()
    stale = [label for label, c in checks.items() if not c.up_to_date]
    if as_json:
        print(json.dumps([check_to_dict(c, label) for label, c in checks.items()], indent=1))
    else:
        for label, c in checks.items():
            print("\n".join(describe_check(c, label)))
        if stale:
            numbers = " ".join(label.removeprefix("cohort-").lstrip("0") for label in stale)
            print(
                "Update with: uv run python scripts/full_pipelines/import_cache.py refresh "
                f"--cohorts {numbers}"
            )
    return NOT_UP_TO_DATE if stale else UP_TO_DATE


def run_refresh(cohort_ids, keep_old: bool) -> int:
    data_root = get_settings().data_root
    for cohort_id in cohort_ids:
        label = cohort_name(cohort_id)
        progress = ListingProgress(label)
        try:
            result = refresh_import_cache(
                cohort_raw_dir(data_root, cohort_id),
                classify_task_types=CLASSIFY_TASK_TYPES,
                keep_old=keep_old,
                progress=progress,
            )
        finally:
            progress.end()
        state = "already up to date" if result.up_to_date else "refreshed"
        skipped = f", {len(result.skipped)} skipped as unreadable" if result.skipped else ""
        print(
            f"{label}: {state}; {len(result.parsed)} file(s) parsed, {len(result.reused)} "
            f"reused{skipped}, snapshot {result.snapshot.path.name}"
        )
    return UP_TO_DATE


def run_prune(cohort_ids, dry_run: bool) -> int:
    cohorts = None
    if cohort_ids:
        data_root = get_unchecked_settings().data_root
        cohorts = [str(cohort_raw_dir(data_root, c)) for c in cohort_ids]
    prune_import_cache(cohorts=cohorts, dry_run=dry_run)
    return UP_TO_DATE


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "check":
            return run_check(args.cohorts, args.json)
        if args.command == "refresh":
            return run_refresh(args.cohorts, args.keep_old)
        return run_prune(args.cohorts, args.dry_run)
    except (FileNotFoundError, ShareReadError, CacheLocationError) as err:
        # the server or a folder is missing, or the cache is misplaced: one clear line
        print(f"error: {err}", file=sys.stderr)
        return SERVER_ERROR
    except ValidationError as err:
        fields = ", ".join(f"BANDIT_{'_'.join(map(str, e['loc'])).upper()}" for e in err.errors())
        print(f"error: the settings are not usable ({fields}): see .env.example", file=sys.stderr)
        return SERVER_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
