"""Run the whole open-field analysis suite for cohorts 1-4 with one command.

Edit the settings below, or override some of them on the command line:

    uv run python scripts/full_pipelines/run_cohorts.py
    uv run python scripts/full_pipelines/run_cohorts.py --cohorts 1 2 3 4
    uv run python scripts/full_pipelines/run_cohorts.py --cohorts 3 --continue-on-error

Redo only what recent sessions changed (the rest of results/ is left as it is):

    uv run python scripts/full_pipelines/run_cohorts.py --cohorts 4 --last-sessions 4
    uv run python scripts/full_pipelines/run_cohorts.py --cohorts 3 4 --since 2026-10-16
    uv run python scripts/full_pipelines/run_cohorts.py --cohorts 4 --last-sessions 4 --dry-run

--last-sessions N takes each mouse's N most recent sessions, --since YYYY-MM-DD the sessions
dated on or after that day (use one of them). The problems containing those sessions get their
figures regenerated, plus the pooled all-drifting-problems analyses if one of them is drifting.
The review tables and trial-block summaries are always rewritten, the stacked diagnostic tables
keep the rows of the other problems, and warnings_log.csv covers only this run. --dry-run prints
the plan (problems, pooled scopes, sessions) and writes no results or tables (the import cache
may be filled). A cohort with no matching session is skipped; if no cohort matches the run stops
with an error.

--last-sessions picks each mouse's last N sessions however old they are, so a mouse that stopped
weeks ago has its final problem regenerated every time; use --since for "what is new since I
last ran".

Redo only the pooled all-drifting-problems analyses, without regenerating any problem folder, and
add a second pooled set without some mice:

    uv run python scripts/full_pipelines/run_cohorts.py --cohorts 3 --pooled-only \\
        --exclude-mice MY_175_N MY_66_LL
    uv run python scripts/full_pipelines/run_cohorts.py --cohorts 3 --pooled-only \\
        --exclude-mice MY_175_N MY_66_LL --dry-run

--pooled-only writes drifting/all-drifting-problems/ (with sessions-per-problem/) and the review
tables, and leaves the problem folders, trial-block-summaries/ and the stacked diagnostic tables
as they are. --exclude-mice also writes drifting/all-drifting-problems-without-<mice>/ with the
same subfolders, every figure marked "without <mice>", and its tables in
tables/open-field/<cohort>/all-drifting-problems-without-<mice>/. It works with a full run and
with --last-sessions/--since too (the extra set is regenerated whenever the pooled set is). Every
cohort in --cohorts must have all the named mice among its drifting mice, so name the cohort
that holds them. Names are checked against the cohort before anything is written.

Without the server (unmounted or unreachable), add --from-cache: each cohort is loaded from its
newest cached import, preferring one made with the current import code, and the raw files are
never read. Sessions added since that import are missing, and import-code changes since then are
not applied; a StaleCacheWarning says so and is kept in warnings_log.csv:

    uv run python scripts/full_pipelines/run_cohorts.py --cohorts 3 --pooled-only --from-cache

Raw data is read from <BANDIT_DATA_ROOT>/3x3_field_magnitude_bandit/rawdata/cohort-0N. Figures
go to results/figures/open-field/cohort-0N/{blocked,drifting}/problem-KK/ and tables to
results/tables/open-field/cohort-0N/ (see src/pipelines/cohort_analysis.py). Every problem also
gets a long-pokes/ folder; drifting problems get value-psychometrics/, reversal-aligned-arms/
and value-change/, which are repeated over all drifting problems in all-drifting-problems/.
Switch an analysis off in the selections below (e.g. run_value_change_analysis=False).
"""

import argparse

from src.behavior_analysis.exclude_mice import ExcludedMiceError, parse_mouse_id
from src.pipelines.cohort_analysis import (
    BLOCKED_SELECTION,
    DEFAULT_RESULTS_ROOT,
    DRIFTING_SELECTION,
    run_cohorts,
)
from src.pipelines.open_field import ChoiceProbabilitySettings

COHORT_IDS = [1, 2, 3, 4]
# Stop at the first cohort that fails (False), or run the others and report all failures (True).
CONTINUE_ON_ERROR = False
RESULTS_ROOT = DEFAULT_RESULTS_ROOT
# Trials per bin of the trial-based reversal-rate plots.
REVERSAL_TRIAL_WINDOW = 100

# Which analyses run per task type; copy one with model_copy(update={"<switch>": False}) to turn
# a switch off. The drifting-only value analyses raise if switched on in the blocked selection.
BLOCKED = BLOCKED_SELECTION
DRIFTING = DRIFTING_SELECTION

# Choice-probability sub-analyses (used for blocked problems only)
CHOICE = ChoiceProbabilitySettings(
    run_all=True,
    skip=True,
    moving_avg=True,
    remove_bad=True,
    split_by_best_change=False,
    split_by_best_change_and_half=False,
    split_by_first_two=True,
    split_by_half=True,
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cohorts", type=int, nargs="+", default=COHORT_IDS, metavar="N")
    parser.add_argument("--continue-on-error", action="store_true", default=CONTINUE_ON_ERROR)
    recent = parser.add_mutually_exclusive_group()
    recent.add_argument(
        "--last-sessions", type=int, metavar="N", help="only each mouse's N most recent sessions"
    )
    recent.add_argument(
        "--since", metavar="YYYY-MM-DD", help="only sessions dated on or after this day"
    )
    parser.add_argument(
        "--pooled-only",
        action="store_true",
        help=(
            "run only the pooled all-drifting-problems analyses and the review tables, no "
            "problem folders (not with --last-sessions or --since)"
        ),
    )
    parser.add_argument(
        "--exclude-mice",
        nargs="+",
        type=parse_mouse_id,
        metavar="MOUSE",
        help=(
            "also write the pooled drifting set without these mice, e.g. MY_175_N MY_66_LL, into "
            "all-drifting-problems-without-<mice>/"
        ),
    )
    parser.add_argument(
        "--from-cache",
        action="store_true",
        help=(
            "use the newest cached import of each cohort without reading the server; sessions "
            "added since then are missing"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "print what --last-sessions, --since or --pooled-only would regenerate and write no "
            "results or tables (the import cache may be filled)"
        ),
    )
    args = parser.parse_args(argv)
    selecting = args.last_sessions is not None or args.since is not None
    if args.pooled_only and selecting:
        parser.error(
            "--pooled-only runs no problems, so it cannot be used with --last-sessions or --since"
        )
    if args.dry_run and not (selecting or args.pooled_only):
        parser.error("--dry-run needs --last-sessions, --since or --pooled-only")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    try:
        run_cohorts(
            args.cohorts,
            continue_on_error=args.continue_on_error,
            last_sessions=args.last_sessions,
            since=args.since,
            dry_run=args.dry_run,
            pooled_only=args.pooled_only,
            exclude_mice=args.exclude_mice,
            from_cache=args.from_cache,
            results_root=RESULTS_ROOT,
            blocked_selection=BLOCKED,
            drifting_selection=DRIFTING,
            choice=CHOICE,
            reversal_trial_window=REVERSAL_TRIAL_WINDOW,
        )
    except ExcludedMiceError as err:
        # a wrong mouse name is a usage error: one line, no traceback
        raise SystemExit(f"error: {err}") from None


if __name__ == "__main__":
    main()
