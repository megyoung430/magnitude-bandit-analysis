"""Pool the drifting problems of several cohorts and run the all-drifting-problems analyses.

    uv run python scripts/full_pipelines/run_all_drifting_problems.py --cohorts 1 3 4
    uv run python scripts/full_pipelines/run_all_drifting_problems.py --cohorts 1 3 4 --no-cache
    uv run python scripts/full_pipelines/run_all_drifting_problems.py --cohorts 1 3 4 \\
        --exclude-mice MY_175_N MY_66_LL

Raw data is read from <BANDIT_DATA_ROOT>/3x3_field_magnitude_bandit/rawdata/cohort-0N. Figures go
to results/figures/open-field/all-cohorts/{choice-stats,value-psychometrics,reversal-aligned-arms,
value-change,port-preference}/ (the folders `run_cohorts.py` writes to a cohort's
drifting/all-drifting-problems/), with cohorts.txt naming the cohorts pooled; tables go to
results/tables/open-field/all-cohorts/. There are no per-mouse figures and no legends naming
mice; those are in the cohort folders. See src/pipelines/all_cohorts_drifting.py.

--exclude-mice also writes the same analyses without the named mice (of any of the cohorts) into
all-cohorts-without-<mice>/ next to all-cohorts/, every figure's caption starting "without <mice>".

--from-cache loads each cohort from its newest cached import without reading the server (for
when it is unmounted or unreachable); sessions added since then are missing:

    uv run python scripts/full_pipelines/run_all_drifting_problems.py --cohorts 1 3 4 --from-cache
"""

import argparse

from src.behavior_analysis.exclude_mice import ExcludedMiceError, parse_mouse_id
from src.pipelines.all_cohorts_drifting import run_all_cohorts_drifting
from src.pipelines.cohort_analysis import DEFAULT_RESULTS_ROOT, DRIFTING_SELECTION

RESULTS_ROOT = DEFAULT_RESULTS_ROOT
# Which drifting analyses run; copy with model_copy(update={"<switch>": False}) to turn one off.
DRIFTING = DRIFTING_SELECTION


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cohorts", type=int, nargs="+", required=True, metavar="N")
    cache = parser.add_mutually_exclusive_group()
    cache.add_argument(
        "--no-cache", action="store_true", help="re-import the raw files instead of the cache"
    )
    cache.add_argument(
        "--from-cache",
        action="store_true",
        help=(
            "use the newest cached import of each cohort without reading the server; sessions "
            "added since then are missing"
        ),
    )
    parser.add_argument(
        "--exclude-mice",
        nargs="+",
        type=parse_mouse_id,
        metavar="MOUSE",
        help="also write the pool without these mice into all-cohorts-without-<mice>/",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    try:
        run_all_cohorts_drifting(
            args.cohorts,
            results_root=RESULTS_ROOT,
            drifting_selection=DRIFTING,
            use_cache=not args.no_cache,
            from_cache=args.from_cache,
            exclude_mice=args.exclude_mice,
        )
    except ExcludedMiceError as err:
        # a wrong mouse name is a usage error: one line, no traceback
        raise SystemExit(f"error: {err}") from None


if __name__ == "__main__":
    main()
