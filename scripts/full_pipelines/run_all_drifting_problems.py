"""Pool the drifting problems of several cohorts and run the all-drifting-problems analyses.

    uv run python scripts/full_pipelines/run_all_drifting_problems.py --cohorts 1 3 4
    uv run python scripts/full_pipelines/run_all_drifting_problems.py --cohorts 1 3 4 --no-cache

Raw data is read from <BANDIT_DATA_ROOT>/3x3_field_magnitude_bandit/rawdata/cohort-0N. Figures go
to results/figures/open-field/all-cohorts/{choice-stats,value-psychometrics,reversal-aligned-arms,
value-change,port-preference}/ (the folders `run_cohorts.py` writes to a cohort's
drifting/all-drifting-problems/), with cohorts.txt naming the cohorts pooled; tables go to
results/tables/open-field/all-cohorts/. There are no per-mouse figures and no legends naming
mice; those are in the cohort folders. See src/pipelines/all_cohorts_drifting.py.
"""

import argparse

from src.pipelines.all_cohorts_drifting import run_all_cohorts_drifting
from src.pipelines.cohort_analysis import DEFAULT_RESULTS_ROOT, DRIFTING_SELECTION

RESULTS_ROOT = DEFAULT_RESULTS_ROOT
# Which drifting analyses run; copy with model_copy(update={"<switch>": False}) to turn one off.
DRIFTING = DRIFTING_SELECTION


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cohorts", type=int, nargs="+", required=True, metavar="N")
    parser.add_argument(
        "--no-cache", action="store_true", help="re-import the raw files instead of the cache"
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    run_all_cohorts_drifting(
        args.cohorts,
        results_root=RESULTS_ROOT,
        drifting_selection=DRIFTING,
        use_cache=not args.no_cache,
    )


if __name__ == "__main__":
    main()
