"""Run the whole open-field analysis suite for cohorts 1-4 with one command.

Edit the settings below, or override some of them on the command line:

    uv run python scripts/full_pipelines/run_cohorts.py
    uv run python scripts/full_pipelines/run_cohorts.py --cohorts 1 2 3 4
    uv run python scripts/full_pipelines/run_cohorts.py --cohorts 3 --continue-on-error

Raw data is read from <BANDIT_DATA_ROOT>/3x3_field_magnitude_bandit/rawdata/cohort-0N. Figures
go to results/figures/open-field/cohort-0N/{blocked,drifting}/problem-KK/ and tables to
results/tables/open-field/cohort-0N/ (see src/pipelines/cohort_analysis.py).
"""

import argparse

from src.pipelines.cohort_analysis import DEFAULT_RESULTS_ROOT, run_cohorts
from src.pipelines.open_field import ChoiceProbabilitySettings

COHORT_IDS = [1, 2, 3, 4]
# Stop at the first cohort that fails (False), or run the others and report all failures (True).
CONTINUE_ON_ERROR = False
RESULTS_ROOT = DEFAULT_RESULTS_ROOT
# Trials per bin of the trial-based reversal-rate plots.
REVERSAL_TRIAL_WINDOW = 100

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
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    run_cohorts(
        args.cohorts,
        continue_on_error=args.continue_on_error,
        results_root=RESULTS_ROOT,
        choice=CHOICE,
        reversal_trial_window=REVERSAL_TRIAL_WINDOW,
    )


if __name__ == "__main__":
    main()
