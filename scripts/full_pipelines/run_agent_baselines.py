"""Fit magnitude and binary agents to each mouse, simulate them, and run the choice regression.

    uv run python scripts/full_pipelines/run_agent_baselines.py --cohorts 1 3
    uv run python scripts/full_pipelines/run_agent_baselines.py --cohorts 1 3 --jobs 4 --starts 5

Raw data is read from <BANDIT_DATA_ROOT>/3x3_field_magnitude_bandit/rawdata/cohort-0N, through
the import cache. Figures go to results/figures/open-field/agent-baselines/cohorts-<ids>/:
summary.png (regression weights of mice and agents, and the per-mouse likelihood comparison)
and one folder per agent with the same pooled drifting figures as all-cohorts/. Tables go to
results/tables/open-field/agent-baselines/cohorts-<ids>/. The agent fits are cached in
.cache/agent_baselines, so a rerun only refits what changed. See src/pipelines/agent_baselines.py.
"""

import argparse

from src.pipelines.agent_baselines import run_agent_baselines
from src.pipelines.cohort_analysis import DEFAULT_RESULTS_ROOT, DRIFTING_SELECTION


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cohorts", type=int, nargs="+", required=True, metavar="N")
    parser.add_argument("--starts", type=int, default=5, help="optimiser starts per fit")
    parser.add_argument("--jobs", type=int, default=1, help="parallel workers for the fits")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-cache", action="store_true", help="re-import the raw files and refit every agent"
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    result = run_agent_baselines(
        args.cohorts,
        results_root=DEFAULT_RESULTS_ROOT,
        selection=DRIFTING_SELECTION,
        n_starts=args.starts,
        seed=args.seed,
        n_jobs=args.jobs,
        use_cache=not args.no_cache,
    )
    print(result.summary.to_string(index=False))


if __name__ == "__main__":
    main()
