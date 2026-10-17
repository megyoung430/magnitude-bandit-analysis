"""One figure per drifting session to check the best-decrease rank-swap windows by eye.

Usage: ``python scripts/individual_analyses/plot_value_change_diagnostic.py --cohorts 1 3``
Writes ``results/figures/value-change-diagnostic/cohort-XX/problem-KK/<mouse>/<session>.png``
and a per-cohort ``index.txt``. Pass ``--no-cache`` to re-import the raw files.
"""

import argparse
from pathlib import Path

from src.pipelines.cohort_analysis import DEFAULT_RESULTS_ROOT
from src.pipelines.value_change_diagnostic import run_cohort_diagnostic


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cohorts", type=int, nargs="+", required=True, metavar="N")
    parser.add_argument("--no-cache", action="store_true", help="re-import instead of the cache")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    args = parser.parse_args(argv)
    for cohort_id in args.cohorts:
        print(f"cohort {cohort_id}")
        folder = run_cohort_diagnostic(
            cohort_id, results_root=args.results_root, use_cache=not args.no_cache
        )
        print(f"  wrote {folder}")


if __name__ == "__main__":
    main()
