"""Which drifting problems enter an analysis: a cutoff on the problem number per cohort.

The trial table of :func:`src.behavior_analysis.data.drifting_trial_table.drifting_trial_table`
holds
drifting problems only, numbered ``k = 1, 2, ...`` within the drifting problems of a cohort (the
blocked problems that precede them are left out and not counted, see
:func:`src.pipelines.common.all_cohorts_drifting.drifting_problems`). A problem key is ``(cohort,
k)``.

Cohort 1's behavior degrades after drifting problem 7: two mice end there and the three that
continue return from a surgery break. The cutoff is one mapping, so changing it is one edit.
Everything here is pure: trial tables in, trial tables out.
"""

from collections.abc import Mapping

import pandas as pd


def _cohort_of(problem) -> int:
    if not (isinstance(problem, tuple) and len(problem) == 2):
        raise ValueError(f"problem key {problem!r} is not (cohort, problem number); no cohort")
    return problem[0]


def _check_cutoffs(cohorts: set, max_problem_by_cohort: Mapping) -> None:
    missing = sorted(cohorts - set(max_problem_by_cohort))
    if missing:
        raise ValueError(
            f"cohort {missing[0]} is in the trials but has no entry in max_problem_by_cohort "
            f"(use None to keep all of its drifting problems); entries: "
            f"{dict(max_problem_by_cohort)}"
        )
    for cohort, cutoff in max_problem_by_cohort.items():
        if cutoff is not None and not (isinstance(cutoff, int) and cutoff >= 1):
            raise ValueError(
                f"cutoff of cohort {cohort} must be a positive integer or None, got {cutoff!r}"
            )


def select_problems(trials: pd.DataFrame, max_problem_by_cohort: Mapping) -> pd.DataFrame:
    """The rows of problems up to each cohort's cutoff.

    Args:
        trials (pd.DataFrame): A drifting trial table whose ``problem`` is ``(cohort, k)``.
        max_problem_by_cohort (Mapping): ``{cohort: last drifting problem to keep}``; ``None``
            keeps every drifting problem of the cohort. Every cohort of *trials* needs an entry,
            so a cohort added to a run is a decision, not a silent default.

    Returns:
        pd.DataFrame: The kept rows, index reset.

    Raises:
        ValueError: If a problem key has no cohort, a cohort has no entry, a cutoff is not a
            positive integer or None, or no problem of a mouse survives.
    """
    cohorts = trials["problem"].map(_cohort_of)
    _check_cutoffs(set(cohorts), max_problem_by_cohort)
    last = cohorts.map(max_problem_by_cohort)
    number = trials["problem"].map(lambda p: p[1])
    keep = last.isna() | (number <= last)
    kept = trials[keep].reset_index(drop=True)
    lost = sorted(set(trials["mouse"]) - set(kept["mouse"]))
    if lost:
        raise ValueError(
            f"the cutoffs {dict(max_problem_by_cohort)} leave no problem for mouse {lost[0]}"
        )
    return kept


def dropped_table(trials: pd.DataFrame, kept: pd.DataFrame) -> pd.DataFrame:
    """Per mouse that lost problems: how many problems and trials the selection dropped.

    Args:
        trials (pd.DataFrame): The table before the selection.
        kept (pd.DataFrame): :func:`select_problems` of it.

    Returns:
        pd.DataFrame: ``mouse, n_problems_dropped, n_trials_dropped, problems_dropped`` (the
        keys, in order of appearance); empty if nothing was dropped.
    """
    kept_keys = set(zip(kept["mouse"], kept["problem"], strict=True))
    gone = trials[
        [key not in kept_keys for key in zip(trials["mouse"], trials["problem"], strict=True)]
    ]
    rows = [
        dict(
            mouse=mouse,
            n_problems_dropped=rows["problem"].nunique(),
            n_trials_dropped=len(rows),
            problems_dropped=", ".join(str(p) for p in dict.fromkeys(rows["problem"])),
        )
        for mouse, rows in gone.groupby("mouse", sort=True)
    ]
    return pd.DataFrame(
        rows, columns=["mouse", "n_problems_dropped", "n_trials_dropped", "problems_dropped"]
    )
