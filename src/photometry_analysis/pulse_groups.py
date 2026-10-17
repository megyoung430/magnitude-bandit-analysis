"""Split outcome-aligned trial windows by the number of reward pulses delivered.

Every function is pure. A trial's pulse count is the ``num_rew_pulses`` of its outcome; 0 means
unrewarded. The windows have already been normalised (per session or per trial) before they are
split, so a category is a plain selection of rows and no normalisation depends on which categories
are shown.
"""

from collections.abc import Iterable, Mapping

import numpy as np
from scipy.stats import spearmanr


def _as_counts(pulses: Iterable[int]) -> np.ndarray:
    """*pulses* as an integer array; raises if any value is negative or not a whole number."""
    values = np.asarray(list(pulses) if not isinstance(pulses, np.ndarray) else pulses)
    if values.size and (np.any(values < 0) or np.any(values != np.round(values))):
        raise ValueError(
            f"pulse counts must be non-negative whole numbers, got {np.unique(values)}."
        )
    return values.astype(int)


def pulse_categories(pulses: Iterable[int]) -> list[int]:
    """The distinct pulse counts observed, ascending (0 = unrewarded).

    Raises:
        ValueError: If a count is negative or not a whole number.
    """
    return [int(p) for p in np.unique(_as_counts(pulses))]


def split_by_pulses(
    views: Mapping[str, np.ndarray], pulses: np.ndarray, categories: list[int]
) -> dict[int, dict[str, np.ndarray]]:
    """Select the trials of each pulse category from every view.

    Args:
        views (Mapping[str, np.ndarray]): View name to ``(n_trials, n_time)`` windows.
        pulses (np.ndarray): Pulse count of each trial (one per row of every view).
        categories (list[int]): The categories to return; a category without trials gets an empty
            ``(0, n_time)`` array so callers can report it rather than miss it.

    Returns:
        dict[int, dict[str, np.ndarray]]: ``{category: {view name: windows}}``.

    Raises:
        ValueError: If a view does not have one row per count, or a count is not in *categories*.
    """
    counts = _as_counts(pulses)
    for name, trials in views.items():
        if len(trials) != len(counts):
            raise ValueError(
                f"need one pulse count per trial: view {name!r} has {len(trials)} trials, "
                f"{len(counts)} counts."
            )
    unlisted = sorted(set(counts.tolist()) - set(categories))
    if unlisted:
        raise ValueError(f"pulse counts {unlisted} are not in the categories {categories}.")
    return {c: {name: trials[counts == c] for name, trials in views.items()} for c in categories}


def categories_with_min_trials(trial_counts: Mapping[int, int], min_trials: int) -> list[int]:
    """Categories with at least *min_trials* trials, in the order of *trial_counts*.

    Raises:
        ValueError: If *min_trials* is below 1 (a category without trials has nothing to draw).
    """
    if min_trials < 1:
        raise ValueError(f"min_trials must be at least 1, got {min_trials}.")
    return [c for c, n in trial_counts.items() if n >= min_trials]


def pulse_response_correlation(responses: np.ndarray, pulses: np.ndarray) -> dict:
    """Spearman correlation of trial response with pulse count across the rewarded trials.

    Unrewarded trials (0 pulses) are left out: they are aligned to a different event (the no-reward
    cue), so they would mix an event difference into the reward-size slope. Trials of one session
    are not independent, so ``n_trials`` is not an effective sample size; no p-value is returned.

    Args:
        responses (np.ndarray): One response per trial.
        pulses (np.ndarray): Pulse count of each trial.

    Returns:
        dict: ``rho``, ``n_trials`` (rewarded trials used) and ``n_categories`` (distinct nonzero
        counts among them). ``rho`` is NaN with fewer than two categories, where it is undefined.

    Raises:
        ValueError: If the two arrays differ in length.
    """
    counts = _as_counts(pulses)
    if len(responses) != len(counts):
        raise ValueError(
            f"need one pulse count per trial: {len(responses)} responses, {len(counts)} counts."
        )
    rewarded = counts > 0
    n_categories = len(np.unique(counts[rewarded]))
    rho = (
        float(spearmanr(counts[rewarded], responses[rewarded]).statistic)
        if n_categories >= 2
        else np.nan
    )
    return {"rho": rho, "n_trials": int(rewarded.sum()), "n_categories": n_categories}
