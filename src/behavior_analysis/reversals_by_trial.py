"""Reversal counts indexed by completed trials, including terminal events."""
import numpy as np
from src.behavior_analysis.get_variables_across_sessions import get_vars_across_all_sessions
from src.behavior_analysis.get_total_reversals import get_total_reversals


def reversal_trial_series(subjects_trials):
    """Return per-mouse cumulative counts at boundaries 0 through n trials.

    Counter index i describes the state before trial i (after i trials).
    The extra endpoint includes reversals triggered by the final trial.
    """
    merged, _ = get_vars_across_all_sessions(subjects_trials)
    result = {}
    for mouse, data in merged.items():
        n = len(data['trial'])
        if not n:
            continue
        totals = get_total_reversals(subjects_trials[mouse])
        series = {}
        for kind in ('good', 'bad'):
            key = kind + '_reversals'
            if key in data:
                series[kind] = np.r_[np.asarray(data[key], dtype=float), totals.get(key, data[key][-1])]
        if not series:
            blocks = np.asarray(data['blocks'], dtype=float)
            series['total'] = np.r_[blocks - blocks[0], totals['total_reversals']]
        for values in series.values():
            if len(values) != n+1 or np.any(np.diff(values) < 0):
                raise ValueError(f'{mouse}: reversal counters are not aligned and cumulative')
        result[mouse] = series
    if not result:
        raise ValueError('No trial observations available')
    return result


def trailing_reversal_rate(cumulative, window=100):
    """Return trailing-window reversals per 100 trials, at trials 1 through n.

    Early windows use their actual observed exposure. Events before the first
    observed trial are included in the first interval. No padding or future
    trials are used.
    """
    if not isinstance(window, int) or isinstance(window, bool) or window < 1:
        raise ValueError('Trial window must be a positive integer')
    cumulative = np.asarray(cumulative, dtype=float)
    ends = np.arange(1,len(cumulative))
    starts = np.maximum(0,ends-window)
    prior = np.where(starts == 0,0,cumulative[starts])
    return 100 * (cumulative[ends] - prior) / (ends-starts)
