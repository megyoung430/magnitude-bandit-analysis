"""Per-session reversal counts and the cumulative / smoothed curves built from them.

Counts come from :func:`get_total_reversals`, one session at a time. Sessions
from performance-dependent paradigms carry ``good`` and ``bad`` counts; the
others only a ``total`` derived from the blocks counter.
"""

import re

import numpy as np
import pandas as pd

from src.behavior_analysis.get_total_reversals import get_total_reversals

_SESSION_RE = re.compile(r"ses-(\d+)")
_COUNT_COLUMNS = {"good_reversals": "good", "bad_reversals": "bad", "total_reversals": "total"}


def session_number(session_key: str) -> int:
    """Return ``N`` from a session key such as ``ses-N_date-YYYYMMDD``."""
    match = _SESSION_RE.search(session_key)
    if match is None:
        raise ValueError(f"Session key does not contain 'ses-<n>': {session_key}")
    return int(match.group(1))


def _count_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows).rename(columns=_COUNT_COLUMNS)
    for col in ("good", "bad"):
        if col not in df:
            df[col] = np.nan
    return df


def reversal_counts_by_session(subjects_trials: dict) -> pd.DataFrame:
    """Reversal counts for every session, one row per (subject, session).

    Args:
        subjects_trials: ``{subject: {session_key: session_dict}}``.

    Returns:
        DataFrame sorted by subject then session number, with columns
        ``subject``, ``session``, ``session_label`` (``ses-NN``), ``good``,
        ``bad`` (NaN for sessions without good/bad counters) and ``total``, plus
        ``suspect_good_reversals`` and ``uncheckable_good_reversals`` (see
        :func:`get_total_reversals`; NaN without good/bad counters).
    """
    rows = [
        {"subject": subject, "session": session_number(key), **get_total_reversals({key: trials})}
        for subject in sorted(subjects_trials)
        for key, trials in subjects_trials[subject].items()
    ]
    if not rows:
        raise ValueError("No sessions found in subjects_trials")
    counts = _count_frame(rows)
    counts["session_label"] = counts["session"].map(lambda n: f"ses-{n:02d}")
    return counts.sort_values(["subject", "session"], kind="stable").reset_index(drop=True)


def reversal_totals_by_subject(subjects_trials: dict) -> pd.DataFrame:
    """Total reversal counts per subject (index), columns ``good``, ``bad``, ``total``."""
    rows = [
        {"subject": s, **get_total_reversals(subjects_trials[s])} for s in sorted(subjects_trials)
    ]
    if not rows:
        raise ValueError("subjects_trials is empty")
    return _count_frame(rows).set_index("subject")


def reversal_kinds(counts: pd.DataFrame) -> tuple[str, ...]:
    """Which count columns to analyse: ``("good", "bad")`` or ``("total",)``.

    Raises if some rows have good/bad counts and others do not: averaging
    across the two paradigms would mix incomparable quantities.
    """
    has_good_bad = counts[["good", "bad"]].notna().all(axis=1)
    if has_good_bad.all():
        return ("good", "bad")
    if not has_good_bad.any():
        return ("total",)
    raise ValueError(
        "Sessions mix performance-dependent (good/bad) and blocks-only reversal counts; "
        "analyse the two paradigms separately."
    )


def cumulative_by_session(counts: pd.DataFrame, kinds: tuple[str, ...]) -> pd.DataFrame:
    """Running total of each count column within each subject, in session order."""
    out = counts.copy()
    out[list(kinds)] = counts.groupby("subject", sort=False)[list(kinds)].cumsum()
    return out


def centered_moving_average(x, window: int) -> np.ndarray:
    """Centered, NaN-ignoring moving average that shrinks at the edges.

    The kernel spans ``window // 2`` points either side, so an even *window*
    widens to the next odd width and stays centred.
    """
    x = np.asarray(x, dtype=float)
    if window <= 1:
        return x.copy()
    width = 2 * (window // 2) + 1
    return pd.Series(x).rolling(width, center=True, min_periods=1).mean().to_numpy()


def moving_average_by_session(
    counts: pd.DataFrame, kinds: tuple[str, ...], window: int
) -> pd.DataFrame:
    """:func:`centered_moving_average` of each count column within each subject."""
    out = counts.copy()
    for kind in kinds:
        out[kind] = counts.groupby("subject", sort=False)[kind].transform(
            lambda s: centered_moving_average(s.to_numpy(), window)
        )
    return out


def curves_by_subject(df: pd.DataFrame, kind: str) -> dict[str, np.ndarray]:
    """``{subject: values}`` for one column, subjects sorted, rows in session order."""
    return {
        subject: g[kind].to_numpy(dtype=float) for subject, g in df.groupby("subject", sort=True)
    }


def _pad_with_last_finite(curve: np.ndarray, length: int) -> np.ndarray:
    padded = pd.Series(np.full(length, np.nan))
    padded[: len(curve)] = curve
    return padded.ffill().bfill().to_numpy()


def mean_across_subjects(curves: dict[str, np.ndarray]) -> np.ndarray:
    """Mean curve across subjects, as long as the longest one.

    Shorter curves are padded with their last finite value: a mouse that has
    stopped training keeps the level it reached rather than dropping out of
    the mean.
    """
    length = max(len(c) for c in curves.values())
    stacked = np.vstack([_pad_with_last_finite(c, length) for c in curves.values()])
    return np.nanmean(stacked, axis=0)
