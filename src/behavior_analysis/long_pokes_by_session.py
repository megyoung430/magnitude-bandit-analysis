"""Per-session mean and SE of non-choice long pokes across mice (no plotting)."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from src.behavior_analysis.stats import mean_se_n
from src.behavior_import.sort_by_session import sort_sessions_chronologically

AVERAGE_KEY = "average_across_subjects"


class PokesBySession(NamedTuple):
    """Long-poke summary on a common, chronologically sorted session axis.

    ``mean`` and ``se`` are across mice, one entry per session (NaN when no mouse
    has a finite value). ``by_mouse`` maps each mouse to its values on the same
    axis (NaN where it has none).
    """

    mice: list[str]
    sessions: list[str]
    mean: np.ndarray
    se: np.ndarray
    by_mouse: dict[str, np.ndarray]


def _is_finite_number(value) -> bool:
    return value is not None and bool(np.isfinite(value))


def _mouse_ids(avg_num_pokes_across_subjects: dict) -> list[str]:
    mice = sorted(m for m in avg_num_pokes_across_subjects.keys() if m != AVERAGE_KEY)
    if len(mice) == 0:
        raise ValueError("avg_num_pokes_across_subjects has no mice.")
    return mice


def _sorted_sessions(avg_num_pokes_across_subjects: dict, mice: list[str]) -> list[str]:
    """Union of session keys over mice, ordered by their numeric ``ses-XX`` index."""
    sessions: set = set()
    for m in mice:
        sessions.update(avg_num_pokes_across_subjects.get(m, {}).keys())
    if len(sessions) == 0:
        raise ValueError("No sessions found in avg_num_pokes_across_subjects.")
    return sorted(sessions, key=sort_sessions_chronologically)


def _mean_and_se(values: np.ndarray) -> tuple[float, float]:
    """Mean and standard error; the SE is NaN with fewer than two values."""
    mean, se, _ = mean_se_n(values)
    return mean, se


def _finite_values(
    avg_num_pokes_across_subjects: dict, mice: list[str], session: str
) -> np.ndarray:
    vals = []
    for m in mice:
        if session in avg_num_pokes_across_subjects.get(m, {}):
            v = avg_num_pokes_across_subjects[m][session]
            if _is_finite_number(v):
                vals.append(float(v))
    return np.asarray(vals, dtype=float)


def _series_for_mouse(per_session: dict, sessions: list[str]) -> np.ndarray:
    y = np.full(len(sessions), np.nan, dtype=float)
    for i, s in enumerate(sessions):
        if s in per_session and _is_finite_number(per_session[s]):
            y[i] = float(per_session[s])
    return y


def summarize_pokes_by_session(avg_num_pokes_across_subjects: dict) -> PokesBySession:
    """Mean and SE across mice for every session, plus each mouse's own series.

    Args:
        avg_num_pokes_across_subjects: ``{mouse: {session_key: mean_pokes}}``, as
            returned by :func:`src.behavior_analysis.get_num_long_pokes.get_num_long_pokes`.
            The ``"average_across_subjects"`` entry is ignored. ``None`` and
            non-finite values are treated as missing.

    Raises:
        ValueError: If there are no mice or no sessions.
    """
    mice = _mouse_ids(avg_num_pokes_across_subjects)
    sessions = _sorted_sessions(avg_num_pokes_across_subjects, mice)
    stats = [_mean_and_se(_finite_values(avg_num_pokes_across_subjects, mice, s)) for s in sessions]
    by_mouse = {
        m: _series_for_mouse(avg_num_pokes_across_subjects.get(m, {}), sessions) for m in mice
    }
    return PokesBySession(
        mice=mice,
        sessions=sessions,
        mean=np.asarray([mean for mean, _ in stats], dtype=float),
        se=np.asarray([se for _, se in stats], dtype=float),
        by_mouse=by_mouse,
    )
