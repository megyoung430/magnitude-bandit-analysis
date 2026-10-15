"""Per-session mean and SE of non-choice long pokes across mice (no plotting)."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from src.behavior_analysis.stats import mean_se_n
from src.behavior_import.session_order import ordered_session_keys
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


def _trial_values(subject: str, session_key: str, session: dict) -> np.ndarray:
    """The session's per-trial ``num_long_pokes`` as floats, trials without a value dropped."""
    if "num_long_pokes" not in session:
        raise ValueError(f"{subject} {session_key}: session has no 'num_long_pokes'")
    values = [v for v in session["num_long_pokes"] if v is not None]
    return np.asarray(values, dtype=float)


def session_mean_long_pokes(subjects_sessions: dict) -> dict[str, dict[str, float]]:
    """Mean non-choice long pokes per trial of every session, sessions in session order.

    Unlike :func:`src.behavior_analysis.get_num_long_pokes.get_num_long_pokes`, a session
    without ``num_long_pokes`` raises rather than being skipped.

    Args:
        subjects_sessions: ``{mouse: {session_key: session_dict}}`` of one problem, each
            session holding the importer's merged per-trial ``"num_long_pokes"`` list.

    Returns:
        ``{mouse: {session_key: mean}}``; NaN for a session with no trial values.

    Raises:
        ValueError: If a session lacks ``num_long_pokes`` or a mouse's sessions cannot be
            ordered.
    """
    return {
        mouse: {
            key: float(np.mean(v))
            if (v := _trial_values(mouse, key, sessions[key])).size
            else np.nan
            for key in ordered_session_keys(mouse, sessions)
        }
        for mouse, sessions in subjects_sessions.items()
    }


def to_relative_sessions(per_mouse: dict, *, start_at: int = 1) -> dict[str, dict]:
    """Re-key each mouse's sessions as ``rel_ses-01``, ``rel_ses-02``, ... in session order.

    Mice that started the problem on different absolute sessions then line up on the same
    relative session. The ``"average_across_subjects"`` entry is dropped.

    Args:
        per_mouse: ``{mouse: {session_key: value}}``.
        start_at: Number of each mouse's first session.

    Raises:
        ValueError: If a mouse's session keys cannot be ordered (see ``ordered_session_keys``).
    """
    return {
        mouse: {
            f"rel_ses-{i:02d}": by_session[key]
            for i, key in enumerate(ordered_session_keys(mouse, by_session), start=start_at)
        }
        for mouse, by_session in per_mouse.items()
        if mouse != AVERAGE_KEY
    }


def problem_mean_long_pokes(subjects_sessions: dict) -> dict[str, float]:
    """Each mouse's mean non-choice long pokes per trial over all trials of the problem.

    Trials are pooled across the mouse's sessions, so long sessions weigh more than short ones.
    Arguments and errors as for :func:`session_mean_long_pokes`; NaN for a mouse without trial
    values.
    """
    return {
        mouse: _pooled_mean([_trial_values(mouse, key, s) for key, s in sessions.items()])
        for mouse, sessions in subjects_sessions.items()
    }


def _pooled_mean(arrays: list[np.ndarray]) -> float:
    pooled = np.concatenate(arrays) if arrays else np.empty(0)
    return float(pooled.mean()) if pooled.size else float("nan")
