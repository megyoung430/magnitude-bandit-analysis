"""Utilities for working with per-problem trial and session data.

Functions here operate on ``subjects_trials_by_problem`` structures and
provide helpers for extracting, sorting, and summarising data at the
problem level.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Key-parsing helpers
# ---------------------------------------------------------------------------

_PROBLEM_RE = re.compile(r"(\d+)")
_SESSION_RE = re.compile(r"ses-(\d+)")


def problem_to_int(k: Any) -> int | None:
    """Convert a problem key (int, string, etc.) to a plain Python int.

    Args:
        k: A problem key such as ``3``, ``"problem-03"``, or ``"3"``.

    Returns:
        The integer problem number, or ``None`` if it cannot be determined.
    """
    if isinstance(k, (int, np.integer)):
        return int(k)
    m = _PROBLEM_RE.search(str(k))
    return int(m.group(1)) if m else None


def session_sort_key(session_key: str) -> tuple:
    """Return a sort key for a session string of the form ``ses-NN_date-YYYYMMDD``.

    Args:
        session_key: A session identifier string.

    Returns:
        A tuple ``(session_number, datetime, session_key)`` suitable for
        sorting sessions chronologically.
    """
    m_ses = _SESSION_RE.search(str(session_key))
    m_date = re.search(r"date-(\d{8})", str(session_key))
    ses_num = int(m_ses.group(1)) if m_ses else float("inf")
    dt = datetime.strptime(m_date.group(1), "%Y%m%d") if m_date else datetime.max
    return (ses_num, dt, session_key)


# ---------------------------------------------------------------------------
# Trial / session structure helpers
# ---------------------------------------------------------------------------

def flatten_trials(subject_trials: Any) -> list:
    """Flatten trial data into a single list regardless of its structure.

    Args:
        subject_trials: Either a ``list[trial_dict]`` or a
            ``dict[session_key -> list[trial_dict]]``.

    Returns:
        A flat list of trial dicts.

    Raises:
        TypeError: If ``subject_trials`` is neither a list nor a dict.
    """
    if subject_trials is None:
        return []
    if isinstance(subject_trials, list):
        return subject_trials
    if isinstance(subject_trials, dict):
        out: list = []
        for sk in sorted(subject_trials.keys(), key=session_sort_key):
            out.extend(subject_trials[sk])
        return out
    raise TypeError(f"Unexpected subject_trials type: {type(subject_trials)}")


def get_session_map(subject_trials: Any) -> dict:
    """Return a ``{session_key: list[trial_dict]}`` mapping.

    If the input is already a flat list, it is wrapped under a single
    ``"__all__"`` key.

    Args:
        subject_trials: Either a list of trial dicts or a
            ``dict[session_key -> list[trial_dict]]``.

    Returns:
        A dict mapping session keys to lists of trial dicts.
    """
    if subject_trials is None:
        return {}
    if isinstance(subject_trials, dict):
        return subject_trials
    if isinstance(subject_trials, list):
        return {"__all__": subject_trials}
    raise TypeError(f"Unexpected subject_trials type: {type(subject_trials)}")


def n_trials_in_session(sess: dict) -> int:
    """Count the number of trials in a single session data dict.

    Inspects ``"trial"``, ``"choice"``, and ``"trial_info"`` keys in priority
    order.

    Args:
        sess: A session data dict produced by ``extract_trials()``.

    Returns:
        The number of trials, or 0 if no trial information is found.
    """
    if not isinstance(sess, dict):
        return 0

    trial_seq = sess.get("trial")
    if isinstance(trial_seq, (list, tuple)):
        return len(trial_seq)

    choice_seq = sess.get("choice")
    if isinstance(choice_seq, (list, tuple)):
        return len(choice_seq)

    trial_info = sess.get("trial_info")
    if isinstance(trial_info, list):
        total = 0
        for seg in trial_info:
            if isinstance(seg, list):
                total += len(seg)
        return total

    return 0


def total_trials_in_problem(mouse_sessions: dict) -> int:
    """Sum the total number of trials across all sessions for one mouse+problem.

    Args:
        mouse_sessions: ``{session_id: session_data}`` for a single mouse.

    Returns:
        Total trial count across all sessions.
    """
    if not isinstance(mouse_sessions, dict):
        return 0
    return sum(n_trials_in_session(sess) for sess in mouse_sessions.values())


def flatten_trials_by_session(trials_by_session: Any) -> list:
    """Flatten session-keyed trial data into a single list.

    Unlike :func:`flatten_trials`, this function always expects a dict or
    falls back gracefully to a list.

    Args:
        trials_by_session: ``dict[session_id -> list[trial_dict]]`` or a
            flat list of trial dicts.

    Returns:
        A flat list of trial dicts.
    """
    if not isinstance(trials_by_session, dict):
        return list(trials_by_session)
    out: list = []
    for _, sess_trials in trials_by_session.items():
        if sess_trials:
            out.extend(sess_trials)
    return out


# ---------------------------------------------------------------------------
# Reversal / block helpers
# ---------------------------------------------------------------------------

def mean_block_length_from_indices(idxs: list[int], T: int) -> float:
    """Compute mean block length from reversal indices.

    Block boundaries are defined as ``[0] + sorted(idxs)``.  Block lengths
    are the differences between consecutive boundaries.

    Args:
        idxs: Reversal indices in ``[0, T)``.
        T: Total number of trials.

    Returns:
        Mean block length in trials, or ``nan`` if there are no boundaries.
    """
    if T <= 0:
        return float("nan")

    clean: list[int] = []
    for i in idxs:
        try:
            ii = int(i)
        except Exception:
            continue
        if 0 <= ii < T:
            clean.append(ii)

    clean = sorted(set(clean))
    boundaries = [0] + clean
    if len(boundaries) <= 1:
        return float("nan")

    lens = np.diff(boundaries)
    if lens.size == 0:
        return float("nan")

    return float(np.mean(lens))


def reversals_from_blocks_counter(session_data: dict) -> float:
    """Estimate the number of reversals in a session from the blocks counter.

    Reversals = ``max(blocks_sequence) - 1``.

    Args:
        session_data: A session dict with a ``"blocks"`` key containing a
            per-trial count of blocks experienced.

    Returns:
        Estimated reversal count, or ``nan`` if the blocks field is absent or
        invalid.
    """
    blocks_seq = session_data.get("blocks")
    if blocks_seq is None:
        return float("nan")

    vals = []
    for x in blocks_seq:
        if x is None:
            continue
        try:
            vals.append(float(x))
        except Exception:
            pass

    if not vals:
        return float("nan")

    max_blocks = np.nanmax(np.asarray(vals, dtype=float))
    if not np.isfinite(max_blocks):
        return float("nan")

    return float(max(0.0, max_blocks - 1.0))


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def summarize_by_problem(per_mouse_metric: dict) -> dict:
    """Compute mean ± SE across mice for each problem.

    Args:
        per_mouse_metric: ``{mouse: {problem_int: value}}``

    Returns:
        A dict with keys ``"blocks"``, ``"means"``, ``"ses"``, ``"ns"``,
        ``"per_mouse_blocklens"``, and ``"mice"``.
    """
    mice = sorted(per_mouse_metric.keys())
    problems = sorted({p for m in mice for p in per_mouse_metric[m].keys()})
    means, ses, ns = [], [], []

    for p in problems:
        vals = np.array(
            [per_mouse_metric[m][p] for m in mice if p in per_mouse_metric[m]],
            dtype=float,
        )
        n = int(vals.size)
        ns.append(n)
        if n == 0:
            means.append(float("nan"))
            ses.append(float("nan"))
        elif n == 1:
            means.append(float(vals.mean()))
            ses.append(0.0)
        else:
            means.append(float(vals.mean()))
            ses.append(float(vals.std(ddof=1) / np.sqrt(n)))

    return {
        "blocks": problems,
        "means": means,
        "ses": ses,
        "ns": ns,
        "per_mouse_blocklens": per_mouse_metric,
        "mice": mice,
    }


def summarize_by_problem_safe(per_mouse_metric: dict) -> dict:
    """Compute mean ± SE by problem, handling missing and non-finite values.

    A safer alternative to :func:`summarize_by_problem` that skips ``None``
    and non-finite values and returns per-problem dicts rather than parallel
    lists.

    Args:
        per_mouse_metric: ``{mouse: {problem_int: value}}``

    Returns:
        ``{problem: {"mean": float, "se": float, "n": int, "values": list}}``
    """
    by_p: dict[Any, list[float]] = {}
    for _mouse, pmap in per_mouse_metric.items():
        if not isinstance(pmap, dict):
            continue
        for p, v in pmap.items():
            if v is None or not np.isfinite(float(v)):
                continue
            by_p.setdefault(p, []).append(float(v))

    out: dict = {}
    for p, vals_list in by_p.items():
        vals = np.asarray(vals_list, dtype=float)
        n = int(np.isfinite(vals).sum())
        if n == 0:
            continue
        m = float(np.nanmean(vals))
        se = 0.0 if n == 1 else float(np.nanstd(vals, ddof=1) / np.sqrt(n))
        out[p] = {"mean": m, "se": se, "n": n, "values": vals.tolist()}
    return out


def exclude_last_problem_per_mouse(per_mouse_metric: dict) -> dict:
    """Remove each mouse's last (highest-numbered) problem in place.

    Args:
        per_mouse_metric: ``{mouse: {problem_int: value}}`` — modified in place.

    Returns:
        The same dict with last-problem entries removed.
    """
    for _mouse, pmap in per_mouse_metric.items():
        if isinstance(pmap, dict) and pmap:
            last_p = max(pmap.keys())
            pmap.pop(last_p, None)
    return per_mouse_metric
