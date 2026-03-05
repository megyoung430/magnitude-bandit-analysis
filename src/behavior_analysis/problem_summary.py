"""Summary-statistic functions computed across problems.

Functions here take ``subjects_trials_by_problem`` or similar structures and
compute per-mouse, per-problem summary metrics that can be plotted with
functions in :mod:`src.behavior_visualization.plot_problem_summary`.
"""
from __future__ import annotations

import re
from collections import defaultdict

import numpy as np


def compute_problem_length_summary(data: dict) -> dict:
    """Summarise the number of sessions to complete each problem per mouse.

    Args:
        data: ``{mouse: [n_sessions_problem1, n_sessions_problem2, ...]}``
            where mice may have different numbers of completed problems.

    Returns:
        A summary dict with keys ``"blocks"``, ``"means"``, ``"ses"``,
        ``"ns"``, ``"per_mouse_blocklens"``, and ``"mice"``.
    """
    mice = list(data.keys())
    max_problems = max((len(v) for v in data.values()), default=0)
    blocks = list(range(1, max_problems + 1))

    per_mouse_blocklens: dict = {}
    for m, vals in data.items():
        per_mouse_blocklens[m] = {i + 1: float(v) for i, v in enumerate(vals)}

    means, ses, ns = [], [], []
    for b in blocks:
        vals_b = np.asarray(
            [per_mouse_blocklens[m][b] for m in mice if b in per_mouse_blocklens[m]],
            dtype=float,
        )
        n = int(vals_b.size)
        ns.append(n)
        if n == 0:
            means.append(float("nan"))
            ses.append(float("nan"))
        elif n == 1:
            means.append(float(vals_b.mean()))
            ses.append(0.0)
        else:
            means.append(float(vals_b.mean()))
            ses.append(float(vals_b.std(ddof=1) / np.sqrt(n)))

    return {
        "blocks": blocks,
        "means": means,
        "ses": ses,
        "ns": ns,
        "per_mouse_blocklens": per_mouse_blocklens,
        "mice": mice,
    }


def compute_problem_blocklen_summary(
    problem_block_stats: dict,
    drop_last_problem_per_mouse: bool = False,
) -> dict:
    """Summarise mean block length per problem averaged across mice.

    Args:
        problem_block_stats: ``{problem: {"block_stats": {...}}}`` as returned
            by ``get_block_lengths()``.
        drop_last_problem_per_mouse: If ``True``, each mouse's last (highest)
            problem is excluded before averaging (default: ``False``).

    Returns:
        A summary dict with keys ``"blocks"``, ``"means"``, ``"ses"``,
        ``"ns"``, ``"per_mouse_blocklens"``, and ``"mice"``.
    """
    problems = sorted(problem_block_stats.keys())
    mice = sorted({
        m
        for p in problems
        for m in problem_block_stats[p]["block_stats"]["per_mouse_blocklens"].keys()
    })

    per_mouse_blocklens: dict = {m: {} for m in mice}
    for p in problems:
        per_mouse = problem_block_stats[p]["block_stats"]["per_mouse_blocklens"]
        for m, block_dict in per_mouse.items():
            vals = np.asarray(list(block_dict.values()), dtype=float)
            if vals.size > 0:
                per_mouse_blocklens[m][p] = float(np.mean(vals))

    if drop_last_problem_per_mouse:
        for m in mice:
            if per_mouse_blocklens[m]:
                last_p = max(per_mouse_blocklens[m].keys())
                per_mouse_blocklens[m].pop(last_p, None)
        problems = [p for p in problems if any(p in per_mouse_blocklens[m] for m in mice)]

    means, ses, ns = [], [], []
    for p in problems:
        vals_p = np.asarray(
            [per_mouse_blocklens[m][p] for m in mice if p in per_mouse_blocklens[m]],
            dtype=float,
        )
        n = int(vals_p.size)
        ns.append(n)
        if n == 0:
            means.append(float("nan"))
            ses.append(float("nan"))
        elif n == 1:
            means.append(float(vals_p.mean()))
            ses.append(0.0)
        else:
            means.append(float(vals_p.mean()))
            ses.append(float(vals_p.std(ddof=1) / np.sqrt(n)))

    mice_kept = [m for m in mice if len(per_mouse_blocklens[m]) > 0]
    per_mouse_blocklens = {m: per_mouse_blocklens[m] for m in mice_kept}

    return {
        "blocks": problems,
        "means": means,
        "ses": ses,
        "ns": ns,
        "per_mouse_blocklens": per_mouse_blocklens,
        "mice": mice_kept,
    }


def compute_avg_good_reversals_per_session(subjects_trials: dict) -> dict:
    """Compute each subject's average number of good reversals per session.

    Args:
        subjects_trials: ``{subject: {session_key: trials}}``

    Returns:
        ``{subject: avg_good_reversals_per_session}``
    """
    from src.behavior_analysis.get_total_reversals import get_total_reversals

    _ses_re = re.compile(r"ses-(\d+)")

    def _session_int(session_key: str) -> int:
        """Return the integer session number from a ``ses-N`` session key."""
        m = _ses_re.search(session_key)
        if m is None:
            raise ValueError(f"Session key lacks 'ses-<n>': {session_key}")
        return int(m.group(1))

    good_per_session_by_subj: dict = defaultdict(list)

    for subj in sorted(subjects_trials.keys()):
        for sess_key, trials in subjects_trials[subj].items():
            stats = get_total_reversals({sess_key: trials})
            _ = _session_int(sess_key)  # validate format
            good_per_session_by_subj[subj].append(
                float(stats.get("good_reversals", 0))
            )

    per_mouse_avg: dict = {}
    for subj, vals in good_per_session_by_subj.items():
        arr = np.asarray(vals, dtype=float)
        per_mouse_avg[subj] = float(arr.mean()) if arr.size else float("nan")

    return per_mouse_avg


def compute_problem_goodrev_summary(problem_to_per_mouse_avg: dict) -> dict:
    """Summarise average good reversals per session across mice and problems.

    Args:
        problem_to_per_mouse_avg: ``{problem: {mouse: avg_good_reversals}}``

    Returns:
        A summary dict with keys ``"blocks"``, ``"means"``, ``"ses"``,
        ``"ns"``, ``"per_mouse_blocklens"``, and ``"mice"``.
    """
    problems = sorted(problem_to_per_mouse_avg.keys())
    mice = sorted({m for p in problems for m in problem_to_per_mouse_avg[p].keys()})

    per_mouse_blocklens: dict = {m: {} for m in mice}
    for p in problems:
        for m, v in problem_to_per_mouse_avg[p].items():
            if np.isfinite(float(v)):
                per_mouse_blocklens[m][p] = float(v)

    means, ses, ns = [], [], []
    for p in problems:
        vals = np.asarray(
            [per_mouse_blocklens[m][p] for m in mice if p in per_mouse_blocklens[m]],
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
        "per_mouse_blocklens": per_mouse_blocklens,
        "mice": mice,
    }


def drop_last_problem_per_mouse(problem_to_per_mouse_avg: dict) -> dict:
    """Remove each mouse's last completed problem from a per-problem dict.

    Args:
        problem_to_per_mouse_avg: ``{problem: {mouse: value}}``

    Returns:
        A new dict with the highest-numbered problem per mouse removed.
    """
    last_problem: dict = {}
    for p, d in problem_to_per_mouse_avg.items():
        for m in d:
            last_problem[m] = max(p, last_problem.get(m, p))

    filtered: dict = {}
    for p, d in problem_to_per_mouse_avg.items():
        new_d = {m: v for m, v in d.items() if last_problem.get(m) != p}
        if new_d:
            filtered[p] = new_d

    return filtered
