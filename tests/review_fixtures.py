"""Synthetic cohorts and raw frames for the review-table tests.

``session`` builds the part of an imported session dict the review tables read: the raw
recordings (``df``) with a ``run_start`` and ``run_end`` row each, the per-trial reward
magnitudes and block ids, the problem's towers, and the first and last trial of the session
(``session_edges``) that ``reconcile_reversals`` stores. Nothing here reads a file.
"""

import json

import pandas as pd

from src.behavior_analysis.review_context import ProblemIndex
from src.behavior_import.task_types import (
    number_within_task_type,
    problem_map_table,
    regroup_by_task_type,
)

BLOCKED_NAME = "Meg\\3x3_field_blocked_value_bandit"
DRIFTING_NAME = "Meg\\3x3_field_drifting_value_bandit"
TOWERS = ["A1", "B1", "C1"]

SETTINGS = {
    "ITI_dur": 1000.0,
    "ses_length": 60,
    "poke_hold_dur": 0,
    "p_ud": 0.1,
    "found_ports": True,
    "rew_mags": [1, 2, 3, 4],
}


def recording(run_start=None, run_end=None, path="M1-2026-01-01-100000.tsv"):
    """A raw recording frame: optional ``run_start`` and ``run_end`` variable rows."""
    rows = [(0.0, "info", "task_name", DRIFTING_NAME)]
    if run_start is not None:
        rows.append((0.1, "variable", "run_start", json.dumps(run_start)))
    if run_end is not None:
        rows.append((9.0, "variable", "run_end", json.dumps(run_end)))
    frame = pd.DataFrame(rows, columns=["time", "type", "subtype", "content"])
    frame.attrs["source_path"] = path
    return frame


def side(magnitudes, **totals):
    """One side of a boundary: the magnitudes and the session totals (``block=3`` etc.)."""
    return {"magnitudes": magnitudes, "totals": totals, "counters": {}}


def edge_sides(first_magnitudes, last_magnitudes, *, first_totals=None, last_totals=None):
    """``session_edges`` of a session: its first and last trial."""
    return {
        "first": side(first_magnitudes, **(first_totals or {})),
        "last": side(last_magnitudes, **(last_totals or {})),
    }


def session(
    recordings=None,
    *,
    magnitudes=None,
    blocks=None,
    edges=None,
    task_name=DRIFTING_NAME,
    towers=TOWERS,
    initiation_tower="B2",
):
    """An imported-session-like dict.

    Args:
        recordings: Raw frames; default one recording with equal run_start and run_end.
        magnitudes: ``{tower: [magnitude per trial]}``; default 4 trials of (4, 2, 0).
        blocks: Block id per trial; default all 1.
        edges: ``session_edges``; default the first and last trial of *magnitudes*, with no
            totals. Set ``session["session_edges"] = None`` afterwards for a session without trials.
    """
    if recordings is None:
        recordings = [recording(dict(SETTINGS), dict(SETTINGS))]
    magnitudes = magnitudes or {"A1": [4] * 4, "B1": [2] * 4, "C1": [0] * 4}
    n = len(next(iter(magnitudes.values())))
    if edges is None:
        edges = edge_sides(
            {t: v[0] for t, v in magnitudes.items()}, {t: v[-1] for t, v in magnitudes.items()}
        )
    return {
        "df": recordings[0] if len(recordings) == 1 else recordings,
        "task_name": task_name,
        "choice_towers": set(towers),
        "initiation_tower": initiation_tower,
        "trial": list(range(1, n + 1)),
        "blocks": blocks or [1] * n,
        "trials_in_block": list(range(1, n + 1)),
        "reward_magnitudes_by_tower": magnitudes,
        "choices_by_tower": {},
        "choices_by_rank": {},
        "session_edges": edges,
    }


def cohort(problems):
    """``(problems, ProblemIndex, grouped, problem_map)`` of a synthetic cohort.

    Args:
        problems: ``{overall problem: {mouse: {session key: session dict}}}``.
    """
    numbering = number_within_task_type(problems)
    problem_map = problem_map_table(problems, numbering)
    grouped = regroup_by_task_type(problems, numbering)
    return problems, ProblemIndex.from_cohort(problems, problem_map), grouped, problem_map
