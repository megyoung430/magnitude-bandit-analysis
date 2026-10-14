"""Raw block lengths per problem and mouse, for the block-length histograms.

Inputs are the ``{problem: {"block_stats": {...}}}`` dicts built from
:func:`src.behavior_analysis.get_task_statistics.get_block_lengths`. The plots
in :mod:`src.behavior_visualization.plot_problem_summary` draw what these
functions return.
"""

import numpy as np


def collect_blocklens(
    problem_block_stats: dict,
) -> tuple[list, list, dict]:
    """Collect raw block lengths per problem per mouse.

    Args:
        problem_block_stats: ``{problem: {"block_stats": {...}}}``

    Returns:
        A three-tuple ``(problems, mice, blocklens_by_problem_mouse)`` where
        *blocklens_by_problem_mouse* is
        ``{problem: {mouse: np.ndarray of block lengths}}``.
    """
    problems = sorted(problem_block_stats.keys())
    mice = sorted(
        {
            m
            for p in problems
            for m in problem_block_stats[p]["block_stats"]["per_mouse_blocklens"].keys()
        }
    )

    blocklens_by_problem_mouse: dict = {p: {} for p in problems}
    for p in problems:
        per_mouse = problem_block_stats[p]["block_stats"]["per_mouse_blocklens"]
        for m in mice:
            block_dict = per_mouse.get(m, {})
            vals = np.asarray(list(block_dict.values()), dtype=float)
            blocklens_by_problem_mouse[p][m] = vals[np.isfinite(vals)]

    return problems, mice, blocklens_by_problem_mouse


def drop_last_problem_per_mouse_raw(
    problems: list,
    mice: list,
    blocklens_by_problem_mouse: dict,
) -> tuple[list, list, dict]:
    """Drop each mouse's last problem from raw block-length data.

    A mouse's last problem is the highest-numbered one it has data for.
    Problems and mice left with no data are removed from the returned lists.

    Args:
        problems: Sorted list of problem numbers.
        mice: Sorted list of mouse identifiers.
        blocklens_by_problem_mouse: ``{problem: {mouse: np.ndarray}}``.

    Returns:
        Updated ``(problems_new, mice_new, data_new)`` with last problems
        removed.
    """
    data = {p: {m: blocklens_by_problem_mouse[p][m].copy() for m in mice} for p in problems}

    for m in mice:
        ps_with_data = [p for p in problems if data[p].get(m, np.array([])).size > 0]
        if ps_with_data:
            data[max(ps_with_data)][m] = np.array([], dtype=float)

    problems_new = [p for p in problems if any(data[p][m].size > 0 for m in mice)]
    mice_new = [m for m in mice if any(data[p][m].size > 0 for p in problems_new)]
    data_new = {p: {m: data[p][m] for m in mice_new} for p in problems_new}
    return problems_new, mice_new, data_new


def pooled_blocklens(data: dict, problems: list, mice: list) -> np.ndarray:
    """Block lengths of the given *problems* and *mice*, concatenated (empty if none).

    Args:
        data: ``{problem: {mouse: np.ndarray}}`` from :func:`collect_blocklens`.
        problems, mice: Which entries of *data* to pool; pass one of each to
            get one mouse's lengths in one problem.
    """
    parts = [data[p][m] for p in problems for m in mice if data[p][m].size > 0]
    return np.concatenate(parts, axis=0) if parts else np.array([])
