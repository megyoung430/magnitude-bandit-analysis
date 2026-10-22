"""Entry point for the drifting-task value heatmaps and psychometric curves.

Drifting problems only: the drifting task has block changes, not good or bad reversals.
"""

from pathlib import Path

import numpy as np

from src.behavior_analysis.get_choice_probs_by_value import TIE_MODES, count_choice_probs
from src.behavior_analysis.reversal_format import check_blocks_only_sessions
from src.behavior_analysis.value_psychometrics import heatmap_tables, psychometric_curves
from src.behavior_visualization.plot_value_psychometrics import (
    plot_overall_heatmaps,
    plot_per_mouse_heatmaps,
    plot_psychometric_curves,
    tie_label,
)

_VALUE_BASES = ("true", "believed")


def _check_choices(name: str, values, allowed: tuple) -> None:
    """Raise ValueError unless every entry of *values* is one of *allowed*."""
    unknown = [v for v in values if v not in allowed]
    if unknown:
        raise ValueError(f"{name} must only contain {list(allowed)}, got {unknown}")


def _check_bools(name: str, values) -> None:
    """Raise ValueError unless every entry of *values* is a real ``bool`` (not 1 or "yes")."""
    unknown = [v for v in values if not isinstance(v, (bool, np.bool_))]
    if unknown:
        raise ValueError(f"{name} must only contain True and/or False, got {unknown}")


def _write_condition(
    counts: dict,
    label: str,
    tie_mode: str,
    pooling_modes,
    out_dir: Path,
    *,
    per_mouse_heatmaps: bool,
) -> dict:
    """Draw and save the figures of one value basis and tie mode; return its tables."""
    ties = tie_label(tie_mode)
    tables = heatmap_tables(counts)
    stem = f"({label} Value) (Ties {ties})"
    plot_overall_heatmaps(tables, label, tie_mode, out_dir / f"Heatmaps {stem} Overall")
    if per_mouse_heatmaps:
        plot_per_mouse_heatmaps(tables, label, tie_mode, out_dir / f"Heatmaps {stem} Per Mouse")
    curves_by_mode = {}
    for pool_trials in pooling_modes:
        curves = psychometric_curves(counts, pool_trials=pool_trials, tie_mode=tie_mode)
        name = f"Psychometric Curves {stem} (Pool Trials {pool_trials})"
        plot_psychometric_curves(curves, label, pool_trials, tables.ties, tie_mode, out_dir / name)
        curves_by_mode[pool_trials] = curves
    return {"counts": counts, "heatmaps": tables, "psychometric": curves_by_mode}


def run_value_psychometric_analysis(
    subjects_trials_by_problem: dict,
    out_dir: Path,
    *,
    value_bases: tuple[str, ...] = ("true", "believed"),
    pooling_modes: tuple[bool, ...] = (True, False),
    tie_modes: tuple[str, ...] = ("exclude", "split"),
    per_mouse_heatmaps: bool = True,
) -> dict:
    """Heatmaps and psychometric curves of P(best) / P(second) by true and believed value.

    Works for one problem (``{problem: subjects_trials}``) or several. Trials are counted
    within each problem, so believed values start empty at every problem boundary and carry
    over that problem's sessions in session order; each mouse's counts are then pooled across
    problems. A believed trial counts only once all three arms have been sampled in its
    problem. Mice are the replicates for every mean and standard error. A warning is logged
    once per mouse and value basis for trials outside the value grid (best 2-4, second 1-3).

    Every figure is made once per tie mode: ``"exclude"`` leaves out every trial with a tie
    among the best three values (best/second, second/third or all three), ``"split"`` divides
    the tied rank positions evenly among the tied arms. Each figure prints the number of tied
    trials per category.

    Writes, as PDF, PNG and SVG with a caption file, for each value basis (``<Value>`` is
    ``True`` or ``Believed``) and tie mode (``<Ties>`` is ``Excluded`` or ``Split``):

    - ``Heatmaps (<Value> Value) (Ties <Ties>) Overall`` and ``... Per Mouse`` (cell
      probabilities do not depend on the pooling mode, so there is one of each; the per-mouse
      one only with *per_mouse_heatmaps*);
    - ``Psychometric Curves (<Value> Value) (Ties <Ties>) (Pool Trials <True|False>)`` for
      each pooling mode: ``True`` pools a mouse's trials across the cells sharing an x value,
      ``False`` averages its cell probabilities equally.

    Args:
        subjects_trials_by_problem (dict): ``{problem: {subject: {session: session_data}}}``.
        out_dir (Path): Folder for the figures; created if missing.
        value_bases (tuple[str, ...]): ``"true"`` and/or ``"believed"``.
        pooling_modes (tuple[bool, ...]): ``pool_trials`` values for the psychometric curves.
        tie_modes (tuple[str, ...]): ``"exclude"`` and/or ``"split"``.
        per_mouse_heatmaps (bool): Draw the ``... Per Mouse`` heatmaps. Switched off for pools
            of many mice, whose per-mouse figures belong to the cohort folders. Default
            ``True``. Keyword-only.

    Returns:
        dict: ``{value_basis: {tie_mode: {"counts": {subject: MouseCounts}, "heatmaps":
        HeatmapTables, "psychometric": {pool_trials: {"best"|"second": {"best_val"|"second_val"
        |"diff": Curve}}}}}}``. ``counts[subject].out_of_grid``, ``.dropped`` and ``.ties`` hold
        the uncounted and the tied trials.

    Raises:
        ValueError: If *value_bases*, *pooling_modes* or *tie_modes* has an unknown entry, if
            a session is counter format (the drifting task has only block changes), or no
            mouse has any session.
    """
    check_blocks_only_sessions(subjects_trials_by_problem, "run_value_psychometric_analysis")
    _check_choices("value_bases", value_bases, _VALUE_BASES)
    _check_bools("pooling_modes", pooling_modes)
    _check_choices("tie_modes", tie_modes, TIE_MODES)
    out_dir = Path(out_dir)
    # Out-of-grid counts differ by tie mode (excluded ties never reach the grid), so the
    # warning comes from one mode: split, whose out-of-grid set is the larger.
    warn_mode = "split" if "split" in tie_modes else tie_modes[0] if tie_modes else None
    results = {}
    for basis in value_bases:
        label = basis.capitalize()
        results[basis] = {}
        for tie_mode in tie_modes:
            counts = count_choice_probs(
                subjects_trials_by_problem,
                use_believed=basis == "believed",
                tie_mode=tie_mode,
                warn_out_of_grid=tie_mode == warn_mode,
            )
            if not counts:
                raise ValueError("no subjects in subjects_trials_by_problem")
            results[basis][tie_mode] = _write_condition(
                counts,
                label,
                tie_mode,
                pooling_modes,
                out_dir,
                per_mouse_heatmaps=per_mouse_heatmaps,
            )
    return results
