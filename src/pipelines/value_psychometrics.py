"""Entry point for the drifting-task value heatmaps and psychometric curves."""

from pathlib import Path

from src.behavior_analysis.get_choice_probs_by_value import TIE_MODES, count_choice_probs
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
    unknown = [v for v in values if not isinstance(v, bool)]
    if unknown:
        raise ValueError(f"{name} must only contain True and/or False, got {unknown}")


def _write_condition(counts: dict, label: str, tie_mode: str, pooling_modes, out_dir: Path) -> dict:
    """Draw and save the figures of one value basis and tie mode; return its tables."""
    ties = tie_label(tie_mode)
    tables = heatmap_tables(counts)
    stem = f"({label} Value) (Ties {ties})"
    plot_overall_heatmaps(tables, label, tie_mode, out_dir / f"Heatmaps {stem} Overall")
    plot_per_mouse_heatmaps(tables, label, tie_mode, out_dir / f"Heatmaps {stem} Per Mouse")
    curves_by_mode = {}
    for pool_trials in pooling_modes:
        curves = psychometric_curves(counts, pool_trials=pool_trials)
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
) -> dict:
    """Heatmaps and psychometric curves of P(best) / P(second) by true and believed value.

    Works for one problem (``{problem: subjects_trials}``) or several. Trials are counted
    within each problem, so believed values start empty at every problem boundary and carry
    over that problem's sessions in session order; each mouse's counts are then pooled across
    problems. A believed trial counts only once all three arms have been sampled in its
    problem. Mice are the replicates for every mean and standard error. A warning is logged
    for each mouse with trials outside the value grid (best 2-4, second 1-3; ties at 4/4 and
    1/1 are outside it).

    Every figure is made once per tie mode: ``"exclude"`` leaves out every trial with a tie
    among the best three values (best/second, second/third or all three), ``"split"`` divides
    the tied rank positions evenly among the tied arms. Each figure prints the number of tied
    trials per category.

    Writes, as PNG and PDF, for each value basis (``<Value>`` is ``True`` or ``Believed``) and
    tie mode (``<Ties>`` is ``Excluded`` or ``Split``):

    - ``Heatmaps (<Value> Value) (Ties <Ties>) Overall`` and ``... Per Mouse`` (cell
      probabilities do not depend on the pooling mode, so there is one of each);
    - ``Psychometric Curves (<Value> Value) (Ties <Ties>) (Pool Trials <True|False>)`` for
      each pooling mode: ``True`` pools a mouse's trials across the cells sharing an x value,
      ``False`` averages its cell probabilities equally.

    Args:
        subjects_trials_by_problem (dict): ``{problem: {subject: {session: session_data}}}``.
        out_dir (Path): Folder for the figures; created if missing.
        value_bases (tuple[str, ...]): ``"true"`` and/or ``"believed"``.
        pooling_modes (tuple[bool, ...]): ``pool_trials`` values for the psychometric curves.
        tie_modes (tuple[str, ...]): ``"exclude"`` and/or ``"split"``.

    Returns:
        dict: ``{value_basis: {tie_mode: {"counts": {subject: MouseCounts}, "heatmaps":
        HeatmapTables, "psychometric": {pool_trials: {"best"|"second": {"best_val"|"second_val"
        |"diff": Curve}}}}}}``. ``counts[subject].out_of_grid``, ``.dropped`` and ``.ties`` hold
        the uncounted and the tied trials.

    Raises:
        ValueError: If *value_bases*, *pooling_modes* or *tie_modes* has an unknown entry, or
            no mouse has any session.
    """
    _check_choices("value_bases", value_bases, _VALUE_BASES)
    _check_bools("pooling_modes", pooling_modes)
    _check_choices("tie_modes", tie_modes, TIE_MODES)
    out_dir = Path(out_dir)
    results = {}
    for basis in value_bases:
        label = basis.capitalize()
        results[basis] = {}
        for tie_mode in tie_modes:
            counts = count_choice_probs(
                subjects_trials_by_problem, use_believed=basis == "believed", tie_mode=tie_mode
            )
            if not counts:
                raise ValueError("no subjects in subjects_trials_by_problem")
            results[basis][tie_mode] = _write_condition(
                counts, label, tie_mode, pooling_modes, out_dir
            )
    return results
