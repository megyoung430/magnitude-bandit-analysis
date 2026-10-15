"""Entry point for the drifting-task value heatmaps and psychometric curves."""

from pathlib import Path

from src.behavior_analysis.get_choice_probs_by_value import count_choice_probs
from src.behavior_analysis.value_psychometrics import heatmap_tables, psychometric_curves
from src.behavior_visualization.plot_value_psychometrics import (
    plot_overall_heatmaps,
    plot_per_mouse_heatmaps,
    plot_psychometric_curves,
)

_VALUE_BASES = ("true", "believed")


def _check_value_bases(value_bases) -> None:
    """Raise ValueError unless every entry is ``"true"`` or ``"believed"``."""
    unknown = [basis for basis in value_bases if basis not in _VALUE_BASES]
    if unknown:
        raise ValueError(f"value_bases must be 'true' and/or 'believed', got {unknown}")


def run_value_psychometric_analysis(
    subjects_trials_by_problem: dict,
    out_dir: Path,
    *,
    value_bases: tuple[str, ...] = ("true", "believed"),
    pooling_modes: tuple[bool, ...] = (True, False),
) -> dict:
    """Heatmaps and psychometric curves of P(best) / P(second) by true and believed value.

    Works for one problem (``{problem: subjects_trials}``) or several. Trials are counted
    within each problem, so believed values start empty at every problem boundary and carry
    over that problem's sessions in session order; each mouse's counts are then pooled across
    problems. Mice are the replicates for every mean and standard error. A warning is logged
    for each mouse with trials outside the value grid (best 2-4, second 1-3).

    Writes, as PNG and PDF, for each value basis (``<Value>`` is ``True`` or ``Believed``):

    - ``Heatmaps (<Value> Value) Overall`` and ``Heatmaps (<Value> Value) Per Mouse``
      (cell probabilities do not depend on the pooling mode, so there is one of each);
    - ``Psychometric Curves (<Value> Value) (Pool Trials <True|False>)`` for each pooling
      mode: ``True`` pools a mouse's trials across the cells sharing an x value, ``False``
      averages its cell probabilities equally.

    Args:
        subjects_trials_by_problem (dict): ``{problem: {subject: {session: session_data}}}``.
        out_dir (Path): Folder for the figures; created if missing.
        value_bases (tuple[str, ...]): ``"true"`` and/or ``"believed"``.
        pooling_modes (tuple[bool, ...]): ``pool_trials`` values for the psychometric curves.

    Returns:
        dict: ``{value_basis: {"counts": {subject: MouseCounts}, "heatmaps": HeatmapTables,
        "psychometric": {pool_trials: {"best"|"second": {"best_val"|"second_val"|"diff":
        Curve}}}}}``. ``counts[subject].out_of_grid`` and ``.dropped`` hold the uncounted trials.

    Raises:
        ValueError: If *value_bases* has an unknown entry or no mouse has any session.
    """
    _check_value_bases(value_bases)
    out_dir = Path(out_dir)
    results = {}
    for basis in value_bases:
        label = basis.capitalize()
        counts = count_choice_probs(subjects_trials_by_problem, use_believed=basis == "believed")
        if not counts:
            raise ValueError("no subjects in subjects_trials_by_problem")
        tables = heatmap_tables(counts)
        plot_overall_heatmaps(tables, label, out_dir / f"Heatmaps ({label} Value) Overall")
        plot_per_mouse_heatmaps(tables, label, out_dir / f"Heatmaps ({label} Value) Per Mouse")
        curves_by_mode = {}
        for pool_trials in pooling_modes:
            curves = psychometric_curves(counts, pool_trials=pool_trials)
            name = f"Psychometric Curves ({label} Value) (Pool Trials {pool_trials})"
            plot_psychometric_curves(curves, label, pool_trials, out_dir / name)
            curves_by_mode[pool_trials] = curves
        results[basis] = {"counts": counts, "heatmaps": tables, "psychometric": curves_by_mode}
    return results
