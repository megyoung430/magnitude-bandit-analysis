"""Entry point for the drifting-task stay/switch psychometrics.

P(stay) and P(switch) against the value of the arm chosen on the previous trial and of its
best alternative (see :mod:`src.behavior_analysis.stay_switch_psychometrics` for the
definitions), for each value basis, tie mode and weighting, next to the value psychometrics.
"""

import warnings
from pathlib import Path

from src.behavior_analysis.get_choice_probs_by_value import TIE_MODES
from src.behavior_analysis.reversal_format import check_blocks_only_sessions
from src.behavior_analysis.stay_switch_psychometrics import (
    stay_heatmap,
    stay_switch_curves,
    stay_switch_pairs,
)
from src.behavior_visualization.plot_stay_switch import (
    plot_stay_heatmap,
    plot_stay_switch_curves,
    ties_in_name,
)
from src.pipelines.value_psychometrics import VALUE_BASES, check_bools, check_choices


def _write_condition(pairs, label: str, tie_mode: str, pooling_modes, out_dir: Path) -> dict:
    """Draw and save the figures of one value basis and tie mode; return its results."""
    stem = f"({label} Value) ({ties_in_name(tie_mode)})"
    heatmap = stay_heatmap(pairs.pairs)
    plot_stay_heatmap(heatmap, label, pairs.ties, tie_mode, out_dir / f"Stay Heatmap {stem}")
    curves_by_mode = {}
    for pool_trials in pooling_modes:
        curves = stay_switch_curves(pairs.pairs, pool_trials=pool_trials)
        name = f"Stay-Switch Curves {stem} (Pool Trials {pool_trials})"
        plot_stay_switch_curves(curves, label, pool_trials, pairs.ties, tie_mode, out_dir / name)
        curves_by_mode[pool_trials] = curves
    return {"pairs": pairs, "heatmap": heatmap, "curves": curves_by_mode}


def run_stay_switch_analysis(
    subjects_trials_by_problem: dict,
    out_dir: Path,
    *,
    value_bases: tuple[str, ...] = ("true", "believed"),
    pooling_modes: tuple[bool, ...] = (True, False),
    tie_modes: tuple[str, ...] = ("exclude", "split"),
) -> dict:
    """Stay/switch curves and P(stay) heatmaps by true and believed value.

    Works for one problem (``{problem: subjects_trials}``) or several; a mouse's trial pairs are
    pooled across problems. Mice are the replicates for every mean and standard error.

    Writes into *out_dir* (created), as PDF, PNG and SVG with a caption file, for each value
    basis (``<Value>`` is ``True`` or ``Believed``) and tie mode (``<Ties>`` is ``Excluded`` or
    ``Split``):

    - ``Stay Heatmap (<Value> Value) (Ties <Ties>)``: P(stay) over V(chosen) x V(alt);
    - ``Stay-Switch Curves (<Value> Value) (Ties <Ties>) (Pool Trials <True|False>)`` for each
      pooling mode: P(stay) and P(switch), then P(switch to the best alternative) and P(switch
      to the other arm), against V(chosen) - V(alt), V(chosen) and V(alt). ``True`` pools a
      mouse's pairs at an x, ``False`` averages its (V(chosen), V(alt)) cells equally.

    A value basis and tie mode without any trial pair draws nothing and warns.

    Args:
        subjects_trials_by_problem (dict): ``{problem: {subject: {session: session_data}}}``.
        out_dir (Path): Folder for the figures.
        value_bases (tuple[str, ...]): ``"true"`` and/or ``"believed"``. Keyword-only.
        pooling_modes (tuple[bool, ...]): ``pool_trials`` values for the curves. Keyword-only.
        tie_modes (tuple[str, ...]): ``"exclude"`` and/or ``"split"``. Keyword-only.

    Returns:
        dict: ``{value_basis: {tie_mode: {"pairs": StaySwitchPairs, "heatmap": StayHeatmap,
        "curves": {pool_trials: {outcome: {axis: Curve}}}}}}``; a condition without pairs has
        only ``"pairs"``.

    Raises:
        ValueError: If an option has an unknown entry, a session is counter format (the
            drifting task has only block changes), or there are no mice.
    """
    check_blocks_only_sessions(subjects_trials_by_problem, "run_stay_switch_analysis")
    check_choices("value_bases", value_bases, VALUE_BASES)
    check_bools("pooling_modes", pooling_modes)
    check_choices("tie_modes", tie_modes, TIE_MODES)
    if not any(subjects_trials_by_problem.values()):
        raise ValueError("no subjects in subjects_trials_by_problem")
    out_dir = Path(out_dir)
    results = {}
    for basis in value_bases:
        label = basis.capitalize()
        results[basis] = {}
        for tie_mode in tie_modes:
            pairs = stay_switch_pairs(
                subjects_trials_by_problem, use_believed=basis == "believed", tie_mode=tie_mode
            )
            if pairs.pairs.empty:
                warnings.warn(
                    f"stay/switch ({label} value, ties {tie_mode}): no trial pairs, no figures",
                    stacklevel=2,
                )
                results[basis][tie_mode] = {"pairs": pairs}
                continue
            results[basis][tie_mode] = _write_condition(
                pairs, label, tie_mode, pooling_modes, out_dir
            )
    return results
