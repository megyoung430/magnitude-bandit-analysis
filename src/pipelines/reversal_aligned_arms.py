"""Entry point: choice probability and reward magnitude of the three arms around block changes.

For drifting problems only: the drifting task has block changes, not good or bad reversals.
Windows are computed per problem, then concatenated per subject for the pooled figures.
See :func:`run_reversal_aligned_arm_analysis`.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src.behavior_analysis.merge_across_problems import merge_windows_by_subject
from src.behavior_analysis.reversal_aligned_arms import (
    MOVING_AVG_MODES,
    WEIGHTINGS,
    ReversalAlignedArms,
    build_windows_with_magnitudes,
    check_moving_avg_window,
    check_windows_built_with,
    compute_reversal_aligned_arms,
)
from src.behavior_analysis.reversal_arm_ranks import (
    LABEL_SCHEMES,
    REVERSAL_TYPES,
    SCHEME_TYPES,
)
from src.behavior_analysis.reversal_format import (
    check_block_change_windows,
    check_blocks_only_sessions,
)
from src.behavior_visualization import plot_helpers
from src.behavior_visualization.plot_reversal_aligned_arms import (
    figure_stem,
    save_reversal_aligned_arms,
)
from src.pipelines.weighting_folders import WEIGHTING_FOLDERS

POOLED = "pooled"
_ANALYSIS = "run_reversal_aligned_arm_analysis"


def _problem_name(problem) -> str:
    is_int = isinstance(problem, (int, np.integer)) and not isinstance(problem, bool)
    return f"problem-{int(problem):02d}" if is_int else str(problem)


def _check_choices(label_schemes, reversal_types, moving_avg_mode, moving_avg_window) -> None:
    check_moving_avg_window(moving_avg_window)
    for name, chosen, allowed in (
        ("label scheme", label_schemes, LABEL_SCHEMES),
        ("reversal_type", reversal_types or (), REVERSAL_TYPES),
    ):
        unknown = [c for c in chosen if c not in allowed]
        if unknown:
            raise ValueError(f"unknown {name} {unknown}; choose from {allowed}")
    unused = [
        t for t in reversal_types or () if not any(t in SCHEME_TYPES[s] for s in label_schemes)
    ]
    if unused:
        raise ValueError(
            f"reversal_type {unused} belongs to none of the label schemes {list(label_schemes)}"
        )
    if moving_avg_mode not in MOVING_AVG_MODES:
        raise ValueError(
            f"moving_avg_mode must be one of {MOVING_AVG_MODES}, got {moving_avg_mode!r}"
        )


def _holds_windows(per_problem) -> bool | None:
    """True if the values are block change lists, False if session dicts, None if there are none."""
    kinds = {isinstance(value, list) for value in per_problem.values()}
    if len(kinds) > 1:
        raise ValueError("a problem mixes block change lists and session dicts")
    return kinds.pop() if kinds else None


def _windows_by_problem(by_problem: Mapping, *, pre: int, post: int) -> dict:
    """Enriched windows per problem from trials, or the windows as given.

    Raises:
        ValueError: If there are no problems, trials and windows are mixed, or given windows
            were not built by ``build_windows_with_magnitudes`` with this *pre* and *post*, or
            trials or windows come from counter-format sessions.
    """
    if not by_problem:
        raise ValueError("no problems given")
    kinds = {_holds_windows(per_problem) for per_problem in by_problem.values()} - {None}
    if len(kinds) > 1:
        raise ValueError("cannot mix trials and windows across problems")
    if kinds == {True}:
        for windows in by_problem.values():
            check_windows_built_with(windows, pre=pre, post=post)
            check_block_change_windows(windows, _ANALYSIS)
        return dict(by_problem)
    check_blocks_only_sessions(by_problem, _ANALYSIS)
    return {
        problem: build_windows_with_magnitudes(trials, pre=pre, post=post)
        for problem, trials in by_problem.items()
    }


def _scopes(windows_by_problem: dict) -> dict[str, dict]:
    """``{scope name: windows}``: pooled over problems, plus each problem if there are several."""
    scopes = {POOLED: merge_windows_by_subject(windows_by_problem)}
    if len(windows_by_problem) > 1:
        scopes.update({_problem_name(p): w for p, w in windows_by_problem.items()})
    return scopes


def _types_for(scheme: str, reversal_types: Sequence[str] | None) -> list[str]:
    """The block change types to make for *scheme*: all of its own, or those asked for that fit."""
    if reversal_types is None:
        return list(SCHEME_TYPES[scheme])
    return [t for t in reversal_types if t in SCHEME_TYPES[scheme]]


def _counts_row(scope: str, result: ReversalAlignedArms, figure: str) -> dict:
    c = result.counts
    return {
        "scope": scope,
        "label_scheme": result.scheme,
        "reversal_type": result.reversal_type,
        "weighting": result.weighting,
        "n_subjects": result.num_subjects,
        "n_reversals": result.num_reversals,
        "n_candidates": c.candidates,
        "n_tied_labels": c.tied_labels,
        "n_tied_type": c.tied_type,
        "n_wrong_type": c.wrong_type,
        "figure": figure,
    }


def _draw(
    result: ReversalAlignedArms, scope: str, out_dir: Path | None, title_scope: str
) -> list[Path]:
    """Save (or show, without *out_dir*) the figure of a result; the paths of its files.

    The figure goes in the subfolder of the result's weighting. A result with no block changes
    has no figure.
    """
    if not result.num_reversals:
        return []
    if out_dir is None:
        save_reversal_aligned_arms(result, None, scope=title_scope)
        return []
    base = out_dir / WEIGHTING_FOLDERS[result.weighting] / figure_stem(result, scope)
    save_reversal_aligned_arms(result, base, scope=title_scope)
    # the PNG first: the counts table names it
    return plot_helpers.figure_paths(base, ("png", "pdf", "svg"))


def run_reversal_aligned_arm_analysis(
    windows_or_trials_by_problem: Mapping,
    out_dir: str | Path | None,
    *,
    pre: int = 10,
    post: int = 30,
    moving_avg_window: int = 4,
    label_schemes: Sequence[str] = LABEL_SCHEMES,
    reversal_types: Sequence[str] | None = None,
    moving_avg_mode: str = "centered",
    skip_n_trials_after_reversal: int = 0,
    scope_label: str | None = None,
) -> dict:
    """Figures of arm choice probability and reward magnitude around block changes.

    One figure per scope, label scheme, block change type and weighting. The mouse-weighted
    figure (each mouse's block changes averaged into one curve, then mean and SE across mice)
    goes in ``<out_dir>/mouse-weighted/``; the event-weighted one (every block change counts
    once, mean and SE across block changes) with the same name in ``<out_dir>/event-weighted/``.
    Nothing is written at the top level of *out_dir*. Scopes: ``"pooled"`` (all problems,
    block changes concatenated per subject) and, if more than one problem is given, each problem
    (``"problem-04"``). At a drifting block change the previous best or the previous second arm
    drops to 0 and the previous third arm is rewarded again; each scheme splits by that:
    ``"prev"`` labels arms by their rank before the block change (types "Prev Best to Third" and
    "Prev Second to Third"), ``"new"`` by their rank after it (types "New Best from Third" and
    "New Second from Third"); "All" averages over both types. Block changes with tied magnitudes
    are left out (see :mod:`src.behavior_analysis.reversal_arm_ranks`); the counts say how many.

    Args:
        windows_or_trials_by_problem (Mapping): ``{problem: {subject: {session: session dict}}}``,
            from which windows (without the first block, with magnitudes) are computed per
            problem; or ``{problem: {subject: [block change dict]}}`` already built with
            :func:`~src.behavior_analysis.reversal_aligned_arms.build_windows_with_magnitudes`
            and the same *pre* and *post*. A single problem is a one-entry dict.
        out_dir (str | Path | None): Folder for the two weighting subfolders (created if
            missing); figures are named like their titles, e.g. ``Choice Probabilities Around
            Block Changes (Prev Best to Third) (Arms Labelled by Rank Before) (pooled).png/.pdf``.
            ``None`` or ``""`` shows the figures of both weightings and writes nothing.
        pre (int): Trials before the block change in each window. Default 10. Keyword-only.
        post (int): Trials from the block change on in each window. Default 30. Keyword-only.
        moving_avg_window (int): Moving-average width for the choice probabilities (positive
            int; 1 for none). Default 4. Keyword-only.
        label_schemes (Sequence[str]): Subset of ``("prev", "new")``. Default: all of
            ``LABEL_SCHEMES``. Keyword-only.
        reversal_types (Sequence[str] | None): Types to make: ``"all"``, and for ``prev``
            ``"best_to_third"`` / ``"second_to_third"``, for ``new`` ``"third_to_best"`` /
            ``"third_to_second"``; a type is made only for the schemes that have it. Default
            ``None``: all types of every scheme. Keyword-only.
        moving_avg_mode (str): ``"centered"`` or ``"trailing"``; the same for every figure.
            Default ``"centered"``. Keyword-only.
        skip_n_trials_after_reversal (int): Post trial 0 is kept and the next *k* trials dropped.
            Default 0. Keyword-only.
        scope_label (str | None): Text in the title of the ``"pooled"`` scope's figures instead
            of ``"pooled"`` (for example ``"problem-03"`` when the one problem given is the
            subject of the folder). File names and the counts table keep the scope name.
            Default ``None``. Keyword-only.

    Returns:
        dict: ``{"results": {scope: {scheme: {type: ReversalAlignedArms}}}, "event_results":
        the same with every block change weighted equally, "counts": DataFrame with one row per
        scope, scheme, type and weighting (``scope``, ``label_scheme``, ``reversal_type``,
        ``weighting`` ("subject" or "event"), ``n_subjects``, ``n_reversals``, ``n_candidates``,
        ``n_tied_labels``, ``n_tied_type``, ``n_wrong_type`` and the ``figure`` file name,
        empty for no figure), "figure_paths": [Path, ...] (both weightings),
        "windows_by_problem": {problem: windows}}``. The counts are returned, not written. A
        combination with no block change left makes no figure and raises an
        ``EmptySelectionWarning``.

    Raises:
        ValueError: For unknown schemes, types or modes, no problems, mixed or incomplete
            inputs, windows not built with this *pre*/*post*, or probabilities that do not sum to 1.
    """
    _check_choices(label_schemes, reversal_types, moving_avg_mode, moving_avg_window)
    windows_by_problem = _windows_by_problem(windows_or_trials_by_problem, pre=pre, post=post)
    out_dir = Path(out_dir) if out_dir else None
    settings = dict(
        pre=pre,
        post=post,
        skip_n_trials_after_reversal=skip_n_trials_after_reversal,
        moving_avg_window=moving_avg_window,
        moving_avg_mode=moving_avg_mode,
    )
    results = {weighting: {} for weighting in WEIGHTINGS}
    rows, figure_paths = [], []
    for scope, windows in _scopes(windows_by_problem).items():
        title_scope = scope_label if scope == POOLED and scope_label else scope
        for weighting in WEIGHTINGS:
            results[weighting][scope] = {scheme: {} for scheme in label_schemes}
        for scheme in label_schemes:
            for kind in _types_for(scheme, reversal_types):
                for weighting in WEIGHTINGS:
                    result = compute_reversal_aligned_arms(
                        windows, scheme=scheme, reversal_type=kind, weighting=weighting, **settings
                    )
                    results[weighting][scope][scheme][kind] = result
                    paths = _draw(result, scope, out_dir, title_scope)
                    figure_paths += paths
                    rows.append(_counts_row(scope, result, paths[0].name if paths else ""))
    return {
        "results": results["subject"],
        "event_results": results["event"],
        "counts": pd.DataFrame(rows),
        "figure_paths": figure_paths,
        "windows_by_problem": windows_by_problem,
    }
