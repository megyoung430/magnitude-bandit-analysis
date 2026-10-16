"""Entry point: reversal-aligned choice probability and reward magnitude of the three arms.

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
    ReversalAlignedArms,
    build_windows_with_magnitudes,
    check_moving_avg_window,
    check_windows_built_with,
    compute_reversal_aligned_arms,
)
from src.behavior_analysis.reversal_arm_ranks import LABEL_SCHEMES, REVERSAL_TYPES
from src.behavior_visualization.plot_reversal_aligned_arms import save_reversal_aligned_arms

COUNTS_FILENAME = "reversal_aligned_arms_counts.csv"
POOLED = "pooled"


def _problem_name(problem) -> str:
    is_int = isinstance(problem, (int, np.integer)) and not isinstance(problem, bool)
    return f"problem-{int(problem):02d}" if is_int else str(problem)


def _check_choices(label_schemes, reversal_types, moving_avg_mode, moving_avg_window) -> None:
    check_moving_avg_window(moving_avg_window)
    for name, chosen, allowed in (
        ("label scheme", label_schemes, LABEL_SCHEMES),
        ("reversal_type", reversal_types, REVERSAL_TYPES),
    ):
        unknown = [c for c in chosen if c not in allowed]
        if unknown:
            raise ValueError(f"unknown {name} {unknown}; choose from {allowed}")
    if moving_avg_mode not in MOVING_AVG_MODES:
        raise ValueError(
            f"moving_avg_mode must be one of {MOVING_AVG_MODES}, got {moving_avg_mode!r}"
        )


def _holds_windows(per_problem) -> bool | None:
    """True if the values are reversal lists, False if session dicts, None if there are none."""
    kinds = {isinstance(value, list) for value in per_problem.values()}
    if len(kinds) > 1:
        raise ValueError("a problem mixes reversal lists and session dicts")
    return kinds.pop() if kinds else None


def _windows_by_problem(by_problem: Mapping, *, pre: int, post: int) -> dict:
    """Enriched windows per problem from trials, or the windows as given.

    Raises:
        ValueError: If there are no problems, trials and windows are mixed, or given windows
            were not built by ``build_windows_with_magnitudes`` with this *pre* and *post*.
    """
    if not by_problem:
        raise ValueError("no problems given")
    kinds = {_holds_windows(per_problem) for per_problem in by_problem.values()} - {None}
    if len(kinds) > 1:
        raise ValueError("cannot mix trials and windows across problems")
    if kinds == {True}:
        for windows in by_problem.values():
            check_windows_built_with(windows, pre=pre, post=post)
        return dict(by_problem)
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


def _counts_row(scope: str, result: ReversalAlignedArms, figure: str) -> dict:
    c = result.counts
    return {
        "scope": scope,
        "label_scheme": result.scheme,
        "reversal_type": result.reversal_type,
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
    """Save (or show, without *out_dir*) the figure of a result; its PNG and PDF paths.

    A result with no reversals has no figure.
    """
    if not result.num_reversals:
        return []
    if out_dir is None:
        save_reversal_aligned_arms(result, None, scope=title_scope)
        return []
    base = out_dir / f"reversal_aligned_arms_{scope}_{result.scheme}_{result.reversal_type}"
    save_reversal_aligned_arms(result, base, scope=title_scope)
    return [Path(f"{base}.png"), Path(f"{base}.pdf")]


def run_reversal_aligned_arm_analysis(
    windows_or_trials_by_problem: Mapping,
    out_dir: str | Path | None,
    *,
    pre: int = 10,
    post: int = 30,
    moving_avg_window: int = 4,
    label_schemes: Sequence[str] = LABEL_SCHEMES,
    reversal_types: Sequence[str] = REVERSAL_TYPES,
    moving_avg_mode: str = "centered",
    skip_n_trials_after_reversal: int = 0,
    scope_label: str | None = None,
) -> dict:
    """Figures of arm choice probability and reward magnitude around block changes.

    One figure per scope, label scheme and reversal type. Scopes: ``"pooled"`` (all problems,
    reversals concatenated per subject) and, if more than one problem is given, each problem
    (``"problem-04"``). Schemes: ``"prev"`` labels arms by their rank before the reversal,
    ``"new"`` by their rank after it; the reversal type is defined by the prev ranking in both.
    Reversals with tied magnitudes are left out (see
    :mod:`src.behavior_analysis.reversal_arm_ranks`); the counts say how many.

    Args:
        windows_or_trials_by_problem (Mapping): ``{problem: {subject: {session: session dict}}}``,
            from which windows (without the first block, with magnitudes) are computed per
            problem; or ``{problem: {subject: [reversal dict]}}`` already built with
            :func:`~src.behavior_analysis.reversal_aligned_arms.build_windows_with_magnitudes`
            and the same *pre* and *post*. A single problem is a one-entry dict.
        out_dir (str | Path | None): Folder for
            ``reversal_aligned_arms_<scope>_<scheme>_<type>.png/.pdf`` and
            ``reversal_aligned_arms_counts.csv`` (created if missing). ``None`` or ``""`` shows
            the figures and writes nothing.
        pre (int): Trials before the reversal in each window. Default 10. Keyword-only.
        post (int): Trials from the reversal on in each window. Default 30. Keyword-only.
        moving_avg_window (int): Moving-average width for the choice probabilities (positive
            int; 1 for none). Default 4. Keyword-only.
        label_schemes (Sequence[str]): Subset of ``("prev", "new")``. Default: all of
            ``LABEL_SCHEMES``. Keyword-only.
        reversal_types (Sequence[str]): Subset of ``("all", "best_to_third",
            "second_to_third")``. Default: all of ``REVERSAL_TYPES``. Keyword-only.
        moving_avg_mode (str): ``"centered"`` or ``"trailing"``; the same for every figure.
            Default ``"centered"``. Keyword-only.
        skip_n_trials_after_reversal (int): Post trial 0 is kept and the next *k* trials dropped.
            Default 0. Keyword-only.
        scope_label (str | None): Text in the title of the ``"pooled"`` scope's figures instead
            of ``"pooled"`` (for example ``"problem-03"`` when the one problem given is the
            subject of the folder). File names and the counts table keep the scope name.
            Default ``None``. Keyword-only.

    Returns:
        dict: ``{"results": {scope: {scheme: {type: ReversalAlignedArms}}}, "counts": DataFrame
        with one row per scope, scheme and type, "figure_paths": [Path, ...],
        "windows_by_problem": {problem: windows}}``. A combination with no reversal left makes
        no figure and raises an ``EmptySelectionWarning``.

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
    results, rows, figure_paths = {}, [], []
    for scope, windows in _scopes(windows_by_problem).items():
        results[scope] = {scheme: {} for scheme in label_schemes}
        for scheme in label_schemes:
            for kind in reversal_types:
                result = compute_reversal_aligned_arms(
                    windows, scheme=scheme, reversal_type=kind, **settings
                )
                results[scope][scheme][kind] = result
                title_scope = scope_label if scope == POOLED and scope_label else scope
                paths = _draw(result, scope, out_dir, title_scope)
                figure_paths += paths
                rows.append(_counts_row(scope, result, paths[0].name if paths else ""))
    counts = pd.DataFrame(rows)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        counts.to_csv(out_dir / COUNTS_FILENAME, index=False)
    return {
        "results": results,
        "counts": counts,
        "figure_paths": figure_paths,
        "windows_by_problem": windows_by_problem,
    }
