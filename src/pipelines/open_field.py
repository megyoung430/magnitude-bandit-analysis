"""Standard open-field bandit figures for a cohort, run problem by problem.

Scripts and notebooks call :func:`run_open_field_analysis` (all problems) or
:func:`run_problem` (one problem) with settings objects, instead of editing or
exec-ing a script's source. Every figure is saved under
``<output_root>/<task>/<cohort>/problem-NN/``.
"""

import warnings
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from src.behavior_analysis.get_choice_probs_around_good_reversals import (
    NoReversalsError,
    apply_moving_average_to_choice_probs,
    get_choice_probs_around_good_reversals,
    remove_trials_after_bad_rev,
)
from src.behavior_analysis.get_diagnostic_p_value import (
    pvalue_paired_t_best_vs_second_vs_third,
    pvalue_paired_t_new_vs_third,
)
from src.behavior_analysis.get_first_leave_after_good_reversals import (
    average_first_leave_across_subjects,
    get_first_leave_after_good_reversals,
)
from src.behavior_analysis.get_good_reversal_info import ReversalWarning, get_good_reversal_info
from src.behavior_analysis.get_rank_counts_by_good_reversal import (
    get_rank_counts_by_good_reversal,
    get_rank_counts_by_value,
)
from src.behavior_analysis.get_total_reversals import get_reversal_indices_by_kind
from src.behavior_analysis.merge_across_problems import merge_windows_by_subject
from src.behavior_analysis.poke_counts import mouse_poke_counts
from src.behavior_analysis.port_finding_sessions import (
    PortFindingAssignment,
    port_finding_sessions_for_problem,
)
from src.behavior_analysis.reversal_aligned_arms import build_windows_with_magnitudes
from src.behavior_analysis.split_early_late_good_reversals import split_good_reversals_early_late
from src.behavior_analysis.split_good_reversals_by_best_change import (
    split_good_reversals_by_best_change,
)
from src.behavior_import.task_types import BLOCKED
from src.behavior_visualization.plot_choice_probs_around_good_reversals import (
    plot_choice_probs_around_good_reversals,
)
from src.behavior_visualization.plot_first_leave_after_good_reversals import (
    plot_first_leave_after_good_reversals,
)
from src.behavior_visualization.plot_num_reversals import (
    plot_moving_avg_reversals_over_time,
    plot_num_reversals,
    plot_num_reversals_over_time,
)
from src.behavior_visualization.plot_rank_proportions import plot_rank_proportions
from src.behavior_visualization.plot_single_session import plot_single_session
from src.pipelines.poke_analysis import run_poke_analysis
from src.pipelines.port_preference import run_port_preference_analysis
from src.pipelines.reversal_aligned_arms import run_reversal_aligned_arm_analysis
from src.pipelines.value_change_analysis import run_value_change_analysis
from src.pipelines.value_psychometrics import run_value_psychometric_analysis

VALUE_BASIS_LABELS = {"true": "True Value", "believed": "Believed Value"}
_CHOICE_PROBS_NAME = "Choice Probabilities Around Good Reversals"
# Analyses of the drifting-value task; a blocked task has no drifting values to look at.
DRIFTING_ONLY_SECTIONS = (
    "run_value_psychometric_analysis",
    "run_reversal_aligned_arm_analysis",
    "run_value_change_analysis",
    "run_port_preference_analysis",
)
POOLED_DRIFTING_LABEL = "all drifting problems"
_SECTION_NAMES = {
    "run_value_psychometric_analysis": "value psychometrics",
    "run_reversal_aligned_arm_analysis": "reversal-aligned arms",
    "run_value_change_analysis": "value-change choice probabilities",
    "run_port_preference_analysis": "port preference",
}
LONG_POKES_FOLDER = "long-pokes"
VALUE_PSYCHOMETRICS_FOLDER = "value-psychometrics"
REVERSAL_ARMS_FOLDER = "reversal-aligned-arms"
VALUE_CHANGE_FOLDER = "value-change"
PORT_PREFERENCE_FOLDER = "port-preference"
# One window size for building the reversal windows and for the entry point that checks it.
REVERSAL_ARMS_PRE = 10
REVERSAL_ARMS_POST = 30


class ChoiceProbabilitySettings(BaseModel):
    """Which choice-probability-around-good-reversal figures to make, and their windows.

    Frozen; an unknown field name raises a validation error.

    Attributes:
        run_all (bool): Raw choice probabilities around all good reversals, plus the early
            (first two) and late splits. Default ``True``.
        skip (bool): Figures that skip the first trials after each reversal. Default ``True``.
        moving_avg (bool): Moving-average figures. Default ``True``.
        remove_bad (bool): Moving-average figures with the trials after a bad reversal removed.
            Default ``True``.
        split_by_best_change (bool): Also split by whether the previously best arm became second
            or third best; only used together with ``moving_avg``. Default ``False``.
        split_by_best_change_and_half (bool): Within the best-change split, also split into
            early and late reversals. Default ``False``.
        split_by_first_two (bool): Moving-average figures also for the first two reversals
            versus the later ones. Default ``True``.
        split_by_half (bool): Moving-average figures also for the first versus second half of
            the reversals. Default ``True``.
        pre (int): Trials before the reversal in each window. Default 10.
        post (int): Trials after the reversal in each window. Default 30.
        skip_n_trials_after_reversal (int): Trials skipped after a reversal when ``skip`` is
            on. Default 15.
        moving_avg_window (int): Window length of the moving average, in trials. Default 4.
    """

    # extra="forbid": a misspelled option raises instead of being silently ignored
    model_config = ConfigDict(frozen=True, extra="forbid")

    run_all: bool = True
    skip: bool = True
    moving_avg: bool = True
    remove_bad: bool = True
    split_by_best_change: bool = False
    split_by_best_change_and_half: bool = False
    split_by_first_two: bool = True
    split_by_half: bool = True
    pre: int = 10
    post: int = 30
    skip_n_trials_after_reversal: int = 15
    moving_avg_window: int = 4


class AnalysisSelection(BaseModel):
    """Which analysis sections to run for a problem.

    ``rank_value_bases=None`` gives the true-value rank proportions from
    ``get_rank_counts_by_good_reversal``; a tuple such as ``("true", "believed")``
    uses ``get_rank_counts_by_value`` for each basis and each pooling mode.

    ``run_poke_analysis`` (long pokes and maze heatmaps) applies to any problem. The three
    value analyses (``run_value_psychometric_analysis``, ``run_reversal_aligned_arm_analysis``,
    ``run_value_change_analysis``, ``run_port_preference_analysis``) are for drifting problems
    only; :func:`run_problem` raises if one is on for a blocked problem. All five are off unless
    asked for.

    Frozen; an unknown field name raises a validation error.

    Attributes:
        run_choice_probability_analysis (bool): Choice probabilities around good reversals.
            Default ``True``.
        run_first_leave_analysis (bool): First arm left after good reversals. Default ``True``.
        run_number_of_reversals_analysis (bool): Total, cumulative and moving-average reversal
            counts. Default ``True``.
        run_rank_proportion_analysis (bool): Rank proportions after good reversals.
            Default ``True``.
        run_single_session_analysis (bool): One overview plot per session. Default ``True``.
        run_poke_analysis (bool): Long pokes and maze heatmaps. Default ``False``.
        run_value_psychometric_analysis (bool): Value psychometric curves (drifting problems
            only). Default ``False``.
        run_reversal_aligned_arm_analysis (bool): Reversal-aligned arm choices (drifting problems
            only). Default ``False``.
        run_value_change_analysis (bool): Value-change choice probabilities (drifting problems
            only). Default ``False``.
        run_port_preference_analysis (bool): Chosen-versus-best preference per port and for X and
            plus ports (drifting problems only). Default ``False``.
        rank_value_bases (tuple[str, ...] | None): ``"true"`` and/or ``"believed"``; ``None``
            gives the true-value rank proportions of ``get_rank_counts_by_good_reversal``.
            Default ``None``.
        rank_pooling_modes (tuple[bool, ...]): ``pool_trials_within_subject`` values to run for
            each basis (only used when ``rank_value_bases`` is set). Default ``(True, False)``.
    """

    # extra="forbid": a misspelled option raises instead of being silently ignored
    model_config = ConfigDict(frozen=True, extra="forbid")

    run_choice_probability_analysis: bool = True
    run_first_leave_analysis: bool = True
    run_number_of_reversals_analysis: bool = True
    run_rank_proportion_analysis: bool = True
    run_single_session_analysis: bool = True
    run_poke_analysis: bool = False
    run_value_psychometric_analysis: bool = False
    run_reversal_aligned_arm_analysis: bool = False
    run_value_change_analysis: bool = False
    run_port_preference_analysis: bool = False
    rank_value_bases: tuple[str, ...] | None = None
    rank_pooling_modes: tuple[bool, ...] = (True, False)


DEFAULT_SELECTION = AnalysisSelection()
DEFAULT_CHOICE = ChoiceProbabilitySettings()


def check_selection_for_task_type(selection: AnalysisSelection, task_type: str | None) -> None:
    """Raise ``ValueError`` if a drifting-only analysis is selected for a blocked problem.

    Args:
        selection (AnalysisSelection): The analyses asked for.
        task_type (str | None): ``"blocked"``, ``"drifting"`` or ``None``; anything but
            ``"blocked"`` is accepted without checking.

    Returns:
        None: Returns nothing when the selection is allowed.

    Raises:
        ValueError: If a drifting-only analysis is on and *task_type* is ``"blocked"``.
    """
    if task_type != BLOCKED:
        return
    wrong = [name for name in DRIFTING_ONLY_SECTIONS if getattr(selection, name)]
    if wrong:
        raise ValueError(
            f"{wrong} are analyses of the drifting task and cannot run on a blocked problem"
        )


def problem_dir(
    output_root: Path, task: str, cohort: str, problem_number: int, task_type: str | None = None
) -> Path:
    """``<output_root>/<task>/<cohort>[/<task_type>]/problem-NN``.

    Args:
        output_root (Path): Root folder for figures.
        task (str): Task folder name, e.g. ``"open-field"``.
        cohort (str): Cohort folder name, e.g. ``"cohort-03"``.
        problem_number (int): Problem number, zero-padded to two digits.
        task_type (str | None): Optional ``"blocked"`` / ``"drifting"`` folder level.
            Default ``None``: no such level.

    Returns:
        Path: The folder for this problem's figures (not created).
    """
    base = Path(output_root) / task / cohort
    if task_type is not None:
        base = base / task_type
    return base / f"problem-{problem_number:02d}"


# ---------------------------------------------------------------------------
# Choice probabilities around good reversals
# ---------------------------------------------------------------------------


def problem_place(task_type: str | None, problem_number: int | None) -> str:
    """How a warning names the problem: ``"blocked problem 2"``, ``"problem 2"`` or a fallback.

    Args:
        task_type (str | None): ``"blocked"`` or ``"drifting"``, or ``None`` if unknown.
        problem_number (int | None): The problem's number within its task type, or ``None``.

    Returns:
        str: The place, ``"this problem"`` when there is no number. The task type is what lets
        :func:`~src.behavior_analysis.warnings_log.warnings_log_table` place the warning in the
        mouse's problems.
    """
    if problem_number is None:
        return "this problem"
    return f"{task_type} problem {problem_number}" if task_type else f"problem {problem_number}"


def _report_missing_mice(mice, used, *, figure: str, where: str) -> bool:
    """Warn about mice that add nothing to *figure*; return whether any mouse is left to draw.

    Args:
        mice: Every mouse of the problem.
        used: The mice that have good reversals in the figure's windows.
        figure (str): The figure's name, for the messages.
        where (str): The problem, for the messages (see :func:`problem_place`).

    Returns:
        bool: ``False`` (after a warning) if no mouse is in *used*, so the figure is skipped.
    """
    for mouse in sorted(set(mice) - set(used)):
        warnings.warn(
            f"{figure}: mouse {mouse} has no good reversals in {where}; "
            "it contributes nothing to this figure.",
            ReversalWarning,
            stacklevel=2,
        )
    if not used:
        warnings.warn(
            f"{figure}: no mouse has good reversals in {where}; figure skipped.",
            ReversalWarning,
            stacklevel=2,
        )
    return bool(used)


def _choice_probs(
    windows,
    s: ChoiceProbabilitySettings,
    figure: str,
    where: str,
    *,
    skip=False,
    smooth=False,
):
    """``(x, across)`` for one figure, or ``None`` (after a warning) if it has no data to draw.

    A mouse with no good reversal is named in a warning. If no mouse has one the analysis raises
    :class:`~src.behavior_analysis.get_choice_probs_around_good_reversals.NoReversalsError`;
    that, and only that, skips the figure here.

    Args:
        windows (dict): ``{mouse: [reversal windows]}`` of the figure.
        s (ChoiceProbabilitySettings): Windows and smoothing.
        figure (str): The figure's file name, for the warnings.
        where (str): The problem, for the warnings (see :func:`problem_place`).
        skip (bool): Skip the first trials after each reversal. Default ``False``. Keyword-only.
        smooth (bool): Apply the moving average. Default ``False``. Keyword-only.

    Returns:
        tuple | None: ``(x, across)``, or ``None`` if no mouse has a good reversal.
    """
    extra = {"skip_n_trials_after_reversal": s.skip_n_trials_after_reversal} if skip else {}
    try:
        x, per_subject, across = get_choice_probs_around_good_reversals(
            windows, pre=s.pre, post=s.post, **extra
        )
    except NoReversalsError:
        per_subject = {}
    if not _report_missing_mice(windows, per_subject, figure=figure, where=where):
        return None
    if smooth:
        x, _, across = apply_moving_average_to_choice_probs(
            x, per_subject, moving_avg_window=s.moving_avg_window, mode="centered"
        )
    return x, across


def _plot_with_reversal_axis(x, across, windows, idx, save_path, *, only_good: bool):
    """Choice probabilities plus the cumulative-fraction-removed axis."""
    good_idx, bad_idx, unclassified_idx = idx
    # good-only plots state use_total explicitly; remove-bad plots rely on its default
    extra = {"use_total": False} if only_good else {}
    plot_choice_probs_around_good_reversals(
        x,
        across,
        add_cumulative_axis=True,
        only_good=only_good,
        **extra,
        windows_for_cumulative_axis=windows,
        all_good_idx=good_idx,
        all_bad_idx=bad_idx,
        all_unclassified_idx=unclassified_idx,
        save_path=save_path,
    )


def _windows_and_indices(subjects_trials, s: ChoiceProbabilitySettings):
    # recomputed for every figure family: later steps may modify the windows
    windows = get_good_reversal_info(
        subjects_trials, pre=s.pre, post=s.post, include_first_block=False
    )
    kinds = get_reversal_indices_by_kind(subjects_trials)
    return windows, (kinds["good"], kinds["bad"], kinds["unclassified"])


def _drop_after_bad(windows, idx):
    return remove_trials_after_bad_rev(windows, *idx[:2], include_bad_trial=True)


def _plot_all_reversals(
    subjects_trials, out_dir: Path, s: ChoiceProbabilitySettings, where="this problem"
):
    """Raw choice probabilities: all good reversals, then the first two and the later ones."""
    windows, idx = _windows_and_indices(subjects_trials, s)
    early, late = split_good_reversals_early_late(windows, first_n=2)
    for suffix, w in [("", windows), (" (Early)", early), (" (Late)", late)]:
        path = out_dir / f"{_CHOICE_PROBS_NAME}{suffix}"
        curves = _choice_probs(w, s, path.name, where)
        if curves is not None:
            _plot_with_reversal_axis(*curves, w, idx, path, only_good=True)


def _plot_skipping_initial_trials(
    subjects_trials, out_dir: Path, s: ChoiceProbabilitySettings, where="this problem"
):
    """Choice probabilities that skip the first trials after each reversal."""
    windows, _ = _windows_and_indices(subjects_trials, s)
    early, late = split_good_reversals_early_late(windows, first_n=2)
    for suffix, w in [("", windows), (" (Early)", early), (" (Late)", late)]:
        path = out_dir / f"{_CHOICE_PROBS_NAME}, Skipping Initial Trials{suffix}"
        curves = _choice_probs(w, s, path.name, where, skip=True)
        if curves is None:
            continue
        x, across = curves
        plot_choice_probs_around_good_reversals(
            x,
            across,
            add_cumulative_axis=False,
            skip_n_trials_after_reversal=s.skip_n_trials_after_reversal,
            save_path=path,
        )


def _plot_moving_average(
    subjects_trials, out_dir: Path, s: ChoiceProbabilitySettings, where="this problem"
):
    """Moving-average choice probabilities, optionally split by half and by first two."""
    windows, idx = _windows_and_indices(subjects_trials, s)
    families = [("", windows)]
    if s.split_by_half:
        first, second = split_good_reversals_early_late(windows)
        families += [(" (First Half)", first), (" (Second Half)", second)]
    if s.split_by_first_two:
        early, late = split_good_reversals_early_late(windows, first_n=2)
        families += [(" (Early)", early), (" (Late)", late)]
    for suffix, w in families:
        path = out_dir / f"{_CHOICE_PROBS_NAME} (Moving Average){suffix}"
        curves = _choice_probs(w, s, path.name, where, smooth=True)
        if curves is not None:
            _plot_with_reversal_axis(*curves, w, idx, path, only_good=True)


def _plot_remove_bad(
    subjects_trials, out_dir: Path, s: ChoiceProbabilitySettings, where="this problem"
):
    """Moving-average choice probabilities with the trials after a bad reversal removed."""
    windows, idx = _windows_and_indices(subjects_trials, s)
    windows = _drop_after_bad(windows, idx)
    early, late = split_good_reversals_early_late(windows, first_n=2)
    families = [
        ("", windows),
        (" (Early)", _drop_after_bad(early, idx)),
        (" (Late)", _drop_after_bad(late, idx)),
    ]
    for suffix, w in families:
        path = out_dir / f"{_CHOICE_PROBS_NAME} (Remove Bad){suffix}"
        curves = _choice_probs(w, s, path.name, where, smooth=True)
        if curves is not None:
            _plot_with_reversal_axis(*curves, w, idx, path, only_good=False)


def run_choice_probability_analysis(
    subjects_trials,
    out_dir: Path,
    s: ChoiceProbabilitySettings,
    *,
    problem_number=None,
    task_type=None,
):
    """All enabled choice-probability figures for one problem, saved in *out_dir*.

    A mouse without good reversals in a figure's windows adds nothing to that figure and is
    named in a warning; a figure that no mouse has data for is skipped, also with a warning.

    Args:
        subjects_trials (dict): ``{subject: {session_key: session_dict}}`` for one problem.
        out_dir (Path): Folder that receives the figures.
        s (ChoiceProbabilitySettings): Which figure families to make and their windows.
        problem_number (int | None): The problem's number within its task type, for the
            warnings. Default ``None``. Keyword-only.
        task_type (str | None): ``"blocked"`` or ``"drifting"``, for the warnings. Default
            ``None``. Keyword-only.

    Returns:
        None: Returns nothing; the figures are written to *out_dir*.
    """
    where = problem_place(task_type, problem_number)
    if s.run_all:
        _plot_all_reversals(subjects_trials, out_dir, s, where)
    if s.skip:
        _plot_skipping_initial_trials(subjects_trials, out_dir, s, where)
    if s.moving_avg:
        _plot_moving_average(subjects_trials, out_dir, s, where)
    if s.remove_bad:
        _plot_remove_bad(subjects_trials, out_dir, s, where)
    if s.split_by_best_change and s.moving_avg:
        _run_best_change(subjects_trials, out_dir, s, where)


def _run_best_change(
    subjects_trials, out_dir: Path, s: ChoiceProbabilitySettings, where="this problem"
):
    name = "Choice Probs Good Reversals"
    windows, idx = _windows_and_indices(subjects_trials, s)
    best2, best3 = split_good_reversals_by_best_change(windows)
    early, late = split_good_reversals_early_late(windows, first_n=2)
    best2_early, best3_early = split_good_reversals_by_best_change(early)
    best2_late, best3_late = split_good_reversals_by_best_change(late)

    families = [(" (Best->Second)", best2), (" (Best->Third)", best3)]
    if s.split_by_best_change_and_half:
        families += [
            (" (Best->Second) (Early)", best2_early),
            (" (Best->Third) (Early)", best3_early),
            (" (Best->Second) (Late)", best2_late),
            (" (Best->Third) (Late)", best3_late),
        ]
    for suffix, w in families:
        path = out_dir / f"{name}{suffix}"
        curves = _choice_probs(w, s, path.name, where, smooth=True)
        if curves is not None:
            _plot_with_reversal_axis(*curves, w, idx, path, only_good=True)

    if not s.remove_bad:
        return
    # Known issue: the full-problem "(Remove Bad)" figures use best2/best3 without
    # remove_trials_after_bad_rev; only the early/late splits are filtered.
    families = [(" (Best->Second) (Remove Bad)", best2), (" (Best->Third) (Remove Bad)", best3)]
    if s.split_by_best_change_and_half:
        families += [
            (" (Best->Second) (Remove Bad) (Early)", _drop_after_bad(best2_early, idx)),
            (" (Best->Third) (Remove Bad) (Early)", _drop_after_bad(best3_early, idx)),
            (" (Best->Second) (Remove Bad) (Late)", _drop_after_bad(best2_late, idx)),
            (" (Best->Third) (Remove Bad) (Late)", _drop_after_bad(best3_late, idx)),
        ]
    for suffix, w in families:
        path = out_dir / f"{name}{suffix}"
        curves = _choice_probs(w, s, path.name, where, smooth=True)
        if curves is not None:
            _plot_with_reversal_axis(*curves, w, idx, path, only_good=False)


# ---------------------------------------------------------------------------
# Other sections
# ---------------------------------------------------------------------------


def run_first_leave_analysis(
    subjects_trials, out_dir: Path, *, problem_number=None, task_type=None
):
    """First arm left after good reversals: all, first two, and later reversals.

    A mouse without good reversals in a figure's windows adds nothing to that figure and is
    named in a warning; a figure that no mouse has data for is skipped, also with a warning.

    Args:
        subjects_trials (dict): ``{subject: {session_key: session_dict}}`` for one problem.
        out_dir (Path): Folder that receives the three figures.
        problem_number (int | None): The problem's number within its task type, for the
            warnings. Default ``None``. Keyword-only.
        task_type (str | None): ``"blocked"`` or ``"drifting"``, for the warnings. Default
            ``None``. Keyword-only.

    Returns:
        None: Returns nothing; the figures are written to *out_dir*.
    """
    where = problem_place(task_type, problem_number)
    windows = get_good_reversal_info(subjects_trials, include_first_block=False)
    early, late = split_good_reversals_early_late(windows, first_n=2)
    for suffix, w in [("", windows), (" (Early)", early), (" (Late)", late)]:
        path = out_dir / f"First Leave After Good Reversals{suffix}"
        per_subject = get_first_leave_after_good_reversals(w)
        with_reversals = [m for m, counts in per_subject.items() if counts["total"] > 0]
        if not _report_missing_mice(w, with_reversals, figure=path.name, where=where):
            continue
        mean, se, _ = average_first_leave_across_subjects(per_subject)
        p_value = pvalue_paired_t_new_vs_third(per_subject, alternative="greater")
        plot_first_leave_after_good_reversals(mean, se, per_subject, p_value, save_path=path)


def run_number_of_reversals_analysis(subjects_trials, out_dir: Path, *, reversal_trial_window=None):
    """Total, cumulative and moving-average reversal counts.

    Args:
        subjects_trials (dict): ``{subject: {session_key: session_dict}}`` for one problem.
        out_dir (Path): Folder that receives the three figures.
        reversal_trial_window (int | None): Trials per bin of the moving-average plot.
            Default ``None``: the plot function's own default. Keyword-only.

    Returns:
        None: Returns nothing; the figures are written to *out_dir*.
    """
    plot_num_reversals(subjects_trials, save_path=out_dir / "Total Reversals")
    plot_num_reversals_over_time(
        subjects_trials, threshold=10, save_path=out_dir / "Cumulative Reversals Over Time"
    )
    extra = {} if reversal_trial_window is None else {"trial_window": reversal_trial_window}
    plot_moving_avg_reversals_over_time(
        subjects_trials, **extra, save_path=out_dir / "Moving Average Reversals Over Time"
    )


def run_rank_proportion_analysis(subjects_trials, out_dir: Path, selection: AnalysisSelection):
    """Rank proportions after good reversals; returns counts and p-values when by value basis.

    Args:
        subjects_trials (dict): ``{subject: {session_key: session_dict}}`` for one problem.
        out_dir (Path): Folder that receives the figures.
        selection (AnalysisSelection): Its ``rank_value_bases`` (``None`` for the plain true-value
            figure) and ``rank_pooling_modes`` decide which figures are made.

    Returns:
        dict: ``{}`` when ``selection.rank_value_bases`` is ``None``; otherwise
        ``{value_basis: {pool_trials: {"counts": ..., "p_values": ...}}}``.
    """
    windows = get_good_reversal_info(subjects_trials, include_first_block=True)
    if selection.rank_value_bases is None:
        counts = get_rank_counts_by_good_reversal(windows)
        p_values = pvalue_paired_t_best_vs_second_vs_third(counts)
        # plot_rank_proportions appends suffixes to a str save_path
        plot_rank_proportions(
            counts,
            average_across_mice_pvalues=p_values,
            save_path=str(out_dir / "Rank Proportions"),
        )
        return {}

    results = {}
    for value_basis in selection.rank_value_bases:
        label = VALUE_BASIS_LABELS[value_basis]
        print(f"Running rank proportions: {out_dir.parent.name}, {label}")
        counts = get_rank_counts_by_value(subjects_trials, windows, value_basis=value_basis)
        results[value_basis] = {}
        for pool_trials in selection.rank_pooling_modes:
            print(f"  pool_trials_within_subject={pool_trials}")
            p_values = pvalue_paired_t_best_vs_second_vs_third(
                counts, pool_trials_within_subject=pool_trials
            )
            results[value_basis][pool_trials] = {"counts": counts, "p_values": p_values}
            plot_rank_proportions(
                counts,
                average_across_mice_pvalues=p_values,
                pool_trials_within_subject=pool_trials,
                save_path=str(out_dir / f"Rank Proportions ({label}) (Pool Trials {pool_trials})"),
            )
    return results


def run_single_session_analysis(
    subjects_trials, out_dir: Path, *, previous_choice_towers=None, poke_counts=None
):
    """One overview plot per session that has trials.

    Args:
        subjects_trials (dict): ``{subject: {session_key: session_dict}}`` for one problem.
        out_dir (Path): Folder that receives ``<subject>/<session>/Session Plot``.
        previous_choice_towers (dict | None): ``{mouse: ports of the mouse's previous problem,
            or None}`` for the red outlines of the maze row; ``None`` draws none. Keyword-only.
        poke_counts (dict | None): ``{mouse: {session: SessionPokeCounts}}`` already parsed, so
            the raw events are read once; ``None`` lets each figure parse its own session.
            Keyword-only.

    Returns:
        None: Returns nothing; the figures are written to *out_dir*. Sessions without
        trials are skipped with a message.
    """
    for subject, sessions in subjects_trials.items():
        for session, session_data in sessions.items():
            if not session_data["trial_info"]:
                warnings.warn(f"Skipping {subject} | {session}. There are no trials.", stacklevel=2)
                continue
            print(f"Plotting {subject} | {session}")
            plot_single_session(
                session_data,
                title=f"{subject} | {session}",
                save_path=out_dir / subject / session / "Session Plot",
                previous_choice_towers=(
                    None if previous_choice_towers is None else previous_choice_towers[subject]
                ),
                poke_counts=None if poke_counts is None else poke_counts[subject][session],
            )


# ---------------------------------------------------------------------------
# Drifting-value analyses
# ---------------------------------------------------------------------------


def pooled_arm_windows(problems: dict) -> dict:
    """Reversal windows of every problem, concatenated per mouse, as a one-entry problem dict.

    Windows are built inside each problem, so none crosses a problem boundary. The entry point
    would add a figure set per problem next to the pooled one; handing it one merged entry
    keeps a folder to the scope it stands for.

    Args:
        problems (dict): ``{problem: {mouse: {session_key: session_dict}}}``.

    Returns:
        dict: ``{"pooled": windows}``, where *windows* is the per-mouse concatenation of the
        reversal windows (with magnitudes) of all problems, built with ``REVERSAL_ARMS_PRE``
        trials before and ``REVERSAL_ARMS_POST`` after each reversal.
    """
    per_problem = {
        problem: build_windows_with_magnitudes(
            trials, pre=REVERSAL_ARMS_PRE, post=REVERSAL_ARMS_POST
        )
        for problem, trials in problems.items()
    }
    return {"pooled": merge_windows_by_subject(per_problem)}


def run_drifting_value_analyses(
    problems: dict,
    out_dir: Path,
    selection: AnalysisSelection,
    *,
    arm_scope_label: str,
    previous_choice_towers: dict | None = None,
    x_vs_plus: bool | None = None,
    per_mouse_figures: bool = True,
    mouse_legends: bool = True,
) -> dict:
    """The selected drifting-only analyses over *problems*, each in a subfolder of *out_dir*.

    Args:
        problems (dict): ``{problem: {mouse: {session_key: session_dict}}}``: one problem for a
            problem folder, several for the pool over a mouse's drifting problems. Every
            analysis computes inside each problem and then combines the mice's results, so
            beliefs reset and windows end at each problem boundary.
        out_dir (Path): Folder that receives ``value-psychometrics``, ``reversal-aligned-arms``,
            ``value-change`` and ``port-preference``.
        selection (AnalysisSelection): Which of the four to run.
        arm_scope_label (str): Scope named in the captions of the reversal-aligned and
            port-preference figures, given
            by the caller (``"problem-03"`` for a problem folder, ``POOLED_DRIFTING_LABEL`` for
            the pool) because a pool of one problem is still the pool. Keyword-only.
        previous_choice_towers (dict | None): ``{mouse: choice ports of the mouse's previous
            problem, or None}`` for the red outlines of the port-preference heatmaps of one
            problem. Default None: no red outlines (as for the pool).
        x_vs_plus (bool | None): Draw the port-preference X versus plus figure. Default
            ``None``: only when *arm_scope_label* is ``POOLED_DRIFTING_LABEL``, because a single
            problem's three ports usually lie in one category. Keyword-only.
        per_mouse_figures (bool): Draw the figures with one panel per mouse (the per-mouse
            value heatmaps and port-preference mazes). Default ``True``. Keyword-only.
        mouse_legends (bool): Name the mice in the legend of the port-preference X versus plus
            figure; the lines are drawn either way. Default ``True``. Keyword-only.

    Returns:
        dict[str, pd.DataFrame]: The figures are written under *out_dir*. The diagnostic tables
        the analyses return, for the analyses that ran: ``"reversal_aligned_arms_counts"`` and
        ``"value_change_events"`` / ``"value_change_bin_counts"`` /
        ``"value_change_symmetric_exclusions"`` (see
        :mod:`src.behavior_analysis.diagnostic_tables`).
    """
    out_dir = Path(out_dir)
    tables = {}
    if selection.run_value_psychometric_analysis:
        run_value_psychometric_analysis(
            problems,
            out_dir / VALUE_PSYCHOMETRICS_FOLDER,
            per_mouse_heatmaps=per_mouse_figures,
        )
    if selection.run_reversal_aligned_arm_analysis:
        arms = run_reversal_aligned_arm_analysis(
            pooled_arm_windows(problems),
            out_dir / REVERSAL_ARMS_FOLDER,
            pre=REVERSAL_ARMS_PRE,
            post=REVERSAL_ARMS_POST,
            scope_label=arm_scope_label,
        )
        tables["reversal_aligned_arms_counts"] = arms["counts"]
    if selection.run_value_change_analysis:
        value_change = run_value_change_analysis(problems, out_dir / VALUE_CHANGE_FOLDER)
        tables["value_change_events"] = value_change["contamination_table"]
        tables["value_change_bin_counts"] = value_change["bin_counts_table"]
        tables["value_change_symmetric_exclusions"] = value_change["symmetric_exclusions_table"]
    if selection.run_port_preference_analysis:
        run_port_preference_analysis(
            problems,
            out_dir / PORT_PREFERENCE_FOLDER,
            scope_label=arm_scope_label,
            previous_choice_towers=previous_choice_towers,
            x_vs_plus=arm_scope_label == POOLED_DRIFTING_LABEL if x_vs_plus is None else x_vs_plus,
            per_mouse_heatmap=per_mouse_figures,
            mouse_legend=mouse_legends,
        )
    return tables


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def run_problem(
    problem_number: int,
    subjects_trials: dict,
    *,
    task: str,
    cohort: str,
    output_root: Path,
    selection: AnalysisSelection = DEFAULT_SELECTION,
    choice: ChoiceProbabilitySettings = DEFAULT_CHOICE,
    reversal_trial_window: int | None = None,
    task_type: str | None = None,
    previous_choice_towers: dict | None = None,
    diagnostics: dict | None = None,
    port_finding_sessions: dict | None = None,
) -> dict:
    """Run the selected analyses for one problem; returns rank-proportion results.

    The selection alone decides which analyses run; the caller picks it per task type.
    With *task_type*, figures go under ``<cohort>/<task_type>/problem-NN``.
    *previous_choice_towers* is ``{mouse: ports of the mouse's previous problem, or None}``
    (see ``previous_choice_towers_for_problem``) for the red outlines of the maze heatmaps; the
    poke analysis (``long-pokes`` folder) and the single-session figures share one parse of the
    raw events. The poke analysis also writes the port-finding summary heatmaps into
    ``long-pokes`` (``summary_scope``; see :mod:`src.pipelines.port_finding_summary`).
    The blocked-task guard below needs *task_type*: without it every selection
    is accepted.

    Args:
        problem_number (int): The problem's number, used in the folder name and messages.
        subjects_trials (dict): ``{subject: {session_key: session_dict}}`` for this problem.
        task (str): Task folder name under *output_root*, e.g. ``"open-field"``. Keyword-only.
        cohort (str): Cohort folder name, e.g. ``"cohort-03"``. Keyword-only.
        output_root (Path): Root folder for figures. Keyword-only.
        selection (AnalysisSelection): Which analyses to run. Default ``DEFAULT_SELECTION``.
        choice (ChoiceProbabilitySettings): Choice-probability figure toggles and windows.
            Default ``DEFAULT_CHOICE``.
        reversal_trial_window (int | None): Trials per bin of the moving-average reversal
            plot. Default ``None``: the plot function's default.
        task_type (str | None): ``"blocked"`` / ``"drifting"``, or ``None`` for no task-type
            folder level and no blocked-task guard. Default ``None``.
        previous_choice_towers (dict | None): ``{mouse: previous-problem choice ports or None}``
            for the maze outlines. Default ``None``: no outlines.
        diagnostics (dict | None): If given, receives the diagnostic tables of the analyses that
            ran, as ``{kind: DataFrame}`` (``"poke_trial_summary"``, ``"port_finding_check"``
            and those of :func:`run_drifting_value_analyses`; see
            :mod:`src.behavior_analysis.diagnostic_tables`). Default ``None``. Keyword-only.
        port_finding_sessions (dict | None): ``{mouse: {session_key: record}}`` of the
            port-finding-only sessions that belong to this problem (see
            :func:`~src.behavior_analysis.port_finding_sessions.port_finding_sessions_for_problem`);
            the poke analysis counts their trials in its port-finding maps. Default ``None``:
            none. Keyword-only.

    Returns:
        dict: The rank-proportion results of :func:`run_rank_proportion_analysis`: ``{}`` if
        that analysis is off or uses the plain true-value figure, else
        ``{value_basis: {pool_trials: {"counts": ..., "p_values": ...}}}``.

    Raises:
        ValueError: If a drifting-only analysis is selected for ``task_type="blocked"``.
        PokeAlignmentError: If parsed pokes do not match the imported trials of a session.
    """
    check_selection_for_task_type(selection, task_type)
    out = problem_dir(output_root, task, cohort, problem_number, task_type)
    if selection.run_choice_probability_analysis:
        print(f"Problem {problem_number}: choice probabilities around good reversals")
        run_choice_probability_analysis(
            subjects_trials,
            out / "reversal-stats",
            choice,
            problem_number=problem_number,
            task_type=task_type,
        )
    if selection.run_first_leave_analysis:
        print(f"Problem {problem_number}: first leave after good reversals")
        run_first_leave_analysis(
            subjects_trials,
            out / "reversal-stats",
            problem_number=problem_number,
            task_type=task_type,
        )
    if selection.run_number_of_reversals_analysis:
        print(f"Problem {problem_number}: number of reversals")
        run_number_of_reversals_analysis(
            subjects_trials, out / "reversal-stats", reversal_trial_window=reversal_trial_window
        )
    rank_results = {}
    if selection.run_rank_proportion_analysis:
        print(f"Problem {problem_number}: rank proportions")
        rank_results = run_rank_proportion_analysis(
            subjects_trials, out / "choice-stats", selection
        )
    if any(getattr(selection, name) for name in DRIFTING_ONLY_SECTIONS):
        names = [
            _SECTION_NAMES[name] for name in DRIFTING_ONLY_SECTIONS if getattr(selection, name)
        ]
        print(f"Problem {problem_number}: {', '.join(names)}")
        tables = run_drifting_value_analyses(
            {problem_number: subjects_trials},
            out,
            selection,
            arm_scope_label=f"problem-{problem_number:02d}",
            previous_choice_towers=previous_choice_towers,
        )
        if diagnostics is not None:
            diagnostics.update(tables)
    # parse once: the same counts feed the poke analysis and the single-session maze row
    parsed = None
    if selection.run_poke_analysis and selection.run_single_session_analysis:
        parsed = {m: mouse_poke_counts(m, sessions) for m, sessions in subjects_trials.items()}
    if selection.run_poke_analysis:
        print(f"Problem {problem_number}: long pokes and maze heatmaps")
        poke = run_poke_analysis(
            subjects_trials,
            out / LONG_POKES_FOLDER,
            previous_choice_towers=previous_choice_towers,
            poke_counts=parsed,
            port_finding_sessions=port_finding_sessions,
            summary_scope=" ".join(
                filter(None, (cohort, task_type, f"problem-{problem_number:02d}"))
            ),
        )
        if diagnostics is not None:
            diagnostics["poke_trial_summary"] = poke["summary"]
            diagnostics["port_finding_check"] = poke["port_finding_check"]
    if selection.run_single_session_analysis:
        print(f"Problem {problem_number}: single sessions")
        run_single_session_analysis(
            subjects_trials,
            out / "single-sessions",
            previous_choice_towers=previous_choice_towers,
            poke_counts=parsed,
        )
    return rank_results


def run_open_field_analysis(
    subjects_trials_by_problem: dict,
    *,
    port_finding_assignment: PortFindingAssignment | None = None,
    **kwargs,
) -> dict:
    """:func:`run_problem` for every problem; returns ``{problem: rank results}``.

    Args:
        subjects_trials_by_problem (dict): ``{problem: {subject: {session_key: session_dict}}}``.
            Problems run in sorted order.
        port_finding_assignment (PortFindingAssignment | None): Where the port-finding-only
            sessions of the import log go (see
            :func:`~src.behavior_analysis.port_finding_sessions.assign_port_finding_sessions`);
            each problem's poke analysis counts the ones assigned to it. Default ``None``: none.
            Keyword-only.
        **kwargs: Passed to :func:`run_problem` for every problem; ``task``, ``cohort`` and
            ``output_root`` are required.

    Returns:
        dict: ``{problem: rank_results}``, each as returned by :func:`run_problem`.
    """
    return {
        problem: run_problem(
            problem,
            trials,
            port_finding_sessions=(
                None
                if port_finding_assignment is None
                else port_finding_sessions_for_problem(port_finding_assignment, trials)
            ),
            **kwargs,
        )
        for problem, trials in sorted(subjects_trials_by_problem.items())
    }
