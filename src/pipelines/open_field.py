"""Standard open-field bandit figures for a cohort, run problem by problem.

Scripts and notebooks call :func:`run_open_field_analysis` (all problems) or
:func:`run_problem` (one problem) with settings objects, instead of editing or
exec-ing a script's source. Every figure is saved under
``<output_root>/<task>/<cohort>/problem-NN/``.
"""

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from src.behavior_analysis.get_choice_probs_around_good_reversals import (
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
from src.behavior_analysis.get_good_reversal_info import get_good_reversal_info
from src.behavior_analysis.get_rank_counts_by_good_reversal import (
    get_rank_counts_by_good_reversal,
    get_rank_counts_by_value,
)
from src.behavior_analysis.get_total_reversals import get_reversal_indices_by_kind
from src.behavior_analysis.merge_across_problems import merge_windows_by_subject
from src.behavior_analysis.poke_counts import mouse_poke_counts
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
)
LONG_POKES_FOLDER = "long-pokes"
VALUE_PSYCHOMETRICS_FOLDER = "value-psychometrics"
REVERSAL_ARMS_FOLDER = "reversal-aligned-arms"
VALUE_CHANGE_FOLDER = "value-change"
# One window size for building the reversal windows and for the entry point that checks it.
REVERSAL_ARMS_PRE = 10
REVERSAL_ARMS_POST = 30


class ChoiceProbabilitySettings(BaseModel):
    """Which choice-probability-around-good-reversal figures to make, and their windows."""

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
    ``run_value_change_analysis``) are for drifting problems only; :func:`run_problem` raises if
    one is on for a blocked problem. All four are off unless asked for.
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
    rank_value_bases: tuple[str, ...] | None = None
    rank_pooling_modes: tuple[bool, ...] = (True, False)


DEFAULT_SELECTION = AnalysisSelection()
DEFAULT_CHOICE = ChoiceProbabilitySettings()


def check_selection_for_task_type(selection: AnalysisSelection, task_type: str | None) -> None:
    """Raise ``ValueError`` if a drifting-only analysis is selected for a blocked problem."""
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
    """``<output_root>/<task>/<cohort>[/<task_type>]/problem-NN``."""
    base = Path(output_root) / task / cohort
    if task_type is not None:
        base = base / task_type
    return base / f"problem-{problem_number:02d}"


# ---------------------------------------------------------------------------
# Choice probabilities around good reversals
# ---------------------------------------------------------------------------


def _choice_probs(windows, s: ChoiceProbabilitySettings, *, skip=False, smooth=False):
    extra = {"skip_n_trials_after_reversal": s.skip_n_trials_after_reversal} if skip else {}
    x, per_subject, across = get_choice_probs_around_good_reversals(
        windows, pre=s.pre, post=s.post, **extra
    )
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


def _plot_all_reversals(subjects_trials, out_dir: Path, s: ChoiceProbabilitySettings):
    """Raw choice probabilities: all good reversals, then the first two and the later ones."""
    windows, idx = _windows_and_indices(subjects_trials, s)
    early, late = split_good_reversals_early_late(windows, first_n=2)
    for suffix, w in [("", windows), (" (Early)", early), (" (Late)", late)]:
        _plot_with_reversal_axis(
            *_choice_probs(w, s), w, idx, out_dir / f"{_CHOICE_PROBS_NAME}{suffix}", only_good=True
        )


def _plot_skipping_initial_trials(subjects_trials, out_dir: Path, s: ChoiceProbabilitySettings):
    """Choice probabilities that skip the first trials after each reversal."""
    windows, _ = _windows_and_indices(subjects_trials, s)
    early, late = split_good_reversals_early_late(windows, first_n=2)
    for suffix, w in [("", windows), (" (Early)", early), (" (Late)", late)]:
        x, across = _choice_probs(w, s, skip=True)
        plot_choice_probs_around_good_reversals(
            x,
            across,
            add_cumulative_axis=False,
            skip_n_trials_after_reversal=s.skip_n_trials_after_reversal,
            save_path=out_dir / f"{_CHOICE_PROBS_NAME}, Skipping Initial Trials{suffix}",
        )


def _plot_moving_average(subjects_trials, out_dir: Path, s: ChoiceProbabilitySettings):
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
        _plot_with_reversal_axis(
            *_choice_probs(w, s, smooth=True),
            w,
            idx,
            out_dir / f"{_CHOICE_PROBS_NAME} (Moving Average){suffix}",
            only_good=True,
        )


def _plot_remove_bad(subjects_trials, out_dir: Path, s: ChoiceProbabilitySettings):
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
        _plot_with_reversal_axis(
            *_choice_probs(w, s, smooth=True),
            w,
            idx,
            out_dir / f"{_CHOICE_PROBS_NAME} (Remove Bad){suffix}",
            only_good=False,
        )


def run_choice_probability_analysis(subjects_trials, out_dir: Path, s: ChoiceProbabilitySettings):
    """All enabled choice-probability figures for one problem, saved in *out_dir*."""
    if s.run_all:
        _plot_all_reversals(subjects_trials, out_dir, s)
    if s.skip:
        _plot_skipping_initial_trials(subjects_trials, out_dir, s)
    if s.moving_avg:
        _plot_moving_average(subjects_trials, out_dir, s)
    if s.remove_bad:
        _plot_remove_bad(subjects_trials, out_dir, s)
    if s.split_by_best_change and s.moving_avg:
        _run_best_change(subjects_trials, out_dir, s)


def _run_best_change(subjects_trials, out_dir: Path, s: ChoiceProbabilitySettings):
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
        _plot_with_reversal_axis(
            *_choice_probs(w, s, smooth=True), w, idx, out_dir / f"{name}{suffix}", only_good=True
        )

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
        _plot_with_reversal_axis(
            *_choice_probs(w, s, smooth=True), w, idx, out_dir / f"{name}{suffix}", only_good=False
        )


# ---------------------------------------------------------------------------
# Other sections
# ---------------------------------------------------------------------------


def run_first_leave_analysis(subjects_trials, out_dir: Path):
    """First arm left after good reversals: all, first two, and later reversals."""
    windows = get_good_reversal_info(subjects_trials, include_first_block=False)
    early, late = split_good_reversals_early_late(windows, first_n=2)
    for suffix, w in [("", windows), (" (Early)", early), (" (Late)", late)]:
        per_subject = get_first_leave_after_good_reversals(w)
        mean, se, _ = average_first_leave_across_subjects(per_subject)
        p_value = pvalue_paired_t_new_vs_third(per_subject, alternative="greater")
        plot_first_leave_after_good_reversals(
            mean,
            se,
            per_subject,
            p_value,
            save_path=out_dir / f"First Leave After Good Reversals{suffix}",
        )


def run_number_of_reversals_analysis(subjects_trials, out_dir: Path, *, reversal_trial_window=None):
    """Total, cumulative and moving-average reversal counts."""
    plot_num_reversals(subjects_trials, save_path=out_dir / "Total Reversals")
    plot_num_reversals_over_time(
        subjects_trials, threshold=10, save_path=out_dir / "Cumulative Reversals Over Time"
    )
    extra = {} if reversal_trial_window is None else {"trial_window": reversal_trial_window}
    plot_moving_avg_reversals_over_time(
        subjects_trials, **extra, save_path=out_dir / "Moving Average Reversals Over Time"
    )


def run_rank_proportion_analysis(subjects_trials, out_dir: Path, selection: AnalysisSelection):
    """Rank proportions after good reversals; returns counts and p-values when by value basis."""
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
        previous_choice_towers: ``{mouse: ports of the mouse's previous problem, or None}`` for
            the red outlines of the maze row; ``None`` draws none.
        poke_counts: ``{mouse: {session: SessionPokeCounts}}`` already parsed, so the raw events
            are read once; ``None`` lets each figure parse its own session.
    """
    for subject, sessions in subjects_trials.items():
        for session, session_data in sessions.items():
            if not session_data["trial_info"]:
                print(f"Skipping {subject} | {session}. There are no trials.")
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
    """
    per_problem = {
        problem: build_windows_with_magnitudes(
            trials, pre=REVERSAL_ARMS_PRE, post=REVERSAL_ARMS_POST
        )
        for problem, trials in problems.items()
    }
    return {"pooled": merge_windows_by_subject(per_problem)}


def _arm_scope_label(problems: dict) -> str:
    """Title text for the arm figures: the problem's name for one problem, else the pool's."""
    if len(problems) == 1:
        (problem,) = problems
        return f"problem-{problem:02d}"
    return "all drifting problems"


def run_drifting_value_analyses(problems: dict, out_dir: Path, selection: AnalysisSelection):
    """The selected drifting-only analyses over *problems*, each in a subfolder of *out_dir*.

    Args:
        problems: ``{problem: {mouse: {session_key: session_dict}}}``: one problem for a
            problem folder, several for the pool over a mouse's drifting problems. Every
            analysis computes inside each problem and then combines the mice's results, so
            beliefs reset and windows end at each problem boundary.
        out_dir: Folder that receives ``value-psychometrics``, ``reversal-aligned-arms`` and
            ``value-change``.
        selection: Which of the three to run.
    """
    out_dir = Path(out_dir)
    if selection.run_value_psychometric_analysis:
        run_value_psychometric_analysis(problems, out_dir / VALUE_PSYCHOMETRICS_FOLDER)
    if selection.run_reversal_aligned_arm_analysis:
        run_reversal_aligned_arm_analysis(
            pooled_arm_windows(problems),
            out_dir / REVERSAL_ARMS_FOLDER,
            pre=REVERSAL_ARMS_PRE,
            post=REVERSAL_ARMS_POST,
            scope_label=_arm_scope_label(problems),
        )
    if selection.run_value_change_analysis:
        run_value_change_analysis(problems, out_dir / VALUE_CHANGE_FOLDER)


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
) -> dict:
    """Run the selected analyses for one problem; returns rank-proportion results.

    The selection alone decides which analyses run; the caller picks it per task type.
    With *task_type*, figures go under ``<cohort>/<task_type>/problem-NN``.
    *previous_choice_towers* is ``{mouse: ports of the mouse's previous problem, or None}``
    (see ``previous_choice_towers_for_problem``) for the red outlines of the maze heatmaps; the
    poke analysis (``long-pokes`` folder) and the single-session figures share one parse of the
    raw events.

    Raises:
        ValueError: If a drifting-only analysis is selected for ``task_type="blocked"``.
        PokeAlignmentError: If parsed pokes do not match the imported trials of a session.
    """
    check_selection_for_task_type(selection, task_type)
    out = problem_dir(output_root, task, cohort, problem_number, task_type)
    if selection.run_choice_probability_analysis:
        print(f"Problem {problem_number}: choice probabilities around good reversals")
        run_choice_probability_analysis(subjects_trials, out / "reversal-stats", choice)
    if selection.run_first_leave_analysis:
        print(f"Problem {problem_number}: first leave after good reversals")
        run_first_leave_analysis(subjects_trials, out / "reversal-stats")
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
        print(f"Problem {problem_number}: value psychometrics and reversal-aligned analyses")
        run_drifting_value_analyses({problem_number: subjects_trials}, out, selection)
    # parse once: the same counts feed the poke analysis and the single-session maze row
    parsed = None
    if selection.run_poke_analysis and selection.run_single_session_analysis:
        parsed = {m: mouse_poke_counts(m, sessions) for m, sessions in subjects_trials.items()}
    if selection.run_poke_analysis:
        print(f"Problem {problem_number}: long pokes and maze heatmaps")
        run_poke_analysis(
            subjects_trials,
            out / LONG_POKES_FOLDER,
            previous_choice_towers=previous_choice_towers,
            poke_counts=parsed,
        )
    if selection.run_single_session_analysis:
        print(f"Problem {problem_number}: single sessions")
        run_single_session_analysis(
            subjects_trials,
            out / "single-sessions",
            previous_choice_towers=previous_choice_towers,
            poke_counts=parsed,
        )
    return rank_results


def run_open_field_analysis(subjects_trials_by_problem: dict, **kwargs) -> dict:
    """:func:`run_problem` for every problem; returns ``{problem: rank results}``."""
    return {
        problem: run_problem(problem, trials, **kwargs)
        for problem, trials in sorted(subjects_trials_by_problem.items())
    }
