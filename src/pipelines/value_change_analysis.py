"""Entry point: choice-probability plots aligned to changes in the true best / second value.

:func:`run_value_change_analysis` makes, for every (rank, direction, filter) configuration of
:func:`src.behavior_analysis.value_change_windows.value_change_configs`, one figure per
truncation mode, and returns the contamination diagnostic and per-bin counts as tables.

Events are found per problem and per session, then merged per mouse, so the same code serves
one problem and a pool of problems. Believed-value events are deliberately not computed: the
analysis was defined on the true values only.

Besides the regular full-window and truncated figures, a *complete-window* version keeps only
the events whose whole ``-pre .. +post`` window is intact (see
:mod:`src.behavior_analysis.value_change_complete_window`); :data:`COMPLETE_WINDOW` sets its
default ``(pre, post)``.
"""

import re
from collections.abc import Mapping
from numbers import Integral
from pathlib import Path

import pandas as pd

from src.behavior_analysis.reversal_format import check_blocks_only_sessions
from src.behavior_analysis.value_change_complete_window import (
    ANY_CAUSE_KEYS,
    EXCLUSION_KEYS,
    select_complete_windows,
)
from src.behavior_analysis.value_change_windows import (
    DEFAULT_POST,
    DEFAULT_PRE,
    ValueChangeConfig,
    ValueChangeCurves,
    ValueChangeWindows,
    check_window,
    contamination_summary,
    describe_value_change,
    get_value_change_windows,
    merge_value_change_windows,
    mode_name,
    summarize_windows,
    truncate_windows_at_rank_swap,
    value_change_configs,
)
from src.behavior_visualization.plot_helpers import figure_paths
from src.behavior_visualization.plot_value_change_choice_probs import (
    plot_value_change_choice_probs,
)
from src.pipelines.weighting_folders import EVENT_WEIGHTED_FOLDER, MOUSE_WEIGHTED_FOLDER

# (pre, post) of the complete-window figures: trials before and after the value change.
COMPLETE_WINDOW = (2, 4)

# A figure name of any complete window, as written by ``complete_window_text``.
_COMPLETE_WINDOW_NAME = re.compile(r", (Symmetric ±\d+ Trials|-\d+ to \+\d+ Trials)\)\.")


def _figure_stem(windows: ValueChangeWindows) -> str:
    """File name (without extension) of the figure of *windows*, e.g. ``Choice Probabilities
    Around Best-Value Decreases (Rank Swap, -2 to +4 Trials)``."""
    complete_window = (windows.pre, windows.post) if windows.complete_window else None
    heading = describe_value_change(windows.config, windows.truncate_at_rank_swap, complete_window)
    return f"Choice Probabilities Around {heading}"


def complete_window_mode_name(pre: int, post: int) -> str:
    """Mode name of the complete-window version in summaries and tables.

    Equal sides keep the name ``symmetric_N``; others are e.g. ``complete_pre2_post4``.
    """
    return f"symmetric_{pre}" if pre == post else f"complete_pre{pre}_post{post}"


def _is_window_size(value) -> bool:
    """Whether *value* is an integer >= 1; a bool is not, any ``numbers.Integral`` is."""
    return not isinstance(value, bool) and isinstance(value, Integral) and value >= 1


def _check_complete_window(complete_window) -> None:
    """Raise ValueError unless *complete_window* is None or a tuple ``(pre, post)`` of integers
    >= 1 (NumPy integers included)."""
    if complete_window is None:
        return
    if not (
        isinstance(complete_window, tuple)
        and len(complete_window) == 2
        and all(map(_is_window_size, complete_window))
    ):
        raise ValueError(
            "complete_window must be None or a tuple (pre, post) of integers >= 1, "
            f"got {complete_window!r}"
        )


def _remove_stale_complete_window_figures(out_dir: Path) -> None:
    """Delete the complete-window figures an earlier run left in the two weighting folders.

    The complete-window size is a setting, so a run with another one would otherwise leave the
    old figures next to the new ones, looking current. The figures of this run are written
    afterwards; nothing else in the folders is touched.
    """
    for folder in (MOUSE_WEIGHTED_FOLDER, EVENT_WEIGHTED_FOLDER):
        directory = out_dir / folder
        if directory.is_dir():
            for path in directory.iterdir():
                if path.is_file() and _COMPLETE_WINDOW_NAME.search(path.name):
                    path.unlink()


def _check_arguments(
    subjects_trials_by_problem: Mapping,
    pre: int,
    post: int,
    modes: tuple,
    complete_window: tuple[int, int] | None,
) -> None:
    if not subjects_trials_by_problem:
        raise ValueError("no problems given: pass {problem: {subject: {session: ...}}}")
    check_blocks_only_sessions(subjects_trials_by_problem, "run_value_change_analysis")
    if not modes or not all(isinstance(m, bool) for m in modes):
        raise ValueError(f"truncate_modes must be a non-empty tuple of bools, got {modes!r}")
    check_window(pre, post)
    _check_complete_window(complete_window)


def pooled_windows(
    subjects_trials_by_problem: Mapping, config: ValueChangeConfig, pre: int, post: int
) -> ValueChangeWindows:
    """Untruncated events of *config*, found per problem and merged per mouse."""
    per_problem = [
        get_value_change_windows(
            subjects_trials_by_problem[problem],
            config.rank,
            config.direction,
            config.rank_filter,
            pre=pre,
            post=post,
        )
        for problem in sorted(subjects_trials_by_problem)
    ]
    return merge_value_change_windows(per_problem)


def _curve_summary(curves: ValueChangeCurves) -> dict:
    return {
        "n_events": curves.num_events,
        "n_subjects": curves.num_subjects,
        "n_truncated": curves.num_truncated,
        "n_events_per_bin": curves.n_events_per_bin,
        "n_subjects_per_bin": curves.n_subjects_per_bin,
        "counts": curves.counts,
        "x": curves.x,
        "mean": curves.mean,
        "se": curves.se,
        "plotted": False,
        "files": [],
    }


def _plot_curves(curves: ValueChangeCurves, base: Path, show_individual_mice: bool) -> list[Path]:
    """Save the figure as ``<base>.<format>``; returns the figure paths."""
    plot_value_change_choice_probs(curves, base, show_individual_mice=show_individual_mice)
    return figure_paths(base)


def _mode_summary(windows: ValueChangeWindows, out_dir: Path, show_individual_mice: bool) -> dict:
    """Aggregate, summarise and (if there are events) plot one configuration and mode.

    The mouse-weighted figure goes in ``out_dir / MOUSE_WEIGHTED_FOLDER``; the event-weighted
    one, with the same name, in ``out_dir / EVENT_WEIGHTED_FOLDER``. Its mean and SE are under
    ``event_weighted``.
    """
    curves = summarize_windows(windows)
    by_event = summarize_windows(windows, weighting="event")
    summary = _curve_summary(curves)
    summary["event_weighted"] = {"mean": by_event.mean, "se": by_event.se}
    if curves.num_events > 0:
        stem = _figure_stem(windows)
        summary["files"] = _plot_curves(
            curves, out_dir / MOUSE_WEIGHTED_FOLDER / stem, show_individual_mice
        )
        summary["files"] += _plot_curves(
            by_event, out_dir / EVENT_WEIGHTED_FOLDER / stem, show_individual_mice
        )
        summary["plotted"] = True
    return summary


def _contamination_row(windows: ValueChangeWindows) -> dict:
    """One table row: the configuration, its contamination summary and its event accounting."""
    config = windows.config
    contamination = contamination_summary(windows.events)
    if contamination["n_events"] != windows.counts["n_events"]:
        raise ValueError(
            f"{config.label}: {contamination['n_events']} events summarised but "
            f"{windows.counts['n_events']} counted"
        )
    return {
        "config": config.label,
        "rank": config.rank,
        "direction": config.direction,
        "rank_filter": config.rank_filter,
        **contamination,
        **windows.counts,
    }


def _bin_count_rows(config: ValueChangeConfig, mode: str, summary: dict) -> list[dict]:
    """One row per time point: how many events and mice have a trial there."""
    return [
        {
            "config": config.label,
            "mode": mode,
            "t": int(t),
            "n_events": int(n_events),
            "n_subjects": int(n_subjects),
        }
        for t, n_events, n_subjects in zip(
            summary["x"], summary["n_events_per_bin"], summary["n_subjects_per_bin"], strict=True
        )
    ]


def _exclusion_row(config: ValueChangeConfig, pre: int, post: int, exclusions: dict) -> dict:
    """One table row: how many events the complete window kept and why it dropped the rest."""
    return {
        "config": config.label,
        "pre": pre,
        "post": post,
        "n_events_before": exclusions["n_events_before"],
        "n_events_kept": exclusions["n_events_kept"],
        "n_excluded": exclusions["n_events_before"] - exclusions["n_events_kept"],
        **{key: exclusions[key] for key in EXCLUSION_KEYS + ANY_CAUSE_KEYS},
    }


def run_value_change_analysis(
    subjects_trials_by_problem: Mapping[object, Mapping[str, Mapping[str, Mapping]]],
    out_dir: str | Path,
    *,
    pre: int = DEFAULT_PRE,
    post: int = DEFAULT_POST,
    truncate_modes: tuple[bool, ...] = (False, True),
    show_individual_mice: bool = False,
    complete_window: tuple[int, int] | None = COMPLETE_WINDOW,
) -> dict:
    """Value-change-aligned choice-probability figures, contamination table and per-bin counts.

    Writes, under *out_dir*, ``Choice Probabilities Around <rank>-Value <Increases|Decreases>
    (<filter>, <mode>).png/.pdf`` for each configuration with at least one event and each mode
    (``Full Window`` or ``Truncated at Swap``), in ``out_dir / "mouse-weighted"``: mean and SE
    across mice, each mouse weighted equally. The same figures with every event weighted equally
    (mean and SE across events) go in ``out_dir / "event-weighted"``; nothing is written at the
    top level of *out_dir*. Only figures are written; the tables are returned for the caller to
    save.

    With *complete_window* = ``(a, b)`` the same figures are also made for the complete-window
    version, in the same two folders and named ``... (<filter>, -a to +b Trials)`` (``...
    (<filter>, Symmetric ±a Trials)`` when ``a == b``): windows are built with ``pre = a``,
    ``post = b`` and only events whose whole ``-a .. +b`` window is intact are kept (no block or
    third-arm change, session edge, best-arm change before the event or missing choice), so every
    time point averages the same events. The x axis runs -a .. +b. There is no rank-swap
    truncation in the complete-window version: blanking post trials would make the later time
    points average fewer events again. It is independent of *pre* and *post*. In its summary
    ``n_events`` and ``counts["n_events"]`` are the events kept, and
    ``counts["n_events_before_exclusion"]`` those found. Complete-window figures of any size
    left in the two folders by an earlier run are deleted first, so a changed size leaves no
    stale figure.

    Args:
        subjects_trials_by_problem (dict): ``{problem: {subject: {session_key: session}}}``.
            One problem or many; a mouse's events are pooled across problems.
        out_dir (str | Path): Output folder (created).
        pre (int): Trials before the value change in each window.
        post (int): Trials after the value-change trial in each window.
        truncate_modes (tuple[bool]): Which truncation modes to plot: False for full post
            windows, True for windows blanked from the first later best/second exchange.
        show_individual_mice (bool): Also draw each mouse's curve in grey.
        complete_window (tuple[int, int] | None): ``(pre, post)`` of the complete-window
            version, integers >= 1; None switches it off. Default :data:`COMPLETE_WINDOW`,
            ``(2, 4)``.

    Returns:
        dict: ``summaries`` (``{config label: {mode: {n_events, n_subjects, n_truncated,
        counts, x, mean, se, event_weighted, n_events_per_bin, n_subjects_per_bin, plotted,
        files}}}`` (``event_weighted`` is ``{mean, se}`` across events), mode
        ``untruncated``, ``truncated_at_rank_swap`` or the complete-window mode
        ``complete_pre<a>_post<b>`` (``symmetric_<a>`` when ``a == b``)), ``contamination``
        (``{config label:
        contamination dict}``), ``contamination_table`` (DataFrame, one row per configuration:
        contamination of the post windows and event accounting, from the untruncated events),
        ``bin_counts_table`` (DataFrame, per configuration, mode and time point ``t``: how many
        events and mice have a trial there, so thinning is visible; the complete-window mode has
        the same count at every t), ``symmetric_exclusions_table`` (DataFrame, one row per
        configuration: the window's ``pre`` and ``post``, events before, kept and excluded in
        total and by reason, each excluded event once (``n_excluded_*``), and the events having
        each cause, which can overlap (``n_with_*``); empty without a complete window; the name
        is kept from when the window was always symmetric, so the CSV keeps its name) and
        ``files`` (every saved figure path).

    Raises:
        ValueError: On empty input, a counter-format session (the drifting task has only block
            changes), bad ``pre``/``post``/``truncate_modes``/``complete_window``, or any error
            from ``get_value_change_windows``.
    """
    _check_arguments(subjects_trials_by_problem, pre, post, truncate_modes, complete_window)
    if complete_window is not None:
        complete_window = tuple(map(int, complete_window))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _remove_stale_complete_window_figures(out_dir)
    summaries, contamination, rows, bin_rows, exclusion_rows, files = {}, {}, [], [], [], []
    for config in value_change_configs():
        pooled = pooled_windows(subjects_trials_by_problem, config, pre, post)
        rows.append(_contamination_row(pooled))
        contamination[config.label] = contamination_summary(pooled.events)
        summaries[config.label] = {}
        for truncated in truncate_modes:
            windows = truncate_windows_at_rank_swap(pooled) if truncated else pooled
            summary = _mode_summary(windows, out_dir, show_individual_mice)
            summaries[config.label][mode_name(truncated)] = summary
            files += summary["files"]
            bin_rows += _bin_count_rows(config, mode_name(truncated), summary)
        if complete_window is not None:
            windows, exclusions = select_complete_windows(
                pooled_windows(subjects_trials_by_problem, config, *complete_window)
            )
            summary = _mode_summary(windows, out_dir, show_individual_mice)
            mode = complete_window_mode_name(*complete_window)
            summaries[config.label][mode] = summary
            files += summary["files"]
            bin_rows += _bin_count_rows(config, mode, summary)
            exclusion_rows.append(_exclusion_row(config, *complete_window, exclusions))
    return {
        "summaries": summaries,
        "contamination": contamination,
        "contamination_table": pd.DataFrame(rows),
        "bin_counts_table": pd.DataFrame(bin_rows),
        "symmetric_exclusions_table": pd.DataFrame(exclusion_rows),
        "files": files,
    }
