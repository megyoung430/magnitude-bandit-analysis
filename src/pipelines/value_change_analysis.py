"""Entry point: choice-probability plots aligned to changes in the true best / second value.

:func:`run_value_change_analysis` makes, for every (rank, direction, filter) configuration of
:func:`src.behavior_analysis.value_change_windows.value_change_configs`, one figure per
truncation mode, and returns the contamination diagnostic and per-bin counts as tables.

Events are found per problem and per session, then merged per mouse, so the same code serves
one problem and a pool of problems. Believed-value events are deliberately not computed: the
analysis was defined on the true values only.

Besides the regular full-window and truncated figures, a *symmetric* version keeps only the
events whose whole ``-N .. +N`` window is intact (see
:mod:`src.behavior_analysis.value_change_symmetric`).
"""

from collections.abc import Mapping
from numbers import Integral
from pathlib import Path

import pandas as pd

from src.behavior_analysis.reversal_format import check_blocks_only_sessions
from src.behavior_analysis.value_change_symmetric import (
    ANY_CAUSE_KEYS,
    EXCLUSION_KEYS,
    select_symmetric_windows,
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

DEFAULT_SYMMETRIC_WINDOW = 6


def _figure_stem(windows: ValueChangeWindows) -> str:
    """File name (without extension) of the figure of *windows*, e.g. ``Choice Probabilities
    Around Best-Value Decreases (Rank Swap, Symmetric ±6 Trials)``."""
    symmetric_window = windows.pre if windows.symmetric else None
    heading = describe_value_change(windows.config, windows.truncate_at_rank_swap, symmetric_window)
    return f"Choice Probabilities Around {heading}"


def _symmetric_mode_name(symmetric_window: int) -> str:
    """Mode name of the symmetric version in summaries and tables, e.g. ``symmetric_6``."""
    return f"symmetric_{symmetric_window}"


def _check_symmetric_window(symmetric_window: int | None) -> None:
    """Raise ValueError unless *symmetric_window* is None or an integer >= 1 (a bool is not).

    Any ``numbers.Integral`` is accepted, so a NumPy integer works.
    """
    if symmetric_window is None:
        return
    if (
        isinstance(symmetric_window, bool)
        or not isinstance(symmetric_window, Integral)
        or symmetric_window < 1
    ):
        raise ValueError(
            f"symmetric_window must be None or an integer >= 1, got {symmetric_window!r}"
        )


def _check_arguments(
    subjects_trials_by_problem: Mapping,
    pre: int,
    post: int,
    modes: tuple,
    symmetric_window: int | None,
) -> None:
    if not subjects_trials_by_problem:
        raise ValueError("no problems given: pass {problem: {subject: {session: ...}}}")
    check_blocks_only_sessions(subjects_trials_by_problem, "run_value_change_analysis")
    if not modes or not all(isinstance(m, bool) for m in modes):
        raise ValueError(f"truncate_modes must be a non-empty tuple of bools, got {modes!r}")
    check_window(pre, post)
    _check_symmetric_window(symmetric_window)


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


def _exclusion_row(config: ValueChangeConfig, symmetric_window: int, exclusions: dict) -> dict:
    """One table row: how many events the symmetric window kept and why it dropped the rest."""
    return {
        "config": config.label,
        "symmetric_window": symmetric_window,
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
    symmetric_window: int | None = DEFAULT_SYMMETRIC_WINDOW,
) -> dict:
    """Value-change-aligned choice-probability figures, contamination table and per-bin counts.

    Writes, under *out_dir*, ``Choice Probabilities Around <rank>-Value <Increases|Decreases>
    (<filter>, <mode>).png/.pdf`` for each configuration with at least one event and each mode
    (``Full Window`` or ``Truncated at Swap``), in ``out_dir / "mouse-weighted"``: mean and SE
    across mice, each mouse weighted equally. The same figures with every event weighted equally
    (mean and SE across events) go in ``out_dir / "event-weighted"``; nothing is written at the
    top level of *out_dir*. Only figures are written; the tables are returned for the caller to
    save.

    With *symmetric_window* = N the same figures are also made for the symmetric version, in the
    same two folders and named ``... (<filter>, Symmetric ±N Trials)``: windows are built with
    ``pre = post = N`` and only events whose whole ``-N .. +N`` window is intact are kept (no
    block or third-arm change, session edge, best-arm change before the event or missing
    choice), so every time point averages the same events. The x axis runs -N .. +N. There is no
    rank-swap truncation in the symmetric version: blanking post trials would break the
    symmetry. It is independent of *pre* and *post*. In its summary ``n_events`` and
    ``counts["n_events"]`` are the events kept, and ``counts["n_events_before_exclusion"]`` those
    found.

    Args:
        subjects_trials_by_problem (dict): ``{problem: {subject: {session_key: session}}}``.
            One problem or many; a mouse's events are pooled across problems.
        out_dir (str | Path): Output folder (created).
        pre (int): Trials before the value change in each window.
        post (int): Trials after the value-change trial in each window.
        truncate_modes (tuple[bool]): Which truncation modes to plot: False for full post
            windows, True for windows blanked from the first later best/second exchange.
        show_individual_mice (bool): Also draw each mouse's curve in grey.
        symmetric_window (int | None): N of the symmetric complete-window version, an integer
            >= 1; None switches it off. Default 6.

    Returns:
        dict: ``summaries`` (``{config label: {mode: {n_events, n_subjects, n_truncated,
        counts, x, mean, se, event_weighted, n_events_per_bin, n_subjects_per_bin, plotted,
        files}}}`` (``event_weighted`` is ``{mean, se}`` across events), mode
        ``untruncated``, ``truncated_at_rank_swap`` or ``symmetric_N``), ``contamination``
        (``{config label:
        contamination dict}``), ``contamination_table`` (DataFrame, one row per configuration:
        contamination of the post windows and event accounting, from the untruncated events),
        ``bin_counts_table`` (DataFrame, per configuration, mode and time point ``t``: how many
        events and mice have a trial there, so thinning is visible; the mode ``symmetric_N`` has
        the same count at every t), ``symmetric_exclusions_table`` (DataFrame, one row per
        configuration: events before, kept and excluded in total and by reason, each excluded
        event once (``n_excluded_*``), and the events having each cause, which can overlap
        (``n_with_*``); empty without a symmetric window) and ``files`` (every saved figure path).

    Raises:
        ValueError: On empty input, a counter-format session (the drifting task has only block
            changes), bad ``pre``/``post``/``truncate_modes``/``symmetric_window``, or any error
            from ``get_value_change_windows``.
    """
    _check_arguments(subjects_trials_by_problem, pre, post, truncate_modes, symmetric_window)
    symmetric_window = None if symmetric_window is None else int(symmetric_window)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
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
        if symmetric_window is not None:
            pooled_symmetric = pooled_windows(
                subjects_trials_by_problem, config, symmetric_window, symmetric_window
            )
            windows, exclusions = select_symmetric_windows(pooled_symmetric)
            summary = _mode_summary(windows, out_dir, show_individual_mice)
            mode = _symmetric_mode_name(symmetric_window)
            summaries[config.label][mode] = summary
            files += summary["files"]
            bin_rows += _bin_count_rows(config, mode, summary)
            exclusion_rows.append(_exclusion_row(config, symmetric_window, exclusions))
    return {
        "summaries": summaries,
        "contamination": contamination,
        "contamination_table": pd.DataFrame(rows),
        "bin_counts_table": pd.DataFrame(bin_rows),
        "symmetric_exclusions_table": pd.DataFrame(exclusion_rows),
        "files": files,
    }
