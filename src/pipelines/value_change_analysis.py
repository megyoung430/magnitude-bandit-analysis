"""Entry point: choice-probability plots aligned to changes in the true best / second value.

:func:`run_value_change_analysis` makes, for every (rank, direction, filter) configuration of
:func:`src.behavior_analysis.value_change_windows.value_change_configs`, one figure per
truncation mode, plus a CSV of the contamination diagnostic and event accounting.

Events are found per problem and per session, then merged per mouse, so the same code serves
one problem and a pool of problems. Believed-value events are deliberately not computed: the
analysis was defined on the true values only.
"""

from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from src.behavior_analysis.value_change_windows import (
    ValueChangeConfig,
    ValueChangeCurves,
    ValueChangeWindows,
    check_window,
    contamination_summary,
    get_value_change_windows,
    merge_value_change_windows,
    summarize_windows,
    truncate_windows_at_rank_swap,
    value_change_configs,
)
from src.behavior_visualization.plot_value_change_choice_probs import (
    plot_value_change_choice_probs,
)

CONTAMINATION_CSV = "value_change_contamination.csv"
BIN_COUNTS_CSV = "value_change_bin_counts.csv"
_FILTER_NAMES = {
    "all": "all_events",
    "rank_preserving": "rank_preserving",
    "rank_swap": "rank_swap",
}


def _mode_name(truncated: bool) -> str:
    return "truncated_at_rank_swap" if truncated else "untruncated"


def _figure_stem(config: ValueChangeConfig, truncated: bool) -> str:
    filter_name = _FILTER_NAMES[config.rank_filter]
    return f"value_change_{config.rank}_{config.direction}_{filter_name}_{_mode_name(truncated)}"


def _check_arguments(
    subjects_trials_by_problem: Mapping, pre: int, post: int, modes: tuple
) -> None:
    if not subjects_trials_by_problem:
        raise ValueError("no problems given: pass {problem: {subject: {session: ...}}}")
    if not modes or not all(isinstance(m, bool) for m in modes):
        raise ValueError(f"truncate_modes must be a non-empty tuple of bools, got {modes!r}")
    check_window(pre, post)


def _pooled_windows(
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
    """Save the figure as ``<base>.png`` and ``.pdf``; returns the two paths."""
    plot_value_change_choice_probs(curves, base, show_individual_mice=show_individual_mice)
    return [base.with_name(f"{base.name}.png"), base.with_name(f"{base.name}.pdf")]


def _mode_summary(windows: ValueChangeWindows, out_dir: Path, show_individual_mice: bool) -> dict:
    """Aggregate, summarise and (if there are events) plot one configuration and mode."""
    curves = summarize_windows(windows)
    summary = _curve_summary(curves)
    if curves.num_events > 0:
        base = out_dir / _figure_stem(windows.config, windows.truncate_at_rank_swap)
        summary["files"] = _plot_curves(curves, base, show_individual_mice)
        summary["plotted"] = True
    return summary


def _contamination_row(windows: ValueChangeWindows) -> dict:
    """One CSV row: the configuration, its contamination summary and its event accounting."""
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


def run_value_change_analysis(
    subjects_trials_by_problem: Mapping[object, Mapping[str, Mapping[str, Mapping]]],
    out_dir: str | Path,
    *,
    pre: int = 5,
    post: int = 15,
    truncate_modes: tuple[bool, ...] = (False, True),
    show_individual_mice: bool = False,
) -> dict:
    """Value-change-aligned choice-probability figures, contamination CSV and per-bin counts.

    Writes, under *out_dir*, ``value_change_<rank>_<direction>_<filter>_<mode>.png/.pdf`` for
    each configuration with at least one event and each mode (``untruncated`` or
    ``truncated_at_rank_swap``), and ``value_change_contamination.csv`` (one row per
    configuration: contamination of the post windows and event accounting, from the
    untruncated events) and ``value_change_bin_counts.csv`` (per configuration, mode and
    time point t: how many events and mice have a trial there, so thinning is visible).

    Args:
        subjects_trials_by_problem (dict): ``{problem: {subject: {session_key: session}}}``.
            One problem or many; a mouse's events are pooled across problems.
        out_dir (str | Path): Output folder (created).
        pre (int): Trials before the value change in each window.
        post (int): Trials after the value-change trial in each window.
        truncate_modes (tuple[bool]): Which truncation modes to plot: False for full post
            windows, True for windows blanked from the first later best/second exchange.
        show_individual_mice (bool): Also draw each mouse's curve in grey.

    Returns:
        dict: ``summaries`` (``{config label: {mode: {n_events, n_subjects, n_truncated,
        counts, x, mean, se, n_events_per_bin, n_subjects_per_bin, plotted, files}}}``),
        ``contamination`` (``{config label: contamination dict}``), ``contamination_csv`` and
        ``bin_counts_csv`` (Paths) and ``files`` (every saved figure path).

    Raises:
        ValueError: On empty input, bad ``pre``/``post``/``truncate_modes``, or any error
            from ``get_value_change_windows``.
    """
    _check_arguments(subjects_trials_by_problem, pre, post, truncate_modes)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries, contamination, rows, bin_rows, files = {}, {}, [], [], []
    for config in value_change_configs():
        pooled = _pooled_windows(subjects_trials_by_problem, config, pre, post)
        rows.append(_contamination_row(pooled))
        contamination[config.label] = contamination_summary(pooled.events)
        summaries[config.label] = {}
        for truncated in truncate_modes:
            windows = truncate_windows_at_rank_swap(pooled) if truncated else pooled
            summary = _mode_summary(windows, out_dir, show_individual_mice)
            summaries[config.label][_mode_name(truncated)] = summary
            files += summary["files"]
            bin_rows += _bin_count_rows(config, _mode_name(truncated), summary)
    csv_path = out_dir / CONTAMINATION_CSV
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    bin_path = out_dir / BIN_COUNTS_CSV
    pd.DataFrame(bin_rows).to_csv(bin_path, index=False)
    return {
        "summaries": summaries,
        "contamination": contamination,
        "contamination_csv": csv_path,
        "bin_counts_csv": bin_path,
        "files": files,
    }
