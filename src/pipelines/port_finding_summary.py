"""Write the port-finding summary figures (see :mod:`src.behavior_analysis.port_finding_summary`).

- :func:`write_problem_port_finding_heatmaps`: one problem of one cohort; a row of three mean
  maze heatmaps per port layout, for each poke kind. The cohort runner writes them into the
  problem's ``long-pokes/`` folder.
- :func:`write_layout_port_finding_heatmaps`: drifting problems of several cohorts pooled by
  port layout, one figure per layout and poke kind (cohorts number their problems differently, so
  the same layout is often a different problem number). The all-cohorts runner writes them into
  ``all-cohorts/port-finding-summary/``.
- :func:`write_category_fractions`: the fraction of pokes at current, previous, initiation and
  other ports in the three windows, averaged over each mouse's problems and then across mice.
  Written per cohort into ``drifting/all-drifting-problems/port-finding-summary/`` and for the
  cohorts together into ``all-cohorts/port-finding-summary/``.

Every figure has a ``.caption.txt`` that defines the windows and lists the mice left out.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
from pathlib import Path

import pandas as pd

from src.behavior_analysis.poke_counts import (
    KINDS,
    previous_choice_towers_for_problem,
    previous_problem_choice_towers,
)
from src.behavior_analysis.port_finding_sessions import (
    PortFindingAssignment,
    port_finding_sessions_for_problem,
)
from src.behavior_analysis.port_finding_summary import (
    N_FIRST_BLOCKS,
    N_FIRST_TRIALS,
    WINDOWS,
    MousePortFinding,
    category_fraction_table,
    exclusion_reasons,
    group_by_layout,
    mean_percent_map,
    mouse_mean_fractions,
    problem_port_finding_summaries,
)
from src.behavior_visualization.plot_helpers import Title, stats_line
from src.behavior_visualization.plot_port_finding_summary import (
    KIND_LABELS,
    HeatmapRow,
    plot_category_fractions,
    plot_port_finding_heatmaps,
    window_labels,
)

PORT_FINDING_SUMMARY_FOLDER = "port-finding-summary"
HEATMAP_STEM = "Port Finding Summary"
FRACTIONS_STEM = "Port Finding Summary - Port Categories"
FRACTIONS_TABLE = "port_finding_category_fractions"

WINDOW_TEXT = (
    f"Windows: port finding (every port-finding trial of the problem, port-finding-only "
    f"sessions included); the first {N_FIRST_TRIALS} task trials of the problem, over as many "
    f"sessions as it takes; the task trials of the first {N_FIRST_BLOCKS} blocks (block numbers "
    f"restart in every session, so each session starts a new block)."
)
KIND_TEXT = {
    "after_initiation": "Long pokes from initiation to the choice; leading long pokes at the "
    "initiation port are not counted.",
    "after_choice": "Port visits after the choice until the next initiation; leading visits to "
    "the chosen port are not counted, the next initiating poke is.",
}
HEATMAP_TEXT = (
    "Each mouse's pokes are turned into percentages of its own pokes in the window, then "
    "averaged across mice; each map is scaled from 0 to its largest port. Outlines: green "
    "current problem choice ports, red previous problem choice ports, blue initiation port."
)
EXCLUSION_TEXT = (
    f"A mouse is left out of a problem unless it has port-finding trials, at least "
    f"{N_FIRST_TRIALS} task trials and a complete block {N_FIRST_BLOCKS}."
)
CATEGORY_TEXT = (
    "Port categories: a port that is a choice port of both the current and the previous problem "
    "counts as current. Each mouse's fractions are averaged over its problems first; bars are the "
    "mean ± se across mice, dots the mice. Fractions are raw shares of pokes, not corrected for "
    "how many ports each category has."
)


def _excluded_lines(excluded: Mapping[Hashable, Sequence[str]]) -> list[str]:
    if not excluded:
        return ["No mouse left out."]
    return ["Left out:"] + [f"{_label(who)}: {'; '.join(why)}" for who, why in excluded.items()]


def _ports(ports: Iterable[str]) -> str:
    return " ".join(ports) or "None"


def _label(key: Hashable) -> str:
    """How a mouse is named in a caption: the mouse id, or e.g. ``"cohort-03 2 MY_66_N"``."""
    return " ".join(map(str, key)) if isinstance(key, tuple) else str(key)


def _split(entries: Mapping[Hashable, MousePortFinding], kind: str) -> tuple[dict, dict]:
    """``(usable entries, {key: reasons} of the left-out ones)`` for *kind*."""
    reasons = {key: exclusion_reasons(summary, kind) for key, summary in entries.items()}
    usable = {key: entries[key] for key, why in reasons.items() if not why}
    return usable, {key: why for key, why in reasons.items() if why}


def _heatmap_row(label: str, layout, summaries: list[MousePortFinding], kind: str) -> HeatmapRow:
    current, previous, initiation = layout
    return HeatmapRow(
        label=label,
        current_ports=current,
        previous_ports=previous,
        initiation_port=initiation,
        maps={w: mean_percent_map(summaries, w, kind) for w in WINDOWS},
        n_mice=len(summaries),
    )


def write_problem_port_finding_heatmaps(
    summaries: Mapping[str, MousePortFinding], out_dir: Path, *, scope: str
) -> list[Path]:
    """The per-layout mean heatmaps of one problem, one figure per poke kind.

    Args:
        summaries (Mapping[str, MousePortFinding]): ``{mouse: summary}`` of the problem.
        out_dir (Path): Folder for the figures. Keyword-only *scope* starts the caption.
        scope (str): E.g. ``"cohort-03 drifting problem-02"``.

    Returns:
        list[Path]: The base paths written (no extension); a kind without a usable mouse is
        skipped.
    """
    written = []
    for kind in KINDS:
        usable, excluded = _split(summaries, kind)
        if not usable:
            print(f"[INFO] {scope}: no mouse for the port-finding summary of {kind}")
            continue
        rows = [
            _heatmap_row(
                f"Current {_ports(layout[0])}\nPrev {_ports(layout[1])}",
                layout,
                [usable[m] for m in mice],
                kind,
            )
            for layout, mice in group_by_layout(usable.items()).items()
        ]
        title = Title(
            f"{HEATMAP_STEM} - {KIND_LABELS[kind]}",
            stats_line("mean", "mice", [(len(usable), "mouse", "mice")]),
            (scope, KIND_TEXT[kind], WINDOW_TEXT, HEATMAP_TEXT, EXCLUSION_TEXT,
             *_excluded_lines(excluded)),
        )  # fmt: skip
        base = Path(out_dir) / f"{HEATMAP_STEM} - {KIND_LABELS[kind]}"
        plot_port_finding_heatmaps(
            rows,
            kind,
            title=title,
            window_headings=window_labels(N_FIRST_TRIALS, N_FIRST_BLOCKS),
            save_path=base,
        )
        written.append(base)
    return written


def _layout_name(layout) -> str:
    current, previous, initiation = layout
    name = f"{_ports(current)} After {_ports(previous)}"
    return name if initiation == "B2" else f"{name} (Initiation {initiation})"


def write_layout_port_finding_heatmaps(
    entries: Mapping[tuple[str, int, str], MousePortFinding], out_dir: Path, *, scope: str
) -> list[Path]:
    """Mice of several cohorts and problems pooled by port layout: one figure per layout and kind.

    Args:
        entries (Mapping): ``{(cohort name, k, mouse): summary}``.
        out_dir (Path): Folder for the figures.
        scope (str): Starts every caption, e.g. ``"drifting problems of cohorts 1 and 3"``.
            Keyword-only.

    Returns:
        list[Path]: The base paths written (no extension).
    """
    written = []
    for kind in KINDS:
        usable, excluded = _split(entries, kind)
        for layout, keys in group_by_layout(usable.items()).items():
            cohorts = sorted({cohort for cohort, _, _ in keys})
            label = "\n".join(
                f"{cohort}: Problem "
                f"{', '.join(str(k) for k in sorted({k for c, k, _ in keys if c == cohort}))} "
                f"(n={sum(c == cohort for c, _, _ in keys)})"
                for cohort in cohorts
            )
            layout_excluded = {
                key: why for key, why in excluded.items() if entries[key].layout == layout
            }
            name = _layout_name(layout)
            title = Title(
                f"{HEATMAP_STEM} - {name} - {KIND_LABELS[kind]}",
                stats_line("mean", "mice", [(len(keys), "mouse", "mice")]),
                (scope, f"Layout {name}; {label.replace(chr(10), '; ')}", KIND_TEXT[kind],
                 WINDOW_TEXT, HEATMAP_TEXT, EXCLUSION_TEXT, *_excluded_lines(layout_excluded)),
            )  # fmt: skip
            row = _heatmap_row(label, layout, [usable[k] for k in keys], kind)
            base = Path(out_dir) / f"{HEATMAP_STEM} - {name} - {KIND_LABELS[kind]}"
            plot_port_finding_heatmaps(
                [row],
                kind,
                title=title,
                window_headings=window_labels(N_FIRST_TRIALS, N_FIRST_BLOCKS),
                save_path=base,
            )
            written.append(base)
    return written


def write_category_fractions(
    entries: Iterable[tuple[Mapping, MousePortFinding]],
    out_dir: Path,
    *,
    scope: str,
    mouse_columns: Sequence[str] = ("mouse",),
) -> pd.DataFrame:
    """The port-category fractions over problems and mice: one figure, and its table returned.

    Args:
        entries (Iterable[tuple[Mapping, MousePortFinding]]): ``(labels, summary)`` pairs, the
            labels naming the problem (and the cohort when several are pooled).
        out_dir (Path): Folder for the figure.
        scope (str): Starts the caption. Keyword-only.
        mouse_columns (Sequence[str]): Columns naming a mouse; add ``"cohort"`` when pooling
            cohorts. Default ``("mouse",)``. Keyword-only.

    Returns:
        pd.DataFrame: One row per mouse, problem, kind, window and category (see
        ``category_fraction_table``); empty, with no figure written, if no mouse is usable.
    """
    entries = list(entries)
    table = category_fraction_table(entries)
    if table.empty:
        print(f"[INFO] {scope}: no mouse for the port-category fractions")
        return table
    means = mouse_mean_fractions(table, mouse_columns)
    problems = table[[*mouse_columns, *[c for c in ("task_type", "k") if c in table]]]
    n_pairs = problems.drop_duplicates().shape[0]
    excluded = {
        (*labels.values(), s.mouse): exclusion_reasons(s, kind)
        for labels, s in entries
        for kind in KINDS
        if exclusion_reasons(s, kind)
    }
    title = Title(
        FRACTIONS_STEM,
        None,
        (scope, f"{n_pairs} mouse-problem pairs.", KIND_TEXT["after_initiation"],
         KIND_TEXT["after_choice"], WINDOW_TEXT, CATEGORY_TEXT, EXCLUSION_TEXT,
         *_excluded_lines(excluded)),
    )  # fmt: skip
    plot_category_fractions(
        means,
        title=title,
        window_headings=window_labels(N_FIRST_TRIALS, N_FIRST_BLOCKS),
        mouse_columns=mouse_columns,
        save_path=Path(out_dir) / FRACTIONS_STEM,
    )
    return table


def write_cohort_category_fractions(
    trials_by_problem: Mapping,
    grouped: Mapping[tuple[str, int], Mapping],
    assignment: PortFindingAssignment,
    out_dir: Path,
    *,
    cohort: str,
    task_type: str = "drifting",
) -> pd.DataFrame:
    """The port-category fractions over the problems of one task type of one cohort.

    Args:
        trials_by_problem (Mapping): ``{overall problem: {mouse: sessions}}`` of the cohort, for
            each mouse's previous problem.
        grouped (Mapping): ``{(task type, k): {mouse: sessions}}`` of the cohort.
        assignment (PortFindingAssignment): The cohort's port-finding-only sessions.
        out_dir (Path): Folder for the figure.
        cohort (str): Cohort name, e.g. ``"cohort-03"``. Keyword-only.
        task_type (str): The problems to pool. Default ``"drifting"``. Keyword-only.

    Returns:
        pd.DataFrame: As :func:`write_category_fractions`.
    """
    previous = previous_problem_choice_towers(trials_by_problem)
    entries = [
        ({"task_type": kind, "k": k}, summary)
        for (kind, k), subjects in grouped.items()
        if kind == task_type
        for summary in problem_port_finding_summaries(
            subjects,
            previous_choice_towers=previous_choice_towers_for_problem(subjects, previous),
            port_finding_sessions=port_finding_sessions_for_problem(assignment, subjects),
        ).values()
    ]
    return write_category_fractions(entries, out_dir, scope=f"{cohort} {task_type} problems")
