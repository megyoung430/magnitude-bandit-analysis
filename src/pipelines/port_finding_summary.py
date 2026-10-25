"""Write the port-finding summary figures (see :mod:`src.behavior_analysis.port_finding_summary`).

- :func:`write_problem_port_finding_heatmaps`: one problem of one cohort; a row of three mean
  maze heatmaps per port layout, for each poke kind. The cohort runner writes them into the
  problem's ``long-pokes/`` folder.
- :func:`write_layout_port_finding_heatmaps`: drifting problems of several cohorts pooled by
  port layout, one figure per layout and poke kind (cohorts number their problems differently, so
  the same layout is often a different problem number). A mouse that ran one layout in several
  problems is averaged over them first. The all-cohorts runner writes them into
  ``all-cohorts/port-finding-summary/``.
- :func:`write_category_fractions`: the fraction of pokes at current, previous, initiation and
  other ports in the three windows, averaged over each mouse's problems and then across mice.
  Written per cohort into ``drifting/all-drifting-problems/port-finding-summary/`` and for the
  cohorts together into ``all-cohorts/port-finding-summary/``.

Every figure has a ``.caption.txt`` that defines the windows and lists the mice left out of it,
and every folder gets :data:`LEFT_OUT_FILE`, listing all mice left out of any figure written
there (also those of a layout or kind with no figure at all). A figure that a rerun no longer
draws is removed, so no stale figure stays behind.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
from pathlib import Path

import pandas as pd

from src.behavior_analysis.poke_counts import KINDS
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
)
from src.behavior_import.problem_numbering import DEFAULT_INITIATION_TOWER
from src.behavior_visualization.plot_helpers import FIGURE_FORMATS, Title, caption_path, stats_line
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
LEFT_OUT_FILE = "Port Finding Summary - Left Out.txt"

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
    f"{N_FIRST_TRIALS} task trials and a complete block {N_FIRST_BLOCKS} (block "
    f"{N_FIRST_BLOCKS + 1} has started)."
)
CATEGORY_TEXT = (
    "Port categories: a port that is a choice port of both the current and the previous problem "
    "counts as current. Each mouse's fractions are averaged over its problems first; bars are the "
    "mean ± se across mice, dots the mice. A mouse's first problem has no previous ports, so it "
    "adds nothing to the previous-port bars (a mouse with only first problems is not in them). "
    "Fractions are raw shares of pokes, not corrected for how many ports each category has."
)


# ---------------------------------------------------------------------------- helpers


def _label(key: Hashable) -> str:
    """How a mouse is named in a caption: the mouse id, or e.g. ``"cohort-03 2 MY_66_N"``."""
    return " ".join(map(str, key)) if isinstance(key, tuple) else str(key)


def _split(entries: Mapping[Hashable, MousePortFinding], kind: str) -> tuple[dict, dict]:
    """``(usable entries, {key: reasons} of the left-out ones)`` for *kind*."""
    reasons = {key: exclusion_reasons(summary, kind) for key, summary in entries.items()}
    usable = {key: entries[key] for key, why in reasons.items() if not why}
    return usable, {key: why for key, why in reasons.items() if why}


def _excluded_lines(excluded: Mapping[Hashable, Sequence[str]]) -> list[str]:
    if not excluded:
        return ["No mouse left out."]
    return ["Left out:"] + [f"{_label(who)}: {'; '.join(why)}" for who, why in excluded.items()]


def _left_out_text(scope: str, by_kind: Mapping[str, Mapping[Hashable, Sequence[str]]]) -> str:
    lines = [scope, EXCLUSION_TEXT]
    for kind, excluded in by_kind.items():
        lines += ["", f"{KIND_LABELS[kind]}:", *_excluded_lines(excluded)]
    return "\n".join(lines) + "\n"


def _write_left_out(out_dir: Path, scope: str, by_kind: Mapping) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / LEFT_OUT_FILE).write_text(_left_out_text(scope, by_kind))


def remove_figure(base: Path) -> None:
    """Delete the files of the figure saved as ``<base>.<format>`` and its caption, if any."""
    for path in [Path(f"{base}.{fmt}") for fmt in FIGURE_FORMATS] + [caption_path(base)]:
        path.unlink(missing_ok=True)


def _ports(ports: Iterable[str]) -> str:
    return " ".join(ports) or "None"


def _layout_name(layout) -> str:
    current, previous, initiation = layout
    name = f"{_ports(current)} After {_ports(previous)}"
    return name if initiation == DEFAULT_INITIATION_TOWER else f"{name} (Initiation {initiation})"


def _heatmap_row(label: str, layout, summaries, kind: str, mice=None) -> HeatmapRow:
    current, previous, initiation = layout
    return HeatmapRow(
        label=label,
        current_ports=current,
        previous_ports=previous,
        initiation_port=initiation,
        maps={w: mean_percent_map(summaries, w, kind, mice) for w in WINDOWS},
        n_mice=len(set(mice)) if mice is not None else len(summaries),
    )


# ---------------------------------------------------------------------------- one problem


def write_problem_port_finding_heatmaps(
    summaries: Mapping[str, MousePortFinding], out_dir: Path, *, scope: str
) -> list[Path]:
    """The per-layout mean heatmaps of one problem, one figure per poke kind.

    A kind without a usable mouse gets no figure (an earlier one is removed); the mice left out
    are listed in :data:`LEFT_OUT_FILE` either way.

    Args:
        summaries (Mapping[str, MousePortFinding]): ``{mouse: summary}`` of the problem.
        out_dir (Path): Folder for the figures.
        scope (str): Starts every caption, e.g. ``"cohort-03 drifting problem-02"``.
            Keyword-only.

    Returns:
        list[Path]: The base paths written (no extension).
    """
    written, left_out = [], {}
    for kind in KINDS:
        usable, left_out[kind] = _split(summaries, kind)
        base = Path(out_dir) / f"{HEATMAP_STEM} - {KIND_LABELS[kind]}"
        if not usable:
            print(f"[INFO] {scope}: no mouse for the port-finding summary of {kind}")
            remove_figure(base)
            continue
        rows = [
            _heatmap_row(
                f"Current {_ports(layout[0])}\nPrev {_ports(layout[1])}"
                + ("" if layout[2] == DEFAULT_INITIATION_TOWER else f"\nInitiation {layout[2]}"),
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
             *_excluded_lines(left_out[kind])),
        )  # fmt: skip
        plot_port_finding_heatmaps(
            rows,
            kind,
            title=title,
            window_headings=window_labels(N_FIRST_TRIALS, N_FIRST_BLOCKS),
            save_path=base,
        )
        written.append(base)
    _write_left_out(out_dir, scope, left_out)
    return written


# ---------------------------------------------------------------------------- pooled by layout


def _layout_label(keys: list[tuple[str, int, str]]) -> str:
    """``"cohort-01: Problem 1 (n=1)\\ncohort-03: Problem 2, 4 (n=2)"`` (n counts mice)."""
    lines = []
    for cohort in sorted({c for c, _, _ in keys}):
        mine = [(k, m) for c, k, m in keys if c == cohort]
        problems = ", ".join(str(k) for k in sorted({k for k, _ in mine}))
        lines.append(f"{cohort}: Problem {problems} (n={len({m for _, m in mine})})")
    return "\n".join(lines)


def write_layout_port_finding_heatmaps(
    entries: Mapping[tuple[str, int, str], MousePortFinding], out_dir: Path, *, scope: str
) -> list[Path]:
    """Mice of several cohorts and problems pooled by port layout: one figure per layout and kind.

    A mouse that ran one layout in several problems is averaged over those problems first, so
    every mouse weighs once and ``n`` counts mice.

    Args:
        entries (Mapping): ``{(cohort name, k, mouse): summary}``.
        out_dir (Path): Folder for the figures.
        scope (str): Starts every caption, e.g. ``"drifting problems of cohorts 1 and 3"``.
            Keyword-only.

    Returns:
        list[Path]: The base paths written (no extension).
    """
    written, left_out = [], {}
    for kind in KINDS:
        usable, left_out[kind] = _split(entries, kind)
        for layout, keys in group_by_layout(usable.items()).items():
            label = _layout_label(keys)
            name = _layout_name(layout)
            mice = [(c, m) for c, _, m in keys]
            in_layout = {k: why for k, why in left_out[kind].items() if entries[k].layout == layout}
            title = Title(
                f"{HEATMAP_STEM} - {name} - {KIND_LABELS[kind]}",
                stats_line("mean", "mice", [(len(set(mice)), "mouse", "mice")]),
                (scope, f"Layout {name}; {label.replace(chr(10), '; ')}", KIND_TEXT[kind],
                 WINDOW_TEXT, HEATMAP_TEXT,
                 "A mouse with this layout in several problems is averaged over them first.",
                 EXCLUSION_TEXT, *_excluded_lines(in_layout)),
            )  # fmt: skip
            row = _heatmap_row(label, layout, [usable[k] for k in keys], kind, mice)
            base = Path(out_dir) / f"{HEATMAP_STEM} - {name} - {KIND_LABELS[kind]}"
            plot_port_finding_heatmaps(
                [row],
                kind,
                title=title,
                window_headings=window_labels(N_FIRST_TRIALS, N_FIRST_BLOCKS),
                save_path=base,
            )
            written.append(base)
    _write_left_out(out_dir, scope, left_out)
    return written


# ---------------------------------------------------------------------------- port categories


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
        ``category_fraction_table``); empty, with no figure (an earlier one is removed), if no
        mouse is usable. The mice left out are listed in :data:`LEFT_OUT_FILE` either way.
    """
    entries = list(entries)
    keyed = {(*labels.values(), s.mouse): s for labels, s in entries}
    left_out = {kind: _split(keyed, kind)[1] for kind in KINDS}
    _write_left_out(out_dir, scope, left_out)
    table = category_fraction_table(entries)
    base = Path(out_dir) / FRACTIONS_STEM
    if table.empty:
        print(f"[INFO] {scope}: no mouse for the port-category fractions")
        remove_figure(base)
        return table
    means = mouse_mean_fractions(table, mouse_columns)
    columns = [*mouse_columns, *[c for c in ("task_type", "k") if c in table]]
    n_pairs = table[columns].drop_duplicates().shape[0]
    excluded_lines = [
        f"{KIND_LABELS[kind]}: {line}" for kind in KINDS for line in _excluded_lines(left_out[kind])
    ]
    title = Title(
        FRACTIONS_STEM,
        None,
        (scope, f"{n_pairs} mouse-problem pairs.", KIND_TEXT["after_initiation"],
         KIND_TEXT["after_choice"], WINDOW_TEXT, CATEGORY_TEXT, EXCLUSION_TEXT,
         *excluded_lines),
    )  # fmt: skip
    plot_category_fractions(
        means,
        title=title,
        window_headings=window_labels(N_FIRST_TRIALS, N_FIRST_BLOCKS),
        mouse_columns=mouse_columns,
        save_path=base,
    )
    return table


def problem_entries(
    summaries: Mapping[tuple[str, int], Mapping[str, MousePortFinding]],
    *,
    exclude_mice: Iterable[str] = (),
    **labels,
) -> list[tuple[dict, MousePortFinding]]:
    """``(labels, summary)`` pairs of ``{(task type, k): {mouse: summary}}``, for
    :func:`write_category_fractions`; the labels are *labels* plus ``task_type`` and ``k``.

    Args:
        summaries (Mapping): As ``cohort_port_finding_summaries`` returns.
        exclude_mice (Iterable[str]): Mice to leave out. Default none. Keyword-only.
        **labels: Extra labels in front, e.g. ``cohort="cohort-03"``.

    Returns:
        list[tuple[dict, MousePortFinding]]: The pairs, problems in order.
    """
    excluded = set(exclude_mice)
    return [
        ({**labels, "task_type": task_type, "k": k}, summary)
        for (task_type, k), mice in summaries.items()
        for mouse, summary in mice.items()
        if mouse not in excluded
    ]
