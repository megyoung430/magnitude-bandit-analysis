"""Figures of where mice poke during port finding and just after it.

:func:`plot_port_finding_heatmaps` draws one row of three maze heatmaps per port layout (port
finding, first trials, first blocks), each the mean across mice of the percentage of pokes per
port. :func:`plot_category_fractions` draws, per poke kind, the fraction of pokes at the current
problem's ports, the previous problem's ports, the initiation port and the other ports in the
three windows, as grouped bars (mean ± se across mice, a dot per mouse). Both take results;
nothing here computes them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import src.behavior_visualization.plot_style  # noqa: F401  # applies rcParams
from src.behavior_analysis.port_finding_summary import (
    CATEGORIES,
    CURRENT,
    INITIATION,
    OTHER,
    PREVIOUS,
    WINDOWS,
)
from src.behavior_visualization.plot_helpers import (
    Title,
    despine,
    save_or_show,
    set_suptitle,
    set_title,
    stats_line,
)
from src.behavior_visualization.plot_maze_heatmaps import (
    CURRENT_COLOR,
    INITIATION_COLOR,
    PREVIOUS_COLOR,
    draw_maze_heatmap,
    outline_legend_handles,
)

KIND_LABELS = {
    "after_initiation": "Long Pokes After Initiation",
    "after_choice": "Pokes After Choice",
}
CATEGORY_LABELS = {
    CURRENT: "Current Problem Ports",
    PREVIOUS: "Previous Problem Ports",
    INITIATION: "Initiation Port",
    OTHER: "Other Ports",
}
CATEGORY_COLORS = {
    CURRENT: CURRENT_COLOR,
    PREVIOUS: PREVIOUS_COLOR,
    INITIATION: INITIATION_COLOR,
    OTHER: "#7F7F7F",
}


def window_labels(n_trials: int, n_blocks: int) -> dict[str, str]:
    """``{window: heading}`` of the three windows."""
    return dict(
        zip(
            WINDOWS,
            (
                "Port Finding",
                f"First {n_trials} Trials After Finding",
                f"First {n_blocks} Blocks After Finding",
            ),
            strict=True,
        )
    )


class HeatmapRow(NamedTuple):
    """One row of :func:`plot_port_finding_heatmaps`: a port layout and its mean maps.

    Attributes:
        label (str): Text written left of the row (the layout, or the cohorts and problems).
        current_ports (tuple[str, ...]): Outlined green.
        previous_ports (tuple[str, ...]): Outlined red; empty for none.
        initiation_port (str): Outlined blue.
        maps (Mapping[str, Mapping[str, float]]): ``{window: {port: mean percent}}``.
        n_mice (int): Mice averaged.
    """

    label: str
    current_ports: tuple[str, ...]
    previous_ports: tuple[str, ...]
    initiation_port: str
    maps: Mapping[str, Mapping[str, float]]
    n_mice: int


def plot_port_finding_heatmaps(
    rows: Sequence[HeatmapRow],
    kind: str,
    *,
    title: Title,
    window_headings: Mapping[str, str],
    save_path: str | Path | None = None,
) -> None:
    """One row of three mean maze heatmaps (one per window) per port layout.

    Each map is scaled from 0 to its own largest port, in percent of pokes, numbers rounded.

    Args:
        rows (Sequence[HeatmapRow]): The layouts, top to bottom.
        kind (str): ``"after_initiation"`` or ``"after_choice"``; names the colorbar.
        title (Title): Figure title; its caption lines go to the caption file. Keyword-only.
        window_headings (Mapping[str, str]): ``{window: panel heading}``. Keyword-only.
        save_path (str | Path | None): Base path without extension; ``None`` shows the figure.
            Default None.

    Raises:
        ValueError: If there are no rows.
    """
    if not rows:
        raise ValueError("no layouts to draw")
    fig, axes = plt.subplots(len(rows), len(WINDOWS), figsize=(12.6, 3.9 * len(rows) + 1.2),
                             squeeze=False)  # fmt: skip
    try:
        for row, row_axes in zip(rows, axes, strict=True):
            for window, ax in zip(WINDOWS, row_axes, strict=True):
                draw_maze_heatmap(
                    row.maps[window],
                    colorbar_label=f"{KIND_LABELS[kind]} (% of Pokes)",
                    ax=ax,
                    current_ports=row.current_ports,
                    previous_ports=row.previous_ports or None,
                    initiation_port=row.initiation_port,
                )
                set_title(
                    ax,
                    Title(window_headings[window], stats_line("mean", "mice",
                          [(row.n_mice, "mouse", "mice")])),
                    fontsize=12, pad=6,
                )  # fmt: skip
            row_axes[0].annotate(row.label, xy=(-0.16, 0.5), xycoords="axes fraction",
                                 rotation=90, ha="center", va="center", fontsize=11)  # fmt: skip
        fig.legend(handles=outline_legend_handles(True, True, True), loc="lower center", ncol=3,
                   frameon=False, fontsize=11)  # fmt: skip
        fig.tight_layout(rect=(0.03, 0.05, 1, 0.95))
        set_suptitle(fig, title, fontsize=14)
    except BaseException:
        plt.close(fig)
        raise
    save_or_show(fig, save_path)


def _mean_se(values: np.ndarray) -> tuple[float, float]:
    """Mean and standard error (``nan`` with one value)."""
    se = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else np.nan
    return float(values.mean()), float(se)


def draw_category_fractions(
    mouse_means: pd.DataFrame,
    *,
    ax: plt.Axes,
    window_headings: Mapping[str, str],
    mouse_columns: Sequence[str] = ("mouse",),
) -> int:
    """Draw one poke kind as grouped bars: a group per window, a bar per port category.

    Bars are the mean across mice with ± se, dots the mice.

    Args:
        mouse_means (pd.DataFrame): One kind's rows of ``mouse_mean_fractions``.
        ax (plt.Axes): Axes to draw into. Keyword-only.
        window_headings (Mapping[str, str]): ``{window: tick label}``. Keyword-only.
        mouse_columns (Sequence[str]): Columns naming a mouse. Default ``("mouse",)``.

    Returns:
        int: The number of mice.

    Raises:
        ValueError: If *mouse_means* is empty.
    """
    if mouse_means.empty:
        raise ValueError("no mice to draw")
    wide = mouse_means.pivot_table(
        index=[*mouse_columns, "category"], columns="window", values="fraction"
    ).reindex(columns=list(WINDOWS))
    n_mice = mouse_means[list(mouse_columns)].drop_duplicates().shape[0]
    width = 0.8 / len(CATEGORIES)
    rng = np.random.default_rng(0)  # fixed jitter, so a redraw is identical
    for i, category in enumerate(CATEGORIES):
        values = wide.xs(category, level="category").to_numpy(dtype=float)
        x = np.arange(len(WINDOWS)) + (i - (len(CATEGORIES) - 1) / 2) * width
        stats = [_mean_se(values[:, j]) for j in range(len(WINDOWS))]
        means, ses = (np.array(v) for v in zip(*stats, strict=True))
        color = CATEGORY_COLORS[category]
        ax.bar(x, means, width=width * 0.9, color=color, alpha=0.75)
        ax.errorbar(x, means, yerr=ses, fmt="none", ecolor="black", capsize=2, linewidth=1)
        for j in range(len(WINDOWS)):
            jitter = rng.uniform(-width * 0.25, width * 0.25, len(values))
            ax.scatter(x[j] + jitter, values[:, j], s=8, color="black", alpha=0.5, zorder=3,
                       linewidths=0)  # fmt: skip
    labels = [window_headings[w].replace(" After", "\nAfter") for w in WINDOWS]
    ax.set_xticks(np.arange(len(WINDOWS)), labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction of Pokes")
    despine(ax)
    return n_mice


def category_legend_handles() -> list[Line2D]:
    """One square handle per port category, in :data:`CATEGORIES` order."""
    return [
        Line2D(
            [0],
            [0],
            color=CATEGORY_COLORS[c],
            marker="s",
            linestyle="",
            markersize=10,
            label=CATEGORY_LABELS[c],
        )  # fmt: skip
        for c in CATEGORIES
    ]


def plot_category_fractions(
    mouse_means: pd.DataFrame,
    *,
    title: Title,
    window_headings: Mapping[str, str],
    mouse_columns: Sequence[str] = ("mouse",),
    save_path: str | Path | None = None,
) -> None:
    """Two panels (long pokes after initiation, pokes after choice) of the port-category fractions.

    Args:
        mouse_means (pd.DataFrame): Output of ``mouse_mean_fractions``.
        title (Title): Figure title. Keyword-only.
        window_headings (Mapping[str, str]): ``{window: tick label}``. Keyword-only.
        mouse_columns (Sequence[str]): Columns naming a mouse. Default ``("mouse",)``.
        save_path (str | Path | None): Base path without extension; ``None`` shows the figure.
            Default None.

    Raises:
        ValueError: If a poke kind has no rows.
    """
    kinds = list(KIND_LABELS)
    fig, axes = plt.subplots(1, len(kinds), figsize=(11, 4.8), sharey=True)
    try:
        for kind, ax in zip(kinds, axes, strict=True):
            rows = mouse_means[mouse_means["kind"] == kind]
            n = draw_category_fractions(
                rows, ax=ax, window_headings=window_headings, mouse_columns=mouse_columns
            )
            average = "mean ± se" if n > 1 else "mean"  # one mouse has no se
            stats = stats_line(average, "mice", [(n, "mouse", "mice")])
            set_title(ax, Title(KIND_LABELS[kind], stats), fontsize=12)
        fig.legend(handles=category_legend_handles(), loc="lower center", ncol=len(CATEGORIES),
                   frameon=False, fontsize=11)  # fmt: skip
        fig.tight_layout(rect=(0, 0.08, 1, 0.92))
        set_suptitle(fig, title, fontsize=14)
    except BaseException:
        plt.close(fig)
        raise
    save_or_show(fig, save_path)
