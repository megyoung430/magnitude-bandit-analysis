"""Figures of port preference, ``P(chosen) / P(best)`` (see ``behavior_analysis.port_preference``).

- :func:`plot_category_ratios_panel` / :func:`plot_category_ratios`: X versus plus ports, mean ±
  SE across mice with every mouse as a point joined by a line.
- :func:`plot_port_ratio_panel` / :func:`plot_port_ratio_heatmaps`: the nine-port maze of one
  mouse coloured by the ratio, on the maze layout of ``plot_maze_heatmaps``.

Each panel function draws into an axes you pass (a new figure is made only if none is given).
The figure functions save PNG and PDF.

Heatmap colours. The ratio is shown on a log2 scale so that "twice as often as best" and "half
as often" are equally far from 1. The diverging map is centred on 1; the scale limit is the
same for every panel of a figure. Grey cells have no ratio: the port was never offered, or never
best.
"""

import math
from collections.abc import Mapping
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.patches import RegularPolygon

import src.behavior_visualization.plot_style  # noqa: F401  # applies rcParams
from src.behavior_analysis.poke_events import ARMS9
from src.behavior_visualization.plot_helpers import (
    axes_or_new,
    despine,
    mouse_color_map,
    save_or_show,
)

# The maze geometry is shared with the other maze heatmaps so the cells line up with them.
from src.behavior_visualization.plot_maze_heatmaps import (
    _NODE_RADIUS,
    _OCTAGON_ROTATION,
    _center,
    _draw_grid_edges,
    _port_rings,
    _ring_patch,
    _text_color,
)

Y_LABEL = "P(chosen) / P(best)"
CATEGORY_LABELS = {
    "X": "X\n(diagonal)",
    "plus": "Plus\n(adjacent)",
}
BAR_COLOR = "#9db4c0"
UNDEFINED_COLOR = "#d9d9d9"
DIVERGING_CMAP = "RdBu_r"
MIN_LOG2_LIMIT = 1.0  # the colour scale always spans at least a factor of 2 either way


def _mice_phrase(n_mice: int) -> str:
    return f"{n_mice} mouse" if n_mice == 1 else f"{n_mice} mice"


# ---------------------------------------------------------------- X versus plus bars


def plot_category_ratios_panel(
    per_mouse: pd.DataFrame, summary: pd.DataFrame, *, ax: Axes | None = None
) -> Axes:
    """Bars of mean ± SE with one point per mouse joined by a line, and a reference line at 1.

    A mouse with an undefined ratio (category never best) has no point in that category and
    no line.

    Args:
        per_mouse (pd.DataFrame): Columns ``mouse``, ``category``, ``ratio`` with a row for each
            mouse and plotted category.
        summary (pd.DataFrame): ``category_summary`` output, indexed by category with ``mean``
            and ``se``; its index order is the bar order.
        ax (Axes | None): Axes to draw into; a new figure if None. Default None.

    Returns:
        Axes: The axes drawn on.
    """
    categories = list(summary.index)
    wide = per_mouse.pivot(index="mouse", columns="category", values="ratio")[categories]
    colors = mouse_color_map(wide.index)
    with axes_or_new(ax, figsize=(5.0, 4.6)) as ax:
        ax.bar(
            range(len(categories)), summary["mean"], color=BAR_COLOR, width=0.6, zorder=1,
            yerr=summary["se"].fillna(0.0), capsize=6, ecolor="black",
        )  # fmt: skip
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1, zorder=0)
        for mouse, values in wide.iterrows():
            ax.plot(
                range(len(categories)), values, marker="o", color=colors[mouse],
                linewidth=1.2, label=mouse, zorder=3,
            )  # fmt: skip
        for x, mean in enumerate(summary["mean"]):
            if np.isnan(mean):
                ax.text(x, 0.02, "no ratio\n(never offered\nor never best)", ha="center",
                        va="bottom", fontsize=8, color="0.4")  # fmt: skip
        ax.set_xticks(range(len(categories)), [CATEGORY_LABELS.get(c, c) for c in categories])
        ax.set_xlabel("Ports relative to the initiation port")
        ax.set_ylabel(Y_LABEL)
        ax.set_xlim(-0.6, len(categories) - 0.4)
        ax.set_ylim(bottom=0)
        despine(ax)
    return ax


def plot_category_ratios(
    per_mouse: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    scope_label: str,
    save_path: str | Path | None = None,
) -> None:
    """One-panel figure of :func:`plot_category_ratios_panel`; saves PNG and PDF or shows.

    Args:
        per_mouse (pd.DataFrame): As for :func:`plot_category_ratios_panel`.
        summary (pd.DataFrame): As for :func:`plot_category_ratios_panel`.
        scope_label (str): Names the data in the title, e.g. ``"all drifting problems"``.
        save_path (str | Path | None): Base path without extension; None shows the figure.

    Raises:
        ValueError: If *per_mouse* has no mice.
    """
    n_mice = per_mouse["mouse"].nunique()
    if n_mice == 0:
        raise ValueError("no mice to plot")
    ax = plot_category_ratios_panel(per_mouse, summary)
    fig = ax.figure
    average = "mean ± se" if n_mice >= 2 else "mean"
    ax.set_title(
        f"Port Preference: X vs Plus [{scope_label}]\n"
        f"({average} across subjects | n={_mice_phrase(n_mice)})",
        pad=12,
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False, title="Mouse")
    fig.tight_layout()
    save_or_show(fig, save_path)


# ---------------------------------------------------------------- per-mouse maze heatmaps


def log2_limit(ratios: np.ndarray) -> float:
    """Half-width of the log2 colour scale: the largest ``|log2 ratio|``, rounded up, at least 1.

    Zero ratios (log2 of -inf) and NaN are ignored; zero is drawn at the end of the scale.

    Args:
        ratios (np.ndarray): Ratios of all panels of a figure.

    Returns:
        float: The limit L; colours span ``2**-L`` to ``2**L``.
    """
    positive = ratios[np.isfinite(ratios) & (ratios > 0)]
    if positive.size == 0:
        return MIN_LOG2_LIMIT
    return max(MIN_LOG2_LIMIT, float(math.ceil(np.abs(np.log2(positive)).max())))


def _face_color(ratio: float, norm: Normalize, cmap) -> tuple[float, ...]:
    if np.isnan(ratio):
        return mpl.colors.to_rgba(UNDEFINED_COLOR)
    with np.errstate(divide="ignore"):
        position = np.log2(ratio)
    return cmap(norm(np.clip(position, norm.vmin, norm.vmax)))


def _cell_labels(ratio: float, n_trials: float) -> tuple[str, str]:
    """(second line, third line) of a cell: the ratio and n, or why there is no ratio."""
    if n_trials == 0:
        return "n/a", "n=0"
    if np.isnan(ratio):
        return "no best", f"n={int(n_trials)}"
    return f"{ratio:.2f}", f"n={int(n_trials)}"


def _draw_ratio_cell(ax: Axes, port: str, ratio: float, n_trials: float, face, outline: bool):
    center = _center(port)
    ax.add_patch(
        RegularPolygon(
            center, numVertices=8, radius=_NODE_RADIUS, orientation=_OCTAGON_ROTATION,
            facecolor=face, edgecolor="black", linewidth=2.0, zorder=2,
        )
    )  # fmt: skip
    if outline:
        for spec, role in _port_rings(port, (), (), port):
            ax.add_patch(_ring_patch(center, spec, f"ring:{port}:{role}"))
    color = _text_color(face)
    value_text, n_text = _cell_labels(ratio, n_trials)
    small = 8 if np.isnan(ratio) else 9
    for dy, text, size, weight in (
        (0.07, port, 11, "bold"),
        (-0.01, value_text, small, "normal"),
        (-0.085, n_text, 8, "normal"),
    ):
        ax.text(center[0], center[1] + dy, text, ha="center", va="center", fontsize=size,
                fontweight=weight, color=color, zorder=5)  # fmt: skip


def plot_port_ratio_panel(
    ports: pd.DataFrame,
    *,
    limit: float,
    initiation_ports: set[str] = frozenset(),
    ax: Axes | None = None,
    title: str | None = None,
) -> Axes:
    """The 3 x 3 maze of one mouse, each port coloured by its ratio on a centred log2 scale.

    Each cell prints the port, its ratio and the number of trials it was offered (n). Grey cells
    have no ratio: "n/a" (never offered as a choice tower) or "no best" (offered but never
    best). Ports in *initiation_ports* get a blue outline.

    Args:
        ports (pd.DataFrame): Indexed by port, with ``ratio`` and ``n_trials``; all nine ports.
        limit (float): Colours span ``2**-limit`` .. ``2**limit``, centred on 1 (see
            :func:`log2_limit`).
        initiation_ports (set[str]): Ports to outline in blue. Default none.
        ax (Axes | None): Axes to draw into; a new figure if None. Default None.
        title (str | None): Axes title. Default None.

    Returns:
        Axes: The axes drawn on.

    Raises:
        KeyError: If *ports* lacks one of the nine ports.
    """
    missing = [p for p in ARMS9 if p not in ports.index]
    if missing:
        raise KeyError(f"ports has no row for {missing}")
    norm, cmap = Normalize(-limit, limit), mpl.colormaps[DIVERGING_CMAP]
    with axes_or_new(ax, figsize=(3.6, 3.4)) as ax:
        _draw_grid_edges(ax)
        for port in ARMS9:
            ratio, n_trials = float(ports.loc[port, "ratio"]), float(ports.loc[port, "n_trials"])
            face = _face_color(ratio, norm, cmap)
            _draw_ratio_cell(ax, port, ratio, n_trials, face, port in initiation_ports)
        if title is not None:
            ax.set_title(title, fontsize=14, pad=6)
        ax.set_xlim(-0.35, 1.65)
        ax.set_ylim(-0.35, 1.65)
        ax.set_aspect("equal")
        ax.axis("off")
    return ax


def _add_log2_colorbar(fig: plt.Figure, axes, limit: float) -> None:
    mappable = mpl.cm.ScalarMappable(norm=Normalize(-limit, limit), cmap=DIVERGING_CMAP)
    ticks = np.arange(-limit, limit + 1)
    colorbar = fig.colorbar(mappable, ax=axes, fraction=0.04, pad=0.02, shrink=0.7)
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels([f"{2.0**t:g}" for t in ticks])
    colorbar.set_label(f"{Y_LABEL} (log2 scale; 1 = chosen as often as best)", fontsize=9)


def plot_port_ratio_heatmaps(
    port_ratios: pd.DataFrame,
    *,
    scope_label: str,
    save_path: str | Path | None = None,
    initiation_ports: Mapping[str, set[str]] | None = None,
) -> None:
    """One maze heatmap per mouse on a shared log2 scale; saves PNG and PDF or shows.

    Args:
        port_ratios (pd.DataFrame): Columns ``mouse``, ``port``, ``ratio``, ``n_trials``: all
            nine ports for every mouse (``add_ratio`` of the pooled per-port counts).
        scope_label (str): Names the data in the title, e.g. ``"problem-03"``.
        save_path (str | Path | None): Base path without extension; None shows the figure.
        initiation_ports (Mapping[str, set[str]] | None): ``{mouse: initiation ports}`` to
            outline in blue. Default None: no outlines.

    Raises:
        ValueError: If *port_ratios* has no mice.
    """
    mice = sorted(port_ratios["mouse"].unique())
    if not mice:
        raise ValueError("no mice to plot")
    limit = log2_limit(port_ratios["ratio"].to_numpy(dtype=float))
    nrows = min(2, len(mice))
    ncols = math.ceil(len(mice) / nrows)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * 4.0 + 1.2, nrows * 4.0 + 0.8), squeeze=False,
        layout="constrained",
    )  # fmt: skip
    try:
        for i, mouse in enumerate(mice):
            plot_port_ratio_panel(
                port_ratios[port_ratios["mouse"] == mouse].set_index("port"),
                limit=limit,
                initiation_ports=(initiation_ports or {}).get(mouse, set()),
                ax=axes[i // ncols, i % ncols],
                title=mouse,
            )
        for j in range(len(mice), nrows * ncols):
            axes[j // ncols, j % ncols].axis("off")
        _add_log2_colorbar(fig, axes.ravel().tolist(), limit)
        fig.suptitle(
            f"Port Preference by Port [{scope_label}]\n"
            f"({Y_LABEL} | grey: n/a = never offered, no best = never best | "
            f"blue outline = initiation port | n={_mice_phrase(len(mice))})",
            fontsize=11,
        )
    except BaseException:
        plt.close(fig)
        raise
    save_or_show(fig, save_path)
