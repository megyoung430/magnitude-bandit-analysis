"""Maze heatmaps: a 3 x 3 map of the ports coloured by how often each was poked.

One map is drawn by :func:`draw_maze_heatmap` into an axes you pass (so it can be a panel of any
figure); :func:`plot_maze_heatmaps_across_mice` makes the one-panel-per-mouse figure.

Layout. The ports sit on a 3 x 3 grid, letters running left to right (A, B, C) and numbers
running bottom to top (1 at the bottom, 3 at the top), so A1 is the bottom-left port and C3 the
top-right one. That is the layout the original notebook drew; the notebook's own comment says
"row 1 is top", but its coordinates put row 1 at the bottom. Whether this matches the physical
maze has not been checked against the rig (see :data:`MAZE_POSITIONS`).

Scale. Each map is coloured from 0 to its own maximum count, and its colorbar says so with real
numbers at both ends.

Outlines. Rings are drawn outside a port's octagon: green for the current problem's choice
ports, red for the previous problem's, blue for the initiation port. A port that needs several
gets concentric rings of identical radial width, so no outline is thinner or hidden behind
another; the inner ring is green, then red, then blue.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import NamedTuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch, RegularPolygon
from matplotlib.path import Path as MplPath

import src.behavior_visualization.plot_style  # noqa: F401  # applies rcParams
from src.behavior_analysis.poke_events import ARMS9
from src.behavior_visualization.plot_helpers import axes_or_new, save_or_show

# (column, row) of each port on the grid; row 0 is the bottom. Same as the notebook's drawing.
MAZE_POSITIONS: dict[str, tuple[int, int]] = {
    f"{letter}{number}": (column, number - 1)
    for column, letter in enumerate("ABC")
    for number in (1, 2, 3)
}

CURRENT_COLOR = "#3A982E"
PREVIOUS_COLOR = "#D1393D"
INITIATION_COLOR = "#5DA5DA"

_GRID_SPACING = 0.65
_NODE_RADIUS = 0.18
_RING_WIDTH = 0.04
_OCTAGON_ROTATION = np.pi / 8
_EDGE_COLOR = "#c9d2d7"


class RingSpec(NamedTuple):
    """One outline ring: its colour and its inner and outer radius around the port centre."""

    color: str
    inner: float
    outer: float


def count_norm(counts: Mapping[str, float]) -> Normalize:
    """Colour scale from 0 to the largest count (1 if nothing was poked)."""
    top = max(float(counts[port]) for port in ARMS9)
    return Normalize(vmin=0.0, vmax=max(top, 1.0))


def ring_specs(colors: list[str], *, base_radius: float, width: float) -> list[RingSpec]:
    """Concentric rings of equal radial *width* around radius *base_radius*, first colour inside."""
    return [
        RingSpec(color, base_radius + i * width, base_radius + (i + 1) * width)
        for i, color in enumerate(colors)
    ]


def _octagon_vertices(center: tuple[float, float], radius: float) -> np.ndarray:
    """Corners of the port octagon (flat sides up, down, left, right), counter-clockwise."""
    angles = np.pi / 2 + _OCTAGON_ROTATION + 2 * np.pi * np.arange(8) / 8
    return np.column_stack(
        [center[0] + radius * np.cos(angles), center[1] + radius * np.sin(angles)]
    )


def _ring_patch(center: tuple[float, float], spec: RingSpec, gid: str) -> PathPatch:
    """A filled octagonal band between the radii of *spec* (inner octagon wound the other way)."""
    outer = _octagon_vertices(center, spec.outer)
    inner = _octagon_vertices(center, spec.inner)[::-1]
    vertices = np.vstack([outer, outer[:1], inner, inner[:1]])
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * 7 + [MplPath.CLOSEPOLY]
    codes = codes + [MplPath.MOVETO] + [MplPath.LINETO] * 7 + [MplPath.CLOSEPOLY]
    patch = PathPatch(
        MplPath(vertices, codes), facecolor=spec.color, edgecolor="none", linewidth=0, zorder=4
    )
    patch.set_gid(gid)
    return patch


def _check_ports(ports: Collection[str], role: str) -> None:
    unknown = sorted(set(ports) - set(ARMS9))
    if unknown:
        raise ValueError(f"unknown {role} ports {unknown}; expected a subset of {ARMS9}")


def _center(port: str) -> tuple[float, float]:
    column, row = MAZE_POSITIONS[port]
    return _GRID_SPACING * column, _GRID_SPACING * row


def _draw_grid_edges(ax: Axes) -> None:
    for port, (column, row) in MAZE_POSITIONS.items():
        for neighbour in ((column + 1, row), (column, row + 1)):
            other = next((p for p, pos in MAZE_POSITIONS.items() if pos == neighbour), None)
            if other is not None:
                (x1, y1), (x2, y2) = _center(port), _center(other)
                ax.plot(
                    [x1, x2], [y1, y2], linewidth=25, color=_EDGE_COLOR, alpha=0.9,
                    solid_capstyle="round", zorder=1,
                )  # fmt: skip


def _text_color(face_rgba: tuple[float, ...]) -> str:
    """White on dark fills, black on light ones."""
    r, g, b = face_rgba[:3]
    return "white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.5 else "black"


def _draw_port(ax: Axes, port: str, count: float, norm: Normalize, cmap, rings: list[RingSpec]):
    center = _center(port)
    face = cmap(norm(count))
    ax.add_patch(
        RegularPolygon(
            center, numVertices=8, radius=_NODE_RADIUS, orientation=_OCTAGON_ROTATION,
            facecolor=face, edgecolor="black", linewidth=2.0, zorder=2,
        )
    )  # fmt: skip
    for spec, role in rings:
        ax.add_patch(_ring_patch(center, spec, f"ring:{port}:{role}"))
    color = _text_color(face)
    ax.text(center[0], center[1] + 0.03, port, ha="center", va="center", fontsize=12,
            fontweight="bold", color=color, zorder=5)  # fmt: skip
    ax.text(center[0], center[1] - 0.075, f"{int(count)}", ha="center", va="center", fontsize=10,
            color=color, zorder=5)  # fmt: skip


def _port_rings(
    port: str, current: Collection[str], previous: Collection[str], initiation: str | None
) -> list[tuple[RingSpec, str]]:
    """The ``(ring, role)`` outlines of one port, innermost first."""
    roles = [
        (role, color)
        for role, color, ports in (
            ("current", CURRENT_COLOR, current),
            ("previous", PREVIOUS_COLOR, previous),
            ("initiation", INITIATION_COLOR, {initiation} if initiation else set()),
        )
        if port in ports
    ]
    specs = ring_specs([color for _, color in roles], base_radius=_NODE_RADIUS, width=_RING_WIDTH)
    return [(spec, role) for spec, (role, _) in zip(specs, roles, strict=True)]


def draw_maze_heatmap(
    counts: Mapping[str, float],
    *,
    colorbar_label: str,
    ax: Axes | None = None,
    title: str | None = None,
    current_ports: Collection[str] = (),
    previous_ports: Collection[str] | None = None,
    initiation_port: str | None = None,
    cmap: str = "Blues",
) -> Axes:
    """Draw one maze heatmap with its own 0-to-max colorbar.

    Args:
        counts: Count for each of the nine ports (``A1`` .. ``C3``).
        colorbar_label: Says what the colour shows, e.g. "Pokes after choice (count)".
        ax: Axes to draw into; a new figure is created if ``None``.
        title: Axes title.
        current_ports: Ports to outline in green (the current problem's choice ports).
        previous_ports: Ports to outline in red (the previous problem's choice ports);
            ``None`` for a mouse without a previous problem.
        initiation_port: Port to outline in blue, or ``None``.
        cmap: Name of a matplotlib colormap.

    Returns:
        The axes the map was drawn on.

    Raises:
        KeyError: If *counts* lacks a port.
        ValueError: If an outline port is not one of the nine.
    """
    previous = set() if previous_ports is None else set(previous_ports)
    _check_ports(current_ports, "current")
    _check_ports(previous, "previous")
    _check_ports([] if initiation_port is None else [initiation_port], "initiation")
    missing = [port for port in ARMS9 if port not in counts]
    if missing:
        raise KeyError(f"counts has no entry for {missing}")
    norm, colormap = count_norm(counts), mpl.colormaps[cmap]
    with axes_or_new(ax, figsize=(3.6, 3.4)) as ax:
        _draw_grid_edges(ax)
        for port in ARMS9:
            rings = _port_rings(port, current_ports, previous, initiation_port)
            _draw_port(ax, port, counts[port], norm, colormap, rings)
        if title is not None:
            ax.set_title(title, fontsize=14, pad=6)
        # Fixed limits (centres span 0..1.3, three rings reach 0.3 beyond): all maps share a size.
        ax.set_xlim(-0.35, 1.65)
        ax.set_ylim(-0.35, 1.65)
        ax.set_aspect("equal")
        ax.axis("off")
        _add_colorbar(ax, norm, colormap, colorbar_label)
    return ax


def _add_colorbar(ax: Axes, norm: Normalize, colormap, label: str) -> None:
    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=colormap)
    colorbar = ax.figure.colorbar(mappable, ax=ax, fraction=0.045, pad=0.02, shrink=0.8)
    colorbar.set_label(label, rotation=90, fontsize=9)
    colorbar.set_ticks([norm.vmin, norm.vmax])
    colorbar.set_ticklabels([f"{norm.vmin:g}", f"{norm.vmax:g}"])


def _per_mouse(value: Mapping | None, mice: list[str], name: str) -> Mapping:
    """*value* if it has an entry for every mouse, else a mapping of ``None`` for all mice."""
    if value is None:
        return dict.fromkeys(mice)
    missing = [m for m in mice if m not in value]
    if missing:
        raise ValueError(f"{name} has no entry for mice {missing}")
    return value


def outline_legend_handles(
    uses_current: bool, uses_previous: bool, initiation_port: str | None
) -> list[Line2D]:
    """Legend handles for the outlines in use (green current, red previous, blue initiation)."""
    entries = [
        (uses_current, CURRENT_COLOR, "Current problem choice ports"),
        (uses_previous, PREVIOUS_COLOR, "Previous problem choice ports"),
        (initiation_port is not None, INITIATION_COLOR, "Initiation port"),
    ]
    return [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            markerfacecolor=color,
            markeredgecolor=color,
            markersize=10,
            label=text,
        )  # fmt: skip
        for used, color, text in entries
        if used
    ]


def plot_maze_heatmaps_across_mice(
    per_mouse_counts: Mapping[str, Mapping[str, float]],
    *,
    colorbar_label: str,
    current_ports: Mapping[str, Collection[str]] | None = None,
    previous_ports: Mapping[str, Collection[str] | None] | None = None,
    initiation_port: str | None = None,
    titles: Mapping[str, str] | None = None,
    suptitle: str | None = None,
    cmap: str = "Blues",
    save_path: str | Path | None = None,
) -> None:
    """One maze heatmap per mouse, two rows, each scaled from 0 to that mouse's own maximum.

    Args:
        per_mouse_counts: ``{mouse: {port: count}}``.
        colorbar_label: What the colour shows, written on every colorbar.
        current_ports: ``{mouse: ports}`` to outline in green; every mouse needs an entry.
            ``None`` draws no green outlines.
        previous_ports: ``{mouse: ports or None}`` to outline in red; every mouse needs an
            entry (``None`` for a mouse without a previous problem). ``None`` draws no red.
        initiation_port: Port to outline in blue on every map, or ``None``.
        titles: ``{mouse: panel title}``; default is the mouse id.
        suptitle: Figure title.
        cmap: Name of a matplotlib colormap.
        save_path: Base path (without extension) for ``.png`` and ``.pdf``; ``None`` shows the
            figure.

    Raises:
        ValueError: If there are no mice or an outline mapping misses a mouse.
    """
    mice = sorted(per_mouse_counts)
    if not mice:
        raise ValueError("per_mouse_counts has no mice.")
    current = _per_mouse(current_ports, mice, "current_ports")
    previous = _per_mouse(previous_ports, mice, "previous_ports")
    nrows = min(2, len(mice))
    ncols = math.ceil(len(mice) / nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.4, nrows * 4.2), squeeze=False)
    try:
        for i, mouse in enumerate(mice):
            draw_maze_heatmap(
                per_mouse_counts[mouse],
                colorbar_label=colorbar_label,
                ax=axes[i // ncols, i % ncols],
                title=(titles or {}).get(mouse, mouse),
                current_ports=current[mouse] or (),
                previous_ports=previous[mouse],
                initiation_port=initiation_port,
                cmap=cmap,
            )
        for j in range(len(mice), nrows * ncols):
            axes[j // ncols, j % ncols].axis("off")
        handles = outline_legend_handles(
            current_ports is not None and any(current[m] for m in mice),
            previous_ports is not None and any(previous[m] for m in mice),
            initiation_port,
        )
        if handles:
            fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False)
        fig.tight_layout(rect=(0, 0.07 if handles else 0, 1, 0.9 if suptitle else 1))
        if suptitle:
            fig.suptitle(suptitle, fontsize=13, y=0.99)
    except BaseException:
        plt.close(fig)
        raise
    save_or_show(fig, save_path)
