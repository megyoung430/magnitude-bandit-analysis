"""Maze schematic: which tower is the initiation tower and which are choice towers in a problem.

Drawn for the connected-platform maze, where all nine towers stand on one solid floor rather
than being joined by bridges. The floor is the tightest outline around the towers, so the outer
towers form its edge and the corner towers its corners.

Layout and tower shape come from :mod:`plot_maze_heatmaps` (A1 bottom-left, C3 top-right), so a
schematic and a heatmap of the same problem line up port for port.
"""

from __future__ import annotations

from collections.abc import Collection
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull

import src.behavior_visualization.plot_style  # noqa: F401  # applies rcParams
from src.behavior_analysis.poke_events import ARMS9
from src.behavior_visualization.plot_helpers import axes_or_new, save_or_show
from src.behavior_visualization.plot_maze_heatmaps import (
    NODE_RADIUS,
    octagon_vertices,
    port_center,
)

INITIATION_FILL = "#74abd3"
CHOICE_FILL = "#b1e3f7"
IDLE_FILL = "white"
PLATFORM_COLOR = "#e4e9ec"

LABEL_FONTSIZE = 12
LEGEND_FONTSIZE = 10
_LIMIT_MARGIN = 0.03  # maze-grid units around the floor


def port_fills(initiation_port: str, choice_ports: Collection[str]) -> dict[str, str]:
    """Fill colour of every port according to its role in the problem.

    Args:
        initiation_port (str): The initiation tower, e.g. ``"B2"``.
        choice_ports (Collection[str]): The choice towers, e.g. ``["A2", "B1", "C3"]``.

    Returns:
        dict[str, str]: ``{port: colour}`` for all nine ports: ``INITIATION_FILL``,
        ``CHOICE_FILL`` or ``IDLE_FILL``.

    Raises:
        ValueError: If a port is not one of the nine, or is both initiation and choice tower.
    """
    unknown = sorted({initiation_port, *choice_ports} - set(ARMS9))
    if unknown:
        raise ValueError(f"unknown ports {unknown}; expected a subset of {ARMS9}")
    if initiation_port in choice_ports:
        raise ValueError(f"{initiation_port} cannot be both the initiation and a choice tower")

    def fill(port: str) -> str:
        if port == initiation_port:
            return INITIATION_FILL
        return CHOICE_FILL if port in choice_ports else IDLE_FILL

    return {port: fill(port) for port in ARMS9}


def platform_outline(radius: float = NODE_RADIUS) -> np.ndarray:
    """Corners of the floor: the convex hull of all nine tower octagons, counter-clockwise.

    Args:
        radius (float): Centre-to-corner distance of a tower octagon. Default ``NODE_RADIUS``.

    Returns:
        np.ndarray: Outline corners, shape ``(n, 2)``.
    """
    corners = np.vstack([octagon_vertices(port_center(port), radius) for port in ARMS9])
    return corners[ConvexHull(corners).vertices]


def _draw_platform(ax: Axes) -> None:
    ax.add_patch(Polygon(platform_outline(), closed=True, facecolor=PLATFORM_COLOR,
                         edgecolor="none", zorder=1))  # fmt: skip


def _draw_port(ax: Axes, port: str, fill: str) -> None:
    center = port_center(port)
    ax.add_patch(Polygon(octagon_vertices(center, NODE_RADIUS), closed=True, facecolor=fill,
                         edgecolor="black", linewidth=2.0, zorder=2))  # fmt: skip
    ax.text(*center, port, ha="center", va="center", fontsize=LABEL_FONTSIZE, color="black",
            zorder=3)  # fmt: skip


def role_legend_handles() -> list[Line2D]:
    """Legend handles for the initiation and choice tower fills.

    Returns:
        list[Line2D]: Square-marker handles, initiation first.
    """
    return [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            markerfacecolor=color,
            markeredgecolor="black",
            markersize=10,
            label=label,
        )  # fmt: skip
        for color, label in ((INITIATION_FILL, "Initiation tower"), (CHOICE_FILL, "Choice tower"))
    ]


def draw_maze_schematic(
    initiation_port: str,
    choice_ports: Collection[str],
    *,
    ax: Axes | None = None,
    title: str | None = None,
    legend: bool = True,
) -> Axes:
    """Draw the maze for one problem: towers on the floor, filled by role.

    Args:
        initiation_port (str): The initiation tower, e.g. ``"B2"``.
        choice_ports (Collection[str]): The choice towers, e.g. ``["A2", "B1", "C3"]``.
        ax (Axes | None): Axes to draw into; a new 4.8 x 3.0 inch figure is created if None.
            Default None.
        title (str | None): Axes title; none if None. Default None.
        legend (bool): Put the tower-role legend to the right of the maze. Default True.

    Returns:
        Axes: The axes the schematic was drawn on.

    Raises:
        ValueError: If a port is not one of the nine, or is both initiation and choice tower.
    """
    fills = port_fills(initiation_port, choice_ports)
    with axes_or_new(ax, figsize=(4.8, 3.0)) as ax:
        _draw_platform(ax)
        for port in ARMS9:
            _draw_port(ax, port, fills[port])
        if title is not None:
            ax.set_title(title, pad=6)
        if legend:
            ax.legend(handles=role_legend_handles(), loc="upper left", bbox_to_anchor=(1.02, 1),
                      fontsize=LEGEND_FONTSIZE, frameon=False)  # fmt: skip
        # Fixed limits from the floor (not the data) so every schematic has the same size.
        # The margin leaves room for the half of the tower outline that lies outside the floor.
        outline = platform_outline()
        ax.set_xlim(outline[:, 0].min() - _LIMIT_MARGIN, outline[:, 0].max() + _LIMIT_MARGIN)
        ax.set_ylim(outline[:, 1].min() - _LIMIT_MARGIN, outline[:, 1].max() + _LIMIT_MARGIN)
        ax.set_aspect("equal")
        ax.axis("off")
    return ax


def plot_maze_schematic(
    initiation_port: str,
    choice_ports: Collection[str],
    *,
    title: str | None = None,
    save_path: str | Path | None = None,
    formats: tuple[str, ...] = ("svg", "pdf", "png"),
) -> None:
    """One schematic as its own figure, saved as ``<save_path>.<format>`` or shown.

    Args:
        initiation_port (str): The initiation tower, e.g. ``"B2"``.
        choice_ports (Collection[str]): The choice towers, e.g. ``["A2", "B1", "C3"]``.
        title (str | None): Axes title; none if None. Default None.
        save_path (str | Path | None): Base path without extension; None shows the figure.
            Default None.
        formats (tuple[str, ...]): File formats to write. Default ``("svg", "pdf", "png")``.

    Raises:
        ValueError: If a port is not one of the nine, or is both initiation and choice tower.
    """
    ax = draw_maze_schematic(initiation_port, choice_ports, title=title)
    save_or_show(ax.figure, save_path, formats=formats)
