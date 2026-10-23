"""Maze schematic: which tower is the initiation tower and which are choice towers in a problem.

Drawn for the connected-platform maze, where all nine towers stand on one solid floor rather
than being joined by bridges. The floor is the tightest outline around the towers, so the outer
towers form its edge and the corner towers its corners.

Layout and tower shape come from :mod:`plot_maze_heatmaps` (A1 bottom-left, C3 top-right), so a
schematic and a heatmap of the same problem line up port for port. The towers of a recorded
problem come from :func:`src.behavior_import.task_types.problem_layout`.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Sequence
from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib import font_manager
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull

from src.behavior_analysis.poke_events import ARMS9
from src.behavior_import.problem_numbering import DEFAULT_INITIATION_TOWER
from src.behavior_visualization.plot_helpers import FIGURE_FORMATS, axes_or_new, save_or_show
from src.behavior_visualization.plot_maze_heatmaps import (
    NODE_RADIUS,
    octagon_vertices,
    port_center,
)
from src.behavior_visualization.plot_style import FONT_RC

INITIATION_FILL = "#74abd3"
CHOICE_FILL = "#b1e3f7"
IDLE_FILL = "white"
PLATFORM_COLOR = "#e4e9ec"

LABEL_FONTSIZE = 12
LEGEND_FONTSIZE = 10

# The shared font list of all figures (Arial where installed; DejaVu Sans ships with matplotlib,
# so one of them is always found).
SCHEMATIC_FONTS = tuple(FONT_RC["font.sans-serif"])


def first_installed_font(names: Sequence[str]) -> str:
    """The first of *names* that matplotlib can find on this machine.

    Args:
        names (Sequence[str]): Concrete font family names such as ``"Arial"``, in order of
            preference; generic names such as ``"sans-serif"`` are not resolved here.

    Returns:
        str: The first installed one.

    Raises:
        ValueError: If none of them is installed.
    """
    for name in names:
        try:
            font_manager.findfont(
                font_manager.FontProperties(family=name), fallback_to_default=False
            )
        except ValueError:
            continue
        return name
    raise ValueError(f"none of the fonts {list(names)} is installed")


def schematic_font() -> str:
    """The font of all schematic text: the first installed of :data:`SCHEMATIC_FONTS`.

    One name only: matplotlib looks up every name in a family list each time it draws text, and
    logs a warning for each one that is missing. Looked up on every call (matplotlib caches the
    search), so a font registered later in the session is picked up.

    Returns:
        str: ``"Arial"`` where it is installed.
    """
    return first_installed_font(SCHEMATIC_FONTS)


def schematic_rc() -> dict[str, object]:
    """rcParams applied only while a schematic is drawn and saved; other figures keep theirs.

    Text is in :func:`schematic_font` and stays editable in saved PDF and SVG files.

    Returns:
        dict[str, object]: rcParams for ``mpl.rc_context``.
    """
    return {
        "font.family": [schematic_font()],
        "pdf.fonttype": FONT_RC["pdf.fonttype"],
        "ps.fonttype": FONT_RC["pdf.fonttype"],
        "svg.fonttype": FONT_RC["svg.fonttype"],
    }


def _choice_tower_set(choice_towers: Collection[str]) -> frozenset[str]:
    """The choice towers as a set of values, after checking they name each tower once."""
    if isinstance(choice_towers, str):
        raise ValueError(
            f"choice towers must be a list of towers, got the string {choice_towers!r}"
        )
    # list() reads the values: `in` on a pandas Series would test its index instead
    towers = list(choice_towers)
    if not towers:
        raise ValueError("give at least one choice tower")
    repeated = sorted(t for t, n in Counter(towers).items() if n > 1)
    if repeated:
        raise ValueError(f"choice towers {repeated} are listed more than once")
    return frozenset(towers)


def port_fills(
    choice_towers: Collection[str], initiation_tower: str = DEFAULT_INITIATION_TOWER
) -> dict[str, str]:
    """Fill colour of every port according to its role in the problem.

    Args:
        choice_towers (Collection[str]): The choice towers, e.g. ``["A2", "B1", "C3"]``; a
            list, tuple, set or pandas Series.
        initiation_tower (str): The initiation tower. Default ``"B2"``, the protocol default.

    Returns:
        dict[str, str]: ``{port: colour}`` for all nine ports: ``INITIATION_FILL``,
        ``CHOICE_FILL`` or ``IDLE_FILL``.

    Raises:
        ValueError: If the choice towers are a bare string, empty or repeat a tower, a tower is
            not one of the nine, or the initiation tower is also a choice tower.
    """
    choices = _choice_tower_set(choice_towers)
    unknown = sorted({initiation_tower, *choices} - set(ARMS9))
    if unknown:
        raise ValueError(f"unknown towers {unknown}; expected a subset of {ARMS9}")
    if initiation_tower in choices:
        raise ValueError(f"{initiation_tower} cannot be both the initiation and a choice tower")

    def fill(port: str) -> str:
        if port == initiation_tower:
            return INITIATION_FILL
        return CHOICE_FILL if port in choices else IDLE_FILL

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
    ax.add_patch(
        Polygon(
            platform_outline(), closed=True, facecolor=PLATFORM_COLOR, edgecolor="none", zorder=1
        )
    )


def _draw_tower(ax: Axes, port: str, fill: str) -> None:
    center = port_center(port)
    # Unclipped: the axes stop at the floor, and half of the outer towers' stroke lies beyond it.
    ax.add_patch(
        Polygon(
            octagon_vertices(center, NODE_RADIUS),
            closed=True,
            facecolor=fill,
            edgecolor="black",
            linewidth=2.0,
            zorder=2,
            clip_on=False,
        )
    )
    ax.text(
        *center, port, ha="center", va="center", fontsize=LABEL_FONTSIZE, color="black", zorder=3
    )


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
        )
        for color, label in ((INITIATION_FILL, "Initiation tower"), (CHOICE_FILL, "Choice tower"))
    ]


def draw_maze_schematic(
    choice_towers: Collection[str],
    initiation_tower: str = DEFAULT_INITIATION_TOWER,
    *,
    ax: Axes | None = None,
    title: str | None = None,
    legend: bool = True,
) -> Axes:
    """Draw the maze for one problem: towers on the floor, filled by role, text in Arial.

    Whoever saves the figure decides how its text is stored. :func:`plot_maze_schematic` keeps
    it editable; when saving a figure of your own, wrap the save in
    ``mpl.rc_context(schematic_rc())`` for the same.

    Args:
        choice_towers (Collection[str]): The choice towers, e.g. ``["A2", "B1", "C3"]``.
        initiation_tower (str): The initiation tower. Default ``"B2"``.
        ax (Axes | None): Axes to draw into; a new 4.8 x 3.0 inch figure is created if None.
            Default None.
        title (str | None): Axes title; none if None. Default None.
        legend (bool): Put the tower-role legend to the right of the maze. Default True.

    Returns:
        Axes: The axes the schematic was drawn on.

    Raises:
        ValueError: As for :func:`port_fills`.
    """
    fills = port_fills(choice_towers, initiation_tower)
    with mpl.rc_context(schematic_rc()), axes_or_new(ax, figsize=(4.8, 3.0)) as ax:
        _draw_platform(ax)
        for port in ARMS9:
            _draw_tower(ax, port, fills[port])
        if title is not None:
            # the title artist exists before this function, so set its font directly
            ax.set_title(title, pad=6, fontfamily=[schematic_font()])
        if legend:
            ax.legend(
                handles=role_legend_handles(),
                loc="upper left",
                bbox_to_anchor=(1.02, 1),
                fontsize=LEGEND_FONTSIZE,
                frameon=False,
            )
        # Limits from the floor, not the data, so every schematic has the same size.
        outline = platform_outline()
        ax.set_xlim(outline[:, 0].min(), outline[:, 0].max())
        ax.set_ylim(outline[:, 1].min(), outline[:, 1].max())
        ax.set_aspect("equal")
        ax.axis("off")
    return ax


def plot_maze_schematic(
    choice_towers: Collection[str],
    initiation_tower: str = DEFAULT_INITIATION_TOWER,
    *,
    title: str | None = None,
    save_path: str | Path | None = None,
    formats: tuple[str, ...] = FIGURE_FORMATS,
) -> None:
    """One schematic as its own figure, saved as ``<save_path>.<format>`` or shown.

    Text is in Arial and stays editable in the PDF and SVG files.

    Args:
        choice_towers (Collection[str]): The choice towers, e.g. ``["A2", "B1", "C3"]``.
        initiation_tower (str): The initiation tower. Default ``"B2"``.
        title (str | None): Axes title; none if None. Default None.
        save_path (str | Path | None): Base path without extension; None shows the figure.
            Default None.
        formats (tuple[str, ...]): File formats to write. Default ``FIGURE_FORMATS`` (pdf, png and
            svg).

    Raises:
        ValueError: As for :func:`port_fills`.
    """
    with mpl.rc_context(schematic_rc()):  # the PDF/SVG font embedding is read when saving
        ax = draw_maze_schematic(choice_towers, initiation_tower, title=title)
        save_or_show(ax.figure, save_path, formats=formats)
