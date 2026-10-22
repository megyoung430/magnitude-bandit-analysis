"""A Nature/Cell figure style applied to finished figures, so no plot function needs to know it.

Two parts work together. :func:`paper_style` is an rc context (fonts, line and tick widths,
editable text in PDF and SVG). :func:`polish` then walks a figure a plot function has already
drawn and fixes what the plot hard-coded: text sizes, line widths, marker sizes, legend frames,
colours, spines trimmed to the outermost ticks, the figure size (one of the journal column
widths) and long titles, which it moves out of the figure into a caption string.
:func:`paper_figures` switches both on for a block, and the save step in ``plot_helpers`` reads
:func:`in_paper_mode` to polish each figure and write the caption beside it.

The trimmed spines follow the ``set_bounds`` idea of sciplotlib
(https://github.com/Timothysit/sciplotlib); the code here is our own.
"""

import re
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar

import matplotlib as mpl
import matplotlib.layout_engine
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import to_hex, to_rgba
from matplotlib.lines import Line2D

from src.behavior_visualization.plot_style import MOUSE_COLORS

MM_PER_INCH = 25.4
PAPER_DPI = 600
# Column widths of Nature and Cell in mm.
COLUMN_WIDTH_MM = {"single": 89, "one_half": 120, "double": 183}
MAX_HEIGHT_MM = 230
# Narrower figures are kept from becoming taller than this multiple of their width; only the
# double-column grids of per-mouse panels are allowed to run down the page.
MAX_NARROW_ASPECT = 0.8

FONT_SIZE = 7
TITLE_SIZE = 8
SPINE_OFFSET_PT = 2.5
MAX_MARKER_SIZE = 3
BAND_ALPHA = 0.25
# A title longer than this, or on several lines, is a description and goes to the caption.
MAX_TITLE_CHARS = 36

# Okabe-Ito colours without black, lower case so they compare with ``to_hex`` output.
BLUE, SKY, GREEN, YELLOW, ORANGE, VERMILLION, PURPLE = (
    "#0072b2",
    "#56b4e9",
    "#009e73",
    "#f0e442",
    "#e69f00",
    "#d55e00",
    "#cc79a7",
)
OKABE_ITO = (BLUE, SKY, GREEN, YELLOW, ORANGE, VERMILLION, PURPLE)
# Mice cycle through the colours not used for the best/second roles first.
MOUSE_CYCLE = (ORANGE, SKY, VERMILLION, PURPLE, YELLOW, BLUE, GREEN)

# The original hues of the best / second / third roles, mapped to colourblind-safe ones that
# keep their meaning (blue best, green second, grey third).
ROLE_RECOLOUR = {"#5da5da": BLUE, "#60bd68": GREEN, "#7f7f7f": "#7f7f7f"}


def _recolour_map() -> dict[str, str]:
    """``{original hex: paper hex}`` for the role colours and every mouse colour."""
    mice = {c.lower(): MOUSE_CYCLE[i % len(MOUSE_CYCLE)] for i, c in enumerate(MOUSE_COLORS)}
    return {**mice, **ROLE_RECOLOUR}


_PAPER_MODE: ContextVar[bool] = ContextVar("paper_mode", default=False)


def in_paper_mode() -> bool:
    """Whether the code is running inside :func:`paper_figures`."""
    return _PAPER_MODE.get()


def paper_rc() -> dict:
    """The rc settings of the style."""
    return {
        "font.family": "sans-serif",
        # Arial or Helvetica where installed; Liberation Sans is metric-compatible on Linux
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE,
        "legend.title_fontsize": FONT_SIZE,
        "legend.frameon": False,
        "legend.handlelength": 1.5,
        "axes.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "lines.linewidth": 0.75,
        "lines.markersize": MAX_MARKER_SIZE,
        "patch.linewidth": 0.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.pad": 2,
        "ytick.major.pad": 2,
        "savefig.dpi": PAPER_DPI,
        # editable text: TrueType in PDF, live <text> elements in SVG
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }


@contextmanager
def paper_style() -> Iterator[None]:
    """Apply the Nature/Cell rc settings for the block; the previous settings return after it."""
    with mpl.rc_context(paper_rc()):
        yield


@contextmanager
def paper_figures() -> Iterator[None]:
    """:func:`paper_style` plus the flag :func:`in_paper_mode` reads, for the block."""
    with ExitStack() as stack:
        stack.enter_context(paper_style())
        token = _PAPER_MODE.set(True)
        stack.callback(_PAPER_MODE.reset, token)
        yield


def figure_size(kind: str, aspect: float) -> tuple[float, float]:
    """Width and height in inches of a figure of column width *kind*.

    Args:
        kind (str): ``"single"`` (89 mm), ``"one_half"`` (120 mm) or ``"double"`` (183 mm).
        aspect (float): Height divided by width. The height is capped at ``MAX_HEIGHT_MM`` so
            a tall grid of panels still fits a page, and for the single and one-and-a-half
            column widths at ``MAX_NARROW_ASPECT`` times the width.

    Returns:
        tuple[float, float]: ``(width, height)`` in inches.

    Raises:
        ValueError: If *kind* is not one of the column widths.
    """
    if kind not in COLUMN_WIDTH_MM:
        raise ValueError(f"unknown figure kind {kind!r}; use one of {sorted(COLUMN_WIDTH_MM)}")
    width_mm = COLUMN_WIDTH_MM[kind]
    if kind != "double":
        aspect = min(aspect, MAX_NARROW_ASPECT)
    height_mm = min(aspect * width_mm, MAX_HEIGHT_MM)
    return width_mm / MM_PER_INCH, height_mm / MM_PER_INCH


_TITLE_CASE_WORD = re.compile(r"\(?[A-Z][a-z]+[0-9]*\)?")


def sentence_case(text: str) -> str:
    """*text* with its Title-Case words lower-cased, except the first word.

    Words with other capitals or underscores (``P(best)``, ``MY_175_L``) are left alone, so ids
    and symbols survive.
    """
    words = text.split(" ")
    return " ".join(
        w.lower() if i > 0 and _TITLE_CASE_WORD.fullmatch(w) else w for i, w in enumerate(words)
    )


# --- caption ------------------------------------------------------------------------------------


def _is_description(title: str) -> bool:
    return "\n" in title or len(title) > MAX_TITLE_CHARS


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _move_titles_to_caption(fig: plt.Figure) -> list[str]:
    """Remove the suptitle, figure-level texts and descriptive axes titles; return their text."""
    moved = []
    suptitle = fig.get_suptitle()
    if suptitle:
        moved.append(_one_line(suptitle))
        fig.suptitle("")
    for text in list(fig.texts):
        if text.get_text():
            moved.append(_one_line(text.get_text()))
        text.remove()
    for ax in fig.axes:
        for title in (ax.title, ax._left_title, ax._right_title):
            if title.get_text() and _is_description(title.get_text()):
                moved.append(_one_line(title.get_text()))
                title.set_text("")
    return moved


def _caption(banner: str | None, moved: Sequence[str]) -> str:
    parts = ([banner] if banner else []) + list(moved)
    return "\n".join(dict.fromkeys(parts))


# --- size ---------------------------------------------------------------------------------------


def _is_colorbar(ax: plt.Axes) -> bool:
    return ax.get_label() == "<colorbar>"


def _panels(fig: plt.Figure) -> list[plt.Axes]:
    return [ax for ax in fig.axes if not _is_colorbar(ax)]


def _has_outside_legend(ax: plt.Axes) -> bool:
    legend = ax.get_legend()
    return legend is not None and legend.get_bbox_to_anchor().x0 > ax.bbox.x1 + 1


def column_kind(fig: plt.Figure) -> str:
    """The column width a figure gets, by its number of panel columns.

    One panel is single-column; two panels, a legend beside the axes or a second y axis need
    one and a half columns; three or more panel columns take the full page width.
    """
    panels = _panels(fig)
    positions = {tuple(np.round(ax.get_position().bounds, 2)) for ax in panels}
    columns = len({round(ax.get_position().x0, 2) for ax in panels})
    has_twin = len(positions) < len(panels)
    if columns >= 3:
        return "double"
    if columns == 2 or has_twin or any(_has_outside_legend(ax) for ax in panels):
        return "one_half"
    return "single"


def _resize(fig: plt.Figure) -> None:
    """Set the column width for the figure, keeping its height-to-width ratio."""
    width, height = fig.get_size_inches()
    new = figure_size(column_kind(fig), height / width)
    fig.set_size_inches(*new)
    _set_layout(fig)


def _set_layout(fig: plt.Figure) -> None:
    """Constrained layout; tight layout for a figure whose colorbar was made without constrained.

    Matplotlib cannot change the layout engine of a figure to constrained once a colorbar has
    been attached under another engine, and raises ``RuntimeError`` then.
    """
    try:
        fig.set_layout_engine("constrained", w_pad=2 / 72, h_pad=2 / 72, wspace=0.04, hspace=0.04)
    except RuntimeError:
        fig.set_layout_engine("tight", pad=0.4)


def _is_constrained(fig: plt.Figure) -> bool:
    return isinstance(fig.get_layout_engine(), mpl.layout_engine.ConstrainedLayoutEngine)


# --- per-artist rules ---------------------------------------------------------------------------


def _thin_width(width: float) -> float:
    """Line width in the paper: 0.5 for helper lines, 0.75 for data, 1.0 for means."""
    if width <= 1.0:
        return 0.5
    return 0.75 if width < 2.0 else 1.0


def _recoloured(color, mapping: dict[str, str]):
    """*color* mapped through *mapping* (keeping its alpha), or ``None`` if it is not mapped."""
    try:
        rgba = to_rgba(color)
    except ValueError:
        return None
    new = mapping.get(to_hex(rgba[:3]))
    return None if new is None else to_rgba(new, alpha=rgba[3])


def _recolour_rows(colors, mapping: dict[str, str]):
    """The (n, 4) colour array of a collection with mapped rows replaced; ``None`` if no change."""
    rows = [_recoloured(c, mapping) or tuple(c) for c in np.atleast_2d(colors)]
    changed = any(not np.allclose(a, b) for a, b in zip(rows, np.atleast_2d(colors), strict=True))
    return rows if changed and len(rows) else None


def _style_line(line: Line2D, mapping: dict[str, str]) -> None:
    line.set_linewidth(_thin_width(line.get_linewidth()))
    line.set_markersize(min(line.get_markersize(), MAX_MARKER_SIZE))
    line.set_markeredgewidth(0.5)
    for getter, setter in (
        (line.get_color, line.set_color),
        (line.get_markerfacecolor, line.set_markerfacecolor),
        (line.get_markeredgecolor, line.set_markeredgecolor),
    ):
        new = _recoloured(getter(), mapping)
        if new is not None:
            setter(new)


def _style_collection(collection, mapping: dict[str, str]) -> None:
    if isinstance(collection, PolyCollection):
        # an error band: translucent, no outline
        collection.set_linewidth(0)
        if collection.get_alpha() is not None:
            collection.set_alpha(BAND_ALPHA)
    elif isinstance(collection, LineCollection):
        collection.set_linewidth([_thin_width(w) for w in collection.get_linewidth()])
    if isinstance(collection, (PolyCollection, LineCollection)):
        for getter, setter in (
            (collection.get_facecolor, collection.set_facecolor),
            (collection.get_edgecolor, collection.set_edgecolor),
        ):
            rows = _recolour_rows(getter(), mapping) if len(getter()) else None
            if rows is not None:
                setter(rows)


def _style_patch(patch, mapping: dict[str, str]) -> None:
    patch.set_linewidth(min(patch.get_linewidth(), 0.5))
    for getter, setter in (
        (patch.get_facecolor, patch.set_facecolor),
        (patch.get_edgecolor, patch.set_edgecolor),
    ):
        new = _recoloured(getter(), mapping)
        if new is not None:
            setter(new)


def _style_text(text, mapping: dict[str, str]) -> None:
    text.set_fontsize(FONT_SIZE)
    new = _recoloured(text.get_color(), mapping)
    if new is not None:
        text.set_color(new)


def _is_below(ax: plt.Axes, legend) -> bool:
    return legend.get_bbox_to_anchor().y1 < ax.bbox.y0 + 1


def _move_legend_below_figure(ax: plt.Axes) -> None:
    """Replace a legend hanging below *ax* by a figure legend the layout makes room for.

    A legend anchored below the axes in axes coordinates ignores the layout, so it overlaps the
    x label once the figure is resized. Two columns keep it narrower than the axes.
    """
    legend = ax.get_legend()
    handles, labels = legend.legend_handles, [t.get_text() for t in legend.get_texts()]
    legend.remove()
    ax.figure.legend(handles, labels, loc="outside lower center", ncols=2, frameon=False)


def _style_legend(legend, mapping: dict[str, str]) -> None:
    legend.set_frame_on(False)
    for text in [*legend.get_texts(), legend.get_title()]:
        text.set_fontsize(FONT_SIZE)
    for line in legend.get_lines():
        _style_line(line, mapping)


# --- axes ---------------------------------------------------------------------------------------


def _ticks_within(ticks: Sequence[float], limits: tuple[float, float]) -> list[float]:
    low, high = sorted(limits)
    return [t for t in ticks if low <= t <= high]


def _trim_spines(ax: plt.Axes) -> None:
    """Shorten the left and bottom spines to span only the outermost ticks."""
    y_spine = ax.spines["right"] if _is_twin_y_axes(ax) else ax.spines["left"]
    pairs = [(y_spine, ax.get_yticks(), ax.get_ylim())]
    if ax.xaxis.get_visible():
        pairs.append((ax.spines["bottom"], ax.get_xticks(), ax.get_xlim()))
    for spine, ticks, limits in pairs:
        inside = _ticks_within(ticks, limits)
        if inside:
            spine.set_bounds(min(inside), max(inside))


def _is_twin_y_axes(ax: plt.Axes) -> bool:
    """An axes made by ``twinx``: its y axis is on the right and it has no x axis of its own."""
    return not ax.xaxis.get_visible() and ax.yaxis.get_label_position() == "right"


def _style_frame(ax: plt.Axes) -> None:
    """Thin spines; open (offset, trimmed, no top and one y side) for line and bar axes.

    Image and colorbar axes keep their box. A ``twinx`` axes keeps only its right spine.
    """
    boxed = bool(ax.images) or _is_colorbar(ax)
    twin = _is_twin_y_axes(ax)
    shown = {"left", "bottom"} if not twin else {"right"}
    for name, spine in ax.spines.items():
        spine.set_linewidth(0.5)
        if not boxed and name not in shown:
            spine.set_visible(False)
    if boxed:
        return
    for name in shown:
        ax.spines[name].set_position(("outward", SPINE_OFFSET_PT))
    _trim_spines(ax)


def _style_axes(ax: plt.Axes, mapping: dict[str, str]) -> None:
    ax.tick_params(
        which="major", direction="out", length=2.5, width=0.5, labelsize=FONT_SIZE, pad=2
    )
    ax.set_xlabel(sentence_case(ax.get_xlabel()))
    ax.set_ylabel(sentence_case(ax.get_ylabel()))
    if ax.get_title():
        ax.set_title(sentence_case(ax.get_title()))
    for artist in ax.lines:
        _style_line(artist, mapping)
    for artist in ax.collections:
        _style_collection(artist, mapping)
    for artist in ax.patches:
        _style_patch(artist, mapping)
    legend = ax.get_legend()
    if legend is not None and _is_below(ax, legend) and _is_constrained(ax.figure):
        _move_legend_below_figure(ax)
    elif legend is not None:
        _style_legend(legend, mapping)
    if ax.axison:
        _style_frame(ax)


def polish(fig: plt.Figure, banner: str | None = None) -> str:
    """Restyle *fig* in place to the paper style and return the text moved out of it.

    Run it inside :func:`paper_style` so artists created afterwards share the settings. Sets the
    column width (see :func:`column_kind`), constrained layout, 7 pt text, thin lines, small
    markers, unboxed legends, sentence-case axis labels and colourblind-safe colours; trims the
    spines to the ticks; and removes the suptitle, figure-level notes and long axes titles.

    Args:
        fig (plt.Figure): The finished figure; changed in place.
        banner (str | None): A marker for the figure's scope (e.g. ``"without MY_1_A"``) that
            goes into the caption instead of onto the figure. Default None.

    Returns:
        str: The caption: the banner, suptitle, notes and long titles, one per line; empty when
        nothing was moved.
    """
    caption = _caption(banner, _move_titles_to_caption(fig))
    _resize(fig)
    mapping = _recolour_map()
    for ax in fig.axes:
        _style_axes(ax, mapping)
    for text in fig.findobj(mpl.text.Text):
        _style_text(text, mapping)
    for legend in fig.legends:
        _style_legend(legend, mapping)
    for ax in fig.axes:
        for title in (ax.title, ax._left_title, ax._right_title):
            title.set_fontsize(TITLE_SIZE)
    return caption
