"""Drawing and output helpers shared by the figure modules.

One place for the things every plotting module needs: palette colours, despining,
x-jitter, axes that may or may not be supplied by the caller, titles, and saving or showing
finished figures.

A title on a figure is short: its heading and, for a panel that shows a mean with error, one
line saying which statistic across which units and the n (:func:`stats_line`). Everything else
a title used to carry (scope, settings, caveats) goes to ``<name>.caption.txt``, written next to
every saved figure (:func:`set_title`, :func:`add_caption`), together with the scope of an
output set (:func:`figure_banner`). Figures a function creates itself are closed when drawing or
saving raises; figures passed in through ``ax=`` belong to the caller and are never
closed here.
"""

from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from types import MappingProxyType
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib._pylab_helpers import Gcf  # private matplotlib API; may need updating on an upgrade

from src.behavior_visualization.plot_legends import tidy_legends
from src.behavior_visualization.plot_style import MOUSE_COLORS

# Every figure is written in these formats: vector files for editing, PNG for quick looks.
FIGURE_FORMATS = ("pdf", "png", "svg")
# Dots per inch by file format; a format not listed here is saved at matplotlib's default.
DEFAULT_DPI = MappingProxyType({"png": 300})
CAPTION_SUFFIX = ".caption.txt"
# Formats and resolution of the figures of write_outputs (the review and summary figures).
OUTPUT_FORMATS = ("png", "pdf", "svg")
OUTPUT_DPI = MappingProxyType({"png": 180, "pdf": 180})
# Caption lines are kept on the figure as invisible texts with this gid, so they travel with it.
CAPTION_GID = "figure-caption"


def palette_color(index: int, palette: Sequence[str] | None = None) -> str:
    """Colour of the *index*-th item, cycling past the end of *palette* (default: ``MOUSE_COLORS``).

    A module whose palette a notebook may replace passes its own module-level palette
    here, looked up at call time.

    Args:
        index (int): Position of the item; wraps around modulo the palette length.
        palette (Sequence[str] | None): Colours to cycle through. Default None, meaning
            ``MOUSE_COLORS``.

    Returns:
        str: The colour at ``index % len(palette)``.
    """
    palette = MOUSE_COLORS if palette is None else palette
    return palette[index % len(palette)]


def mouse_color_map(mice: Iterable[str]) -> dict[str, str]:
    """Map each mouse to a palette colour, cycling when there are more mice than colours.

    Args:
        mice (Iterable[str]): Mouse identifiers, in the order that fixes their colours.

    Returns:
        dict[str, str]: ``{mouse: colour}`` using ``MOUSE_COLORS``.
    """
    return {m: palette_color(i) for i, m in enumerate(mice)}


def despine(ax: plt.Axes) -> None:
    """Hide the top and right spines.

    Args:
        ax (plt.Axes): Axes to modify in place.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


JITTER_SEED = 0


def jitter_rng(seed: int = JITTER_SEED) -> np.random.Generator:
    """A generator for the jitter of one figure; one per figure keeps it reproducible.

    Args:
        seed (int): Seed for the generator. Default ``JITTER_SEED`` (0).

    Returns:
        np.random.Generator: A new generator from ``np.random.default_rng(seed)``.
    """
    return np.random.default_rng(seed)


def jittered(
    xs: np.ndarray, half_width: float, rng: np.random.Generator | None = None
) -> np.ndarray:
    """*xs* plus uniform noise of half-width *half_width* (unchanged if it is ``<= 0``).

    Pass one *rng* for all the lines of a figure so each gets its own noise; the default is a
    fresh generator seeded with ``JITTER_SEED``, so the same input always gives the same
    output. Nothing is drawn when the jitter is off.

    Args:
        xs (np.ndarray): Positions to jitter, shape ``(n,)``.
        half_width (float): Half-width of the uniform noise; ``<= 0`` turns jitter off.
        rng (np.random.Generator | None): Generator to draw from. Default None, meaning a
            fresh ``jitter_rng()``.

    Returns:
        np.ndarray: ``xs`` plus noise, shape ``(n,)``; the input array itself when jitter is
        off.
    """
    if half_width > 0:
        rng = jitter_rng() if rng is None else rng
        return xs + rng.uniform(-half_width, half_width, size=len(xs))
    return xs


@contextmanager
def axes_or_new(ax: plt.Axes | None, **subplots_kwargs) -> Iterator[plt.Axes]:
    """Yield *ax*, or the axes of a new figure made with ``plt.subplots(**subplots_kwargs)``.

    A figure made here is closed if the body raises (the error is re-raised), so a
    failed draw leaves nothing open. A figure that is returned normally stays open for
    the caller. An *ax* passed in belongs to the caller and is left alone either way.

    Args:
        ax (plt.Axes | None): Axes to draw into, or None to create a new figure.
        **subplots_kwargs: Passed to ``plt.subplots`` when a figure is created; ignored if
            *ax* is given.

    Yields:
        plt.Axes: *ax*, or the axes of the new figure.
    """
    if ax is not None:
        yield ax
        return
    fig, new_ax = plt.subplots(**subplots_kwargs)
    try:
        yield new_ax
    except BaseException:
        plt.close(fig)
        raise


# Set only by figure_banner; read where figures are written, so a banner reaches every figure of
# a run without each plotting function taking a parameter for it.
_FIGURE_BANNER: ContextVar[str | None] = ContextVar("figure_banner", default=None)


@contextmanager
def figure_banner(text: str) -> Iterator[None]:
    """Start the caption of every figure saved inside the block with *text*.

    For output sets that differ from the standard one in what they contain (e.g. "without
    MY_175_N"): the line is added when the figure is written, so no plot needs to know about
    it and none can be forgotten. Blocks nest; the innermost text wins.

    Args:
        text (str): The first caption line of each figure.

    Yields:
        None: Use as ``with figure_banner("without MY_175_N"): ...``.
    """
    token = _FIGURE_BANNER.set(text)
    try:
        yield
    finally:
        _FIGURE_BANNER.reset(token)


def current_figure_banner() -> str | None:
    """The text of the innermost active :func:`figure_banner`, or ``None`` outside one."""
    return _FIGURE_BANNER.get()


# --- titles and captions ------------------------------------------------------------------------


def count_text(n: int, singular: str, plural: str) -> str:
    """``"1 mouse"``, ``"2 mice"``: *n* with the noun in the right number."""
    return f"{n} {singular if n == 1 else plural}"


def stats_line(average: str, unit: str, counts: Sequence[tuple[int, str, str]]) -> str:
    """The statistics line of a panel title: what the marks show, across what, and the n.

    Args:
        average (str): The statistic drawn, e.g. ``"mean ± se"``, ``"mean ± SD"`` or ``"mean"``.
        unit (str): What it is taken across, plural, e.g. ``"mice"`` or ``"events"``.
        counts (Sequence[tuple[int, str, str]]): ``(n, singular, plural)`` of each count to
            state, the averaged unit first.

    Returns:
        str: E.g. ``"mean ± se across events, n=120 events, n=13 mice"``.
    """
    numbers = ", ".join(f"n={count_text(*count)}" for count in counts)
    return f"{average} across {unit}, {numbers}"


def error_drawn(*errors) -> bool:
    """Whether any of *errors* (arrays or numbers of SE/SD values) is finite, i.e. drawn.

    A title states "± se" only when an error bar or band is actually drawn.
    """
    return any(np.isfinite(np.asarray(e, dtype=float)).any() for e in errors)


class Title(NamedTuple):
    """A figure or panel title split into what is drawn and what goes to the caption.

    Attributes:
        heading (str): The name of the figure or panel, one line.
        stats (str | None): The :func:`stats_line` of a panel showing a mean with error, drawn
            under the heading; ``None`` if the panel shows none. Default None.
        caption (tuple[str, ...]): Everything else (scope, settings, caveats), one caption line
            each. Default empty.
    """

    heading: str
    stats: str | None = None
    caption: tuple[str, ...] = ()

    @property
    def shown(self) -> str:
        """The drawn text: the heading, and the statistics line under it if there is one."""
        return self.heading if self.stats is None else f"{self.heading}\n{self.stats}"


def title_from_text(text: str, stats: str | None = None) -> Title:
    """A :class:`Title` from a caller's free text: its first line is the heading.

    For titles given as one string (e.g. from a notebook): any further lines go to the caption.

    Args:
        text (str): The title text; may span several lines.
        stats (str | None): The statistics line to draw under the heading. Default None.

    Returns:
        Title: Heading, *stats*, and the remaining non-empty lines as caption lines.
    """
    heading, *rest = text.split("\n")
    return Title(heading, stats, tuple(line.strip() for line in rest if line.strip()))


def add_caption(fig: plt.Figure, *lines: str) -> None:
    """Add *lines* to the caption of *fig*; they are written beside it and never drawn.

    Args:
        fig (plt.Figure): The figure the lines describe.
        *lines (str): Caption lines; empty ones are skipped.
    """
    for line in lines:
        if line:
            fig.text(0, 0, line, visible=False, gid=CAPTION_GID)


def set_title(ax: plt.Axes, title: Title, **kwargs) -> None:
    """Draw the shown part of *title* on *ax* and add the rest to its figure's caption.

    Args:
        ax (plt.Axes): The panel.
        title (Title): The title.
        **kwargs: Passed to ``ax.set_title`` (e.g. ``pad``).
    """
    ax.set_title(title.shown, **kwargs)
    add_caption(ax.figure, *title.caption)


def set_suptitle(fig: plt.Figure, title: Title, **kwargs) -> None:
    """As :func:`set_title`, for the figure title.

    Args:
        fig (plt.Figure): The figure.
        title (Title): The title.
        **kwargs: Passed to ``fig.suptitle``.
    """
    fig.suptitle(title.shown, **kwargs)
    add_caption(fig, *title.caption)


def figure_caption(fig: plt.Figure) -> str:
    """The caption of *fig*: the active banner, then its caption lines, each line once.

    Args:
        fig (plt.Figure): The figure.

    Returns:
        str: The lines joined by newlines; ``""`` when there is nothing to say.
    """
    banner = current_figure_banner()
    lines = [banner] if banner else []
    lines += [t.get_text() for t in fig.texts if t.get_gid() == CAPTION_GID]
    return "\n".join(dict.fromkeys(lines))


def caption_path(base: str | Path) -> Path:
    """``<base>.caption.txt``, the caption file of the figure saved as ``<base>.<format>``."""
    return Path(f"{base}{CAPTION_SUFFIX}")


def figure_paths(base: str | Path, formats: Iterable[str] = FIGURE_FORMATS) -> list[Path]:
    """The figure files a save of ``<base>`` writes, in *formats* order (without the caption)."""
    return [Path(f"{base}.{fmt}") for fmt in formats]


def _write_figure_files(
    fig: plt.Figure, base: str | Path, formats: Iterable[str], dpi: Mapping[str, int]
) -> list[Path]:
    """Save *fig* as ``<base>.<format>`` for each of *formats*, in order; return the paths.

    The legends are tidied first (see
    :func:`~src.behavior_visualization.plot_legends.tidy_legends`).
    ``<base>.caption.txt`` is written too, empty when the figure has no caption, so every
    figure has one.
    """
    paths = figure_paths(base, formats)
    add_caption(fig, *tidy_legends(fig))
    for path in paths:
        fmt = path.suffix.removeprefix(".")
        dpi_kwargs = {"dpi": dpi[fmt]} if fmt in dpi else {}
        fig.savefig(path, bbox_inches="tight", **dpi_kwargs)
    caption = figure_caption(fig)
    caption_path(base).write_text(f"{caption}\n" if caption else "")
    return paths


def save_or_show(
    figures: plt.Figure | Mapping[str, plt.Figure],
    save_path: str | Path | None,
    *,
    formats: tuple[str, ...] = FIGURE_FORMATS,
    dpi: Mapping[str, int] = DEFAULT_DPI,
    make_parent: bool = True,
) -> None:
    """Save *figures* as ``<save_path>.<format>`` (in *formats* order), or show them if no path.

    Each saved figure also gets ``<save_path>.caption.txt`` (see :func:`figure_caption`).

    *figures* is one figure, or ``{suffix: figure}`` for several that share a base path
    (saved as ``<save_path><suffix>.<format>``). With no path they are all shown together,
    in a single ``plt.show()``. Every figure is closed afterwards, also when saving
    raises. With ``make_parent=False`` the parent directory must already exist and
    *save_path* is used as given.

    Args:
        figures (plt.Figure | Mapping[str, plt.Figure]): One figure, or ``{suffix: figure}``.
        save_path (str | Path | None): Base path without extension; None shows the figures
            instead of saving.
        formats (tuple[str, ...]): File formats to write. Default ``FIGURE_FORMATS`` (pdf, png
            and svg).
        dpi (Mapping[str, int]): Dots per inch by format; formats not listed use matplotlib's
            default. Default ``DEFAULT_DPI`` (png at 300).
        make_parent (bool): Create *save_path*'s parent directory (and convert the path to a
            ``Path``). Default True.
    """
    named = {"": figures} if isinstance(figures, plt.Figure) else dict(figures)
    try:
        if save_path is None:
            plt.show()
        else:
            base = Path(save_path) if make_parent else save_path
            if make_parent:
                base.parent.mkdir(parents=True, exist_ok=True)
            for suffix, fig in named.items():
                _write_figure_files(fig, f"{base}{suffix}", formats, dpi)
    finally:
        for fig in named.values():
            plt.close(fig)


class TableOutput(NamedTuple):
    """A table to write as ``<directory>/<filename>`` (CSV).

    Attributes:
        filename (str): File name within the output directory, including extension.
        frame (pd.DataFrame): The table to write.
        index (bool): Whether to write the DataFrame index. Default False.
    """

    filename: str
    frame: pd.DataFrame
    index: bool = False


class FigureOutput(NamedTuple):
    """A finished figure to save as ``<directory>/<name>.png``, ``.pdf`` and ``.svg``.

    Attributes:
        name (str): File name without extension.
        figure (plt.Figure): The figure to save; it is closed after saving.
    """

    name: str
    figure: plt.Figure


def _open_figures() -> set[plt.Figure]:
    """The figure objects pyplot currently holds open (by identity, not by number)."""
    return {manager.canvas.figure for manager in Gcf.get_all_fig_managers()}


def write_outputs(
    outputs: Iterable[TableOutput | FigureOutput], directory: Path, *, show: bool
) -> list[Path]:
    """Write tables and save figures in the order *outputs* yields them.

    *outputs* is consumed lazily: each figure is saved and closed before the next
    output is built, so the builders do no file I/O and this is the only place that
    touches the disk. If anything raises part-way, the files written so far are kept
    and every figure opened while *outputs* was being consumed is closed (a builder
    that raises cannot hand back its unfinished figure, so it is found in pyplot).
    Figures that were already open beforehand are left alone.

    Args:
        outputs (Iterable[TableOutput | FigureOutput]): Tables and figures to write; may be a
            generator.
        directory (Path): Existing directory the files are written into.
        show (bool): Also display each figure after saving it.

    Returns:
        list[Path]: The saved figure paths, PNG, PDF then SVG for each figure (table and
        caption paths are not included).

    Raises:
        TypeError: If an item of *outputs* is neither a ``TableOutput`` nor a ``FigureOutput``.
    """
    already_open = _open_figures()
    paths: list[Path] = []
    try:
        for output in outputs:
            if isinstance(output, TableOutput):
                output.frame.to_csv(directory / output.filename, index=output.index)
            elif isinstance(output, FigureOutput):
                paths += _save_figure(output.figure, directory, output.name, show=show)
            else:
                raise TypeError(
                    f"expected TableOutput or FigureOutput, got {type(output).__name__}"
                )
    except BaseException:
        for fig in _open_figures() - already_open:
            plt.close(fig)
        raise
    return paths


def _save_figure(fig: plt.Figure, directory: Path, name: str, *, show: bool) -> list[Path]:
    """Tight-layout *fig*, save ``<directory>/<name>`` as PNG, PDF and SVG, then close it.

    Returns the three saved paths (PNG first); the caption file is written too. With ``show``
    the figure is also displayed after saving. The figure is closed even if saving raises.
    """
    try:
        fig.tight_layout()
        paths = _write_figure_files(fig, directory / name, OUTPUT_FORMATS, OUTPUT_DPI)
        if show:
            plt.show()
    finally:
        plt.close(fig)
    return paths
