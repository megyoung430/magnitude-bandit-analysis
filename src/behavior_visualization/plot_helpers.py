"""Drawing and output helpers shared by the figure modules.

One place for the things every plotting module needs: palette colours, despining,
x-jitter, axes that may or may not be supplied by the caller, and saving or showing
finished figures. Figures a function creates itself are closed when drawing or
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

from src.behavior_visualization.plot_style import MOUSE_COLORS

# Dots per inch by file format; a format not listed here is saved at matplotlib's default.
DEFAULT_DPI = MappingProxyType({"png": 300})


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
    """Mark every figure saved inside the block with *text*, in bold above its top-left corner.

    For output sets that differ from the standard one in what they contain (e.g. "without
    MY_175_N"): the marker is added when the figure is written, so no plot needs to know about
    it and none can be forgotten. Blocks nest; the innermost text wins.

    Args:
        text (str): The text to print on each figure.

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


@contextmanager
def _banner_drawn(fig: plt.Figure) -> Iterator[None]:
    """Draw the active banner on *fig* for the block and take it off again afterwards."""
    text = current_figure_banner()
    if text is None:
        yield
        return
    # y=1 with va="bottom" sits above the suptitle; bbox_inches="tight" grows the file to fit
    banner = fig.text(0.01, 1.0, text, ha="left", va="bottom", fontsize=11, weight="bold")
    try:
        yield
    finally:
        banner.remove()


def _write_figure_files(
    fig: plt.Figure, base: str | Path, formats: Iterable[str], dpi: Mapping[str, int]
) -> list[Path]:
    """Save *fig* as ``<base>.<format>`` for each of *formats*, in order; return the paths."""
    paths = []
    with _banner_drawn(fig):
        for fmt in formats:
            path = Path(f"{base}.{fmt}")
            dpi_kwargs = {"dpi": dpi[fmt]} if fmt in dpi else {}
            fig.savefig(path, bbox_inches="tight", **dpi_kwargs)
            paths.append(path)
    return paths


def save_or_show(
    figures: plt.Figure | Mapping[str, plt.Figure],
    save_path: str | Path | None,
    *,
    formats: tuple[str, ...] = ("pdf", "png"),
    dpi: Mapping[str, int] = DEFAULT_DPI,
    make_parent: bool = True,
) -> None:
    """Save *figures* as ``<save_path>.<format>`` (in *formats* order), or show them if no path.

    *figures* is one figure, or ``{suffix: figure}`` for several that share a base path
    (saved as ``<save_path><suffix>.<format>``). With no path they are all shown together,
    in a single ``plt.show()``. Every figure is closed afterwards, also when saving
    raises. With ``make_parent=False`` the parent directory must already exist and
    *save_path* is used as given.

    Args:
        figures (plt.Figure | Mapping[str, plt.Figure]): One figure, or ``{suffix: figure}``.
        save_path (str | Path | None): Base path without extension; None shows the figures
            instead of saving.
        formats (tuple[str, ...]): File formats to write. Default ``("pdf", "png")``.
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
    """A finished figure to save as ``<directory>/<name>.png`` and ``.pdf``.

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
        list[Path]: The saved figure paths, PNG then PDF for each figure (table paths are
        not included).

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
    """Tight-layout *fig*, save ``<directory>/<name>.png`` and ``.pdf``, then close it.

    Returns the two saved paths (PNG first). With ``show`` the figure is also
    displayed after saving. The figure is closed even if saving raises.
    """
    try:
        fig.tight_layout()
        paths = _write_figure_files(fig, directory / name, ("png", "pdf"), {"png": 180, "pdf": 180})
        if show:
            plt.show()
    finally:
        plt.close(fig)
    return paths
