"""Drawing and output helpers shared by the figure modules.

One place for the things every plotting module needs: palette colours, despining,
x-jitter, axes that may or may not be supplied by the caller, and saving or showing
finished figures. Figures a function creates itself are closed when drawing or
saving raises; figures passed in through ``ax=`` belong to the caller and are never
closed here.
"""

from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from types import MappingProxyType
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.behavior_visualization.plot_style import MOUSE_COLORS

# Dots per inch by file format; a format not listed here is saved at matplotlib's default.
DEFAULT_DPI = MappingProxyType({"png": 300})


def palette_color(index: int) -> str:
    """Palette colour for the *index*-th item, cycling past the end of the palette."""
    return MOUSE_COLORS[index % len(MOUSE_COLORS)]


def mouse_color_map(mice: Iterable[str]) -> dict[str, str]:
    """Map each mouse to a palette colour, cycling when there are more mice than colours."""
    return {m: palette_color(i) for i, m in enumerate(mice)}


def despine(ax: plt.Axes) -> None:
    """Hide the top and right spines."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def jittered(xs: np.ndarray, half_width: float) -> np.ndarray:
    """*xs* plus uniform noise of half-width *half_width* (unchanged if it is ``<= 0``).

    Draws from the global ``np.random`` stream, so ``np.random.seed`` makes it
    repeatable; nothing is drawn when the jitter is off, which leaves the stream alone.
    """
    if half_width > 0:
        return xs + np.random.uniform(-half_width, half_width, size=len(xs))
    return xs


@contextmanager
def axes_or_new(ax: plt.Axes | None, **subplots_kwargs) -> Iterator[plt.Axes]:
    """Yield *ax*, or the axes of a new figure made with ``plt.subplots(**subplots_kwargs)``.

    A figure made here is closed if the body raises (the error is re-raised), so a
    failed draw leaves nothing open. A figure that is returned normally stays open for
    the caller. An *ax* passed in belongs to the caller and is left alone either way.
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


def _write_figure_files(
    fig: plt.Figure, base: str | Path, formats: Iterable[str], dpi: Mapping[str, int]
) -> list[Path]:
    """Save *fig* as ``<base>.<format>`` for each of *formats*, in order; return the paths."""
    paths = []
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
    *save_path* must be a ``str``, used as given.
    """
    named = {"": figures} if isinstance(figures, plt.Figure) else dict(figures)
    try:
        if save_path is None:
            plt.show()
        else:
            if make_parent:
                base = Path(save_path)
                base.parent.mkdir(parents=True, exist_ok=True)
            elif isinstance(save_path, str):
                base = save_path
            else:
                raise TypeError(
                    f"save_path must be a str with make_parent=False, got {save_path!r}"
                )
            for suffix, fig in named.items():
                _write_figure_files(fig, f"{base}{suffix}", formats, dpi)
    finally:
        for fig in named.values():
            plt.close(fig)


class TableOutput(NamedTuple):
    """A table to write as ``<directory>/<filename>`` (CSV)."""

    filename: str
    frame: pd.DataFrame
    index: bool = False


class FigureOutput(NamedTuple):
    """A finished figure to save as ``<directory>/<name>.png`` and ``.pdf``."""

    name: str
    figure: plt.Figure


def write_outputs(
    outputs: Iterable[TableOutput | FigureOutput], directory: Path, *, show: bool
) -> list[Path]:
    """Write tables and save figures in the order *outputs* yields them.

    *outputs* is consumed lazily: each figure is saved and closed before the next
    output is built, so the builders do no file I/O and this is the only place that
    touches the disk. If anything raises part-way, the files written so far are kept
    and every figure opened while *outputs* was being consumed is closed (a builder
    that raises cannot hand back its unfinished figure, so it is found by number).
    Figures that were already open beforehand are left alone.

    Returns:
        The saved figure paths, PNG then PDF for each figure.
    """
    already_open = set(plt.get_fignums())
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
        for number in set(plt.get_fignums()) - already_open:
            plt.close(number)
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
