"""Small drawing and output helpers shared by the summary figure modules."""

from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import pandas as pd

from src.behavior_visualization.plot_style import MOUSE_COLORS


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


def save_or_show(
    fig: plt.Figure, save_path: str | Path | None, *, make_parent: bool = True
) -> None:
    """Save *fig* as ``<save_path>.pdf`` and ``<save_path>.png``, or show it if no path.

    The figure is closed afterwards. With ``make_parent=False`` the parent
    directory must already exist and the path is used as given.

    Known issue: if saving raises, *fig* stays open.
    """
    if save_path is None:
        plt.show()
    else:
        base = Path(save_path) if make_parent else save_path
        if make_parent:
            base.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(base) + ".pdf", bbox_inches="tight")
        fig.savefig(str(base) + ".png", dpi=300, bbox_inches="tight")
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
    touches the disk. A builder that raises part-way leaves the files written so far
    and its own unfinished figure open.

    Returns:
        The saved figure paths, PNG then PDF for each figure.
    """
    paths: list[Path] = []
    for output in outputs:
        if isinstance(output, TableOutput):
            output.frame.to_csv(directory / output.filename, index=output.index)
        elif isinstance(output, FigureOutput):
            paths += _save_figure(output.figure, directory, output.name, show=show)
        else:
            raise TypeError(f"expected TableOutput or FigureOutput, got {type(output).__name__}")
    return paths


def _save_figure(fig: plt.Figure, directory: Path, name: str, *, show: bool) -> list[Path]:
    """Tight-layout *fig*, save ``<directory>/<name>.png`` and ``.pdf``, then close it.

    Returns the two saved paths (PNG first). With ``show`` the figure is also
    displayed after saving.
    """
    fig.tight_layout()
    paths = []
    for extension in ("png", "pdf"):
        path = directory / f"{name}.{extension}"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        paths.append(path)
    if show:
        plt.show()
    plt.close(fig)
    return paths
