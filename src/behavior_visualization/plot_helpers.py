"""Small drawing and output helpers shared by the summary figure modules."""

from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt

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


def save_png_and_pdf(
    fig: plt.Figure, directory: Path, name: str, *, show: bool, dpi: int = 180
) -> list[Path]:
    """Tight-layout *fig*, save ``<directory>/<name>.png`` and ``.pdf``, then close it.

    Returns the two saved paths (PNG first). With ``show`` the figure is also
    displayed after saving.
    """
    fig.tight_layout()
    paths = []
    for extension in ("png", "pdf"):
        path = directory / f"{name}.{extension}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(path)
    if show:
        plt.show()
    plt.close(fig)
    return paths
