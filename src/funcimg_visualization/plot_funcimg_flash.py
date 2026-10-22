"""Figures of the flash check: per-cell event images and traces, and an animated clip."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from src.funcimg_visualization.plot_funcimg_transients import GREEN, RED

N_COLUMNS = 4
OUTLINE = "#f0e442"  # yellow, visible on grey and on both ends of the diverging colour map


def flash_pdf_name(file_tag: str, variant: str, cut_s: float) -> str:
    """File name of the flash-check PDF; the cut that selects the frames is in it."""
    return f"flash_check_{file_tag}_{variant}_cut{cut_s:.0f}s.pdf"


def clip_gif_name(file_tag: str, variant: str, start_s: float, length_s: float) -> str:
    """File name of a clip; where it starts and how long it is are in it."""
    return f"clip_{file_tag}_{variant}_{start_s:.0f}s-{start_s + length_s:.0f}s.gif"


def _outline(ax, mask: np.ndarray) -> None:
    if mask.any():
        ax.contour(mask.astype(float), levels=[0.5], colors=OUTLINE, linewidths=0.8)


def _image(ax, image, mask, title, **imshow) -> None:
    ax.imshow(image, interpolation="nearest", **imshow)
    _outline(ax, mask)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def _peri(ax, cell: dict) -> None:
    t = cell["offsets_s"]
    for channel, colour in (("green", GREEN), ("red", RED)):
        traces = cell["peri"][channel]
        for row in traces:
            ax.plot(t, row, color=colour, lw=0.5, alpha=0.4)
        if len(traces):
            ax.plot(t, traces.mean(axis=0), color=colour, lw=1.5, label=channel)
    ax.axvline(0, color="0.5", lw=0.6, ls="--")
    ax.set_xlabel("Time from event (s)", fontsize=8)
    ax.set_ylabel("z", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    if len(cell["peaks"]):
        ax.legend(fontsize=7, loc="upper right")


def plot_flash_cell(cell: dict, axes=None, vmax: float | None = None):
    """One cell: green mean image, green and red event ΔF/F0 images, and its peri-event z.

    The two event images share one symmetric colour scale (*vmax*, default the largest absolute
    value of either), so a red image as bright as the green means the change is not calcium.

    Args:
        cell (dict): One :func:`~src.funcimg_analysis.flash_check.flash_check` entry.
        axes: Four axes, or None for a new figure.
        vmax (float | None): Colour limit of the event images.

    Returns:
        matplotlib.figure.Figure: The figure.

    Raises:
        ValueError: If *axes* is not four axes.
    """
    if axes is None:
        _, axes = plt.subplots(1, N_COLUMNS, figsize=(13, 3.2), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    if len(axes) != N_COLUMNS:
        raise ValueError(f"plot_flash_cell needs {N_COLUMNS} axes, not {len(axes)}.")
    n = len(cell["peaks"])
    _image(axes[0], cell["mean_image"], cell["mask"], f"ROI {cell['roi']}: green mean", cmap="gray")
    if n == 0:
        for ax in axes[1:3]:
            ax.text(
                0.5,
                0.5,
                "no event above the threshold",
                ha="center",
                va="center",
                fontsize=8,
                transform=ax.transAxes,
            )
            ax.set_xticks([])
            ax.set_yticks([])
    else:
        both = np.concatenate([np.ravel(cell["image"][c]) for c in ("green", "red")])
        limit = vmax if vmax is not None else float(np.nanmax(np.abs(both)))
        for ax, channel in zip(axes[1:3], ("green", "red"), strict=True):
            _image(
                ax,
                cell["image"][channel],
                cell["mask"],
                f"{channel} ΔF/F0 at events (n={n})",
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
            )
    _peri(axes[3], cell)
    return axes[0].figure


def plot_flash_cells(cells: list[dict], title: str = ""):
    """:func:`plot_flash_cell` for every cell, one row each, in one figure."""
    fig, axes = plt.subplots(
        len(cells),
        N_COLUMNS,
        figsize=(13, 2.8 * len(cells)),
        constrained_layout=True,
        squeeze=False,
    )
    for row, cell in zip(axes, cells, strict=True):
        plot_flash_cell(cell, axes=row)
    fig.suptitle(title)
    return fig


def animate_clip(
    clip: dict,
    dff: dict,
    cells: list[dict],
    period_s: float,
    start_s: float,
    bin_px: int = 1,
    title: str = "",
) -> FuncAnimation:
    """An animation of green F, green ΔF/F0 and red ΔF/F0, with the checked cells outlined.

    Args:
        clip (dict): ``collect_clip`` output (``green``, ``red``, ``origin``).
        dff (dict): ``green`` and ``red`` :func:`~src.funcimg_analysis.flash_check.clip_dff`.
        cells (list[dict]): Flash-check cells to outline (their crops locate them).
        period_s (float): Time between the clip's (binned) frames (s).
        start_s (float): Time of the clip's first frame (s from the run's first frame).
        bin_px (int): The clip's spatial bin, to place the outlines.
        title (str): Figure title.

    Returns:
        matplotlib.animation.FuncAnimation: Save it with ``.save(path, writer=PillowWriter(...))``.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), constrained_layout=True)
    limit = float(np.nanpercentile(np.abs(dff["green"]), 99.5))
    panels = [
        (
            clip["green"],
            "green F",
            {
                "cmap": "gray",
                "vmin": np.percentile(clip["green"], 1),
                "vmax": np.percentile(clip["green"], 99.7),
            },
        ),
        (dff["green"], "green ΔF/F0", {"cmap": "RdBu_r", "vmin": -limit, "vmax": limit}),
        (dff["red"], "red ΔF/F0 (no indicator)", {"cmap": "RdBu_r", "vmin": -limit, "vmax": limit}),
    ]
    images = []
    for ax, (movie, label, imshow) in zip(axes, panels, strict=True):
        images.append(ax.imshow(movie[0], interpolation="nearest", **imshow))
        ax.set_title(label, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        for cell in cells:
            y = (cell["box"][0].start + cell["box"][0].stop) / 2 - clip["origin"][0]
            x = (cell["box"][1].start + cell["box"][1].stop) / 2 - clip["origin"][1]
            ax.plot(x / bin_px - 0.5, y / bin_px - 0.5, "o", mfc="none", mec=OUTLINE, ms=9, mew=0.8)
            ax.text(x / bin_px + 4, y / bin_px, str(cell["roi"]), color=OUTLINE, fontsize=6)
    clock = fig.suptitle("")

    def draw(i):
        for image, (movie, _, _) in zip(images, panels, strict=True):
            image.set_data(movie[i])
        clock.set_text(f"{title}  t = {start_s + i * period_s:6.1f} s")
        return [*images, clock]

    return FuncAnimation(fig, draw, frames=len(clip["green"]), blit=False)
