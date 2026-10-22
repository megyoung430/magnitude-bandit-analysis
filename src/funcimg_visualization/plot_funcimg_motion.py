"""Figure of the motion check: registration motion against the cells' shared signal."""

import matplotlib.pyplot as plt
import numpy as np

from src.funcimg_visualization.plot_funcimg_transients import GREEN, RED

LEFT_OUT_GREY = "0.88"


def motion_check_png_name(file_tag: str, variant: str, cut_s: float) -> str:
    """File name of a motion-check figure; the cut that shapes the panels is in it."""
    return f"motion_check_{file_tag}_{variant}_cut{cut_s:.0f}s.png"


def _offsets(ax, motion: dict, t_s: np.ndarray) -> None:
    ax.plot(t_s, motion["xoff"], lw=0.6, label="xoff (rigid)")
    ax.plot(t_s, motion["yoff"], lw=0.6, label="yoff (rigid)")
    if motion["xoff1"] is not None and motion["yoff1"] is not None:
        ax.plot(t_s, motion["xoff1"], lw=0.6, alpha=0.7, label="xoff1 (median block)")
        ax.plot(t_s, motion["yoff1"], lw=0.6, alpha=0.7, label="yoff1 (median block)")
    ax.set_ylabel("Offset (px)")
    ax.legend(loc="upper right", fontsize=7, ncol=4)


def _registration_quality(ax, corr_xy: np.ndarray | None, bad: np.ndarray, t_s) -> None:
    if corr_xy is None:
        ax.text(0.5, 0.5, "ops has no corrXY", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.plot(t_s, corr_xy, lw=0.6, color="0.3", label="corrXY")
    if bad.any():
        ax.plot(
            t_s[bad],
            np.zeros(bad.sum()),
            "|",
            color="C3",
            ms=8,
            transform=ax.get_xaxis_transform(),
            label=f"Suite2p badframes ({bad.sum()})",
        )
    ax.set_ylabel("corrXY")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="lower right", fontsize=7)


def _shared_signal(ax, median_z: dict, t_s, threshold: float | None) -> None:
    ax.plot(t_s, median_z["red"], lw=0.6, color=RED, label="red, no indicator")
    ax.plot(t_s, median_z["green"], lw=0.6, color=GREEN, alpha=0.8, label="green")
    if threshold is not None:
        for level in (threshold, -threshold):
            ax.plot(t_s[[0, -1]], [level, level], color=RED, ls=":", lw=0.8)
    ax.set_ylabel("Median z across cells")
    ax.set_xlabel("Time from the run's first frame (s)")
    ax.legend(loc="upper right", fontsize=7)


def plot_motion_check(result: dict, axes=None, title: str = "", threshold: float | None = None):
    """Rigid (and median non-rigid) offsets, corrXY with badframes, and each channel's shared z.

    The stretch before the cut is shaded grey and the cut is a black line in every panel.

    Args:
        result (dict): A :func:`~src.funcimg_analysis.motion_check.motion_check` result.
        axes: Three axes sharing x, or None for a new figure.
        title (str): Figure title.
        threshold (float | None): The artefact threshold, drawn at ± in the shared-signal panel.

    Returns:
        matplotlib.figure.Figure: The figure.

    Raises:
        ValueError: If *axes* is not three axes.
    """
    if axes is None:
        _, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True, constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    if len(axes) != 3:
        raise ValueError(f"plot_motion_check needs three axes, not {len(axes)}.")
    t_s, cut_s = result["t_s"], result["cut_s"]
    _offsets(axes[0], result["motion"], t_s)
    _registration_quality(axes[1], result["motion"]["corrXY"], result["badframes"], t_s)
    _shared_signal(axes[2], result["median_z"], t_s, threshold)
    for ax in axes:
        if cut_s > t_s[0]:
            ax.axvspan(t_s[0], cut_s, color=LEFT_OUT_GREY, zorder=0)
        ax.axvline(cut_s, color="k", lw=1)
        ax.spines[["top", "right"]].set_visible(False)
    fig = axes[0].figure
    note = f"grey: left out (cut at {cut_s:.0f} s)"
    fig.suptitle(f"{title}\n{note}" if title else note)
    return fig
