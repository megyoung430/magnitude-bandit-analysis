"""Figures for the Zong et al. 2022 pipeline: signal examples, SNR, responses, motion, z drift.

Each function draws into an axis (or axes) it is given and makes its own figure only when none is
passed, so any panel can be placed in a larger figure. Colours for pulse counts are those of the
other imaging figures.
"""

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from src.funcimg_analysis.event_frames import mean_by_category, sem_by_category
from src.funcimg_visualization.plot_funcimg_responses import (
    PULSE_COLORS,
    outline,
    roi_colors,
    window_x,
)

TRACE_COLORS = {"raw": "0.6", "clear": "#225EA8", "events": "#C44E52", "f0": "#DD8452"}
TRACE_TITLES = {"dff_clear": "ΔF/F clear", "e_clear": "E clear (scaled events)"}
GROUP_TITLES = {"snr_pass": "cells with SNR above threshold", "all": "all cells"}


def zong2022_png_name(kind: str, tag: str, variant: str) -> str:
    """File name of a figure: ``<kind>_<tag>_<variant>.png``.

    *tag* is the pooled tag for figures over all runs and the run's own tag for per-run figures,
    so a name never carries both.
    """
    return f"{kind}_{tag}_{variant}.png"


def _axis(ax, figsize):
    if ax is not None:
        return ax.figure, ax
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    return fig, ax


# ---- signal processing -------------------------------------------------------------------------


def plot_baseline_example(signal, roi: int, ax=None, title: str = ""):
    """Fcorr, the running baseline F0 and the baseline points of one cell, against time (s)."""
    _, ax = _axis(ax, (11, 3.5))
    t = np.arange(signal.fcorr.shape[1]) * signal.period_s
    ax.plot(t, signal.fcorr[roi], color=TRACE_COLORS["raw"], lw=0.6, label="Fcorr")
    ax.plot(t, signal.f0[roi], color=TRACE_COLORS["f0"], lw=1.2, label="F0 = Fs + m")
    points = signal.baseline_points[roi]
    ax.plot(t[points], signal.f0[roi][points], ".", color="k", ms=1.5, label="baseline points")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("F (a.u.)")
    ax.set_title(title or f"ROI {roi}: running baseline")
    ax.legend(frameon=False, ncols=3, fontsize=8, loc="upper right")
    return ax.figure


def plot_signal_examples(signal, rois, window_s: tuple[float, float], axes=None, title: str = ""):
    """ΔF/F, its significant transients and the scaled calcium events for a few cells.

    ``E_clear`` is scaled to the maximum of ``ΔF/F_clear``, so both share one axis.
    """
    rois = list(rois)
    if axes is None:
        _, axes = plt.subplots(
            len(rois), 1, figsize=(11, 2.4 * len(rois)), sharex=True, squeeze=False,
            constrained_layout=True,
        )  # fmt: skip
    axes = np.asarray(axes).reshape(-1)
    frames = slice(*(int(round(s / signal.period_s)) for s in window_s))
    t = np.arange(signal.dff.shape[1])[frames] * signal.period_s
    for ax, roi in zip(axes, rois, strict=True):
        ax.plot(t, signal.dff[roi, frames], color=TRACE_COLORS["raw"], lw=0.6, label="ΔF/F")
        ax.plot(t, signal.dff_clear[roi, frames], color=TRACE_COLORS["clear"], lw=1.0,
                label="ΔF/F clear")  # fmt: skip
        events = signal.e_clear[roi, frames]
        ax.vlines(t[events > 0], 0, events[events > 0], color=TRACE_COLORS["events"], lw=0.8,
                  label="E clear")  # fmt: skip
        ax.set_ylabel(f"ROI {roi}\nΔF/F")
    axes[0].legend(frameon=False, ncols=3, fontsize=8, loc="upper right")
    axes[-1].set_xlabel("Time (s)")
    axes[0].figure.suptitle(title or "ΔF/F, significant transients and calcium events")
    return axes[0].figure


def plot_snr_histogram(table, min_snr: float, ax=None, title: str = ""):
    """Histogram of the cells' SNR with the inclusion threshold and the pass count."""
    _, ax = _axis(ax, (6, 4))
    snr = table["snr"].to_numpy(float)
    finite = snr[np.isfinite(snr)]
    ax.hist(finite, bins=30, color=TRACE_COLORS["clear"], alpha=0.8)
    ax.axvline(min_snr, color="k", ls="--", lw=1)
    ax.set_xlabel("SNR")
    ax.set_ylabel("Cells")
    n_pass = int(table["passes"].sum())
    n_nan = int((~np.isfinite(snr)).sum())
    ax.set_title(
        title or f"{n_pass} of {len(table)} cells above SNR {min_snr:g} ({n_nan} undefined)"
    )
    return ax.figure


def plot_tau_sensitivity(counts: dict, chosen_tau: float, ax=None, title: str = ""):
    """Mean calcium events per cell for each assumed decay time, the chosen one marked.

    Args:
        counts (dict): ``{tau_s: (n_cells,) events}`` from ``event_counts_by_tau``.
        chosen_tau (float): The pipeline's ``tau_s``, drawn in a darker colour.
    """
    _, ax = _axis(ax, (5, 4))
    taus = sorted(counts)
    means = [float(np.mean(counts[t])) for t in taus]
    colors = [TRACE_COLORS["clear"] if np.isclose(t, chosen_tau) else "0.7" for t in taus]
    ax.bar([f"{t:g}" for t in taus], means, color=colors)
    ax.set_xlabel("Assumed decay time τ (s)")
    ax.set_ylabel("Calcium events per cell")
    ax.set_title(title or "Event count against τ")
    return ax.figure


# ---- responses ---------------------------------------------------------------------------------


def plot_clear_responses(
    pooled: dict, offsets, period_s: float, x_units: str = "s", axes=None, title: str = ""
):
    """Mean ± SEM across cells of ΔF/F_clear and E_clear around the outcome, by pulse count.

    Args:
        pooled (dict): ``reward_analysis(...)["pooled"]``: ``{trace: {group: {pulses: stats}}}``
            with traces ``dff_clear``, ``e_clear`` and groups ``snr_pass``, ``all``.
        offsets: Window offsets in frames.
        period_s (float): Frame period.
        x_units (str): ``"s"`` or ``"frames"``.
        axes: A ``(2, 2)`` array of axes (rows traces, columns groups) or None.
        title (str): Figure title.
    """
    if axes is None:
        _, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, constrained_layout=True)
    axes = np.asarray(axes)
    if axes.shape != (2, 2):
        raise ValueError(f"Need a 2 x 2 array of axes, got {axes.shape}.")
    x, xlabel = window_x(np.asarray(offsets), period_s, x_units)
    for row, trace in enumerate(("dff_clear", "e_clear")):
        for col, group in enumerate(("snr_pass", "all")):
            ax = axes[row, col]
            for pulses, stats in sorted(pooled[trace][group].items()):
                label = f"{pulses} pulses ({stats['n_cells']} cells)"
                ax.plot(x, stats["mean"], color=PULSE_COLORS[pulses], lw=1.4, label=label)
                ax.fill_between(x, stats["mean"] - stats["sem"], stats["mean"] + stats["sem"],
                                color=PULSE_COLORS[pulses], alpha=0.2, lw=0)  # fmt: skip
            ax.axvline(0, color="0.5", lw=0.8, ls="--")
            ax.set_title(f"{TRACE_TITLES[trace]}: {GROUP_TITLES[group]}", fontsize=11)
            if col == 0:
                ax.set_ylabel(TRACE_TITLES[trace])
            if row == 1:
                ax.set_xlabel(xlabel)
    axes[0, 1].legend(frameon=False, fontsize=8)
    axes[0, 0].figure.suptitle(title or "Outcome-aligned response, mean ± SEM across cells")
    return axes[0, 0].figure


def plot_score_vs_null(table, kind: str, ax=None, title: str = ""):
    """Each cell's score against its own shuffled 95th percentile; responsive cells in colour."""
    _, ax = _axis(ax, (5.5, 5))
    score = table[f"{kind}_score"].to_numpy(float)
    null = table[f"{kind}_null_95th"].to_numpy(float)
    hit = table[f"{kind}_responsive"].to_numpy(bool)
    ax.scatter(null[~hit], score[~hit], s=10, color="0.6", label="not responsive")
    ax.scatter(null[hit], score[hit], s=14, color=TRACE_COLORS["events"], label="responsive")
    low = np.nanmin(np.r_[null, score, 0])
    high = np.nanmax(np.r_[null, score, 0])
    ax.plot([low, high], [low, high], color="k", lw=0.8, ls="--")
    ax.set_xlabel("95th percentile of shuffled scores")
    ax.set_ylabel("Score (A_event - A_other) / (A_event + A_other)")
    ax.set_title(title or f"{kind.replace('_', ' ')}: {int(hit.sum())} of {len(table)} cells above")
    ax.legend(frameon=False, fontsize=8)
    return ax.figure


def _fov(ax, mean_image, labels, rois, names, colors):
    low, high = np.nanpercentile(mean_image, [1, 99.5])
    ax.imshow(mean_image, cmap="gray", vmin=low, vmax=high)
    for roi, name, color in zip(rois, names, colors, strict=True):
        edge = outline(np.where(labels == roi + 1, labels, 0))
        overlay = np.zeros((*labels.shape, 4))
        overlay[edge] = matplotlib.colors.to_rgba(color)
        ax.imshow(overlay)
        ys, xs = np.nonzero(labels == roi + 1)
        if len(ys):
            ax.text(xs.mean(), ys.mean(), str(name), color=color, fontsize=6, ha="center")
    ax.set_axis_off()
    ax.set_title("Mean image; cells outlined")


def plot_top_responsive(
    windows: np.ndarray,
    pulses: np.ndarray,
    rois,
    table,
    kind: str,
    offsets,
    period_s: float,
    mean_image: np.ndarray,
    labels: np.ndarray,
    x_units: str = "s",
    title: str = "",
):
    """The most responsive cells: ΔF/F_clear around the outcome by pulse count, and the field.

    One small panel per cell (trial mean ± SEM across trials for each pulse count) titled with
    the ROI, its SNR and its score, and the mean image with the cells outlined in the colour of
    their title.

    Args:
        windows (np.ndarray): ``(n_cells_in_run, n_events, n_window)`` ΔF/F_clear windows.
        pulses (np.ndarray): Reward pulses of each event.
        rois: ROI numbers to draw, best first.
        table: The run's cell table (``snr`` and ``<kind>_score`` columns).
        kind (str): ``"reward"`` or ``"no_reward"``.
        offsets: Window offsets in frames.
        period_s (float): Frame period.
        mean_image (np.ndarray): Image for the field of view.
        labels (np.ndarray): ROI label image, row ``i`` of the traces drawn as ``i + 1``.
            Cells are named by the table's ``cellpose_label`` column.
        x_units (str): ``"s"`` or ``"frames"``.
        title (str): Figure title.
    """
    rois = list(rois)
    n_cols = max(1, min(4, len(rois)))
    n_rows = max(1, int(np.ceil(len(rois) / n_cols)))
    fig = plt.figure(figsize=(16, 2.6 * n_rows + 1), constrained_layout=True)
    grid = fig.add_gridspec(n_rows, n_cols + 2)
    colors = roi_colors(len(rois))
    x, xlabel = window_x(np.asarray(offsets), period_s, x_units)
    categories = sorted(set(pulses.tolist()))
    for k, (roi, color) in enumerate(zip(rois, colors, strict=True)):
        ax = fig.add_subplot(grid[k // n_cols, k % n_cols])
        means = mean_by_category(windows[roi : roi + 1], pulses, categories)
        sems = sem_by_category(windows[roi : roi + 1], pulses, categories)
        for p in means:
            ax.plot(x, means[p][0], color=PULSE_COLORS[p], lw=1.2, label=f"{p}")
            ax.fill_between(x, means[p][0] - sems[p][0], means[p][0] + sems[p][0],
                            color=PULSE_COLORS[p], alpha=0.2, lw=0)  # fmt: skip
        ax.axvline(0, color="0.5", lw=0.8, ls="--")
        row = table.iloc[roi]
        name = f"ROI {table['cellpose_label'].iloc[roi]} | SNR {row['snr']:.1f}"
        ax.set_title(f"{name} | score {row[f'{kind}_score']:.2f}", fontsize=9, color=color)
        if k // n_cols == n_rows - 1:
            ax.set_xlabel(xlabel)
        if k % n_cols == 0:
            ax.set_ylabel("ΔF/F clear")
        if k == 0:
            ax.legend(title="pulses", frameon=False, fontsize=7, ncols=2)
    names = [table["cellpose_label"].iloc[roi] for roi in rois]
    _fov(fig.add_subplot(grid[:, n_cols:]), mean_image, labels, rois, names, colors)
    fig.suptitle(title or f"Most responsive cells ({kind.replace('_', ' ')})")
    return fig


# ---- motion and z drift ------------------------------------------------------------------------


def plot_motion_summary(drift, offsets: dict, nonrigid: np.ndarray | None, period_s: float,
                        axes=None, title: str = ""):  # fmt: skip
    """Residual drift per spatial PC, and the rigid and non-rigid motion over the recording.

    Args:
        drift (pd.DataFrame): From ``residual_drift`` (``pc``, ``rigid_px``, ``mean_nonrigid_px``,
            ``max_nonrigid_px``).
        offsets (dict): Suite2p's recorded ``xoff``, ``yoff`` (frames); missing keys are skipped.
        nonrigid (np.ndarray | None): Per-frame non-rigid motion (px), or None if not recorded.
        period_s (float): Frame period, for the time axis.
        axes: Three axes or None.
        title (str): Figure title.
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)
    ax_pc, ax_rigid, ax_nonrigid = axes
    ax_pc.plot(drift["pc"], drift["rigid_px"], "o-", ms=4, lw=1, label="rigid")
    ax_pc.plot(drift["pc"], drift["mean_nonrigid_px"], "s-", ms=4, lw=1, label="mean over blocks")
    ax_pc.plot(drift["pc"], drift["max_nonrigid_px"], "^-", ms=4, lw=1, label="max over blocks")
    ax_pc.set_xlabel("Spatial PC")
    ax_pc.set_ylabel("Residual drift, Itop vs Ibottom (px)")
    ax_pc.set_title("Residual xy drift per PC")
    ax_pc.legend(frameon=False, fontsize=8)
    for key, color in (("xoff", "#225EA8"), ("yoff", "#C44E52")):
        if key in offsets:
            t = np.arange(len(offsets[key])) * period_s
            ax_rigid.plot(t, offsets[key], color=color, lw=0.6, label=key)
    ax_rigid.set_xlabel("Time (s)")
    ax_rigid.set_ylabel("Rigid shift (px)")
    ax_rigid.set_title("Rigid motion")
    ax_rigid.legend(frameon=False, fontsize=8)
    if nonrigid is not None:
        ax_nonrigid.plot(np.arange(len(nonrigid)) * period_s, nonrigid, color="0.3", lw=0.6)
    else:
        ax_nonrigid.text(
            0.5, 0.5, "no non-rigid shifts in ops", ha="center", transform=ax_nonrigid.transAxes
        )
    ax_nonrigid.set_xlabel("Time (s)")
    ax_nonrigid.set_ylabel("Largest block-to-block shift (px)")
    ax_nonrigid.set_title("Non-rigid motion")
    ax_pc.figure.suptitle(title or "Motion correction quality")
    return ax_pc.figure


def plot_zdrift(i_top, i_bottom, labels: np.ndarray, pcs, n_cols: int = 2, title: str = ""):
    """Ibottom - Itop of chosen PCs, each as a plain panel and a panel with the ROI contours.

    If a cell's nucleus and cytoplasm have opposite signs (one light, one dark) the difference
    cannot be calcium and points to z drift; both should brighten and dim together. The plain
    panel shows the pixels untouched; the other has a thin contour just outside each ROI, so the
    ROI's own pixels stay visible.

    Args:
        i_top: ``(n_pcs, Ly, Lx)`` mean of the highest-scoring frames of each PC.
        i_bottom: Same, lowest-scoring frames.
        labels (np.ndarray): ROI label image cut to the same crop as the images.
        pcs: 0-based PCs to draw.
        n_cols (int): PCs per row; each takes two panels.
        title (str): Figure title.
    """
    pcs = list(pcs)
    n_rows = int(np.ceil(len(pcs) / n_cols))
    fig, axes = plt.subplots(n_rows, 2 * n_cols, figsize=(5 * 2 * n_cols, 5 * n_rows),
                             squeeze=False, constrained_layout=True)  # fmt: skip
    # one pixel of dilation puts the 0.5 contour 1.5 px outside the ROI edge
    ring = ndimage.binary_dilation(labels > 0).astype(float)
    for ax in axes.reshape(-1):
        ax.set_axis_off()
    for k, pc in enumerate(pcs):
        plain, outlined = (
            axes[k // n_cols, 2 * (k % n_cols)],
            axes[k // n_cols, 2 * (k % n_cols) + 1],
        )
        diff = np.asarray(i_bottom[pc], float) - np.asarray(i_top[pc], float)
        limit = np.nanpercentile(np.abs(diff), 99.5) or 1.0
        for ax in (plain, outlined):
            ax.imshow(diff, cmap="gray", vmin=-limit, vmax=limit)
        plain.set_title(f"PC {pc + 1}: Ibottom - Itop")
        outlined.contour(ring, levels=[0.5], colors=[(1.0, 0.8, 0.0)], linewidths=0.6)
        outlined.set_title(f"PC {pc + 1}: with ROI outlines")
    fig.suptitle(title or "z-drift review: nucleus and cytoplasm should not have opposite signs")
    return fig
