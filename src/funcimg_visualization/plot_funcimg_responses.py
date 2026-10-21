"""Figures for reward-aligned imaging responses and a run's most responsive cells."""

import warnings

import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np

# 0 pulses (no reward) in red; 1-4 pulses from teal to dark blue, as in the photometry figures.
PULSE_COLORS = {0: "#C44E52", 1: "#41B6C4", 2: "#1D91C0", 3: "#225EA8", 4: "#0C2C84"}
CHANNEL_TITLES = {"green": "Green (functional)", "red": "Red (structural)"}
RANKING_TITLES = {"max_dff": "maximum dF/F", "p_cell": "p(cell)"}
MEASURE_LABELS = {
    "F": ("raw F (not neuropil-corrected)", "F (a.u.)"),
    "dF/F0": ("ΔF/F0", "ΔF/F0"),
    "F z": (
        "raw F (not neuropil-corrected), robust z, baseline-subtracted",
        "F (robust z, baseline-subtracted)",
    ),
    "dF/F0 z": ("ΔF/F0, robust z, baseline-subtracted", "ΔF/F0 (robust z, baseline-subtracted)"),
}


def window_x(offsets: np.ndarray, period_s: float, x_units: str) -> tuple[np.ndarray, str]:
    """x values and axis label of window *offsets* (frames), in ``"s"`` or ``"frames"``."""
    if x_units == "frames":
        return offsets, "Frames from the outcome's frame"
    if x_units == "s":
        return offsets * period_s, "Time from outcome to frame start (s)"
    raise ValueError(f"x_units must be 's' or 'frames', not {x_units!r}.")


def plot_pulse_responses(
    pooled: dict,
    offsets: np.ndarray,
    period_s: float,
    x_units: str = "s",
    axes=None,
    title: str = "",
    measures: tuple[str, ...] = ("F z", "dF/F0 z"),
    unit: str | None = "cells",
):
    """Mean ± SEM across cells of each view, one line per number of reward pulses.

    Args:
        pooled (dict): ``{(channel, measure): {pulses: {"mean", "sem", "n_cells", "n_runs"}}}``.
        offsets (np.ndarray): Window offsets in frames (0 = outcome frame).
        period_s (float): Frame period, for an x axis in seconds.
        x_units (str): ``"s"`` or ``"frames"``.
        axes: A ``(len(measures), 2)`` array of axes (columns green, red), or None for a new
            figure with a row per measure.
        title (str): Figure title.
        measures (tuple[str, ...]): The rows: ``"F"``, ``"dF/F0"`` or their robust-z,
            baseline-subtracted versions ``"F z"``, ``"dF/F0 z"`` (the default pair).
        unit (str | None): What the pooled items are, named in the legend with their count
            (``"cells"``); None names only the number of runs.

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    if not measures:
        raise ValueError("Give at least one measure to plot.")
    n_rows = len(measures)
    if axes is None:
        _, axes = plt.subplots(
            n_rows, 2, figsize=(11, 4 * n_rows), sharex=True, constrained_layout=True, squeeze=False
        )
    axes = np.asarray(axes)
    if axes.shape != (n_rows, 2):
        raise ValueError(
            f"Need a {n_rows} x 2 array of axes for {n_rows} measures, got {axes.shape}."
        )
    fig = axes[0, 0].figure
    x, xlabel = window_x(np.asarray(offsets), period_s, x_units)
    for col, channel in enumerate(("green", "red")):
        for row, measure in enumerate(measures):
            ax = axes[row, col]
            for pulses, stats in sorted(pooled[(channel, measure)].items()):
                counts = f"{stats['n_cells']} {unit}, " if unit else ""
                n_runs = stats["n_runs"]
                sem_note = ", SEM across trials" if stats.get("sem_kind") == "trials" else ""
                label = f"{pulses} pulses ({counts}{n_runs} run{'s' * (n_runs != 1)}{sem_note})"
                ax.plot(x, stats["mean"], color=PULSE_COLORS[pulses], label=label)
                ax.fill_between(
                    x,
                    stats["mean"] - stats["sem"],
                    stats["mean"] + stats["sem"],
                    color=PULSE_COLORS[pulses],
                    alpha=0.2,
                    linewidth=0,
                )
            ax.axvline(0, color="0.5", linewidth=0.8, linestyle="--")
            name, ylabel = MEASURE_LABELS[measure]
            ax.set_title(f"{CHANNEL_TITLES[channel]}: {name}")
            ax.set_ylabel(ylabel)
            if row == len(measures) - 1:
                ax.set_xlabel(xlabel)
    axes[0, 1].legend(fontsize=8, frameon=False)
    fig.suptitle(title or "Outcome-aligned response, mean ± SEM across cells")
    return fig


def outline(labels: np.ndarray) -> np.ndarray:
    """Boolean image of the pixels on the boundary of a labelled ROI (label > 0)."""
    edge = np.zeros(labels.shape, bool)
    edge[:-1] |= labels[:-1] != labels[1:]
    edge[1:] |= labels[:-1] != labels[1:]
    edge[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    edge[:, 1:] |= labels[:, :-1] != labels[:, 1:]
    return edge & (labels > 0)


def roi_colors(n: int) -> list[str]:
    """Hex colours for *n* ROIs, cycling matplotlib's default ``tab10`` order."""
    cycle = plt.get_cmap("tab10").colors
    return [matplotlib.colors.to_hex(cycle[i % len(cycle)]) for i in range(n)]


SPACING_MARGIN = 0.1  # fraction of the spacing left between neighbouring rows
SPACING_FLOOR = 0.05  # ΔF/F0 units; keeps rows apart when every trace is flat


def check_clip_percentiles(clip_percentiles) -> None:
    """Raise unless *clip_percentiles* is None or two numbers with ``0 <= low < high <= 100``."""
    if clip_percentiles is None:
        return
    message = (
        "clip_percentiles must be None or (low, high) with 0 <= low < high <= 100, "
        f"not {clip_percentiles!r}."
    )
    try:
        low, high = clip_percentiles
    except (TypeError, ValueError):
        raise ValueError(message) from None
    numbers = all(
        isinstance(value, int | float | np.integer | np.floating) and not isinstance(value, bool)
        for value in (low, high)
    )
    if not (numbers and 0 <= low < high <= 100):
        raise ValueError(message)


def row_percentiles(traces: np.ndarray, percentiles) -> np.ndarray:
    """``(len(percentiles), n_rows)`` NaN-safe percentiles; NaN for a row with no finite value."""
    out = np.full((len(percentiles), len(traces)), np.nan)
    has_data = np.isfinite(traces).any(axis=1)
    if has_data.any():
        out[:, has_data] = np.nanpercentile(traces[has_data], percentiles, axis=1)
    return out


def clip_traces(dff: np.ndarray, clip_percentiles: tuple[float, float] | None) -> np.ndarray:
    """Each trace limited to its own percentile range (NaN-safe), for display only.

    Args:
        dff (np.ndarray): ``(n_cells, n_frames)`` ΔF/F0.
        clip_percentiles (tuple[float, float] | None): Low and high percentile, e.g. ``(1, 99)``;
            None returns *dff* unchanged. Values above the high percentile are flattened to it,
            so a trace's true maximum is not visible in the clipped trace.

    Raises:
        ValueError: If *clip_percentiles* is not None or two numbers with
            ``0 <= low < high <= 100``.
    """
    check_clip_percentiles(clip_percentiles)
    if clip_percentiles is None:
        return dff
    low, high = row_percentiles(dff, clip_percentiles)
    return np.clip(dff, low[:, None], high[:, None])


def trace_spacing(traces: np.ndarray, percentiles: tuple[float, float] = (0.5, 99.5)) -> float:
    """Vertical distance between stacked traces, constant over the rows.

    The largest, over adjacent rows (in plotted order), of the lower trace's high percentile
    minus the upper trace's low percentile, so the bulk of one row never reaches the next; a
    brief transient beyond the percentiles may. A trace's own range counts too. A margin
    is added and a floor keeps all-flat traces apart. A row with no finite value (NaN
    throughout) constrains nothing; if no row has a value the spacing is the floor.

    Args:
        traces (np.ndarray): ``(n_cells, n_frames)`` ΔF/F0, best cell first.
        percentiles (tuple[float, float]): Low and high percentile of a trace's extent. Use
            ``(0, 100)`` for traces already clipped to their percentile range.
    """
    low, high = row_percentiles(traces, percentiles)
    needed = np.concatenate([high - low, high[1:] - low[:-1]])
    if not np.isfinite(needed).any():
        return SPACING_FLOOR
    return float(max((1 + SPACING_MARGIN) * np.nanmax(needed), SPACING_FLOOR))


def trace_baselines(
    traces: np.ndarray, percentiles: tuple[float, float] = (0.5, 99.5)
) -> np.ndarray:
    """Baseline (y of ΔF/F0 = 0) of each stacked trace: row 0 at 0, each next one lower."""
    return -trace_spacing(traces, percentiles) * np.arange(len(traces))


def middle_frames(n_frames: int, n_shown: int | None) -> slice:
    """The centred stretch of *n_shown* frames out of *n_frames* (all of them if fewer).

    Args:
        n_frames (int): Frames in the run.
        n_shown (int | None): Frames to keep, e.g. ``8000``; None keeps every frame.

    Raises:
        TypeError: If *n_shown* is not None or an integer (a bool or a float is refused).
        ValueError: If *n_shown* is not positive.
    """
    if n_shown is None:
        return slice(0, n_frames)
    if isinstance(n_shown, bool) or not isinstance(n_shown, int | np.integer):
        raise TypeError(
            f"middle_frames_shown must be None or an integer number of frames, not {n_shown!r}."
        )
    if n_shown < 1:
        raise ValueError(f"middle_frames_shown must be positive, not {n_shown}.")
    if n_shown >= n_frames:
        return slice(0, n_frames)
    start = (n_frames - n_shown) // 2
    return slice(start, start + n_shown)


def peak_frames(dff: np.ndarray) -> np.ndarray:
    """Frame of each row's maximum; -1 for a row with no finite value."""
    finite = np.isfinite(dff)
    peaks = np.where(finite, dff, -np.inf).argmax(axis=1)
    return np.where(finite.any(axis=1), peaks, -1)


def roi_label(cell: int, p_cell: float, peak: float, has_data: bool) -> str:
    """Tick label: ROI number, classifier probability, full-run max ΔF/F0, and a no-data note."""
    label = f"ROI {cell} (p={p_cell:.2f}, max {peak:.3g})"
    return label if has_data else f"{label} (no data in shown frames)"


def _x_axis(top: dict, x_units: str) -> tuple[np.ndarray, str, np.ndarray]:
    """Frame positions, axis label and outcome positions for ``x_units``."""
    times = top["outcomes"]["outcome_time"].to_numpy(float)
    if x_units == "s":
        return top["time_s"], "pyControl time (s)", times
    if x_units == "frames":
        known = np.isfinite(top["time_s"])
        imaged = (times >= top["time_s"][known][0]) & (times <= top["time_s"][known][-1])
        # outside the imaged span np.interp would pin the mark to the first or last frame
        events = np.where(
            imaged, np.interp(times, top["time_s"][known], np.flatnonzero(known)), np.nan
        )
        return np.arange(top["dff"].shape[1]), "Frames", events
    raise ValueError(f"x_units must be 's' or 'frames', not {x_units!r}.")


def _draw_traces(
    ax,
    top: dict,
    x_units: str,
    colors: list[str],
    clip_percentiles: tuple[float, float] | None,
    frames: slice,
) -> bool:
    """Stacked traces, one colour per ROI, with ROI tick labels, outcome markers and peak ticks.

    The tick label carries the ROI's full-run maximum ΔF/F0 (clipping flattens it in the
    trace), and a small tick on the x axis marks where that maximum falls, if within *frames*.
    A ROI with no finite value in *frames* is an empty row, labelled as such.

    Returns:
        bool: False if the frames shown have no pyControl time (``x_units="s"``), so nothing
        could be placed; True otherwise.
    """
    if len(top["cells"]) == 0:
        ax.text(0.5, 0.5, "no ROIs to show", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return True
    x_all, xlabel, events = _x_axis(top, x_units)
    x = x_all[frames]
    if not np.isfinite(x).any():
        message = "no frame times in the frames shown"
        ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return False
    # an outcome outside the frames shown would stretch the axis, so it is not marked
    events = np.where((events >= np.nanmin(x)) & (events <= np.nanmax(x)), events, np.nan)
    shown = clip_traces(top["dff"][:, frames], clip_percentiles)
    has_data = np.isfinite(shown).any(axis=1)
    # clipped traces end exactly at their percentiles, so the spacing is measured at the ends
    spacing_percentiles = (0, 100) if clip_percentiles else (0.5, 99.5)
    spacing = trace_spacing(shown, spacing_percentiles)
    baselines = -spacing * np.arange(len(shown))
    for base, trace, color in zip(baselines, shown, colors, strict=True):
        ax.plot(x, base + trace, color=color, linewidth=0.8)
    # the axis spans every row's baseline (so every tick label is visible), and the data
    stacked = baselines[:, None] + shown
    top_y = max(np.nanmax(stacked), baselines[0]) if has_data.any() else baselines[0]
    bottom_y = min(np.nanmin(stacked), baselines[-1]) if has_data.any() else baselines[-1]
    marker_bottom = top_y + 0.1 * spacing
    for time, pulses in zip(events, top["outcomes"]["num_rew_pulses"], strict=True):
        ax.plot(
            [time, time],
            [marker_bottom, marker_bottom + 0.4 * spacing],
            color=PULSE_COLORS[pulses],
        )
    for pulses in sorted(set(top["outcomes"]["num_rew_pulses"])):
        ax.plot([], [], color=PULSE_COLORS[pulses], label=f"{pulses} pulses")
    ax.legend(loc="upper right", fontsize=7, frameon=False, ncols=5)
    for frame, color in zip(peak_frames(top["dff"]), colors, strict=True):
        if frames.start <= frame < frames.stop and np.isfinite(x_all[frame]):
            ax.plot(
                [x_all[frame]],
                [0],
                marker="|",
                markersize=9,
                markeredgewidth=1.5,
                linestyle="none",
                color=color,
                transform=ax.get_xaxis_transform(),  # y in axes units: on the x axis itself
                clip_on=False,
            )
    ax.set_yticks(
        baselines,
        [
            roi_label(cell, p, peak, row_has_data)
            for cell, p, peak, row_has_data in zip(
                top["cells"], top["p_cell"], top["peaks"], has_data, strict=True
            )
        ],
    )
    ax.set_ylim(bottom_y - 0.1 * spacing, marker_bottom + 0.6 * spacing)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("ROI")
    ax.spines[["top", "right"]].set_visible(False)
    return True


def draw_fov(ax, top: dict, colors: list[str]) -> None:
    """Mean image with each top ROI outlined and numbered in its trace colour."""
    image = top["mean_image"]
    low, high = np.nanpercentile(image, [1, 99.5])
    ax.imshow(image, cmap="gray", vmin=low, vmax=high)
    labels = top["labels"]
    chosen = np.where(np.isin(labels, top["cells"] + 1), labels, 0)
    # sized by the ROI numbers too: a top ROI fully hidden by overlaps has no label pixels
    n_labels = max(int(labels.max()), int(top["cells"].max()) + 1 if len(top["cells"]) else 0)
    lut = np.zeros((n_labels + 1, 4))
    if len(top["cells"]):
        lut[top["cells"] + 1] = [matplotlib.colors.to_rgba(c) for c in colors]
    overlay = np.zeros((*image.shape, 4))
    edge = outline(chosen)
    overlay[edge] = lut[chosen[edge]]
    ax.imshow(overlay)
    hidden = []
    for cell, color in zip(top["cells"], colors, strict=True):
        ys, xs = np.nonzero(labels == cell + 1)
        if len(ys):
            ax.text(xs.mean(), ys.mean(), str(cell), color=color, fontsize=6, ha="center")
        else:
            hidden.append(f"ROI {cell}")
    ax.set_axis_off()
    note = f"\nnot visible (hidden by overlaps): {', '.join(hidden)}" if hidden else ""
    ax.set_title("Mean image; top cells outlined" + note)


def top_cells_pdf_name(
    n: int,
    rank_by: str,
    tag: str,
    variant: str,
    x_units: str,
    middle_frames_shown: int | None = None,
    clip_percentiles: tuple[float, float] | None = None,
) -> str:
    """File name of the PDF of one ranking's top cells.

    The display settings are part of the name, so a figure made with other settings does not
    overwrite it: ``top30_cells_by_p_cell_<tag>_<variant>_frames_mid8000_clip1-99.pdf``. A
    setting that is None adds nothing.
    """
    name = f"top{n}_cells_by_{rank_by}_{tag}_{variant}_{x_units}"
    if middle_frames_shown is not None:
        name += f"_mid{middle_frames_shown}"
    if clip_percentiles is not None:
        low, high = clip_percentiles
        name += f"_clip{low:g}-{high:g}"
    return name + ".pdf"


def window_tag(window_s: tuple[float, float]) -> str:
    """The window around the outcome for a file name, e.g. ``win-10to30s``."""
    start, stop = window_s
    return f"win{start:g}to{stop:g}s"


def pulse_responses_png_name(
    file_tag: str,
    variant: str,
    measure_tag: str,
    x_units: str,
    window_s: tuple[float, float],
) -> str:
    """File name of the pooled response figure; the window is part of it."""
    return (
        f"pulse_responses_{file_tag}_{variant}_{measure_tag}_{x_units}_{window_tag(window_s)}.png"
    )


def _ordinal(value: float) -> str:
    number = int(value) if float(value).is_integer() else value
    suffix = (
        "th"
        if 10 <= number % 100 < 20 or number % 10 not in (1, 2, 3)
        else ("st", "nd", "rd")[number % 10 - 1]
    )
    return f"{number}{suffix}"


def plot_top_cells(
    top: dict,
    x_units: str = "s",
    title: str = "",
    variant: str = "",
    clip_percentiles: tuple[float, float] | None = (1, 99),
    middle_frames_shown: int | None = None,
):
    """Stacked ΔF/F0 traces of a run's most responsive cells, and where they are.

    Each ROI has one colour, used for its trace and for its outline and number in the field of
    view. The best ROI is on top; ``ROI <n> (p=<p(cell)>)`` tick labels mark each trace's
    baseline. Outcomes are marked above the traces, coloured by number of reward pulses.
    Each trace is drawn clipped to its own percentile range, so a brief transient does not set
    the row spacing; the ranking behind *top* uses the unclipped values. Clipping flattens any
    peak above the high percentile, so each tick label carries the ROI's true maximum ΔF/F0 over
    the whole run (``ROI 12 (p=0.93, max 2.4)``) and a small tick on the x axis, in the ROI's
    colour, marks the frame of that maximum when it is among the frames shown. A ROI with no
    finite value in the frames shown is an empty row, labelled "(no data in shown frames)".

    Args:
        top (dict): From ``src.funcimg_analysis.collect.top_cells``.
        x_units (str): ``"s"`` (pyControl time) or ``"frames"`` (frame index).
        title (str): Figure title, e.g. the session and run.
        variant (str): Destriping variant, named in the traces' title.
        clip_percentiles (tuple[float, float] | None): Per-trace display range in percentiles,
            e.g. ``(1, 99)``; None draws the traces unclipped. The percentiles are taken over the
            frames shown. Must be None or ``0 <= low < high <= 100``.
        middle_frames_shown (int | None): Draw only this many frames from the middle of the
            run, e.g. ``8000``; outcomes outside them are not marked. Default None, every frame.

    Returns:
        matplotlib.figure.Figure: The figure.

    Raises:
        ValueError: If *clip_percentiles* or *middle_frames_shown* is invalid.
        TypeError: If *middle_frames_shown* is not an integer.

    Warns:
        UserWarning: If ``x_units="s"`` and none of the frames shown has a pyControl time; the
            traces panel is then empty and the title says so.
    """
    check_clip_percentiles(clip_percentiles)
    n_frames = top["dff"].shape[1]
    frames = middle_frames(n_frames, middle_frames_shown)  # validated before a figure exists
    fig, (ax_traces, ax_fov) = plt.subplots(
        1, 2, figsize=(16, 9), gridspec_kw={"width_ratios": [3, 1]}, constrained_layout=True
    )
    colors = roi_colors(len(top["cells"]))
    has_times = _draw_traces(ax_traces, top, x_units, colors, clip_percentiles, frames)
    n = len(top["cells"])
    ranking = RANKING_TITLES[top["rank_by"]]
    heading = f"{variant} green: top {n} ROIs by {ranking}".strip()
    if clip_percentiles:
        low, high = clip_percentiles
        heading += f"\nclipped to {_ordinal(low)}–{_ordinal(high)} percentile per ROI"
    if frames.stop - frames.start < n_frames:
        heading += f"; middle {frames.stop - frames.start} of {n_frames} frames"
    if not has_times:
        warnings.warn(
            f"{title or 'top cells'}: none of the frames shown has a pyControl time, so the "
            "traces cannot be placed on the time axis; use x_units='frames'.",
            UserWarning,
            stacklevel=2,
        )
        heading += "\nno pyControl time in the frames shown"
    ax_traces.set_title(heading)
    draw_fov(ax_fov, top, colors)
    fig.suptitle(title)
    return fig
