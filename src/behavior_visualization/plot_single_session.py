"""Visualise a single behavioural session: reward magnitudes, EMA, and choice distributions.

Each panel function draws into an ``ax`` you pass (or a new figure if you
don't), so it can be one panel of a larger figure. :func:`plot_single_session`
composes the panels into the standard four-panel figure and saves or shows it.

When the session carries its raw recordings (``"df"``), two maze heatmaps of that session's
pokes are added in a row below the standard panels, each centred under the bar chart above it:
long pokes after initiation (under the counts by tower) and pokes after the choice (under the
counts by rank; see :mod:`src.behavior_analysis.poke_events`). The standard panels keep their
size and their distance from the top of the figure.
"""

import warnings
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

import src.behavior_visualization.plot_style  # noqa: F401  # applies rcParams
from src.behavior_analysis.poke_counts import (
    SessionPokeCounts,
    session_choice_ports,
    session_poke_counts,
)
from src.behavior_analysis.poke_events import PokeParseError
from src.behavior_analysis.single_session import (
    Reversals,
    choice_counts_by_tower,
    choice_probability_matrix,
    chosen_magnitudes,
    event_indices_from_cumulative,  # noqa: F401  # kept importable here; it was part of this module's public API
    has_ema_traces,
    latest_rank_counts,
    ordered_rank_counts,
    percent_of_total,
    reversal_event_indices,
)
from src.behavior_import.problem_numbering import DEFAULT_INITIATION_TOWER
from src.behavior_visualization.plot_helpers import save_or_show
from src.behavior_visualization.plot_maze_heatmaps import (
    draw_maze_heatmap,
    outline_legend_handles,
)

# Figure styling, not part of the module API.
_TOWER_COLORS = ["#A6CEE3", "#FDBF6F", "#B2DF8A"]
_EMA_LABELS = {"ema_best_arm_choices": "Best", "ema_second_arm_choices": "Second"}
_EMA_COLORS = {"ema_best_arm_choices": "#000000", "ema_second_arm_choices": "#858585"}
_RANK_COLORS = {
    "best": _EMA_COLORS["ema_best_arm_choices"],
    "second": _EMA_COLORS["ema_second_arm_choices"],
    "third": "#CCCCCC",
}
_GOOD_REVERSAL_COLOR = "#3A982E"
_BAD_REVERSAL_COLOR = "#F97979"


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------


def _ensure_ax(ax: Axes | None) -> Axes:
    """Return *ax*, or the axes of a new figure when none is given."""
    if ax is None:
        _, ax = plt.subplots()
    return ax


def _vlines(ax: Axes, indices: list[int], color: str, label: str) -> None:
    """Dashed vertical line before each trial in *indices*; only the first is labelled."""
    for j, i in enumerate(indices):
        ax.axvline(
            i - 0.5, linestyle="--", color=color, linewidth=2.5, label=label if j == 0 else None
        )


def add_reversal_lines(
    ax: Axes, good_idx: list[int], bad_idx: list[int], block_idx: list[int]
) -> None:
    """Draw dashed lines for good/bad reversals, or for blocks when neither exists.

    Args:
        ax (Axes): Axes to draw on, in place; the x axis is trial number.
        good_idx (list[int]): Trial indices of good reversals (green lines).
        bad_idx (list[int]): Trial indices of bad reversals (red lines).
        block_idx (list[int]): Trial indices of block starts, drawn only when both *good_idx*
            and *bad_idx* are empty.
    """
    if good_idx or bad_idx:
        _vlines(ax, good_idx, _GOOD_REVERSAL_COLOR, "Good Rev")
        _vlines(ax, bad_idx, _BAD_REVERSAL_COLOR, "Bad Rev")
    elif block_idx:
        _vlines(ax, block_idx, _GOOD_REVERSAL_COLOR, "Rev")


def plot_magnitudes_and_choices(
    mags_by: Mapping[str, Sequence[float]],
    choices_by: Mapping[str, Sequence[float]],
    towers: Sequence[str],
    reversals: Reversals,
    *,
    title: str | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Per-tower reward magnitude traces with the chosen magnitude as black dots.

    Args:
        mags_by (Mapping[str, Sequence[float]]): ``{tower: magnitude per trial}``; all the same
            length.
        choices_by (Mapping[str, Sequence[float]]): ``{tower: one-hot choice per trial}``.
        towers (Sequence[str]): Tower names to draw, in order; at most the first three are
            coloured and drawn.
        reversals (Reversals): ``(good_idx, bad_idx, block_idx)`` as from
            :func:`reversal_event_indices`.
        title (str | None): Axes title; ``"Reward Magnitudes & Choices"`` if None or empty.
            Default None.
        ax (Axes | None): Axes to draw into; a new figure if None. Default None.

    Returns:
        Axes: The axes drawn into.
    """
    ax = _ensure_ax(ax)
    N = len(mags_by[towers[0]])
    x = np.arange(N)
    # Towers beyond the palette length (3) are not drawn, though their choices
    # still count in the bar chart.
    for tower, color in zip(towers, _TOWER_COLORS, strict=False):
        ax.plot(
            x, np.asarray(mags_by[tower], dtype=float), linewidth=2.5, color=color, label=f"{tower}"
        )
    chosen = chosen_magnitudes(mags_by, choices_by, towers=towers, N=N)
    ax.scatter(x, chosen, s=30, color="black", marker="o", label="Choice", zorder=5)
    add_reversal_lines(ax, *reversals)
    ax.set_ylabel("Reward Magnitude", fontsize=12)
    ax.set_yticks([0, 1, 4])
    ax.set_title(title or "Reward Magnitudes & Choices", fontsize=14)
    ax.legend(loc="upper left", fontsize=10)
    return ax


def add_choice_probabilities(probabilities: np.ndarray, towers: Sequence[str], ax: Axes) -> Axes:
    """Overlay dashed per-tower model choice probabilities on a right-hand twin of *ax*.

    Args:
        probabilities (np.ndarray): ``(N, len(towers))`` array, one column per tower.
        towers (Sequence[str]): Tower names, in column order; at most the first three are
            drawn.
        ax (Axes): Axes to attach the twin to.

    Returns:
        Axes: The new twin axis.
    """
    probability_axis = ax.twinx()
    x = np.arange(len(probabilities))
    for arm, (tower, color) in enumerate(zip(towers, _TOWER_COLORS, strict=False)):
        probability_axis.plot(
            x,
            probabilities[:, arm],
            color=color,
            linestyle="--",
            linewidth=1.8,
            label=f"RNN P({tower})",
            alpha=0.95,
        )
    probability_axis.set_ylim(0, 1)
    probability_axis.set_ylabel("Vanilla RNN choice probability", fontsize=12)
    probability_axis.spines["top"].set_visible(False)
    probability_axis.legend(loc="upper right", fontsize=9)
    return probability_axis


def plot_ema_traces(
    session_data: Mapping[str, Any],
    ema_keys: Sequence[str],
    reversals: Reversals,
    *,
    N: int,
    ax: Axes | None = None,
) -> Axes:
    """EMA choice curves over *N* trials for the keys in *ema_keys* that the session holds.

    Args:
        session_data (Mapping[str, Any]): Session dict holding one length-*N* array per EMA key.
        ema_keys (Sequence[str]): Keys to plot; keys absent from *session_data* are skipped.
        reversals (Reversals): ``(good_idx, bad_idx, block_idx)`` as from
            :func:`reversal_event_indices`.
        N (int): Number of trials on the x axis.
        ax (Axes | None): Axes to draw into; a new figure if None. Default None.

    Returns:
        Axes: The axes drawn into.
    """
    ax = _ensure_ax(ax)
    x = np.arange(N)
    for k in ema_keys:
        if k in session_data:
            ax.plot(
                x,
                np.asarray(session_data[k], dtype=float),
                linewidth=3,
                color=_EMA_COLORS.get(k),
                label=_EMA_LABELS.get(k, k),
            )
    ax.axhline(0.7, linestyle="--", linewidth=2, color="#DCDCDC", alpha=0.7, label="Threshold")
    add_reversal_lines(ax, *reversals)
    ax.set_ylabel("EMA", fontsize=12)
    ax.set_xlabel("Trial", fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.legend(loc="upper left", fontsize=10)
    return ax


def _bar_with_percent_labels(
    ax: Axes, names: Sequence[str], counts: Sequence[int], colors: Sequence[str], ylabel: str
) -> None:
    """Bar chart of *counts* with each bar's share of the total written above it."""
    bars = ax.bar(names, counts, color=colors)
    ax.set_ylabel(ylabel, fontsize=12)
    for rect, pct in zip(bars, percent_of_total(counts), strict=True):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height(),
            f"{pct:.0f}%",
            fontsize=12,
            ha="center",
            va="bottom",
        )


def plot_choices_by_tower(
    choices_by: Mapping[str, Sequence[float]],
    towers: Sequence[str],
    *,
    N: int,
    ax: Axes | None = None,
) -> Axes:
    """Bar chart of how often each tower was chosen over the first *N* trials.

    Args:
        choices_by (Mapping[str, Sequence[float]]): ``{tower: one-hot choice per trial}``.
        towers (Sequence[str]): Towers to count, in bar order.
        N (int): Number of leading trials counted.
        ax (Axes | None): Axes to draw into; a new figure if None. Default None.

    Returns:
        Axes: The axes drawn into.
    """
    ax = _ensure_ax(ax)
    counts = choice_counts_by_tower(choices_by, towers, N)
    _bar_with_percent_labels(ax, towers, counts, _TOWER_COLORS[: len(towers)], "Count by Tower")
    return ax


def plot_choices_by_rank(rank_counts: Mapping[str, float] | None, ax: Axes | None = None) -> Axes:
    """Bar chart of choices per rank; a placeholder text when *rank_counts* is empty/None.

    Args:
        rank_counts (Mapping[str, float] | None): ``{rank: count}`` with ranks ``"best"``,
            ``"second"``, ``"third"``; None or empty draws "No rank data".
        ax (Axes | None): Axes to draw into; a new figure if None. Default None.

    Returns:
        Axes: The axes drawn into.
    """
    ax = _ensure_ax(ax)
    if rank_counts:
        names, counts = ordered_rank_counts(rank_counts)
        colors = [_RANK_COLORS.get(k, "#CCCCCC") for k in names]
        _bar_with_percent_labels(ax, names, counts, colors, "Count by Rank")
    else:
        ax.text(0.5, 0.5, "No rank data", ha="center", va="center")
        ax.set_xticks([])
        ax.set_ylabel("Count by Rank", fontsize=12)
    return ax


# ---------------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------------


# Inches added below the standard panels for the two maze maps.
_MAP_ROW_INCHES = 4.4
# Side of each (square) maze map, and the colorbar's gap, width and height relative to the side.
_MAP_SIDE_INCHES = 3.3
_COLORBAR_GAP_INCHES = 0.12
_COLORBAR_WIDTH_INCHES = 0.16
_COLORBAR_HEIGHT_FRACTION = 0.8
# Distance from the top of the added row to the top of the maps (room for their two-line titles),
# and from the bottom of the figure to the bottom of the added row (room for the legend).
_MAP_TOP_MARGIN_INCHES = 0.45
_MAP_BOTTOM_MARGIN_INCHES = 0.5


class MapAxes(NamedTuple):
    """The axes of one maze map and of its colorbar."""

    map: Axes
    colorbar: Axes


class PokeMaps(NamedTuple):
    """What the two maze panels of the single-session figure show.

    Attributes:
        counts (SessionPokeCounts): The session's poke counts; ``after_initiation`` and
            ``after_choice`` give the count per port for the left and right map.
        current_ports (frozenset[str]): The session's choice ports, outlined in green.
        previous_ports (frozenset[str] | None): The previous problem's choice ports, outlined
            in red; None for a mouse without a previous problem.
        initiation_port (str): The initiation port, outlined in blue.
    """

    counts: SessionPokeCounts
    current_ports: frozenset[str]
    previous_ports: frozenset[str] | None
    initiation_port: str


def _panel_grid_margins(base_height: float, extra: float) -> dict[str, float]:
    """GridSpec margins that keep the standard panels in place when *extra* inches are added.

    The figure grows at the bottom, so the panels keep their size and distance from the top.
    """
    if extra == 0:
        return {}
    total = base_height + extra
    bottom, top = plt.rcParams["figure.subplot.bottom"], plt.rcParams["figure.subplot.top"]
    return {
        "bottom": (extra + bottom * base_height) / total,
        "top": (extra + top * base_height) / total,
    }


def _add_map_axes(fig: Figure, under: Axes, base_height: float, extra: float) -> MapAxes:
    """A square map axes centred horizontally under *under*, and its colorbar axes beside it.

    The map is centred on the plot area of the panel above it, so the colorbar (placed outside
    the map's box) does not push the map off-centre; the map sits in the *extra* inches below
    the standard panels.
    """
    width, total = fig.get_size_inches()[0], base_height + extra
    box = under.get_position()
    centre = (box.x0 + box.x1) / 2
    side_x, side_y = _MAP_SIDE_INCHES / width, _MAP_SIDE_INCHES / total
    top = (extra - _MAP_TOP_MARGIN_INCHES) / total
    bottom = top - side_y
    map_ax = fig.add_axes((centre - side_x / 2, bottom, side_x, side_y))
    bar_height = _COLORBAR_HEIGHT_FRACTION * side_y
    colorbar_ax = fig.add_axes(
        (
            centre + side_x / 2 + _COLORBAR_GAP_INCHES / width,
            bottom + (side_y - bar_height) / 2,
            _COLORBAR_WIDTH_INCHES / width,
            bar_height,
        )
    )
    return MapAxes(map_ax, colorbar_ax)


def _make_axes(
    has_ema: bool, with_maps: bool
) -> tuple[Figure, tuple[Axes, Axes | None, Axes, Axes], tuple[MapAxes, MapAxes] | None]:
    """Figure, ``(ax_magnitudes, ax_ema | None, ax_towers, ax_ranks)`` and the two map axes.

    The map axes are ``None`` unless *with_maps*; then the figure is taller by the map row. The
    first map sits under the counts-by-tower chart and the second under the counts-by-rank chart.
    """
    extra = _MAP_ROW_INCHES if with_maps else 0
    base_height = 11 if has_ema else 8
    fig = plt.figure(figsize=(14, base_height + extra))
    margins = _panel_grid_margins(base_height, extra)
    if has_ema:
        gs = GridSpec(3, 2, height_ratios=[2.2, 1.8, 1.5], hspace=0.25, wspace=0.25, **margins)
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, :], sharex=ax1)
        axes = (ax1, ax2, fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1]))
    else:
        gs = GridSpec(2, 2, height_ratios=[2.5, 1.5], hspace=0.25, wspace=0.25, **margins)
        ax1 = fig.add_subplot(gs[0, :])
        axes = (ax1, None, fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]))
    maps = None
    if with_maps:
        maps = (
            _add_map_axes(fig, axes[2], base_height, extra),
            _add_map_axes(fig, axes[3], base_height, extra),
        )
    return fig, axes, maps


def poke_maps_for_session(
    session_data: Mapping[str, Any],
    previous_choice_towers: Collection[str] | None,
    *,
    label: str = "session",
    poke_counts: SessionPokeCounts | None = None,
) -> PokeMaps | None:
    """The data of the two maze panels, or ``None`` when there is nothing to draw.

    ``None`` when the session has no raw recordings, and when its raw events cannot be read
    (a :class:`~src.behavior_analysis.poke_events.PokeParseError`): that warns with *label* and
    the reason, so the standard panels are still drawn. With *poke_counts* (already parsed, e.g.
    by ``session_poke_counts`` or ``mouse_poke_counts``) the raw events are not read again; a
    result with a ``parse_error`` draws no maps and does not warn a second time.

    Args:
        session_data (Mapping[str, Any]): Session dict; ``"df"`` holds the raw recordings and
            ``"initiation_tower"`` (optional) the initiation port.
        previous_choice_towers (Collection[str] | None): Choice ports of the mouse's previous
            problem; None for a mouse without one.
        label (str): Name of the session used in the warning. Default ``"session"``.
        poke_counts (SessionPokeCounts | None): Already parsed poke counts; None parses
            ``session_data["df"]`` here. Default None.

    Returns:
        PokeMaps | None: The maze-panel data, or None when there is nothing to draw.

    Raises:
        ValueError: From :func:`src.behavior_analysis.poke_counts.session_choice_ports` if the
            session has no choice towers.
        PokeAlignmentError: If the parsed trials do not match the session's imported trials.
    """
    if "df" not in session_data and poke_counts is None:
        return None
    if poke_counts is not None:
        counts = poke_counts
    else:
        try:
            counts = session_poke_counts(dict(session_data), label=label)
        except PokeParseError as error:
            warnings.warn(f"{label}: maze maps not drawn: {error}", UserWarning, stacklevel=2)
            return None
    if counts.parse_error is not None:
        return None
    return PokeMaps(
        counts=counts,
        current_ports=session_choice_ports(session_data),
        previous_ports=None
        if previous_choice_towers is None
        else frozenset(previous_choice_towers),
        initiation_port=session_data.get("initiation_tower") or DEFAULT_INITIATION_TOWER,
    )


def plot_poke_maps(
    maps: PokeMaps,
    axes: tuple[Axes, Axes],
    colorbar_axes: tuple[Axes, Axes] | None = None,
) -> None:
    """Draw long pokes after initiation (left) and pokes after the choice (right) of one session.

    Each map is coloured from 0 to its own largest count; the colorbar says so. Green outlines
    are the session's choice ports, red ones the previous problem's, blue the initiation port.

    Args:
        maps (PokeMaps): The data of the two maps.
        axes (tuple[Axes, Axes]): ``(initiation_ax, choice_ax)``, the axes for the long pokes
            after initiation and for the pokes after the choice.
        colorbar_axes (tuple[Axes, Axes] | None): Axes for the two colorbars, in the same order;
            None lets each colorbar take its room from its map. Default None.
    """
    scale = "colour scale 0 to this session's maximum"
    counts = maps.counts
    initiation_ax, choice_ax = axes
    initiation_cax, choice_cax = colorbar_axes or (None, None)
    common = {"current_ports": maps.current_ports, "previous_ports": maps.previous_ports}
    draw_maze_heatmap(
        counts.after_initiation,
        colorbar_label="Long pokes after initiation (count per port)",
        ax=initiation_ax,
        colorbar_ax=initiation_cax,
        title=f"Long pokes after initiation\n({scale})",
        initiation_port=maps.initiation_port,
        **common,
    )
    draw_maze_heatmap(
        counts.after_choice,
        colorbar_label="Pokes after choice (count per port)",
        ax=choice_ax,
        colorbar_ax=choice_cax,
        title=f"Pokes after choice\n({scale})",
        **common,
    )


def _style_spines(axes: Sequence[Axes | None]) -> None:
    for ax in axes:
        if ax is None:
            continue
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.2)
        ax.spines["bottom"].set_linewidth(1.2)


def _draw_session(
    session_data: Mapping[str, Any],
    *,
    mag_key: str,
    choice_key: str,
    ema_keys: Sequence[str],
    title: str | None,
    choice_probabilities_by_tower: Mapping[str, Sequence[float]] | None,
    poke_maps: PokeMaps | None,
) -> Figure:
    """Build the four-panel figure for one session (plus the maze maps, if any) and return it.

    The figure is created before the inputs are fully validated, so an error
    raised afterwards (e.g. invalid probabilities) leaves it open in pyplot.
    """
    has_ema = has_ema_traces(session_data, ema_keys)
    mags_by = session_data[mag_key]
    choices_by = session_data[choice_key]
    towers = list(mags_by.keys())
    N = len(mags_by[towers[0]])

    fig, (ax1, ax2, ax3, ax4), map_axes = _make_axes(has_ema, poke_maps is not None)
    reversals = reversal_event_indices(session_data, N)
    plot_magnitudes_and_choices(mags_by, choices_by, towers, reversals, title=title, ax=ax1)
    if choice_probabilities_by_tower is not None:
        probabilities = choice_probability_matrix(choice_probabilities_by_tower, towers, N)
        add_choice_probabilities(probabilities, towers, ax1)
    if has_ema:
        plot_ema_traces(session_data, ema_keys, reversals, N=N, ax=ax2)
    else:
        ax1.set_xlabel("Trial", fontsize=12)
    plot_choices_by_tower(choices_by, towers, N=N, ax=ax3)
    plot_choices_by_rank(latest_rank_counts(session_data), ax=ax4)
    _style_spines((ax1, ax2, ax3, ax4))
    if poke_maps is not None:
        plot_poke_maps(
            poke_maps,
            (map_axes[0].map, map_axes[1].map),
            (map_axes[0].colorbar, map_axes[1].colorbar),
        )
        handles = outline_legend_handles(True, poke_maps.previous_ports is not None, True)
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False)
    # The GridSpecs set hspace/wspace explicitly, which tight_layout reports as
    # "This figure includes Axes that are not compatible with tight_layout"
    # (with or without the twin axis) and then leaves every axis where the GridSpecs
    # put it. The call has no effect on this figure; it stays only so the figure
    # matches main, and that one UserWarning is hidden.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*not compatible with tight_layout", category=UserWarning
        )
        fig.tight_layout()
    return fig


def plot_single_session(
    session_data: Mapping[str, Any],
    mag_key: str = "reward_magnitudes_by_tower",
    choice_key: str = "choices_by_tower",
    ema_keys: Sequence[str] = ("ema_best_arm_choices", "ema_second_arm_choices"),
    title: str | None = None,
    save_path: str | Path | None = None,
    choice_probabilities_by_tower: Mapping[str, Sequence[float]] | None = None,
    previous_choice_towers: Collection[str] | None = None,
    poke_counts: SessionPokeCounts | None = None,
) -> None:
    """Plot a four-panel summary of a single experimental session.

    Panels include reward magnitude traces (with chosen-arm scatter),
    exponential moving-average (EMA) choice curves, and bar charts of choices
    by tower and by rank.  Reversal events are shown as vertical dashed lines
    on the top two panels.  A session that carries its raw recordings (``"df"``) also gets
    two maze heatmaps of its own pokes below the panels: long pokes after initiation and pokes
    after the choice, each coloured from 0 to its own maximum.

    Args:
        session_data (Mapping[str, Any]): Session dict containing at minimum
            ``"reward_magnitudes_by_tower"`` and ``"choices_by_tower"``, and
            optionally ``"good_reversals"``, ``"bad_reversals"``, ``"blocks"``,
            ``"rank_counts"``, and EMA keys.
        mag_key (str): Key for the per-tower magnitude dict
            ``{tower: magnitude per trial}``. Default ``"reward_magnitudes_by_tower"``.
        choice_key (str): Key for the per-tower one-hot choice dict
            ``{tower: choice per trial}``. Default ``"choices_by_tower"``.
        ema_keys (Sequence[str]): Keys whose values are plotted as EMA traces. Default
            ``("ema_best_arm_choices", "ema_second_arm_choices")``.
        title (str | None): Custom figure title for the top panel. Default None.
        save_path (str | Path | None): Base path (without extension) for saving ``.pdf`` and
            ``.png`` output. If None or empty, the figure is shown interactively. Default None.
        choice_probabilities_by_tower (Mapping[str, Sequence[float]] | None): Optional
            normalized pre-choice probabilities by physical tower, aligned to session trials;
            dashed lines on a right axis. Default None.
        previous_choice_towers (Collection[str] | None): Choice ports of the mouse's previous
            problem, outlined in red on the maze maps (the session's own choice ports are
            outlined in green and the initiation port in blue); None for a mouse without a
            previous problem. Ignored when the session has no raw recordings. Default None.
        poke_counts (SessionPokeCounts | None): The session's already parsed poke counts (a
            :class:`~src.behavior_analysis.poke_counts.SessionPokeCounts`, e.g. from
            ``session_poke_counts``), so the raw events are parsed once per session when
            several analyses use them. None parses ``session_data["df"]`` here. Default None.

    Raises:
        KeyError: if *mag_key* or *choice_key* is missing from *session_data*.
        TypeError: if *session_data* is not a mapping.
        ValueError: if the session has no choice towers, or *previous_choice_towers* names
            a port other than A1..C3; if *choice_probabilities_by_tower* does not use the
            session's tower labels, or is not finite, within [0, 1], aligned
            to every trial and normalized on each trial; also from matplotlib
            when one EMA key holds data and another is ``None``.

    Notes:
        ``session_data["reversal_boundary_events"]`` (optional) is a list of
        ``{"kind": "good" | "bad" | "block", "index": int}`` dicts that add reversal lines
        the cumulative ``good_reversals`` / ``bad_reversals`` counters did not
        mark; a ``"block"`` event is a block-start line, any other kind than ``"good"``
        is drawn as a bad reversal.

        Raw events that cannot be read as trials (``PokeParseError``) warn with the title
        and the reason, and the figure is drawn without the maze maps. A
        ``PokeAlignmentError`` (parsed trials differ from the imported ones) is not caught.

        The figure is closed on success and when saving it raises, but left open
        in pyplot when an error is raised while drawing it.
    """
    # Parsed before the figure exists; unreadable raw events warn and skip the maps.
    poke_maps = poke_maps_for_session(
        session_data, previous_choice_towers, label=title or "session", poke_counts=poke_counts
    )
    fig = _draw_session(
        session_data,
        mag_key=mag_key,
        choice_key=choice_key,
        ema_keys=ema_keys,
        title=title,
        choice_probabilities_by_tower=choice_probabilities_by_tower,
        poke_maps=poke_maps,
    )
    save_or_show(fig, save_path or None)  # an empty path means "show"
