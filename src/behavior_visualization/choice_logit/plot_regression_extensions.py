"""Figures of the choice-logit extensions: believed value, binary or pulse, variance, calibration.

Drawing only: the numbers come from
:mod:`src.behavior_analysis.choice_logit.choice_logit_extensions` (tables
per source from :func:`src.pipelines.choice_logit.regression_extensions.extension_tables`). Each
``plot_*``
function builds a figure, then saves it (pdf, png, svg and a caption file) or shows it. Every figure
puts the mice next to the controls (simulated Q-learner, null agents) as one panel or one
column per source, one dot per mouse and black bars for the mean ± se across mice. A source with
fewer than ``choice_logit_extensions.MIN_TEST_MICE`` mice shows its mice only: no mean, error
bar or test.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.behavior_analysis.choice_logit import choice_logit_extensions as ext
from src.behavior_analysis.data.stats import nan_mean, nan_standard_error
from src.behavior_visualization.common.plot_helpers import (
    Title,
    count_text,
    despine,
    error_drawn,
    jitter_rng,
    jittered,
    mouse_color_map,
    save_or_show,
    set_title,
)
from src.behavior_visualization.value.plot_value_psychometrics import finish_figure, new_figure

DOT_JITTER = 0.14
CALIBRATION_LABELS = {"full": "Full Model", "choice only": "Choice History Only"}
# (column, panel heading, y label) of each believed-value panel.
BELIEVED_PANELS = (
    ("believed_weight", "Believed Value Weight", "Weight (Logit per Pulse)"),
    ("unvisited_weight", "Never-Visited Weight", "Weight (Logit)"),
    ("gain_believed", "Believed Value Added to Full Model", "Held-Out Gain (Nats per Choice)"),
    (
        "gain_no_reward_believed",
        "Believed Value Instead of Reward Lags",
        "Held-Out Gain (Nats per Choice)",
    ),
)
BINARY_PANELS = (
    ("gain_binary", "Binary Instead of Pulse"),
    ("gain_binary_pulse", "Binary and Pulse Instead of Pulse"),
)
PARTITION_COLUMNS = (
    ("unique_tower", "Unique Tower"),
    ("unique_reward", "Unique Reward"),
    ("unique_choice", "Unique Choice"),
    ("shared_reward_choice", "Shared Reward/Choice"),
)


# ---------------------------------------------------------------------------
# Shared drawing
# ---------------------------------------------------------------------------


def _check_sources(sources: Mapping[str, pd.DataFrame]) -> None:
    if not sources:
        raise ValueError("no sources to plot")
    empty = [name for name, table in sources.items() if table.empty]
    if empty:
        raise ValueError(f"no mice in {empty}")


def colors_of(tables: Sequence[pd.DataFrame]) -> dict[str, str]:
    """One color per mouse over all *tables*, in sorted order."""
    return mouse_color_map(sorted({m for t in tables for m in t["mouse"]}))


# Color of the mice that are not in the first source (e.g. simulated Q-learners).
OTHER_MICE_COLOR = "0.55"


def source_colors(tables: Sequence[pd.DataFrame]) -> tuple[dict[str, str], frozenset[str]]:
    """Colors of every mouse, and the mice left out of the legend.

    The mice of the first table (the real mice when there are any) get the per-mouse palette; a
    mouse that is not in the first table (a simulated Q-learner, say) is gray and unnamed, so the
    legend lists the mice only. Null agents fitted to a mouse share its name and its color.

    Args:
        tables (Sequence[pd.DataFrame]): One table per source with a ``mouse`` column, the mice
            first.

    Returns:
        tuple[dict[str, str], frozenset[str]]: ``{mouse: color}`` and the gray mice.
    """
    colors = colors_of(tables[:1])
    others = frozenset(m for t in tables[1:] for m in t["mouse"]) - set(colors)
    return {**colors, **dict.fromkeys(others, OTHER_MICE_COLOR)}, others


def legend_if_named(ax: plt.Axes) -> None:
    """Draw the panel's legend when it names anyone (a panel of gray mice names no one)."""
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=9)


def draw_column(
    ax: plt.Axes,
    x: float,
    values: pd.Series,
    colors: Mapping[str, str],
    rng: np.random.Generator,
    labeled: set,
    bands: Mapping[str, tuple[float, float]] | None = None,
    summary: bool = True,
) -> float:
    """Dots of the mice (index of *values*) at *x*, and the mean ± se bar if enough mice.

    A mouse is named in the legend once per panel (*labeled* collects those already named). The
    bar is drawn for ``MIN_TEST_MICE`` or more mice only.

    Args:
        ax (plt.Axes): The panel.
        x (float): Position of the column.
        values (pd.Series): One value per mouse, indexed by mouse.
        colors (Mapping[str, str]): ``{mouse: color}``.
        rng (np.random.Generator): Jitter of the dots.
        labeled (set): Mice already in the panel's legend; updated.
        bands (Mapping[str, tuple[float, float]] | None): ``{mouse: (low, high)}`` drawn as a gray
            vertical band behind that mouse's dot (e.g. its null interval). Default ``None``.
        summary (bool): Draw the mean ± se bar (with enough mice). Default True.

    Returns:
        float: The standard error drawn, NaN when no bar or no error is drawn.
    """
    for mouse, value in values.dropna().items():
        label = None if mouse in labeled else mouse
        labeled.add(mouse)
        dot_x = jittered(np.array([x]), DOT_JITTER, rng)
        if bands and mouse in bands:
            ax.plot([dot_x[0]] * 2, bands[mouse], color="0.7", lw=5, alpha=0.6,
                    solid_capstyle="butt", zorder=2)  # fmt: skip
        ax.plot(dot_x, [value], "o", color=colors[mouse], ms=5, alpha=0.85, label=label,
                zorder=3)  # fmt: skip
    if not summary or len(values.dropna()) < ext.MIN_TEST_MICE:
        return np.nan
    column = values.to_numpy(dtype=float)[:, None]
    mean, se = float(nan_mean(column)[0]), float(nan_standard_error(column)[0])
    ax.errorbar(x, mean, yerr=se if np.isfinite(se) else None, fmt="_", color="black",
                ms=24, mew=2, capsize=5, lw=2, zorder=5)  # fmt: skip
    return se


def source_stats_line(counts: Sequence[int], errors: Sequence[float]) -> str:
    """The statistics line of a panel that shows several sources, each with its own n.

    Sources with fewer than ``MIN_TEST_MICE`` mice show their mice only. The n is stated when all
    sources have the same one; otherwise the axis gives it (as the legend of Kernels by Source
    does).
    """
    same_n = len(set(counts)) == 1
    n_text = f"n={count_text(counts[0], 'mouse', 'mice')}" if same_n else "n per source on the axis"
    if max(counts) >= ext.MIN_TEST_MICE:
        average = "mean ± se" if error_drawn(*errors) else "mean"
        return f"{average} across mice, {n_text}"
    return f"per mouse, {n_text}"


def source_ticks(sources: Mapping[str, pd.DataFrame]) -> list[str]:
    """``"<source>\nn=<mice>"`` tick labels, one per source (rows of its table are mice)."""
    return [f"{name}\nn={len(t)}" for name, t in sources.items()]


def counts_caption(tables: Mapping[str, pd.DataFrame]) -> str:
    """The caption line naming each source's number of mice and the six-mice rule."""
    counts = ", ".join(f"{name}: n={len(t)} mice" for name, t in tables.items())
    return (
        f"Mice per source, {counts}. A source with fewer than {ext.MIN_TEST_MICE} mice shows "
        "per-mouse values only: no mean, error bar or test."
    )


def style_source_axis(ax: plt.Axes, labels: Sequence[str], ylabel: str) -> None:
    """A zero line, one slanted x tick per label, the y label and no top or right spine."""
    ax.axhline(0, color="0.7", lw=0.8, zorder=1)
    ax.set_xticks(range(len(labels)), labels, rotation=20, ha="right")
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.set_ylabel(ylabel, fontsize=12)
    despine(ax)


# ---------------------------------------------------------------------------
# 1. Believed value
# ---------------------------------------------------------------------------


def plot_believed_value(
    sources: Mapping[str, pd.DataFrame], save_path: str | Path | None = None
) -> None:
    """Believed-value weights and what the believed value adds to the held-out fit, by source.

    Args:
        sources (Mapping[str, pd.DataFrame]): ``{source label: believed_value_table}``, e.g. the
            mice, the simulated Q-learners and each null agent.
        save_path (str | Path | None): Base path without extension; ``None`` shows the figure.

    Raises:
        ValueError: If there are no sources or a source has no mice.
    """
    _check_sources(sources)
    colors, unnamed = source_colors(list(sources.values()))
    caption = (
        "Believed value: the magnitude (pulses) an arm last paid when the mouse chose it in the "
        "current problem; 0 before the arm's first visit, where the never-visited indicator is 1, "
        "so an unvisited arm is neither worth 0 nor dropped. Believed-value weight and "
        "never-visited weight: from the full model plus both columns, fitted to all of each "
        "mouse's trials.",
        "Gains are held-out (leave-one-problem-out) nats per choice of the full model minus that "
        "of the model named; positive is better than the full model. Instead of reward lags: "
        "choice history, tower bias and believed value, without the reward lags.",
        "Dots: mice (simulated agents not fitted to a mouse in gray); black bars: mean ± se "
        "across mice.",
        counts_caption(sources),
    )
    counts = [len(t) for t in sources.values()]
    with new_figure(nrows=2, ncols=2, figsize=(11, 8.6)) as (fig, axs):
        rng = jitter_rng()
        for ax, (column, heading, ylabel) in zip(axs.ravel(), BELIEVED_PANELS, strict=True):
            labeled: set = set(unnamed)
            errors = [
                draw_column(ax, x, t.set_index("mouse")[column], colors, rng, labeled)
                for x, t in enumerate(sources.values())
            ]
            style_source_axis(ax, source_ticks(sources), ylabel)
            set_title(ax, Title(heading, source_stats_line(counts, errors)), fontsize=12)
            legend_if_named(ax)
        finish_figure(fig, Title("Believed Value in the Choice Logit", None, caption))
    save_or_show(fig, save_path or None)


# ---------------------------------------------------------------------------
# 2. Binary against pulse
# ---------------------------------------------------------------------------


def _binary_caption_lines(sources: Mapping[str, pd.DataFrame]) -> list[str]:
    lines = []
    for name, table in sources.items():
        if len(table) < ext.MIN_TEST_MICE:
            lines.append(f"{name}: n={len(table)} mice, per-mouse values only.")
            continue
        for column, heading in BINARY_PANELS:
            row = ext.gain_summary(table, [column]).iloc[0]
            lines.append(
                f"{name}, {heading}: mean {row['mean']:.4f} ± se {row['se']:.4f}; "
                f"{row['n_positive']} of {row['n']} mice gain; Wilcoxon p={row['p_wilcoxon']:.3g}, "
                f"sign test p={row['p_sign']:.3g} (n={row['n']} mice)."
            )
    return lines


def plot_binary_vs_pulse(
    sources: Mapping[str, pd.DataFrame],
    save_path: str | Path | None = None,
    tail_decays: Mapping[str, float | None] | None = None,
) -> None:
    """Held-out gain of the binary and the binary + pulse reward codings over the pulse coding.

    A source with fewer than ``choice_logit_extensions.MIN_TEST_MICE`` mice shows its mice
    individually, with no mean, error bar or test.

    Args:
        sources (Mapping[str, pd.DataFrame]): ``{source label: binary_pulse_table}``.
        save_path (str | Path | None): Base path without extension; ``None`` shows the figure.
        tail_decays (Mapping[str, float | None] | None): ``{source label: tail decay chosen for
            the binary coding}``, named in the caption. Default ``None``.

    Raises:
        ValueError: If there are no sources or a source has no mice.
    """
    _check_sources(sources)
    colors, unnamed = source_colors(list(sources.values()))
    counts = [len(t) for t in sources.values()]
    decays = ""
    if tail_decays:
        named = ", ".join(f"{k}: {'none' if d is None else d}" for k, d in tail_decays.items())
        decays = (
            "The tail decay of the binary coding is chosen from a grid by the binary model's own "
            f"pooled held-out likelihood, as the pulse model's is ({named})."
        )
    caption = (
        "Pulse: reward lags code the pulses received (the standard model). Binary: reward lags "
        "code 1 if the arm paid, 0 otherwise, with the same lags and a tail. Binary and pulse: "
        "both sets of lags.",
        "Gain = held-out (leave-one-problem-out) nats per choice of the pulse model minus that of "
        "the model named; positive is better than the pulse model. Binary Instead of Pulse above "
        "0 means choices follow whether the arm was rewarded, below 0 that they follow the "
        "magnitude.",
        decays,
        "Every weight has the same L2 penalty (a Gaussian prior, SD 1 logit per unit of its "
        "feature). A pulse feature runs 0-4 and a binary feature 0-1, so the same effect needs a "
        "weight up to four times larger on the binary lags and is shrunk more: the comparison "
        "slightly favors the pulse coding. The columns are not standardized, to leave the "
        "standard model unchanged.",
        "Dots: mice (simulated agents not fitted to a mouse in gray); black bars: mean ± se "
        "across mice.",
        counts_caption(sources),
        *_binary_caption_lines(sources),
    )
    with new_figure(ncols=2, figsize=(13.5, 5.2), sharey=True) as (fig, axs):
        rng = jitter_rng()
        for ax, (column, heading) in zip(axs, BINARY_PANELS, strict=True):
            labeled: set = set(unnamed)
            errors = [
                draw_column(ax, x, t.set_index("mouse")[column], colors, rng, labeled)
                for x, t in enumerate(sources.values())
            ]
            style_source_axis(
                ax,
                source_ticks(sources),
                "Held-Out Gain over Pulse (Nats per Choice)" if ax is axs[0] else "",
            )
            set_title(ax, Title(heading, source_stats_line(counts, errors)), fontsize=12)
            legend_if_named(ax)
        finish_figure(fig, Title("Binary and Pulse Reward Coding", None, caption))
    save_or_show(fig, save_path or None)


# ---------------------------------------------------------------------------
# 3. Unique and shared variance
# ---------------------------------------------------------------------------


def plot_variance_partition(
    sources: Mapping[str, pd.DataFrame], save_path: str | Path | None = None
) -> None:
    """Unique variance of each regressor group, and the shared reward/choice term, by source.

    One figure with a panel per source on a common y axis, so the mice can be read against the
    controls directly.

    Args:
        sources (Mapping[str, pd.DataFrame]): ``{source label: variance_partition}``.
        save_path (str | Path | None): Base path without extension; ``None`` shows the figure.

    Raises:
        ValueError: If there are no sources or a source has no mice.
    """
    _check_sources(sources)
    colors, unnamed = source_colors(list(sources.values()))
    labels = [label for _, label in PARTITION_COLUMNS]
    caption = (
        "Held-out (leave-one-problem-out) nats per choice. Unique tower, reward and choice: the "
        "held-out negative log likelihood of the full model without that group minus the full "
        "model's: what only that group explains, given the other two.",
        "Shared reward/choice = [gain(reward + tower) + gain(choice + tower)] - gain(full), with "
        "gain(model) = loss(tower only) - loss(model). It is the sum of what each history explains "
        "beyond the tower bias minus what both explain together: positive when they explain the "
        "same choices, negative when they complement each other. Unique reward + unique "
        "choice + shared = the full model's gain over the tower only model.",
        "Dots: mice (simulated agents not fitted to a mouse in gray); black bars: mean ± se "
        "across mice.",
        counts_caption(sources),
    )
    width = max(5.5, 3.6 * len(sources))
    with new_figure(ncols=len(sources), figsize=(width, 4.8), sharey=True, squeeze=False) as (
        fig,
        axs,
    ):
        rng = jitter_rng()
        for ax, (name, table) in zip(axs[0], sources.items(), strict=True):
            labeled: set = set(unnamed)
            errors = [
                draw_column(ax, x, table.set_index("mouse")[column], colors, rng, labeled)
                for x, (column, _) in enumerate(PARTITION_COLUMNS)
            ]
            style_source_axis(ax, labels, "Held-Out Gain (Nats per Choice)")
            set_title(ax, Title(name, source_stats_line([len(table)], errors)), fontsize=12)
            legend_if_named(ax)
        finish_figure(fig, Title("Unique and Shared Variance of Regressor Groups", None, caption))
    save_or_show(fig, save_path or None)


# ---------------------------------------------------------------------------
# 4. Calibration
# ---------------------------------------------------------------------------


def _error_text(summary: pd.DataFrame, column: str) -> str:
    """A summary column across mice: mean ± se for enough mice, else each mouse's value."""
    values = summary.set_index("mouse")[column]
    if len(values) < ext.MIN_TEST_MICE:
        return "per mouse " + ", ".join(f"{m} {v:.4f}" for m, v in values.items())
    mean, se = float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))
    return f"mean {mean:.4f} ± se {se:.4f} across mice (n={len(values)} mice)"


def _plot_calibration_panel(
    table: pd.DataFrame, colors: Mapping[str, str], floor: float, ax: plt.Axes
) -> float:
    """Draw one model's calibration curves into *ax*; return the standard error drawn (or NaN).

    Mice are thin lines; the mean ± se across mice is drawn for ``MIN_TEST_MICE`` or more mice.
    The gray band around the identity is the mean noise floor: how far a perfectly calibrated
    model would typically lie from the identity by sampling noise alone.
    """
    predicted = table.pivot(index="mouse", columns="bin", values="mean_predicted")
    observed = table.pivot(index="mouse", columns="bin", values="observed")
    enough = len(observed) >= ext.MIN_TEST_MICE
    line = dict(lw=1, ms=3, alpha=0.5) if enough else dict(lw=1.6, ms=4, alpha=0.9)
    x = np.array([0.0, 1.0])
    ax.fill_between(x, x - floor, x + floor, color="0.88", lw=0, zorder=0)
    for mouse in observed.index:
        ax.plot(predicted.loc[mouse], observed.loc[mouse], "-o", color=colors[mouse],
                label=mouse, **line)  # fmt: skip
    se = np.nan
    if enough:
        se_bins = nan_standard_error(observed.to_numpy())
        ax.errorbar(nan_mean(predicted.to_numpy(), axis=0), nan_mean(observed.to_numpy(), axis=0),
                    yerr=se_bins, fmt="-o", color="black", ms=5, lw=2, capsize=3,
                    zorder=5)  # fmt: skip
        se = float(se_bins.max())
    ax.plot([0, 1], [0, 1], ":", color="0.4", lw=1)  # identity: observed = predicted
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("Mean Predicted P(Choice)", fontsize=12)
    ax.set_ylabel("Observed Fraction Chosen", fontsize=12)
    despine(ax)
    return se


def _calibration_caption(
    label: str, summaries: Mapping[str, pd.DataFrame], n_bins: int
) -> tuple[str, ...]:
    nats = {m: s["nats_per_choice"].mean() for m, s in summaries.items()}
    nats_text = ", ".join(f"{CALIBRATION_LABELS.get(m, m)} {v:.4f}" for m, v in nats.items())
    return (
        f"{label}. Every (trial, arm) pair is one prediction: the arm's held-out probability "
        f"(leave-one-problem-out), with outcome 1 if that arm was chosen. Predictions are sorted "
        f"and cut into {n_bins} equal-count (quantile) bins per mouse; x is the bin's mean "
        "predicted probability, y the fraction of its arms that were chosen. Dotted: identity "
        "(perfect calibration). Thin lines: mice; black: mean ± se across mice, drawn for at "
        f"least {ext.MIN_TEST_MICE} mice. Gray band: the mean noise floor, the distance from the "
        "identity that a perfectly calibrated model would show by sampling noise alone "
        "(outcomes drawn from the mouse's own predictions, fixed seed).",
        *(
            f"{CALIBRATION_LABELS.get(m, m)}: mean |observed - predicted| across bins, "
            f"{_error_text(s, 'mean_abs_error')}; its noise floor, "
            f"{_error_text(s, 'noise_floor_mean_abs_error')}; weighted by bin size, "
            f"{_error_text(s, 'weighted_abs_error')}."
            for m, s in summaries.items()
        ),
        "Calibration is not predictive quality. The choice-history-only model is calibrated "
        "almost by construction: it has few distinct predictions, so each bin pools many similar "
        "situations and the fraction chosen matches their mean, yet it predicts worse. Held-out "
        f"nats per choice (mean across mice, lower is better): {nats_text}.",
    )


def plot_calibration(
    label: str,
    tables: Mapping[str, pd.DataFrame],
    summaries: Mapping[str, pd.DataFrame],
    save_path: str | Path | None = None,
) -> None:
    """Observed choice frequency against held-out predicted probability, for one source.

    Args:
        label (str): The source's name, e.g. ``"Mice"``.
        tables (Mapping[str, pd.DataFrame]): ``{model: calibration_table}``, one panel each (the
            full model and the choice-history-only model).
        summaries (Mapping[str, pd.DataFrame]): ``{model: summary}`` with the columns
            ``mouse, mean_abs_error, weighted_abs_error, noise_floor_mean_abs_error,
            noise_floor_weighted_abs_error, nats_per_choice`` (the ``calibration_summary`` of
            ``regression_extensions.ExtensionTables``).
        save_path (str | Path | None): Base path without extension; ``None`` shows the figure.

    Raises:
        ValueError: If there are no calibration tables, one has no mice, or the models of
            *tables* and *summaries* differ.
    """
    if not tables:
        raise ValueError("no calibration tables to plot")
    _check_sources(tables)
    if set(tables) != set(summaries):
        raise ValueError(f"tables {list(tables)} and summaries {list(summaries)} differ")
    colors = colors_of(list(tables.values()))
    n_bins = max(t["bin"].nunique() for t in tables.values())
    caption = _calibration_caption(label, summaries, n_bins)
    with new_figure(ncols=len(tables), figsize=(4.8 * len(tables), 5.0)) as (fig, axs):
        for ax, (model, table) in zip(np.atleast_1d(axs), tables.items(), strict=True):
            floor = float(summaries[model]["noise_floor_mean_abs_error"].mean())
            se = _plot_calibration_panel(table, colors, floor, ax)
            set_title(
                ax,
                Title(
                    CALIBRATION_LABELS.get(model, model),
                    source_stats_line([table["mouse"].nunique()], [se]),
                ),
                fontsize=12,
            )
            ax.legend(fontsize=9)
        finish_figure(fig, Title(f"Calibration of Held-Out Predictions: {label}", None, caption))
    save_or_show(fig, save_path or None)


# ---------------------------------------------------------------------------
# One call for every figure
# ---------------------------------------------------------------------------


def plot_extensions(tables: Mapping, out_dir: str | Path) -> None:
    """Save every figure of the extensions for several sources into *out_dir*.

    Writes the believed-value, binary-against-pulse and variance figures (one panel or column per
    source) and a calibration figure for each source, each with pdf, png, svg and a caption file.

    Args:
        tables (Mapping): ``{source label: ExtensionTables}``, the mice first.
        out_dir (str | Path): Folder; created if missing.
    """
    out_dir = Path(out_dir)
    plot_believed_value({k: t.believed for k, t in tables.items()}, out_dir / "Believed Value")
    plot_binary_vs_pulse(
        {k: t.binary_pulse for k, t in tables.items()},
        out_dir / "Binary vs Pulse",
        {k: t.binary_tail_decay for k, t in tables.items()},
    )
    plot_variance_partition(
        {k: t.partition for k, t in tables.items()}, out_dir / "Unique and Shared Variance"
    )
    for label, source_tables in tables.items():
        plot_calibration(
            label,
            source_tables.calibration,
            source_tables.calibration_summary,
            out_dir / f"Calibration - {label}",
        )
