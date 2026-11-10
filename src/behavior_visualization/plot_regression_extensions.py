"""Figures of the choice-logit extensions: believed value, binary or pulse, variance, calibration.

Drawing only: the numbers come from :mod:`src.behavior_analysis.choice_logit_extensions` (tables
per source from :func:`src.pipelines.regression_extensions.extension_tables`). Each ``plot_*``
function builds a figure, then saves it (pdf, png, svg and a caption file) or shows it. Every figure
puts the mice next to the controls (simulated Q-learner, null agents) as one panel or one
column per source, one dot per mouse and black bars for the mean ± se across mice.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.behavior_analysis import choice_logit_extensions as ext
from src.behavior_analysis.stats import nan_mean, nan_standard_error
from src.behavior_visualization.plot_choice_logit import mice_stats
from src.behavior_visualization.plot_helpers import (
    Title,
    count_text,
    despine,
    jitter_rng,
    jittered,
    mouse_color_map,
    save_or_show,
    set_title,
)
from src.behavior_visualization.plot_value_psychometrics import finish_figure, new_figure

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


def _colors_of(tables: Sequence[pd.DataFrame]) -> dict[str, str]:
    """One color per mouse over all *tables*, in sorted order."""
    return mouse_color_map(sorted({m for t in tables for m in t["mouse"]}))


def _draw_column(
    ax: plt.Axes,
    x: float,
    values: pd.Series,
    colors: Mapping[str, str],
    rng: np.random.Generator,
    labeled: set,
    show_mean: bool = True,
) -> tuple[float, float]:
    """Dots of the mice (index of *values*) at *x*, and the mean ± se bar; returns ``(mean, se)``.

    A mouse is named in the legend once per panel (*labeled* collects those already named).
    """
    for mouse, value in values.dropna().items():
        label = None if mouse in labeled else mouse
        labeled.add(mouse)
        ax.plot(jittered(np.array([x]), DOT_JITTER, rng), [value], "o", color=colors[mouse],
                ms=5, alpha=0.85, label=label, zorder=3)  # fmt: skip
    column = values.to_numpy(dtype=float)[:, None]
    mean, se = float(nan_mean(column)[0]), float(nan_standard_error(column)[0])
    if show_mean:
        ax.errorbar(x, mean, yerr=se if np.isfinite(se) else None, fmt="_", color="black",
                    ms=24, mew=2, capsize=5, lw=2, zorder=5)  # fmt: skip
    return mean, se


def _counts_caption(tables: Mapping[str, pd.DataFrame]) -> str:
    counts = ", ".join(f"{name}: n={len(t)} mice" for name, t in tables.items())
    return f"Mice per source, {counts}. A panel title states the largest n."


def _style_source_axis(ax: plt.Axes, labels: Sequence[str], ylabel: str) -> None:
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
    colors = _colors_of(list(sources.values()))
    caption = (
        "Believed value: the magnitude (pulses) an arm last paid when the mouse chose it in the "
        "current problem; 0 before the arm's first visit, where the never-visited indicator is 1, "
        "so an unvisited arm is neither worth 0 nor dropped. Believed-value weight and "
        "never-visited weight: from the full model plus both columns, fitted to all of each "
        "mouse's trials.",
        "Gains are held-out (leave-one-problem-out) nats per choice of the full model minus that "
        "of the model named; positive is better than the full model. Instead of reward lags: "
        "choice history, tower bias and believed value, without the reward lags.",
        "Dots: mice; black bars: mean ± se across mice.",
        _counts_caption(sources),
    )
    n_max = max(len(t) for t in sources.values())
    with new_figure(nrows=2, ncols=2, figsize=(11, 8.6)) as (fig, axs):
        rng = jitter_rng()
        for ax, (column, heading, ylabel) in zip(axs.ravel(), BELIEVED_PANELS, strict=True):
            labeled: set = set()
            errors = [
                _draw_column(ax, x, t.set_index("mouse")[column], colors, rng, labeled)[1]
                for x, t in enumerate(sources.values())
            ]
            _style_source_axis(ax, list(sources), ylabel)
            set_title(ax, Title(heading, mice_stats(n_max, *errors)), fontsize=12)
            ax.legend(fontsize=9)
        finish_figure(fig, Title("Believed Value in the Choice Logit", None, caption))
    save_or_show(fig, save_path or None)


# ---------------------------------------------------------------------------
# 2. Binary against pulse
# ---------------------------------------------------------------------------


def _source_label(name: str, n: int) -> str:
    return f"{name}\nn={n}"


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
    sources: Mapping[str, pd.DataFrame], save_path: str | Path | None = None
) -> None:
    """Held-out gain of the binary and the binary + pulse reward codings over the pulse coding.

    A source with fewer than ``choice_logit_extensions.MIN_TEST_MICE`` mice shows its mice
    individually, with no mean, error bar or test.

    Args:
        sources (Mapping[str, pd.DataFrame]): ``{source label: binary_pulse_table}``.
        save_path (str | Path | None): Base path without extension; ``None`` shows the figure.

    Raises:
        ValueError: If there are no sources or a source has no mice.
    """
    _check_sources(sources)
    colors = _colors_of(list(sources.values()))
    ticks = [_source_label(name, len(t)) for name, t in sources.items()]
    n_max = max(len(t) for t in sources.values())
    drawing_means = n_max >= ext.MIN_TEST_MICE
    caption = (
        "Pulse: reward lags code the pulses received (the standard model). Binary: reward lags "
        "code 1 if the arm paid, 0 otherwise, with the same structure and tail. Binary and "
        "pulse: both sets of lags.",
        "Gain = held-out (leave-one-problem-out) nats per choice of the pulse model minus that of "
        "the model named; positive is better than the pulse model. Binary Instead of Pulse above "
        "0 means the mouse follows whether it was rewarded, below 0 that it follows the "
        "magnitude.",
        f"Dots: mice; black bars: mean ± se across mice, drawn only for a source with at least "
        f"{ext.MIN_TEST_MICE} mice (fewer than {ext.MIN_TEST_MICE} mice: per-mouse values only).",
        *_binary_caption_lines(sources),
    )
    with new_figure(ncols=2, figsize=(11, 4.8), sharey=True) as (fig, axs):
        rng = jitter_rng()
        for ax, (column, heading) in zip(axs, BINARY_PANELS, strict=True):
            labeled: set = set()
            errors = [
                _draw_column(
                    ax, x, t.set_index("mouse")[column], colors, rng, labeled,
                    show_mean=len(t) >= ext.MIN_TEST_MICE,
                )[1]
                for x, t in enumerate(sources.values())
            ]  # fmt: skip
            _style_source_axis(ax, ticks, "Held-Out Gain over Pulse (Nats per Choice)")
            stats = (
                mice_stats(n_max, *errors)
                if drawing_means
                else f"per mouse, n={count_text(n_max, 'mouse', 'mice')}"
            )
            set_title(ax, Title(heading, stats), fontsize=12)
            ax.legend(fontsize=9)
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
    colors = _colors_of(list(sources.values()))
    labels = [label for _, label in PARTITION_COLUMNS]
    caption = (
        "Held-out (leave-one-problem-out) nats per choice. Unique tower, reward and choice: the "
        "held-out negative log likelihood of the full model without that group minus the full "
        "model's: what only that group explains, given the other two.",
        "Shared reward/choice = [gain(reward + tower) + gain(choice + tower)] - gain(full), with "
        "gain(model) = loss(tower only) - loss(model). Positive: reward and choice history "
        "explain the same choices; negative: they complement each other. Unique reward + unique "
        "choice + shared = the full model's gain over the tower only model.",
        "Dots: mice; black bars: mean ± se across mice.",
        _counts_caption(sources),
    )
    width = max(5.5, 3.6 * len(sources))
    with new_figure(ncols=len(sources), figsize=(width, 4.8), sharey=True, squeeze=False) as (
        fig,
        axs,
    ):
        rng = jitter_rng()
        for ax, (name, table) in zip(axs[0], sources.items(), strict=True):
            labeled: set = set()
            errors = [
                _draw_column(ax, x, table.set_index("mouse")[column], colors, rng, labeled)[1]
                for x, (column, _) in enumerate(PARTITION_COLUMNS)
            ]
            _style_source_axis(ax, labels, "Held-Out Gain (Nats per Choice)")
            set_title(ax, Title(name, mice_stats(len(table), *errors)), fontsize=12)
            ax.legend(fontsize=9)
        finish_figure(fig, Title("Unique and Shared Variance of Regressor Groups", None, caption))
    save_or_show(fig, save_path or None)


# ---------------------------------------------------------------------------
# 4. Calibration
# ---------------------------------------------------------------------------


def _error_text(summary: pd.DataFrame, column: str) -> str:
    """``"mean ± se across mice (n=… mice)"`` of a summary column; ``mean`` alone for one mouse."""
    values = summary[column].to_numpy(dtype=float)[:, None]
    mean, se = float(nan_mean(values)[0]), float(nan_standard_error(values)[0])
    spread = f" ± se {se:.4f}" if np.isfinite(se) else ""
    return f"mean {mean:.4f}{spread} across mice (n={len(summary)} mice)"


def _plot_calibration_panel(
    table: pd.DataFrame, colors: Mapping[str, str], ax: plt.Axes
) -> np.ndarray:
    """Draw one model's calibration curves into *ax*; return the standard error per bin."""
    predicted = table.pivot(index="mouse", columns="bin", values="mean_predicted")
    observed = table.pivot(index="mouse", columns="bin", values="observed")
    for mouse in observed.index:
        ax.plot(predicted.loc[mouse], observed.loc[mouse], "-o", color=colors[mouse], lw=1,
                ms=3, alpha=0.5, label=mouse)  # fmt: skip
    se = nan_standard_error(observed.to_numpy())
    ax.errorbar(nan_mean(predicted.to_numpy(), axis=0), nan_mean(observed.to_numpy(), axis=0),
                yerr=se if np.isfinite(se).any() else None, fmt="-o", color="black", ms=5, lw=2,
                capsize=3, zorder=5)  # fmt: skip
    ax.plot([0, 1], [0, 1], ":", color="0.4", lw=1)  # identity: observed = predicted
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("Mean Predicted P(Choice)", fontsize=12)
    ax.set_ylabel("Observed Fraction Chosen", fontsize=12)
    despine(ax)
    return se


def plot_calibration(
    label: str, tables: Mapping[str, pd.DataFrame], save_path: str | Path | None = None
) -> None:
    """Observed choice frequency against held-out predicted probability, for one source.

    Args:
        label (str): The source's name, e.g. ``"Mice"``.
        tables (Mapping[str, pd.DataFrame]): ``{model: calibration_table}``, one panel each (the
            full model and the choice-history-only model).
        save_path (str | Path | None): Base path without extension; ``None`` shows the figure.

    Raises:
        ValueError: If there are no calibration tables or one has no mice.
    """
    if not tables:
        raise ValueError("no calibration tables to plot")
    _check_sources(tables)
    colors = _colors_of(list(tables.values()))
    summaries = {m: ext.calibration_summary(t) for m, t in tables.items()}
    n_bins = max(t["bin"].nunique() for t in tables.values())
    caption = (
        f"{label}. Every (trial, arm) pair is one prediction: the arm's held-out probability "
        f"(leave-one-problem-out), with outcome 1 if that arm was chosen. Predictions are sorted "
        f"and cut into {n_bins} equal-count (quantile) bins per mouse; x is the bin's mean "
        "predicted probability, y the fraction of its arms that were chosen. Dotted: identity "
        "(perfect calibration). Thin lines: mice; black: mean ± se across mice.",
        *(
            f"{CALIBRATION_LABELS.get(m, m)}: mean |observed - predicted| across bins, "
            f"{_error_text(s, 'mean_abs_error')}; weighted by bin size, "
            f"{_error_text(s, 'weighted_abs_error')}."
            for m, s in summaries.items()
        ),
    )
    with new_figure(ncols=len(tables), figsize=(4.8 * len(tables), 5.0)) as (fig, axs):
        for ax, (model, table) in zip(np.atleast_1d(axs), tables.items(), strict=True):
            se = _plot_calibration_panel(table, colors, ax)
            heading = CALIBRATION_LABELS.get(model, model)
            set_title(ax, Title(heading, mice_stats(table["mouse"].nunique(), se)), fontsize=12)
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
        {k: t.binary_pulse for k, t in tables.items()}, out_dir / "Binary vs Pulse"
    )
    plot_variance_partition(
        {k: t.partition for k, t in tables.items()}, out_dir / "Unique and Shared Variance"
    )
    for label, source_tables in tables.items():
        plot_calibration(label, source_tables.calibration, out_dir / f"Calibration ({label})")
