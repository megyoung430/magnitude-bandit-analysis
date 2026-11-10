"""Figures of the null agents: how well they fit, the stay rule they imply, and kernels by source.

Drawing only: the numbers come from :mod:`src.pipelines.null_agents` and
:mod:`src.behavior_analysis.choice_logit`. Each ``plot_*`` function builds a figure, then saves it
(pdf, png, svg and a caption file) or shows it. Legends are drawn outside the data by the save
step (:mod:`src.behavior_visualization.plot_legends`).
"""

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.behavior_analysis.stats import nan_mean, nan_standard_error
from src.behavior_visualization.plot_choice_logit import KERNEL_PANELS, kernel_matrix
from src.behavior_visualization.plot_helpers import (
    Title,
    despine,
    error_drawn,
    mouse_color_map,
    save_or_show,
    set_title,
    stats_line,
)
from src.behavior_visualization.plot_value_psychometrics import finish_figure, new_figure

MODEL_LABELS = {
    "wsls": "WSLS",
    "wsls_recency": "WSLS (Recency)",
    "wsls_magnitude": "WSLS (Magnitude)",
    "choice_logit": "Choice Logit (Held-Out)",
}
# Okabe-Ito, black first (the mice or the reference), yellow last (poor contrast on white).
SOURCE_PALETTE = (
    "#000000", "#E69F00", "#56B4E9", "#009E73", "#D55E00", "#CC79A7", "#0072B2", "#F0E442",
)  # fmt: skip


def source_colors(names: Sequence[str]) -> dict[str, str]:
    """One palette colour per source, cycling past the palette's end."""
    return {n: SOURCE_PALETTE[i % len(SOURCE_PALETTE)] for i, n in enumerate(names)}


def _mice_stats(n_mice: int, *errors) -> str:
    average = "mean ± se" if error_drawn(*errors) else "mean"
    return stats_line(average, "mice", [(n_mice, "mouse", "mice")])


def _label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


# ---------------------------------------------------------------------------
# Fits
# ---------------------------------------------------------------------------


def _plot_score_panel(scores: pd.DataFrame, ax: plt.Axes) -> None:
    nats = scores.pivot(index="mouse", columns="model", values="nats_per_choice")
    nats = nats[[m for m in MODEL_LABELS if m in nats]]
    colors = mouse_color_map(nats.index)
    x = np.arange(nats.shape[1])
    mean, se = nan_mean(nats.to_numpy(), axis=0), nan_standard_error(nats.to_numpy())
    for mouse, values in nats.iterrows():
        ax.plot(x, values, "-o", color=colors[mouse], lw=1, ms=3, alpha=0.5)
    ax.errorbar(x, mean, yerr=se if error_drawn(se) else None, fmt="o", color="black", ms=7,
                capsize=4, lw=2, zorder=5)  # fmt: skip
    ax.axhline(np.log(3), color="0.5", lw=0.8, ls=":")
    ax.set_xticks(x, [_label(m) for m in nats.columns], rotation=20, ha="right")
    ax.set_ylabel("Negative Log Likelihood (Nats per Choice)", fontsize=12)
    set_title(ax, Title("Fit to Each Mouse's Choices", _mice_stats(len(nats), se)), fontsize=12)
    despine(ax)


def _plot_stay_panel(observed: pd.DataFrame, implied: pd.DataFrame, ax: plt.Axes) -> None:
    table = observed.pivot(index="mouse", columns="magnitude", values="p_stay")
    magnitudes = table.columns.to_numpy(dtype=float)
    colors = mouse_color_map(table.index)
    mean, se = nan_mean(table.to_numpy(), axis=0), nan_standard_error(table.to_numpy())
    for mouse, values in table.iterrows():
        ax.plot(magnitudes, values, "-o", color=colors[mouse], lw=1, ms=3, alpha=0.4)
    if error_drawn(se):
        ax.fill_between(magnitudes, mean - se, mean + se, color="0.5", alpha=0.25, lw=0, zorder=3)
    ax.plot(magnitudes, mean, "-o", color="black", lw=2.2, ms=5, label="Observed", zorder=5)
    models = [m for m in MODEL_LABELS if m in set(implied["model"])]
    palette = source_colors(["Observed", *(_label(m) for m in models)])
    for model in models:
        curve = implied[implied["model"] == model].pivot(
            index="mouse", columns="magnitude", values="p_stay"
        )
        ax.plot(
            curve.columns, nan_mean(curve.to_numpy(), axis=0), "--s", lw=1.8, ms=4,
            color=palette[_label(model)], label=_label(model), zorder=4,
        )  # fmt: skip
    ax.set_xticks(magnitudes)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Pulses Paid by the Previous Choice", fontsize=12)
    ax.set_ylabel("P(Stay)", fontsize=12)
    set_title(ax, Title("Stay After Each Magnitude", _mice_stats(len(table), se)), fontsize=12)
    ax.legend(fontsize=10)
    despine(ax)


def plot_null_agent_fits(
    scores: pd.DataFrame,
    observed_stay: pd.DataFrame,
    implied_stay: pd.DataFrame,
    save_path: str | Path | None = None,
) -> None:
    """Likelihood of each agent per mouse, and observed against implied stay probabilities.

    Args:
        scores (pd.DataFrame): ``mouse, model, nats_per_choice``, one row per mouse and model
            (the output of ``null_agents.scores_table``, optionally with ``choice_logit`` rows
            holding the logit's held-out nats per choice).
        observed_stay (pd.DataFrame): ``null_agents.stay_by_magnitude`` of the same mice.
        implied_stay (pd.DataFrame): ``null_agents.implied_stay_curves`` of the same mice.
        save_path (str | Path | None): Base path without extension; ``None`` shows the figure.

    Raises:
        ValueError: If *scores* is empty.
    """
    if scores.empty:
        raise ValueError("no scores to plot")
    caption = (
        "Left: negative log likelihood per choice of each agent on the choices it was fitted to "
        "(in-sample; two to four parameters against thousands of choices, so not held out), "
        "thin lines: each mouse; the Choice Logit (Held-Out) column, if present, is its "
        "leave-one-problem-out score. The dotted line is chance, log(3) nats per choice.",
        "Right: observed P(stay) after a choice that paid each number of pulses (pairs of "
        "consecutive trials within a session), thin lines: each mouse; dashed: the mean "
        "probability each fitted agent implies. WSLS and WSLS (Recency) cannot depend on the "
        "magnitude beyond rewarded versus not, and share one stay rule, so their dashed lines "
        "coincide.",
    )
    with new_figure(ncols=2, figsize=(11, 4.6)) as (fig, (left, right)):
        _plot_score_panel(scores, left)
        _plot_stay_panel(observed_stay, implied_stay, right)
        finish_figure(fig, Title("Null Agents Fitted to Each Mouse", None, caption))
    save_or_show(fig, save_path or None)


# ---------------------------------------------------------------------------
# Kernels by source
# ---------------------------------------------------------------------------


def _plot_source_kernel(sources: dict, group: str, heading: str, ylabel: str, ax: plt.Axes):
    colors = source_colors(list(sources))
    n_mice, errors = 0, []
    for name, weights in sources.items():
        matrix = kernel_matrix(weights, group)
        lags = matrix.columns.to_numpy(dtype=float)
        mean, se = nan_mean(matrix.to_numpy(), axis=0), nan_standard_error(matrix.to_numpy())
        if error_drawn(se):
            ax.fill_between(lags, mean - se, mean + se, color=colors[name], alpha=0.2, lw=0)
        ax.plot(lags, mean, "-o", color=colors[name], lw=2, ms=4, label=name)
        n_mice, errors = max(n_mice, len(matrix)), [*errors, se]
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set_xticks(lags)
    ax.set_xlabel("Lag (Trials Back)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    set_title(ax, Title(heading, _mice_stats(n_mice, *errors)), fontsize=12)
    ax.legend(fontsize=10)
    despine(ax)


def plot_kernels_by_source(sources: dict, save_path: str | Path | None = None) -> None:
    """Mean reward and choice kernels of several sources on common axes.

    Args:
        sources (dict): ``{label: choice_logit.weights_table of that source's mice}``, e.g. the
            mice, the simulated Q-learners and each null agent.
        save_path (str | Path | None): Base path without extension; ``None`` shows the figure.

    Raises:
        ValueError: If there are no sources or a source has no weights.
    """
    if not sources:
        raise ValueError("no sources to plot")
    empty = [name for name, weights in sources.items() if weights.empty]
    if empty:
        raise ValueError(f"no weights for {empty}")
    counts = ", ".join(f"{name}: n={w['mouse'].nunique()} mice" for name, w in sources.items())
    caption = (
        "Weights of the choice logit fitted to all of each source's trials; each line is the "
        "mean across the source's mice with its se band.",
        "Reward kernel: change in an arm's logit per pulse received there k trials back. Choice "
        "kernel: change in an arm's logit if it was chosen k trials back.",
        f"Mice per source, {counts}. The title states the largest.",
    )
    with new_figure(ncols=2, figsize=(10, 4.4)) as (fig, axs):
        for ax, (group, heading, ylabel) in zip(axs, KERNEL_PANELS, strict=True):
            _plot_source_kernel(sources, group, heading, ylabel, ax)
        finish_figure(fig, Title("Choice Logit Kernels by Source", None, caption))
    save_or_show(fig, save_path or None)
