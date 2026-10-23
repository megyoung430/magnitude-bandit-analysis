"""Figures comparing the mice with the magnitude and binary agents.

Both panels take an existing axis (and make their own figure only when none is given), so they
can be combined; :func:`plot_agent_baselines_summary` puts them side by side.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.behavior_analysis.choice_regression import FEATURES
from src.behavior_visualization.plot_helpers import Title, save_or_show, set_title, stats_line

SOURCE_LABELS = {"mice": "Mice", "magnitude": "Magnitude agent", "binary": "Binary agent"}
SOURCE_COLORS = {"mice": "#333333", "magnitude": "#4C8BC9", "binary": "#D9822B"}
FEATURE_LABELS = {
    "last_value": "Last value\n(per unit)",
    "last_zero": "Last visit\npaid 0",
    "unseen": "Not yet\nvisited",
    "stay": "Stay",
    "plus_port": "Plus port",
}


def plot_regression_weights(weights: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    """Regression weight per feature for each source: one dot per animal, mean ± se across them.

    Args:
        weights (pd.DataFrame): Columns ``source`` (a key of ``SOURCE_LABELS``), ``mouse``,
            ``feature``, ``weight``: the per-animal weights of every source.
        ax (plt.Axes | None): Axis to draw into. Default ``None``: a new figure.

    Returns:
        plt.Axes: The axis drawn into.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    sources = [s for s in SOURCE_LABELS if s in set(weights["source"])]
    width = 0.8 / max(len(sources), 1)
    for i, source in enumerate(sources):
        rows = weights[weights["source"] == source]
        for j, feature in enumerate(FEATURES):
            values = rows.loc[rows["feature"] == feature, "weight"].to_numpy(dtype=float)
            if len(values) == 0:
                continue
            x = j + (i - (len(sources) - 1) / 2) * width
            jitter = np.linspace(-width / 4, width / 4, len(values)) if len(values) > 1 else [0]
            ax.scatter(x + np.asarray(jitter), values, s=10, alpha=0.35,
                       color=SOURCE_COLORS[source], linewidths=0)  # fmt: skip
            se = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else np.nan
            ax.errorbar(x, values.mean(), yerr=se, fmt="o", color=SOURCE_COLORS[source],
                        capsize=3, label=SOURCE_LABELS[source] if j == 0 else None)  # fmt: skip
    ax.axhline(0, color="0.6", lw=0.8, ls="--")
    ax.set_xticks(range(len(FEATURES)), [FEATURE_LABELS[f] for f in FEATURES])
    ax.set_ylabel("Weight (logit)")
    n_animals = int(weights.groupby("source")["mouse"].nunique().max()) if len(weights) else 0
    # a single animal has no SE (see the errorbars above)
    average = "mean ± se" if n_animals >= 2 else "mean"
    stats = stats_line(average, "animals of each source", [(n_animals, "animal", "animals")])
    set_title(ax, Title("Choice regression: what drives port choice", stats))
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def plot_likelihood_comparison(fits: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    """Per mouse, how much worse the binary agent fits than the magnitude agent, per trial.

    Args:
        fits (pd.DataFrame): Table of ``drifting_simulation.fits_table``.
        ax (plt.Axes | None): Axis to draw into. Default ``None``: a new figure.

    Returns:
        plt.Axes: The axis drawn into. Bars above 0 favour the magnitude agent.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    binary = fits[fits["kind"] == "binary"].sort_values("mouse")
    per_trial = (binary["delta_nll_vs_magnitude"] / binary["n_trials"]).to_numpy()
    colors = [SOURCE_COLORS["magnitude"] if v > 0 else SOURCE_COLORS["binary"] for v in per_trial]
    ax.bar(range(len(binary)), per_trial * 1000, color=colors)
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_xticks(range(len(binary)), binary["mouse"], rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("NLL binary − magnitude\n(×10⁻³ per trial)")
    n_magnitude = int((per_trial > 0).sum())
    ax.set_title(f"Model comparison: magnitude agent better in {n_magnitude}/{len(binary)} mice")
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def plot_agent_baselines_summary(
    weights: pd.DataFrame, fits: pd.DataFrame, save_path: Path | None = None
) -> plt.Figure:
    """Both panels side by side; saved (and closed) when *save_path* (without suffix) is given.

    Args:
        weights (pd.DataFrame): As for :func:`plot_regression_weights`.
        fits (pd.DataFrame): As for :func:`plot_likelihood_comparison`.
        save_path (Path | None): Path without suffix. Default ``None``: not saved.

    Returns:
        plt.Figure: The figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), gridspec_kw={"width_ratios": [3, 2]})
    plot_regression_weights(weights, axes[0])
    plot_likelihood_comparison(fits, axes[1])
    fig.tight_layout()
    if save_path is not None:
        save_or_show(fig, save_path, dpi={"png": 200, "pdf": 200})
    return fig
