"""Figures of the choice logit: weights, model comparison, and psychometrics on model values.

Drawing only: the numbers come from :mod:`src.behavior_analysis.choice_logit`,
:mod:`src.behavior_analysis.choice_logit_cv` and
:mod:`src.behavior_analysis.model_value_psychometrics`. The panel functions take ``ax=None`` and
draw into it; the ``plot_*`` figure functions build a figure, then save it (pdf, png, svg and a
caption file) or show it.
"""

from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.behavior_analysis import choice_logit as cl
from src.behavior_analysis.choice_logit_cv import MODELS
from src.behavior_analysis.model_value_psychometrics import REWARDED, UNREWARDED
from src.behavior_analysis.stats import nan_mean, nan_standard_error
from src.behavior_visualization.plot_helpers import (
    Title,
    axes_or_new,
    despine,
    error_drawn,
    mouse_color_map,
    save_or_show,
    set_title,
    stats_line,
)
from src.behavior_visualization.plot_stay_switch import (
    STAY_SWITCH_SPEC,
    SWITCH_TARGET_SPEC,
    plot_outcome_curves,
)
from src.behavior_visualization.plot_value_psychometrics import (
    finish_figure,
    new_figure,
    plot_psychometric_panel,
)

REFERENCE_COLOR = "0.2"
MODEL_LABELS = {
    "reward": "Reward",
    "reward+choice": "+ Choice History",
    "full": "+ Tower Bias (Full)",
}
KERNEL_PANELS = (
    (cl.REWARD, "Reward Kernel", "Weight (Logit per Pulse)"),
    (cl.CHOICE, "Choice Kernel", "Weight (Logit)"),
)
REWARD_ROW_HEADINGS = {
    REWARDED: "Previous Choice Rewarded",
    UNREWARDED: "Previous Choice Unrewarded",
}
BINNED = " (Quantile Bin)"


def _mice_stats(n_mice: int, *errors) -> str:
    """``"mean ± se across mice, n=… mice"``; ``mean`` alone if no error is drawn."""
    average = "mean ± se" if error_drawn(*errors) else "mean"
    return stats_line(average, "mice", [(n_mice, "mouse", "mice")])


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------


def _kernel_matrix(weights: pd.DataFrame, group: str) -> pd.DataFrame:
    """Lags (columns) by mice (rows) of one group's weights."""
    rows = weights.loc[weights["group"] == group]
    return rows.pivot(index="mouse", columns="lag", values="weight").sort_index(axis=1)


def plot_kernel_panel(
    weights: pd.DataFrame,
    group: str,
    heading: str,
    ylabel: str,
    *,
    reference: pd.DataFrame | None = None,
    reference_label: str = "Reference",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """One group's weights against lag: mice thin, mean thick with se band, optional reference.

    Args:
        weights (pd.DataFrame): ``choice_logit.weights_table`` of all mice.
        group (str): ``"reward"`` or ``"choice"``.
        heading (str): Panel title.
        ylabel (str): Y axis label.
        reference (pd.DataFrame | None): Weights in the same layout drawn dashed (per mouse thin,
            mean thick). Keyword-only.
        reference_label (str): Legend label of the reference mean. Keyword-only.
        ax (plt.Axes | None): Axes to draw into; a new figure if ``None``. Keyword-only.

    Returns:
        plt.Axes: The axes drawn into.
    """
    fitted = _kernel_matrix(weights, group)
    colors = mouse_color_map(fitted.index)
    lags = fitted.columns.to_numpy(dtype=float)
    with axes_or_new(ax, figsize=(4.6, 4.2)) as ax:
        ax.axhline(0, color="0.7", lw=0.8)
        for mouse, values in fitted.iterrows():
            ax.plot(lags, values, "-o", color=colors[mouse], lw=1, ms=2.5, alpha=0.4)
        mean, se = nan_mean(fitted.to_numpy(), axis=0), nan_standard_error(fitted.to_numpy())
        if error_drawn(se):
            ax.fill_between(lags, mean - se, mean + se, color="0.5", alpha=0.25, lw=0, zorder=3)
        ax.plot(lags, mean, "-o", color="black", lw=2.2, ms=5, label="Fitted Weights", zorder=5)
        if reference is not None:
            ref = _kernel_matrix(reference, group).reindex(
                index=fitted.index, columns=fitted.columns
            )
            for mouse, values in ref.iterrows():
                ax.plot(lags, values, "--", color=colors[mouse], lw=1, alpha=0.6)
            ax.plot(
                lags, nan_mean(ref.to_numpy(), axis=0), "--", color=REFERENCE_COLOR, lw=2.2,
                label=reference_label, zorder=4,
            )  # fmt: skip
        ax.set_xticks(lags)
        ax.set_xlabel("Lag (Trials Back)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        set_title(ax, Title(heading, _mice_stats(len(fitted), se)), fontsize=12)
        ax.legend(fontsize=10)
        despine(ax)
    return ax


def plot_tower_recovery(
    weights: pd.DataFrame, reference: pd.DataFrame, ax: plt.Axes | None = None
) -> plt.Axes:
    """Fitted against reference tower bias, one point per tower and mouse, on the identity line."""
    key = ["mouse", "feature"]
    towers = weights.loc[weights["group"] == cl.TOWER, [*key, "weight"]].merge(
        reference.loc[reference["group"] == cl.TOWER, [*key, "weight"]],
        on=key,
        suffixes=("_fitted", "_reference"),
    )
    colors = mouse_color_map(sorted(towers["mouse"].unique()))
    with axes_or_new(ax, figsize=(4.6, 4.2)) as ax:
        ax.scatter(
            towers["weight_reference"], towers["weight_fitted"],
            c=towers["mouse"].map(colors), s=22, alpha=0.8, zorder=3,
        )  # fmt: skip
        both = np.concatenate([towers["weight_reference"], towers["weight_fitted"]])
        limits = (both.min() - 0.1, both.max() + 0.1)
        ax.plot(limits, limits, "--", color=REFERENCE_COLOR, lw=1)
        ax.set(xlim=limits, ylim=limits)
        ax.set_aspect("equal")
        ax.set_xlabel("Reference Tower Bias (Logit)", fontsize=12)
        ax.set_ylabel("Fitted Tower Bias (Logit)", fontsize=12)
        set_title(ax, Title("Tower Bias"), fontsize=12)
        despine(ax)
    return ax


def plot_kernels(
    weights: pd.DataFrame,
    save_path: str | Path | None = None,
    *,
    reference: pd.DataFrame | None = None,
    reference_label: str = "Reference",
) -> None:
    """Reward and choice kernels of all mice (and tower bias recovery given a reference).

    Args:
        weights (pd.DataFrame): ``choice_logit.weights_table`` of all mice.
        save_path (str | Path | None): Base path without extension; ``None`` shows the figure.
        reference (pd.DataFrame | None): Reference weights in the same layout (e.g. of the
            generating rule); drawn dashed, and adds a tower-bias scatter. Keyword-only.
        reference_label (str): What the reference is, for the legend and caption. Keyword-only.

    Raises:
        ValueError: If *weights* is empty.
    """
    if weights.empty:
        raise ValueError("no weights to plot")
    n_panels = len(KERNEL_PANELS) + (reference is not None)
    caption = [
        "Weights of the choice logit fitted to all of each mouse's trials; thin lines: each "
        "mouse; thick line: mean across mice with the se band.",
        "Reward kernel: change in an arm's logit per pulse received there k trials back. Choice "
        "kernel: change in an arm's logit if it was chosen k trials back.",
    ]
    if reference is not None:
        caption.append(
            f"Dashed: {reference_label}, the least-squares weights of the same features that best "
            "reproduce each agent's generating utility. Right: each point is one tower of one "
            "mouse."
        )
    with new_figure(ncols=n_panels, figsize=(4.8 * n_panels, 4.4)) as (fig, axs):
        for ax, (group, heading, ylabel) in zip(axs, KERNEL_PANELS, strict=False):
            plot_kernel_panel(
                weights, group, heading, ylabel, reference=reference,
                reference_label=reference_label, ax=ax,
            )  # fmt: skip
        if reference is not None:
            plot_tower_recovery(weights, reference, ax=axs[-1])
        finish_figure(fig, Title("Choice Logit Weights", None, tuple(caption)))
    save_or_show(fig, save_path or None)


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------


def plot_model_comparison(
    comparison: pd.DataFrame, save_path: str | Path | None = None, ax: plt.Axes | None = None
) -> None:
    """Held-out gain over chance of the nested models, one thin line per mouse.

    Args:
        comparison (pd.DataFrame): ``choice_logit_cv.model_comparison`` output.
        save_path (str | Path | None): Base path without extension; ``None`` shows the figure.
        ax (plt.Axes | None): Axes to draw into instead of a new figure (then nothing is saved).

    Raises:
        ValueError: If *comparison* is empty.
    """
    if comparison.empty:
        raise ValueError("no model comparison to plot")
    gain = comparison.pivot(index="mouse", columns="model", values="gain_over_chance")
    gain = gain[[m for m in MODELS if m in gain]]
    nats = comparison.pivot(index="mouse", columns="model", values="nats_per_choice")
    colors = mouse_color_map(gain.index)
    x = np.arange(gain.shape[1])
    mean, se = nan_mean(gain.to_numpy(), axis=0), nan_standard_error(gain.to_numpy())

    def draw(ax: plt.Axes) -> None:
        for mouse, values in gain.iterrows():
            ax.plot(x, values, "-o", color=colors[mouse], lw=1, ms=3, alpha=0.5)
        ax.errorbar(x, mean, yerr=se if error_drawn(se) else None, fmt="o", color="black",
                    ms=7, capsize=4, lw=2, zorder=5)  # fmt: skip
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_xticks(x, [MODEL_LABELS[m] for m in gain.columns])
        ax.set_xlim(-0.4, len(x) - 0.6)
        ax.set_ylabel("Gain over Chance (Nats per Choice)", fontsize=12)
        set_title(ax, Title("Held-Out Model Comparison", _mice_stats(len(gain), se)), fontsize=12)
        despine(ax)

    if ax is not None:
        draw(ax)
        return
    per_model = ", ".join(
        f"{MODEL_LABELS[m].lower()} {v:.4f}"
        for m, v in nats.mean(axis=0)[list(gain.columns)].items()
    )
    caption = (
        "Leave-one-problem-out: each model is fitted on all but one of a mouse's problems and "
        "scored on the problem it did not see; thin lines: each mouse.",
        f"Gain over chance = log(3) - held-out negative log likelihood per choice; chance is "
        f"log(3) = {np.log(3):.4f} nats per choice.",
        f"Mean held-out nats per choice across mice: {per_model}.",
    )
    with new_figure(figsize=(5.2, 4.4)) as (fig, new_ax):
        draw(new_ax)
        finish_figure(fig, Title("Choice Logit Model Comparison", None, caption))
    save_or_show(fig, save_path or None)


# ---------------------------------------------------------------------------
# Psychometrics on a continuous value
# ---------------------------------------------------------------------------


class PsychometricRow(NamedTuple):
    """One row of psychometric panels: the curves of one value basis.

    Attributes:
        label (str): Name of the value basis (not drawn; the x labels name it).
        curves (dict[str, dict[str, Curve]]): ``curves[choice][axis]`` as returned by
            ``binned_psychometric_curves`` or ``psychometric_curves``.
        xlabels (tuple[str, str, str]): X labels of the best, second and difference panels.
    """

    label: str
    curves: dict
    xlabels: tuple


def value_labels(basis: str, binned: bool = True) -> tuple[str, str, str]:
    """X labels of the three psychometric panels for the value *basis*, e.g. ``"Model Value"``."""
    suffix = BINNED if binned else ""
    return (
        f"{basis} of Best{suffix}",
        f"{basis} of Second{suffix}",
        f"{basis} of Best - Second{suffix}",
    )


def plot_psychometric_rows(
    rows: list[PsychometricRow],
    heading: str,
    caption: tuple[str, ...],
    save_path: str | Path | None = None,
) -> None:
    """P(best) and P(second) against three x axes, one row per value basis; save or show.

    Args:
        rows (list[PsychometricRow]): The value bases, top row first.
        heading (str): Figure title.
        caption (tuple[str, ...]): Caption lines (the thin-line note is added).
        save_path (str | Path | None): Base path without extension; ``None`` shows the figure.

    Raises:
        ValueError: If there are no rows.
    """
    if not rows:
        raise ValueError("no rows to plot")
    axes_names = ("best_val", "second_val", "diff")
    with new_figure(nrows=len(rows), ncols=3, figsize=(13, 4.4 * len(rows)), squeeze=False) as (
        fig,
        axs,
    ):
        for axes_row, row in zip(axs, rows, strict=True):
            for ax, axis, xlabel in zip(axes_row, axes_names, row.xlabels, strict=True):
                plot_psychometric_panel(
                    row.curves["best"][axis], row.curves["second"][axis], xlabel, ax=ax
                )
        lines = (*caption, "Thin lines: each mouse.")
        finish_figure(fig, Title(heading, None, lines))
    save_or_show(fig, save_path or None)


def _stay_switch_panels(value_label: str) -> tuple[tuple[str, str], ...]:
    return (
        ("diff", f"{value_label} of Chosen - Best Alternative{BINNED}"),
        ("chosen", f"{value_label} of Chosen{BINNED}"),
        ("alt", f"{value_label} of Best Alternative{BINNED}"),
    )


def _stay_switch_caption(value_label: str, pool_trials: bool) -> tuple[str, ...]:
    weighting = "pairs pooled within mouse" if pool_trials else "cell probabilities averaged"
    return (
        "Pairs of consecutive trials (t-1, t) of one session; the chosen option is the arm "
        "chosen on t-1; pairs across a session start or a missed trial are not used.",
        f"Values are the {value_label.lower()} of each arm on trial t: V(chosen) is the chosen "
        "arm's value, V(alt) the higher value of the two other arms.",
        "Stay: trial t picks the chosen arm again. Switch = 1 - stay.",
        "Rows: whether the choice on t-1 was rewarded (a non-zero magnitude) or not; a "
        "continuous value has no 0-value row.",
        "Values are cut into quantile bins per mouse, so a bin number is a rank among that "
        "mouse's trials, not a value; the difference axis bins the raw difference.",
        f"{weighting.capitalize()}; thin lines: each mouse.",
    )


def plot_stay_switch_by_reward(
    curves_by_row: dict,
    value_label: str,
    pool_trials: bool,
    save_path: str | Path | None = None,
) -> None:
    """P(stay) and P(switch) against binned values, rows by whether t-1 was rewarded.

    Args:
        curves_by_row (dict): From ``stay_switch_curves_by_reward``.
        value_label (str): Value basis for titles and labels, e.g. ``"Model Value"``.
        pool_trials (bool): The weighting the curves were computed with, for the caption.
        save_path (str | Path | None): Base path without extension; ``None`` shows the figure.
    """
    title = Title(
        f"Stay-Switch Curves ({value_label})", None, _stay_switch_caption(value_label, pool_trials)
    )
    plot_outcome_curves(
        curves_by_row, STAY_SWITCH_SPEC, title, save_path,
        row_headings=REWARD_ROW_HEADINGS, panels=_stay_switch_panels(value_label),
    )  # fmt: skip


def plot_switch_targets_by_reward(
    curves_by_row: dict,
    value_label: str,
    pool_trials: bool,
    save_path: str | Path | None = None,
) -> None:
    """P(best alternative | switch) and P(other arm | switch), rows as above; save or show.

    Args:
        curves_by_row (dict): From ``switch_target_curves_by_reward``.
        value_label (str): Value basis for titles and labels.
        pool_trials (bool): The weighting the curves were computed with, for the caption.
        save_path (str | Path | None): Base path without extension; ``None`` shows the figure.
    """
    caption = (
        *_stay_switch_caption(value_label, pool_trials),
        "Switch targets are fractions of the switches after the chosen option, so they add up "
        "to 1; stays are left out.",
    )
    title = Title(f"Switch Target Curves ({value_label})", None, caption)
    plot_outcome_curves(
        curves_by_row, SWITCH_TARGET_SPEC, title, save_path,
        row_headings=REWARD_ROW_HEADINGS, panels=_stay_switch_panels(value_label),
    )  # fmt: skip
