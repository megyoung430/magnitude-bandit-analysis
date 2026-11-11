"""Figures of the rank-signature tests: the mice next to replays of their fitted agents.

Drawing only: the numbers come from :mod:`src.pipelines.rank_signatures` (a :class:`FigureData`
bundles the analysis of the mice, the replays and the calibration). One dot is one mouse in its own
color; for an agent it is the mean of that mouse's replays, with the 2.5th to 97.5th percentile of
the replays as a thin bar (cut at the axis, which is fitted to the dots). A source with fewer
than ``MIN_TEST_MICE`` mice shows dots only (no mean,
error bar or test). Curves draw the mice's mean ± se (the mice alone below that number) and the
agents' mean over mice. Each figure is saved as pdf, png and svg with a ``.caption.txt``; the long
text (definitions, the calibrated z and p of every readout, the mice that pass) is in the caption.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from src.behavior_analysis import choice_logit_extensions as ext
from src.behavior_analysis import strategy_models as sm
from src.behavior_analysis.stats import nan_standard_error
from src.behavior_visualization.plot_helpers import (
    Title,
    despine,
    jitter_rng,
    jittered,
    save_or_show,
    set_title,
)
from src.behavior_visualization.plot_regression_extensions import colors_of, source_stats_line
from src.behavior_visualization.plot_value_psychometrics import finish_figure, new_figure
from src.pipelines import rank_signatures as pipe

STAY_WEIGHTS = "Stay Comparison Weights"
CROSSING_STEP = "Rank Crossing Step"
CROSSING_ALIGNED = "Rank Crossing Aligned"
THIRD_OPTION = "Third Option"
RANK_VS_VALUE = "Rank vs Value"
TRANSFER = "Transfer Across Problems"
BLOCK_CHANGE = "Block Change Inference"
FIGURE_NAMES = (STAY_WEIGHTS, CROSSING_STEP, CROSSING_ALIGNED, THIRD_OPTION, RANK_VS_VALUE,
                TRANSFER, BLOCK_CHANGE)  # fmt: skip

SOURCE_LABELS = {
    sm.WSLS_MAGNITUDE: "WSLS (Magnitude)",
    sm.WSLS_HABIT: "WSLS (Habit)",
    sm.KERNEL_Q: "Kernel Q-Learner",
}
# Replays are lines in neutral grays; the key comparator, the kernel Q-learner, is in color.
AGENT_STYLE = {
    sm.WSLS_MAGNITUDE: dict(color="0.45", ls="--"),
    sm.WSLS_HABIT: dict(color="0.65", ls=":"),
    sm.KERNEL_Q: dict(color="#D55E00", ls="-"),
}
READOUT_LABELS = {
    "w_best": "Weight of Best Alternative",
    "w_worst": "Weight of Worst Alternative",
    "w_diff": "Worst Minus Best Weight",
    "q_w_best": "Best Alternative, Q Values",
    "q_w_worst": "Worst Alternative, Q Values",
    "q_w_diff": "Worst Minus Best, Q Values",
    "step_weight": "Step Weight at D < 0",
    "gain_step_vs_smooth": "Step Gain Over Smooth (Nats)",
    "gain_indicator": "Indicator Gain (Nats)",
    "cross_leave_1": "P(Leave) After Crossing",
    "cross_adopt_1": "P(New Best) After Crossing",
    "cross_t50": "Trials to 50% Adoption",
    "worst_slope": "Worst-Arm Slope Over Gap",
    "top2_w_worst": "Worst-Arm Weight, Top Two",
    "rank_minus_value": "Rank Over Value, Stay (Nats)",
    "sw_rank_minus_value": "Rank Over Value, Switch (Nats)",
    "scale_slope_first": "Scale Slope, Rank First",
    "scale_slope_second": "Scale Slope, Rank Second",
    "slope_p100": "Slope of P(Best), First 100",
    "slope_threshold": "Slope of Trials to Threshold",
    "half_p100": "Later Minus Earlier Half, P(Best)",
    "block_inherit": "P(Old Zero Arm), All",
    "block_inherit_below": "P(Old Zero Arm), Value Says Third",
}
MEAN_BAR = dict(color="black", lw=2, zorder=5)


class FigureData(NamedTuple):
    """Everything the figures draw.

    Attributes:
        source (pipe.SourceAnalysis): The mice's readouts and curves.
        replays (pipe.Replays): The fitted agents' replays.
        calibration (pipe.Calibration | None): Each mouse against its replays, or ``None``.
    """

    source: pipe.SourceAnalysis
    replays: pipe.Replays
    calibration: pipe.Calibration | None = None


def _mice(data: FigureData) -> list[str]:
    return sorted(data.source.readouts["mouse"])


def _colors(data: FigureData) -> dict[str, str]:
    return colors_of([data.source.readouts[["mouse"]]])


def _models(data: FigureData) -> list[str]:
    return list(data.replays.readouts)


def _tick_labels(data: FigureData) -> list[str]:
    n = len(_mice(data))
    return [f"Mice\nn={n}", *[f"{SOURCE_LABELS[m]}\nn={n}" for m in _models(data)]]


def _agent_summary(data: FigureData, model: str, readout: str) -> pd.DataFrame:
    """Per mouse: mean, 2.5th and 97.5th percentile of the replays' readout."""
    table = data.replays.readouts[model]
    grouped = table.groupby("mouse")[readout]
    return pd.DataFrame(
        dict(mean=grouped.mean(), low=grouped.quantile(0.025), high=grouped.quantile(0.975))
    )


def _mice_legend(ax, colors: dict) -> None:
    handles = [
        Line2D([], [], marker="o", ls="", color=c, label=m) for m, c in sorted(colors.items())
    ]
    ax.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False, fontsize=9
    )


def _mean_bar(ax, x: float, values: np.ndarray) -> float:
    """Black mean ± se of *values* at *x* (nothing below ``MIN_TEST_MICE`` values); the se."""
    values = values[~np.isnan(values)]
    if len(values) < ext.MIN_TEST_MICE:
        return np.nan
    mean, se = values.mean(), float(nan_standard_error(values[:, None])[0])
    ax.errorbar([x], [mean], yerr=[se], fmt="_", markersize=14, capsize=4, **MEAN_BAR)
    return se


def _limit_to_dots(ax, ys: np.ndarray) -> None:
    """Fit the y-axis to the dots: an agent's replay bar can be far wider than the mice differ
    from one another, and would hide them; a bar longer than the axis is cut at its edge."""
    ys = ys[np.isfinite(ys)]
    if len(ys):
        pad = 0.12 * (ys.max() - ys.min() or 1.0)
        ax.set_ylim(ys.min() - pad, ys.max() + pad)


def scalar_panel(ax, readout: str, data: FigureData, colors: dict, rng) -> list[float]:
    """Dots of one readout: the mice, then each agent's replays; returns the se drawn per column."""
    errors, dots = [], []
    observed = data.source.readouts.set_index("mouse")[readout]
    columns = [("Mice", observed, None)] + [
        (m, _agent_summary(data, m, readout), m) for m in _models(data)
    ]
    for x, (_, values, model) in enumerate(columns):
        if model is None:
            ys = values.to_numpy(float)
            names = list(values.index)
        else:
            ys, names = values["mean"].to_numpy(float), list(values.index)
            for name, low, high in zip(names, values["low"], values["high"], strict=True):
                ax.plot([x, x], [low, high], color=colors[name], lw=1, alpha=0.5, zorder=2)
        xs = jittered(np.full(len(ys), float(x)), 0.12, rng)
        ax.scatter(xs, ys, c=[colors[n] for n in names], s=34, zorder=3, edgecolor="none")
        errors.append(_mean_bar(ax, x, ys))
        dots.append(ys)
    _limit_to_dots(ax, np.concatenate(dots))
    ax.axhline(0, color="0.8", lw=0.8, zorder=1)
    ax.set_xticks(range(len(columns)), _tick_labels(data), rotation=20, ha="right", fontsize=9)
    despine(ax)
    return errors


def _scalar_figure(name: str, readouts: Sequence[str], data: FigureData, caption, save_path):
    colors, rng = _colors(data), jitter_rng()
    n = len(_mice(data))
    with new_figure(nrows=1, ncols=len(readouts), figsize=(4.1 * len(readouts) + 1.5, 4.6),
                    squeeze=False) as (fig, axs):  # fmt: skip
        for ax, readout in zip(axs[0], readouts, strict=True):
            errors = scalar_panel(ax, readout, data, colors, rng)
            stats = source_stats_line([n], [errors[0]])
            set_title(ax, Title(READOUT_LABELS[readout], stats), fontsize=11)
        axs[0, 0].set_ylabel("Readout")
        _mice_legend(axs[0, -1], colors)
        finish_figure(fig, Title(name, None, caption))
    save_or_show(fig, save_path or None)


# ---------------------------------------------------------------------------
# Captions
# ---------------------------------------------------------------------------


def _calibration_lines(data: FigureData, readouts: Sequence[str]) -> list[str]:
    """Calibrated z across mice, and the passing mice, per readout and agent."""
    if data.calibration is None:
        return []
    lines = [
        f"Each mouse's readout is compared with {data.replays.replicates} replays of each agent "
        "fitted to that mouse, played on that mouse's own problems and analyzed identically "
        "(z = (observed - replay mean) / replay sd; p two-sided empirical). A mouse passes a "
        "readout when p < 0.05 in the direction in which a rank algorithm differs from a softmax "
        "learner; the kernel Q-learner is the key comparator."
    ]
    summary = data.calibration.summary
    for readout in readouts:
        for row in summary[summary["readout"] == readout].itertuples():
            label = f"{READOUT_LABELS[readout]} vs {SOURCE_LABELS[row.agent]}"
            stats = (
                f"z mean {row.mean:.2f} ± se {row.se:.2f}, Wilcoxon p={row.p_wilcoxon:.3g}"
                if np.isfinite(row.mean)
                else "z not summarized (fewer than 6 mice)"
            )
            passing = row.passing or "none"
            lines.append(f"{label}: {stats}; passing ({row.n_pass}): {passing}.")
    return lines


def _skipped_line(data: FigureData) -> list[str]:
    skipped = data.source.skipped
    return [f"Skipped, {m}: {why}." for m, why in skipped.items()] if skipped else []


def _failed_line(data: FigureData) -> list[str]:
    failed = data.replays.failed[data.replays.failed["n_failed"] > 0]
    if failed.empty:
        return []
    parts = ", ".join(
        f"{r.mouse} {SOURCE_LABELS[r.agent]} ({r.n_failed})" for r in failed.itertuples()
    )
    return [f"Replays left out for too little data: {parts}."]


def _common_caption(data: FigureData, readouts: Sequence[str], *definitions: str) -> tuple:
    return (
        *definitions,
        "Dots: mice, colored per mouse; for an agent, the mean of that mouse's replays with the "
        "2.5th to 97.5th percentile of the replays as a thin bar (cut at the axis, which "
        "is fitted to the dots). Black bars: mean ± se across "
        f"mice (drawn from {ext.MIN_TEST_MICE} mice).",
        *_skipped_line(data),
        *_failed_line(data),
        *_calibration_lines(data, readouts),
    )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

STAY_READOUTS = ("w_best", "w_worst", "w_diff")


def plot_stay_weights(data: FigureData, save_path=None) -> None:
    """Weights of the best and worst alternative on P(stay), and their difference (test 1)."""
    readouts = STAY_READOUTS
    caption = _common_caption(
        data, readouts,
        "P(stay) on the own payout, the believed value (last payout) of the best and of the worst "
        "alternative and history (run length, choice trace, earlier choices, plus-port bias), "
        "ridge logistic over decisions with all three arms visited. A rank algorithm compares "
        "the own arm with the best alternative only: best weight negative, worst weight about 0 "
        "(worst minus best positive). A softmax over values lowers staying with both.",
    )  # fmt: skip
    _scalar_figure(STAY_WEIGHTS, readouts, data, caption, save_path)


def plot_rank_vs_value(data: FigureData, save_path=None) -> None:
    """Held-out gain of ranks over values (stay and switch models), and scale slope (test 4)."""
    readouts = ("rank_minus_value", "sw_rank_minus_value", "scale_slope_first")
    caption = _common_caption(
        data, readouts,
        "Rank over value: held-out nats per decision (leave-one-problem-out) of the value model "
        "minus the rank-indicator model, positive when the ranks alone predict better; stay: "
        "P(stay) "
        "on the own payout and both alternatives' values against the own arm's first/second-rank "
        "indicators; switch: where a switch goes, believed values of the alternatives against "
        "their "
        "rank indicators, after non-zero payouts. Scale slope: change of the stay logit per pulse "
        "of the payout within the best-ranked own arm; 0 if only the rank matters.",
    )  # fmt: skip
    _scalar_figure(RANK_VS_VALUE, readouts, data, caption, save_path)


def plot_block_change(data: FigureData, save_path=None) -> None:
    """Share of first leaves after a block change that go to the old zero arm (test 5b)."""
    readouts = ("block_inherit", "block_inherit_below")
    caption = _common_caption(
        data, readouts,
        "A block change zeroes the arm chosen most in the block and gives its value to the arm "
        "that "
        "was at zero (the old zero arm). Events: the mouse is on the zeroed arm at the change and "
        "later leaves it; the share going to the old zero arm rather than the third arm. 'Value "
        "says third': events in which the last payouts make the old zero arm worth less than the "
        "third arm, so a learner that follows last payouts goes to the third arm. Chance is 0.5.",
    )  # fmt: skip
    _scalar_figure(BLOCK_CHANGE, readouts, data, caption, save_path)


def _line_stats(n_mice: int, errors_drawn: bool) -> str:
    average = "mean ± se across mice" if errors_drawn else "per mouse"
    return f"{average}, n={n_mice} mice"


def curve_panel(ax, curve: str, series: str, data: FigureData, colors: dict) -> bool:
    """One curve: each mouse's line, the mice's mean ± se and the agents' mean; True if se drawn."""
    mine = data.source.curves.query("curve == @curve and series == @series")
    if mine.empty:
        raise ValueError(f"no curve {curve!r}/{series!r} to draw")
    wide = mine.pivot_table(index="x", columns="mouse", values="y")
    for mouse in wide.columns:
        ax.plot(wide.index, wide[mouse], color=colors[mouse], lw=1, alpha=0.45, marker="o", ms=3)
    drawn = wide.shape[1] >= ext.MIN_TEST_MICE
    if drawn:
        se = wide.apply(lambda row: float(nan_standard_error(row.dropna().to_numpy()[:, None])[0]),
                        axis=1)  # fmt: skip
        ax.fill_between(wide.index, wide.mean(axis=1) - se, wide.mean(axis=1) + se,
                        color="black", alpha=0.18, lw=0, zorder=4)  # fmt: skip
        ax.plot(wide.index, wide.mean(axis=1), color="black", lw=2.2, zorder=5)
    for model in _models(data):
        agent = data.replays.curves[model].query("curve == @curve and series == @series")
        if len(agent):
            line = agent.pivot_table(index="x", columns="mouse", values="y").mean(axis=1)
            ax.plot(line.index, line, lw=2, zorder=6, **AGENT_STYLE[model])
    despine(ax)
    return drawn


def _source_legend(ax, data: FigureData) -> None:
    handles = [Line2D([], [], color="black", lw=2.2, label="Mice")] + [
        Line2D([], [], lw=2, label=SOURCE_LABELS[m], **AGENT_STYLE[m]) for m in _models(data)
    ]
    ax.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False, fontsize=9
    )


def _curve_figure(
    name, panels, data: FigureData, caption, save_path, zero_line=False, sharey=False
):
    """*panels*: ``(curve, series, panel title, xlabel, ylabel)``, drawn in one row."""
    colors, n = _colors(data), len(_mice(data))
    with new_figure(nrows=1, ncols=len(panels), figsize=(4.4 * len(panels) + 1.5, 4.4),
                    squeeze=False, sharey=sharey) as (fig, axs):  # fmt: skip
        for ax, (curve, series, title, xlabel, ylabel) in zip(axs[0], panels, strict=True):
            drawn = curve_panel(ax, curve, series, data, colors)
            ax.set(xlabel=xlabel, ylabel=ylabel)
            if zero_line:
                ax.axvline(0, color="0.8", lw=0.8, zorder=0)
            set_title(ax, Title(title, _line_stats(n, drawn)), fontsize=11)
        _source_legend(axs[0, -1], data)
        finish_figure(fig, Title(name, None, caption))
    save_or_show(fig, save_path or None)


def plot_crossing_step(data: FigureData, save_path=None) -> None:
    """P(leave) by the believed difference, and the step readouts (test 2a)."""
    readouts = ("step_weight", "gain_step_vs_smooth")
    caption = _common_caption(
        data, readouts,
        "D = the value of the arm the mouse is on (its last payout) minus the best alternative's "
        "believed value; D < 0 is a believed rank crossing. Left: P(leave) by D, clipped to +-3. "
        "Step weight: weight of [D < 0] in a logistic on D, the indicator and the run length. Step "
        "gain over smooth: held-out nats per decision of a logistic on D minus a step at D = 0 "
        "with a leaving rate on each side, positive when the step predicts better.",
    )  # fmt: skip
    colors, rng, n = _colors(data), jitter_rng(), len(_mice(data))
    with new_figure(nrows=1, ncols=3, figsize=(15.5, 4.6)) as (fig, axs):
        drawn = curve_panel(axs[0], "crossing_step", "", data, colors)
        axs[0].set(xlabel="D (Own Value Minus Best Alternative)", ylabel="P(Leave)", ylim=(0, 1))
        set_title(
            axs[0], Title("P(Leave) by Believed Difference", _line_stats(n, drawn)), fontsize=11
        )
        for ax, readout in zip(axs[1:], readouts, strict=True):
            errors = scalar_panel(ax, readout, data, colors, rng)
            set_title(ax, Title(READOUT_LABELS[readout], source_stats_line([n], [errors[0]])),
                      fontsize=11)  # fmt: skip
        _source_legend(axs[0], data)
        _mice_legend(axs[2], colors)
        finish_figure(fig, Title(CROSSING_STEP, None, caption))
    save_or_show(fig, save_path or None)


def plot_crossing_aligned(data: FigureData, save_path=None) -> None:
    """Choices around believed and true rank crossings (test 2b)."""
    readouts = ("cross_leave_1", "cross_adopt_1", "cross_t50")
    caption = _common_caption(
        data, readouts,
        "Believed crossing: the trial on which an arm the mouse has been choosing pays below the "
        "best alternative's last payout, having been at or above it the trial before; time 0. The "
        "new believed best is that alternative. True crossing: the trial on which the arm with the "
        "highest magnitude changes (or the second highest, with the best unchanged), from the "
        "schedule. P(New Best): choosing the arm that now holds the rank. Trials to 50% adoption: "
        "first trial after the event on which the new best is chosen at least half the time "
        "(11 if never within 10).",
    )  # fmt: skip
    crossing, true_best, true_second = (
        "Trial From Crossing",
        "Trial From True Best Change",
        "Trial From Second-Best Change",
    )
    panels = [
        ("cross_believed", "new_best", "New Believed Best", crossing, "P(Choose New Best)"),
        ("cross_believed", "own", "Arm Left", crossing, "P(Stay)"),
        ("cross_true_best", "next", "New True Best", true_best, "P(Choose New Best)"),
        ("cross_true_second", "next", "New True Second Best", true_second, "P(Choose New Second)"),
    ]
    _curve_figure(CROSSING_ALIGNED, panels, data, tuple(caption), save_path, zero_line=True)


def plot_third_option(data: FigureData, save_path=None) -> None:
    """P(choose the believed worst) by its gap to the second best, and the slope (test 3)."""
    readouts = ("worst_slope", "top2_w_worst")
    caption = _common_caption(
        data, readouts,
        "Decisions with a unique believed-worst arm. Gap: the second best's last payout minus the "
        "worst's. Slope: change of the logit of choosing the worst per pulse of gap, with whether "
        "the worst arm is the one just chosen and the run length. A rank algorithm treats the "
        "worst as a category: flat at a floor. A softmax learner chooses it less the further it "
        "lies below. Right: among decisions that go to the believed best or second best (strict "
        "order), the weight of the worst arm's value on choosing the best; zero for a softmax "
        "learner too (independence of irrelevant alternatives), so it is a check, not a test "
        "that separates the two.",
    )  # fmt: skip
    colors, rng, n = _colors(data), jitter_rng(), len(_mice(data))
    with new_figure(nrows=1, ncols=3, figsize=(15.5, 4.6)) as (fig, axs):
        drawn = curve_panel(axs[0], "third_option", "", data, colors)
        axs[0].set(xlabel="Gap, Second Best Minus Worst (Pulses)", ylabel="P(Choose Worst)")
        set_title(axs[0], Title("P(Worst) by Gap", _line_stats(n, drawn)), fontsize=11)
        for ax, readout in zip(axs[1:], readouts, strict=True):
            errors = scalar_panel(ax, readout, data, colors, rng)
            set_title(ax, Title(READOUT_LABELS[readout], source_stats_line([n], [errors[0]])),
                      fontsize=11)  # fmt: skip
        _source_legend(axs[0], data)
        _mice_legend(axs[2], colors)
        finish_figure(fig, Title(THIRD_OPTION, None, caption))
    save_or_show(fig, save_path or None)


def plot_transfer(data: FigureData, save_path=None) -> None:
    """P(best) in the first 100 trials and trials to threshold across problems (test 5a)."""
    readouts = ("slope_p100", "slope_threshold")
    caption = _common_caption(
        data, readouts,
        "Per problem, P(choosing the arm with the highest magnitude) over the first 100 trials, "
        "and the trials until a sliding 20-trial window of it first reaches 0.6 (the problem's "
        "length if never). Problems are numbered within the cohort's drifting problems (cohort 1 "
        "up to the selection's cutoff, cohort 3 all). Replays of the fitted agents do not "
        "transfer: their line is the trend of the schedule itself. Slope: change per problem; the "
        "calibrated z is the mouse's slope against the replays' slopes.",
    )  # fmt: skip
    colors, rng, n = _colors(data), jitter_rng(), len(_mice(data))
    with new_figure(nrows=1, ncols=4, figsize=(20, 4.6)) as (fig, axs):
        for ax, (curve, title, ylabel) in zip(
            axs[:2],
            [("transfer_p100", "P(Best), First 100 Trials", "P(Best)"),
             ("transfer_threshold", "Trials to Threshold", "Trials")],
            strict=True,
        ):  # fmt: skip
            drawn = curve_panel(ax, curve, "", data, colors)
            ax.set(xlabel="Problem Number", ylabel=ylabel)
            set_title(ax, Title(title, _line_stats(n, drawn)), fontsize=11)
        for ax, readout in zip(axs[2:], readouts, strict=True):
            errors = scalar_panel(ax, readout, data, colors, rng)
            set_title(ax, Title(READOUT_LABELS[readout], source_stats_line([n], [errors[0]])),
                      fontsize=11)  # fmt: skip
        _source_legend(axs[1], data)
        _mice_legend(axs[3], colors)
        finish_figure(fig, Title(TRANSFER, None, caption))
    save_or_show(fig, save_path or None)


def plot_all(data: FigureData, out_dir: str | Path) -> None:
    """Save every figure of the rank tests under *out_dir*.

    Args:
        data (FigureData): Mice, replays and calibration.
        out_dir (str | Path): Folder for the figures (created if needed).
    """
    out_dir = Path(out_dir)
    for name, plot in (
        (STAY_WEIGHTS, plot_stay_weights),
        (CROSSING_STEP, plot_crossing_step),
        (CROSSING_ALIGNED, plot_crossing_aligned),
        (THIRD_OPTION, plot_third_option),
        (RANK_VS_VALUE, plot_rank_vs_value),
        (TRANSFER, plot_transfer),
        (BLOCK_CHANGE, plot_block_change),
    ):
        plot(data, out_dir / name)
