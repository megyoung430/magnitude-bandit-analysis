"""Figures of the switch regression: value weights, held-out gains and history weights.

Drawing only: the numbers come from :mod:`src.pipelines.switch_regression` (one
``SourceSwitchRegression`` per source, and optionally a ``NullCalibration`` per source). Each
``plot_*`` function builds a figure, then saves it (pdf, png, svg and a caption file) or shows it.
Every figure puts the mice next to the controls (simulated Q-learner, null agents), one dot per
mouse (colored for the mice of the first source, gray and not in the legend for other mice such
as the simulated Q-learners) and black bars for the mean ± se across mice; a source with fewer than
``choice_logit_extensions.MIN_TEST_MICE`` mice shows its mice only, with no mean, error bar or
test. A calibrated mouse has a gray band behind its dot: the middle 95% of its own null. Each
column's mean is also written at the top of its panel, because the panels are narrow.
"""

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from src.behavior_analysis import choice_logit_extensions as ext
from src.behavior_analysis import switch_regression as sr
from src.behavior_visualization.plot_helpers import Title, jitter_rng, save_or_show, set_title
from src.behavior_visualization.plot_regression_extensions import (
    OTHER_MICE_COLOR,  # noqa: F401  (re-exported: the gray of the unnamed mice)
    counts_caption,
    draw_column,
    source_colors,
    source_stats_line,
    style_source_axis,
)
from src.behavior_visualization.plot_value_psychometrics import finish_figure, new_figure
from src.pipelines.null_agents import AGENT_LABELS

VALUE_WEIGHTS = "Switch Regression Value Weights"
MODEL_COMPARISON = "Switch Regression Model Comparison"
HISTORY_WEIGHTS = "Switch Regression History Weights"
FIGURE_NAMES = (VALUE_WEIGHTS, MODEL_COMPARISON, HISTORY_WEIGHTS)
VALUE_LABELS = {sr.TRUE: "True Value", sr.BELIEVED: "Believed Value", sr.MODEL: "Model Value"}
CONDITION_LABELS = {sr.AFTER_ZERO: "After Zero", sr.AFTER_NONZERO: "After Non-Zero"}
WEIGHT_UNITS = {
    sr.TRUE: "Value Weight (Logit per Pulse)",
    sr.BELIEVED: "Value Weight (Logit per Pulse)",
    sr.MODEL: "Value Weight (per Logit of Model Value)",
}
GAIN_LABEL = "Held-Out Gain (Nats per Switch)"
FEATURE_LABELS = {sr.TRACE: "Trace", sr.RECENCY: "Recency", sr.NEVER: "Never Chosen"}

MODEL_CAPTION = (
    "Switch trials: consecutive trials of one segment (a segment starts at a problem's start, a "
    "session's start and after a dropped trial) on which the mouse leaves the arm it chose on "
    "the previous trial. The choice set is the two other arms. After zero: the previous choice "
    "paid 0 pulses; after non-zero: it paid 1 or more. Each condition is fitted separately.",
    "Model: P(first alternative) = sigmoid(U(first) - U(second)), a two-alternative conditional "
    "logit with no intercept (the order of the two arms cannot matter; port preferences are the "
    "tower biases), U(a) = value weight * V(a) + choice lags 2..K + trace weight * h(a) + recency "
    "weight * log(1 + min(trials since a was chosen, cap)) + never-chosen weight + tower bias. "
    "h is the decayed count of past choices of a within the segment (the null agents' trace); "
    "its decay is chosen per mouse from a grid by the held-out likelihood of the history-only "
    "model (no value), both conditions pooled, and used for every value variant of the mouse. "
    "Trials since chosen are counted within the problem across sessions; the never-chosen "
    "indicator is 1 before the arm's first choice in the problem.",
    "Value variants, one fit each: true value = the magnitude the arm offers on the switch trial "
    "(pulses); believed value = the magnitude it paid when last chosen in the problem, 0 before "
    "its first visit (pulses; the never-chosen indicator is its never-visited indicator); model "
    "value = the choice logit's held-out reward part (logits). The model value is a consistency "
    "check, not an independent test: it is built from the same reward history as the believed "
    "value, and the logit fit behind a training problem's model values saw the held-out problem.",
    "Held-out gain: leave-one-problem-out, the history-only model's negative log likelihood per "
    "switch minus that of the model with the value, on the same folds; positive means the value "
    "predicts held-out switch targets beyond choice history, recency and tower bias. Weights and "
    "their standard errors (Laplace, from the Hessian of the penalized likelihood) come from the "
    "fit to all of a condition's switches. Every weight has an L2 penalty (a Gaussian prior with "
    "SD 1); nothing is standardized, so the value weight is in logit per pulse. The standard "
    "errors are model-based (Laplace), not clustered by problem, so they can be too small when "
    "switches within a problem are dependent; the random-effects (DerSimonian-Laird) se built on "
    "them can be too small for real mice as well.",
    "Readouts: the believed value and the true value after non-zero. The true value after zero "
    "is weak even for the simulated Q-learner and cannot separate it from the null agents at these "
    "switch counts. Read a mouse against its own null (gray band, calibrated z and p) rather than "
    "against 0: a value-blind agent's raw weight is not reliably 0 at a mouse's switch counts.",
    "Dots: mice, colored per mouse; a null agent fitted to a mouse has that mouse's color, and "
    "simulated mice not fitted to a mouse (the Q-learners next to the real mice) are gray. Black "
    "bars: mean ± se across mice, with the mean written at the top of the panel.",
)


# ---------------------------------------------------------------------------
# Shared pieces
# ---------------------------------------------------------------------------


def _check_sources(sources: Mapping) -> None:
    if not sources:
        raise ValueError("no sources to plot")
    empty = [name for name, s in sources.items() if s.values.empty]
    if empty:
        raise ValueError(f"no mice in {empty}")


# Dots of mice that are not in the first source (e.g. the simulated Q-learners next to the real
# mice) are drawn in this gray and left out of the legend.


def _mouse_colors(sources: Mapping) -> tuple[dict[str, str], frozenset[str]]:
    """Colors of every mouse, and the mice left out of the legend.

    The mice of the first source (the real mice when there are any) get the per-mouse palette;
    a mouse that is not in the first source (a simulated Q-learner, say) is gray and unnamed, so
    the legend lists the mice only. Null agents fitted to a mouse share its name and its color.

    Args:
        sources (Mapping): ``{source label: SourceSwitchRegression}``, the mice first.

    Returns:
        tuple[dict[str, str], frozenset[str]]: ``{mouse: color}`` and the gray mice.
    """
    return source_colors(list(_mice_tables(sources).values()))


# Headroom added above the data, as a fraction of the y range, for the mean labels.
LABEL_HEADROOM = 0.15


def column_mean(values: pd.Series, summary: bool = True) -> float:
    """The mean that :func:`draw_column` draws as a bar, NaN when it draws none."""
    values = values.dropna()
    if not summary or len(values) < ext.MIN_TEST_MICE:
        return np.nan
    return float(values.mean())


def label_means(ax, means: Mapping[float, float]) -> None:
    """Write each column's mean (``{x: mean}``) at the top of the panel, above its bar.

    The panels are narrow, so the number is easier to read than the bar's height. Columns
    without a bar (too few mice) get no number.
    """
    for x, mean in means.items():
        if np.isfinite(mean):
            ax.text(x, 0.98, f"{mean:.2g}", transform=ax.get_xaxis_transform(), ha="center",
                    va="top", fontsize=8)  # fmt: skip


def add_headroom(ax) -> None:
    """Raise the top of *ax*'s y range (shared axes follow) so the mean labels clear the data."""
    low, high = ax.get_ylim()
    ax.set_ylim(low, high + LABEL_HEADROOM * (high - low))


def _mice_tables(sources: Mapping) -> dict:
    """``{source: table with one row per mouse}``, for the shared caption and color helpers."""
    return {name: s.decays[["mouse"]] for name, s in sources.items()}


def _ticks(sources: Mapping) -> list[str]:
    """``"<source>\nn=<analyzed>"``, with the skipped mice counted when there are any."""
    ticks = []
    for name, s in sources.items():
        skipped = f" ({len(s.skipped)} skipped)" if s.skipped else ""
        ticks.append(f"{name}\nn={len(s.decays)}{skipped}")
    return ticks


def _skipped_lines(sources: Mapping) -> list[str]:
    return [
        f"Skipped, {name}: " + "; ".join(f"{m} ({reason})" for m, reason in s.skipped.items()) + "."
        for name, s in sources.items()
        if s.skipped
    ]


def _primary(calibration) -> pd.DataFrame:
    """Rows of the first null agent of *calibration* (the realistic WSLS (Magnitude) by default)."""
    table = calibration.calibration
    return table[table["agent"] == table["agent"].iloc[0]]


def _bands(calibration, value: str, condition: str, term: str) -> dict:
    """``{mouse: (null_low, null_high)}`` of one panel, or ``{}`` without a calibration."""
    if calibration is None:
        return {}
    rows = _primary(calibration).query(
        "value == @value and condition == @condition and term == @term"
    )
    return {r.mouse: (r.null_low, r.null_high) for r in rows.itertuples()}


def _calibration_lines(calibrations: Mapping, term: str) -> list[str]:
    """Per mouse: calibrated z and empirical p against each null agent; and the across-mice test."""
    lines = []
    for name, calibration in calibrations.items():
        table = calibration.calibration.query("term == @term")
        agents = list(dict.fromkeys(table["agent"]))
        first = AGENT_LABELS.get(agents[0], agents[0])
        lines.append(
            f"Null calibration, {name}: each mouse's fitted null agents ("
            + ", ".join(AGENT_LABELS.get(a, a) for a in agents)
            + f") played on its own task, {calibration.replicates} replicates each, through the "
            "identical analysis; gray band: 2.5th-97.5th percentile of the "
            f"{first} null only (one band is drawn; every agent's z and p are listed). "
            "z = (observed - null mean) / null sd; p: two-sided empirical."
        )
        for (mouse, value, condition), group in table.groupby(
            ["mouse", "value", "condition"], sort=False
        ):
            parts = [
                f"z={r.z:.2f}, p={r.p_empirical:.3g} vs {AGENT_LABELS.get(r.agent, r.agent)}"
                for r in group.itertuples()
            ]
            lines.append(
                f"{name}, {mouse}, {VALUE_LABELS[value]}, {CONDITION_LABELS[condition]}: "
                f"observed {group['observed'].iloc[0]:.4f}; " + "; ".join(parts) + "."
            )
        summary = calibration.summary.query("term == @term")
        for row in summary.itertuples():
            if np.isfinite(row.mean):
                lines.append(
                    f"{name}, calibrated z vs {AGENT_LABELS.get(row.agent, row.agent)}, "
                    f"{VALUE_LABELS[row.value]}, {CONDITION_LABELS[row.condition]}: mean "
                    f"{row.mean:.2f} ± se {row.se:.2f}; Wilcoxon p={row.p_wilcoxon:.3g} "
                    f"(n={row.n} mice)."
                )
    return lines


def _counts_lines(sources: Mapping) -> list[str]:
    lines = []
    for name, s in sources.items():
        wide = s.counts.pivot(index="mouse", columns="condition", values="n_switches")
        per_mouse = "; ".join(
            f"{m}: {row[sr.AFTER_ZERO]} after zero, {row[sr.AFTER_NONZERO]} after non-zero"
            for m, row in wide.iterrows()
        )
        lines.append(f"Switches per mouse, {name}: {per_mouse}.")
    return lines


def _decay_line(sources: Mapping) -> str:
    parts = []
    for name, s in sources.items():
        counts = s.decays["trace_decay"].value_counts().sort_index()
        chosen = ", ".join(
            f"{d:g} ({n} {'mouse' if n == 1 else 'mice'})" for d, n in counts.items()
        )
        parts.append(f"{name}: {chosen}")
    grids = {s.trace_grid for s in sources.values()}
    grid = ", ".join(f"{d:g}" for d in sorted(set().union(*grids)))
    return f"Trace decay chosen per mouse from {{{grid}}}: " + "; ".join(parts) + "."


def _settings_line(sources: Mapping) -> str:
    first = next(iter(sources.values()))
    return (
        f"K = {first.n_lags} (lags 2 to {first.n_lags}; lag 1 is the previous choice, never in "
        f"the choice set), recency cap {first.recency_cap} trials, at least {first.min_switches} "
        f"switches per condition and mouse, L2 = {first.l2:g}."
    )


def _test_lines(sources: Mapping, term: str, unit: str) -> list[str]:
    """Mean ± se, positive mice and Wilcoxon p of *term* for each source with enough mice."""
    lines = []
    for name, s in sources.items():
        summary = sr.group_summary(s.values).query("term == @term")
        for _, row in summary.iterrows():
            heading = f"{name}, {VALUE_LABELS[row['value']]}, {CONDITION_LABELS[row['condition']]}"
            if not np.isfinite(row["mean"]):
                continue
            text = (
                f"{heading}: mean {row['mean']:.4f} ± se {row['se']:.4f} {unit}; "
                f"{row['n_positive']} of {row['n']} mice positive; "
                f"Wilcoxon p={row['p_wilcoxon']:.3g}"
            )
            if term == "beta":
                text += (
                    f"; random-effects (DerSimonian-Laird) mean {row['re_mean']:.4f} ± "
                    f"{row['re_se']:.4f}"
                )
            lines.append(text + f" (n={row['n']} mice).")
    return lines


def _present_values(sources: Mapping) -> list[str]:
    present = {v for s in sources.values() for v in s.values["value"]}
    return [v for v in sr.VALUES if v in present]


def _value_grid(
    sources: Mapping,
    column: str,
    ylabel,
    heading: str,
    caption: tuple,
    save_path,
    calibrations: Mapping,
):
    """One panel per value variant (rows) and condition (columns), x = sources."""
    values = _present_values(sources)
    colors, unnamed = _mouse_colors(sources)
    ticks = _ticks(sources)
    width = max(9.0, 2 * (1.2 * len(sources) + 1.5))
    with new_figure(
        nrows=len(values), ncols=len(sr.CONDITIONS), figsize=(width, 3.9 * len(values)),
        sharey="row", squeeze=False,
    ) as (fig, axs):  # fmt: skip
        rng = jitter_rng()
        for row, value in enumerate(values):
            for col, condition in enumerate(sr.CONDITIONS):
                ax, labeled, counts, errors = axs[row, col], set(unnamed), [], []
                means = {}
                for x, (name, s) in enumerate(sources.items()):
                    mine = s.values.query("value == @value and condition == @condition")
                    if mine.empty:  # e.g. no model value for a source without a logit analysis
                        ax.text(x, 0, "not fitted", rotation=90, ha="center", va="bottom",
                                color="0.5", fontsize=8)  # fmt: skip
                        continue
                    counts.append(len(mine))
                    bands = _bands(calibrations.get(name), value, condition, column)
                    per_mouse = mine.set_index("mouse")[column]
                    errors.append(draw_column(ax, x, per_mouse, colors, rng, labeled, bands=bands))
                    means[x] = column_mean(per_mouse)
                label_means(ax, means)
                style_source_axis(ax, ticks, ylabel(value) if col == 0 else "")
                title = f"{VALUE_LABELS[value]}, {CONDITION_LABELS[condition]}"
                set_title(ax, Title(title, source_stats_line(counts, errors)), fontsize=12)
                if ax.get_legend_handles_labels()[1]:  # a panel of gray mice only has none
                    ax.legend(fontsize=9)
            add_headroom(axs[row, 0])  # once per row: the row shares its y axis
        finish_figure(fig, Title(heading, None, caption))
    save_or_show(fig, save_path or None)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def plot_value_weights(
    sources: Mapping, save_path: str | Path | None = None, calibrations: Mapping | None = None
) -> None:
    """The value weight of each variant after zero and after non-zero, by source.

    Args:
        sources (Mapping): ``{source label: SourceSwitchRegression}``, the mice first.
        save_path (str | Path | None): Base path without extension; ``None`` shows the figure.
        calibrations (Mapping | None): ``{source label: NullCalibration}`` of the sources whose
            mice are calibrated (gray null band per mouse, calibrated z and p in the caption).
            Default ``None``.

    Raises:
        ValueError: If there are no sources or a source has no mice.
    """
    _check_sources(sources)
    caption = (
        *MODEL_CAPTION,
        _settings_line(sources),
        _decay_line(sources),
        counts_caption(_mice_tables(sources)),
        *_counts_lines(sources),
        *_skipped_lines(sources),
        *_test_lines(sources, "beta", "logit per unit of value"),
        *_calibration_lines(calibrations or {}, "beta"),
    )
    _value_grid(
        sources, "beta", WEIGHT_UNITS.get, VALUE_WEIGHTS, caption, save_path, calibrations or {}
    )


def plot_model_comparison(
    sources: Mapping, save_path: str | Path | None = None, calibrations: Mapping | None = None
) -> None:
    """The held-out gain of adding each value variant to the history-only model, by source.

    Args:
        sources (Mapping): ``{source label: SourceSwitchRegression}``, the mice first.
        save_path (str | Path | None): Base path without extension; ``None`` shows the figure.
        calibrations (Mapping | None): ``{source label: NullCalibration}``; as in
            :func:`plot_value_weights`, for the gain. Default ``None``.

    Raises:
        ValueError: If there are no sources or a source has no mice.
    """
    _check_sources(sources)
    caption = (
        "Gain = held-out (leave-one-problem-out) nats per switch of the history-only model minus "
        "that of the model with the value; positive: the value predicts where held-out switches "
        "go beyond choice history, recency and tower bias.",
        *MODEL_CAPTION,
        _settings_line(sources),
        _decay_line(sources),
        counts_caption(_mice_tables(sources)),
        *_counts_lines(sources),
        *_skipped_lines(sources),
        *_test_lines(sources, "gain", "nats per switch"),
        *_calibration_lines(calibrations or {}, "gain"),
    )
    _value_grid(
        sources, "gain", lambda _: GAIN_LABEL, MODEL_COMPARISON, caption, save_path,
        calibrations or {},
    )  # fmt: skip


def _feature_label(feature: str) -> str:
    if feature.startswith("choice_lag"):
        return f"Lag {feature.removeprefix('choice_lag')}"
    return FEATURE_LABELS.get(feature, feature)


def plot_history_weights(sources: Mapping, save_path: str | Path | None = None) -> None:
    """Weights of the history-only model (lags, trace, recency, never chosen) by source.

    These are the terms that absorb where switches go without a value. One row per condition,
    one panel per source, on a common y axis.

    Args:
        sources (Mapping): ``{source label: SourceSwitchRegression}``, the mice first.
        save_path (str | Path | None): Base path without extension; ``None`` shows the figure.

    Raises:
        ValueError: If there are no sources or a source has no mice.
    """
    _check_sources(sources)
    colors, unnamed = _mouse_colors(sources)
    features = list(dict.fromkeys(f for s in sources.values() for f in s.history["feature"]))
    labels = [_feature_label(f) for f in features]
    caption = (
        "Weights of the history-only model (no value) fitted to all of a condition's switches. "
        "Lags: logit for the alternative chosen k trials ago in the segment (per choice). Trace: "
        "logit per unit of the decayed choice count. Recency: logit per unit of log(1 + trials "
        "since chosen), negative when recently chosen arms are preferred. Never chosen: logit for "
        "an arm not yet chosen in the problem. Tower biases are fitted too and not shown.",
        "The trace's unit depends on its decay (a count decayed by 0.5 runs to 2, by 0.95 to 20), "
        "so trace weights of mice with different decays are not on one scale: for a source whose "
        "mice chose different decays the trace column shows the mice only, without a mean.",
        *MODEL_CAPTION,
        _settings_line(sources),
        _decay_line(sources),
        counts_caption(_mice_tables(sources)),
        *_counts_lines(sources),
        *_skipped_lines(sources),
    )
    width = max(6.0, 4.3 * len(sources))
    with new_figure(
        nrows=len(sr.CONDITIONS), ncols=len(sources), figsize=(width, 4.3 * len(sr.CONDITIONS)),
        sharey=True, squeeze=False,
    ) as (fig, axs):  # fmt: skip
        rng = jitter_rng()
        for row, condition in enumerate(sr.CONDITIONS):
            for col, (name, s) in enumerate(sources.items()):
                ax, labeled = axs[row, col], set(unnamed)
                mine = s.history.query("condition == @condition")
                wide = mine.pivot(index="mouse", columns="feature", values="weight")
                empty = pd.Series(np.nan, index=wide.index)  # a feature the source lacks
                one_decay = s.decays["trace_decay"].nunique() == 1
                summaries = [one_decay or f != sr.TRACE for f in features]
                columns = [wide.get(f, empty) for f in features]
                errors = [
                    draw_column(ax, x, c, colors, rng, labeled, summary=summary)
                    for x, (c, summary) in enumerate(zip(columns, summaries, strict=True))
                ]  # fmt: skip
                label_means(ax, {
                    x: column_mean(c, summary)
                    for x, (c, summary) in enumerate(zip(columns, summaries, strict=True))
                })  # fmt: skip
                style_source_axis(ax, labels, "Weight (Logit)" if col == 0 else "")
                heading = f"{name}, {CONDITION_LABELS[condition]}"
                set_title(ax, Title(heading, source_stats_line([len(wide)], errors)), fontsize=12)
                if ax.get_legend_handles_labels()[1]:  # a panel of gray mice only has none
                    ax.legend(fontsize=9)
        add_headroom(axs[0, 0])  # once: every panel shares the y axis
        finish_figure(fig, Title(HISTORY_WEIGHTS, None, caption))
    save_or_show(fig, save_path or None)


def plot_switch_regression(
    sources: Mapping, out_dir: str | Path, calibrations: Mapping | None = None
) -> None:
    """Save the three switch-regression figures into *out_dir* (created if missing).

    Args:
        sources (Mapping): ``{source label: SourceSwitchRegression}``, the mice first.
        out_dir (str | Path): Folder of the figures.
        calibrations (Mapping | None): ``{source label: NullCalibration}`` of the calibrated
            sources. Default ``None``.
    """
    out_dir = Path(out_dir)
    plot_value_weights(sources, out_dir / VALUE_WEIGHTS, calibrations)
    plot_model_comparison(sources, out_dir / MODEL_COMPARISON, calibrations)
    plot_history_weights(sources, out_dir / HISTORY_WEIGHTS)
