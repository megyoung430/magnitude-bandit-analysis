"""Figures of the switch regression: value weights, held-out gains and history weights.

Drawing only: the numbers come from :mod:`src.pipelines.switch_regression` (one
``SourceSwitchRegression`` per source). Each ``plot_*`` function builds a figure, then saves it
(pdf, png, svg and a caption file) or shows it. Every figure puts the mice next to the controls
(simulated Q-learner, null agents), one dot per mouse and black bars for the mean ± se across
mice; a source with fewer than ``choice_logit_extensions.MIN_TEST_MICE`` mice shows its mice only,
with no mean, error bar or test.
"""

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from src.behavior_analysis import switch_regression as sr
from src.behavior_visualization.plot_helpers import Title, jitter_rng, save_or_show, set_title
from src.behavior_visualization.plot_regression_extensions import (
    _colors_of,
    _counts_caption,
    _draw_column,
    _source_ticks,
    _stats_line,
    _style_source_axis,
)
from src.behavior_visualization.plot_value_psychometrics import finish_figure, new_figure

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
    "SD 1); nothing is standardized, so the value weight is in logit per pulse.",
    "Dots: mice; black bars: mean ± se across mice.",
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


def _mice_tables(sources: Mapping) -> dict:
    """``{source: table with one row per mouse}``, for the shared caption and color helpers."""
    return {name: s.decays[["mouse"]] for name, s in sources.items()}


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


def _value_grid(sources: Mapping, column: str, ylabel, heading: str, caption: tuple, save_path):
    """One panel per value variant (rows) and condition (columns), x = sources."""
    values = _present_values(sources)
    colors = _colors_of(list(_mice_tables(sources).values()))
    ticks = _source_ticks(_mice_tables(sources))
    width = max(9.0, 2 * (1.2 * len(sources) + 1.5))
    with new_figure(
        nrows=len(values), ncols=len(sr.CONDITIONS), figsize=(width, 3.9 * len(values)),
        sharey="row", squeeze=False,
    ) as (fig, axs):  # fmt: skip
        rng = jitter_rng()
        for row, value in enumerate(values):
            for col, condition in enumerate(sr.CONDITIONS):
                ax, labeled, counts, errors = axs[row, col], set(), [], []
                for x, s in enumerate(sources.values()):
                    mine = s.values.query("value == @value and condition == @condition")
                    if len(mine):  # a source without this variant (no logit) is left empty
                        counts.append(len(mine))
                    errors.append(
                        _draw_column(ax, x, mine.set_index("mouse")[column], colors, rng, labeled)
                    )
                _style_source_axis(ax, ticks, ylabel(value) if col == 0 else "")
                title = f"{VALUE_LABELS[value]}, {CONDITION_LABELS[condition]}"
                set_title(ax, Title(title, _stats_line(counts, errors)), fontsize=12)
                ax.legend(fontsize=9)
        finish_figure(fig, Title(heading, None, caption))
    save_or_show(fig, save_path or None)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def plot_value_weights(sources: Mapping, save_path: str | Path | None = None) -> None:
    """The value weight of each variant after zero and after non-zero, by source.

    Args:
        sources (Mapping): ``{source label: SourceSwitchRegression}``, the mice first.
        save_path (str | Path | None): Base path without extension; ``None`` shows the figure.

    Raises:
        ValueError: If there are no sources or a source has no mice.
    """
    _check_sources(sources)
    caption = (
        *MODEL_CAPTION,
        _settings_line(sources),
        _decay_line(sources),
        _counts_caption(_mice_tables(sources)),
        *_counts_lines(sources),
        *_test_lines(sources, "beta", "logit per unit of value"),
    )
    _value_grid(sources, "beta", WEIGHT_UNITS.get, VALUE_WEIGHTS, caption, save_path)


def plot_model_comparison(sources: Mapping, save_path: str | Path | None = None) -> None:
    """The held-out gain of adding each value variant to the history-only model, by source.

    Args:
        sources (Mapping): ``{source label: SourceSwitchRegression}``, the mice first.
        save_path (str | Path | None): Base path without extension; ``None`` shows the figure.

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
        _counts_caption(_mice_tables(sources)),
        *_counts_lines(sources),
        *_test_lines(sources, "gain", "nats per switch"),
    )
    _value_grid(sources, "gain", lambda _: GAIN_LABEL, MODEL_COMPARISON, caption, save_path)


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
    colors = _colors_of(list(_mice_tables(sources).values()))
    features = list(dict.fromkeys(f for s in sources.values() for f in s.history["feature"]))
    labels = [_feature_label(f) for f in features]
    caption = (
        "Weights of the history-only model (no value) fitted to all of a condition's switches. "
        "Lags: logit for the alternative chosen k trials ago in the segment (per choice). Trace: "
        "logit per unit of the decayed choice count. Recency: logit per unit of log(1 + trials "
        "since chosen), negative when recently chosen arms are preferred. Never chosen: logit for "
        "an arm not yet chosen in the problem. Tower biases are fitted too and not shown.",
        *MODEL_CAPTION,
        _settings_line(sources),
        _decay_line(sources),
        _counts_caption(_mice_tables(sources)),
        *_counts_lines(sources),
    )
    width = max(6.0, 4.3 * len(sources))
    with new_figure(
        nrows=len(sr.CONDITIONS), ncols=len(sources), figsize=(width, 4.3 * len(sr.CONDITIONS)),
        sharey=True, squeeze=False,
    ) as (fig, axs):  # fmt: skip
        rng = jitter_rng()
        for row, condition in enumerate(sr.CONDITIONS):
            for col, (name, s) in enumerate(sources.items()):
                ax, labeled = axs[row, col], set()
                mine = s.history.query("condition == @condition")
                wide = mine.pivot(index="mouse", columns="feature", values="weight")
                empty = pd.Series(np.nan, index=wide.index)  # a feature the source lacks
                errors = [
                    _draw_column(ax, x, wide.get(f, empty), colors, rng, labeled)
                    for x, f in enumerate(features)
                ]
                _style_source_axis(ax, labels, "Weight (Logit)" if col == 0 else "")
                heading = f"{name}, {CONDITION_LABELS[condition]}"
                set_title(ax, Title(heading, _stats_line([len(wide)], errors)), fontsize=12)
                ax.legend(fontsize=9)
        finish_figure(fig, Title(HISTORY_WEIGHTS, None, caption))
    save_or_show(fig, save_path or None)


def plot_switch_regression(sources: Mapping, out_dir: str | Path) -> None:
    """Save the three switch-regression figures into *out_dir* (created if missing).

    Args:
        sources (Mapping): ``{source label: SourceSwitchRegression}``, the mice first.
        out_dir (str | Path): Folder of the figures.
    """
    out_dir = Path(out_dir)
    plot_value_weights(sources, out_dir / VALUE_WEIGHTS)
    plot_model_comparison(sources, out_dir / MODEL_COMPARISON)
    plot_history_weights(sources, out_dir / HISTORY_WEIGHTS)
