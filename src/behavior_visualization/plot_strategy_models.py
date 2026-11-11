"""Figures of the strategy-model comparison and of its model recovery.

Drawing only: the numbers come from :mod:`src.pipelines.strategy_comparison` (one
``SourceComparison`` per source, one ``Recovery`` on the mice's own schedules). Each ``plot_*``
function builds a figure, then saves it (pdf, png, svg and a caption file) or shows it. A source
with fewer than ``choice_logit_extensions.MIN_TEST_MICE`` mice shows its mice only, with no mean,
error bar or test. Recovery is judged against ``POOR_RECOVERY`` and the caption says plainly when
a model, or a parameter, is not recovered.
"""

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from src.behavior_analysis import choice_logit_extensions as ext
from src.behavior_analysis import strategy_models as sm
from src.behavior_visualization.plot_helpers import (
    Title,
    despine,
    jitter_rng,
    mouse_color_map,
    save_or_show,
    set_title,
)
from src.behavior_visualization.plot_regression_extensions import (
    draw_column,
    source_stats_line,
    style_source_axis,
)
from src.behavior_visualization.plot_switch_regression import (
    OTHER_MICE_COLOR,
    add_headroom,
    column_mean,
    label_means,
)
from src.behavior_visualization.plot_value_psychometrics import finish_figure, new_figure
from src.pipelines import strategy_comparison as pipe

STRATEGY_MODEL_COMPARISON = "Strategy Model Comparison"
STRATEGY_MODEL_RECOVERY = "Strategy Model Recovery"
FIGURE_NAMES = (STRATEGY_MODEL_COMPARISON, STRATEGY_MODEL_RECOVERY)
MODEL_LABELS = {
    sm.WSLS_MAGNITUDE: "WSLS (Magnitude)",
    sm.HEURISTIC_PLUS: "Heuristic+",
    sm.Q_LEARNER: "Q-Learner",
    sm.WSLS_GRADED: "WSLS (Graded)",
}
# Okabe-Ito blue, vermillion, green and reddish purple for the four models.
MODEL_COLORS = {
    sm.WSLS_MAGNITUDE: "#0072B2",
    sm.HEURISTIC_PLUS: "#D55E00",
    sm.Q_LEARNER: "#009E73",
    sm.WSLS_GRADED: "#CC79A7",
}
PARAMETER_LABELS = {
    "v": "Believed-Value Weight (v)",
    "u": "Never-Visited Weight (u)",
    "q_learning_rate": "Q-Learner Learning Rate",
    "q_beta": "Q-Learner Beta",
    "graded_w_m": "WSLS (Graded) Magnitude Weight (w_m)",
    "p_recent": "WSLS (Graded) p_recent",
}
# Panels per row of the comparison figure (differences, wins, then fitted parameters).
PANEL_COLUMNS = 4
# Parameters whose short name is shown as written rather than title-cased.
AS_WRITTEN = ("p_recent",)
# Fraction of simulated mice of a model that must be won by that model to call it recovered, and
# the correlation of a recovered parameter with its generating value to call it recovered.
POOR_RECOVERY = 0.8
POOR_CORRELATION = 0.7
# The parameters shown in the recovery figure: (generating model, parameter).
RECOVERED = (
    (sm.HEURISTIC_PLUS, "v"),
    (sm.HEURISTIC_PLUS, "u"),
    (sm.Q_LEARNER, "learning_rate"),
    (sm.WSLS_GRADED, "p_recent"),
)


def _check_sources(sources: Mapping) -> None:
    if not sources:
        raise ValueError("no sources to plot")
    empty = [name for name, s in sources.items() if s.nats.empty]
    if empty:
        raise ValueError(f"no mice in {empty}")


# ---------------------------------------------------------------------------
# What recovery says (pure text)
# ---------------------------------------------------------------------------


def _parameter_name(parameter: str) -> str:
    """Display name of a generating parameter (``v``, ``u`` and ``p_recent`` stay as written)."""
    if len(parameter) == 1 or parameter in AS_WRITTEN:
        return parameter
    return parameter.replace("_", " ").title()


def diagonal(recovery) -> pd.Series:
    """Fraction of each generating model's simulated mice won by that model (NaN if none)."""
    return pd.Series({m: recovery.fractions.loc[m, m] for m in sm.MODELS}, name="recovered")


def recovery_statement(recovery) -> list[str]:
    """Plain sentences on model recovery and on the recovery of the ``RECOVERED`` parameters.

    A model is called poorly recovered when fewer than ``POOR_RECOVERY`` of its simulated mice
    are won by it; a parameter when its correlation with the generating value is below
    ``POOR_CORRELATION`` (or cannot be computed).

    Args:
        recovery (Recovery): :func:`src.pipelines.strategy_comparison.recover_models`.

    Returns:
        list[str]: Caption lines, the verdict first.
    """
    fraction = diagonal(recovery)
    counts = recovery.counts.sum(axis=1)
    poor = [MODEL_LABELS[m] for m in sm.MODELS if not fraction[m] >= POOR_RECOVERY]
    verdict = (
        f"Model recovery is poor for {', '.join(poor)}: fewer than {POOR_RECOVERY:.0%} of its "
        "simulated mice are won by the model that generated them, so a winner among the real "
        "mice is not strong evidence for that model."
        if poor
        else f"Model recovery is good: each model wins at least {POOR_RECOVERY:.0%} of its own "
        "simulated mice."
    )
    lines = [
        verdict,
        "Heuristic+ nests WSLS (Magnitude): when WSLS (Magnitude) generated the choices the two "
        "differ by about 0.001 nats per choice or less, so either can win by chance and a win "
        "by that margin is not evidence for the extra weights.",
    ]
    for m in sm.MODELS:
        row = recovery.fractions.loc[m]
        lines.append(
            f"Generated by {MODEL_LABELS[m]} ({int(counts[m])} simulated mice): "
            + ", ".join(f"won by {MODEL_LABELS[w]} {row[w]:.0%}" for w in sm.MODELS)
            + "."
        )
    for model, parameter in RECOVERED:
        rows = recovery.summary.query("generator == @model and parameter == @parameter")
        name = f"{MODEL_LABELS[model]} {_parameter_name(parameter)}"
        if rows.empty:
            lines.append(f"{name}: not simulated.")
            continue
        r = rows.iloc[0]
        if np.isfinite(r["correlation"]):
            quality = "recovered" if r["correlation"] >= POOR_CORRELATION else "poorly recovered"
            head = f"{name} is {quality}: correlation of estimate with truth {r['correlation']:.2f}"
        else:
            head = (
                f"{name} cannot be judged by correlation (it needs three simulated mice and "
                "spread in the truth)"
            )
        lines.append(
            f"{head}, bias {r['bias']:.2f}, rmse {r['rmse']:.2f} (n={int(r['n'])} simulated mice)."
        )
    return lines


# ---------------------------------------------------------------------------
# Strategy Model Comparison
# ---------------------------------------------------------------------------


def _source_colors(sources: Mapping) -> tuple[dict[str, str], frozenset[str]]:
    """Per-mouse colors for the first source, gray (unnamed) for mice of other sources."""
    names = [list(s.nats["mouse"].unique()) for s in sources.values()]
    colors = mouse_color_map(sorted(names[0]))
    others = frozenset(m for n in names[1:] for m in n) - set(colors)
    return {**colors, **dict.fromkeys(others, OTHER_MICE_COLOR)}, others


def _ticks(sources: Mapping) -> list[str]:
    return [
        f"{name}\nn={s.nats['mouse'].nunique()}"
        + (f" ({len(s.skipped)} skipped)" if s.skipped else "")
        for name, s in sources.items()
    ]


def _skipped_lines(sources: Mapping) -> list[str]:
    return [
        f"Skipped, {name}: " + "; ".join(f"{m} ({reason})" for m, reason in s.skipped.items()) + "."
        for name, s in sources.items()
        if s.skipped
    ]


def _comparison_title(first: str, second: str) -> str:
    return f"{MODEL_LABELS[first]} Minus {MODEL_LABELS[second]}"


def _summary_lines(sources: Mapping) -> list[str]:
    lines = []
    for row in pipe.source_summaries(sources).itertuples():
        first, _, second = row.comparison.partition("_minus_")
        if np.isfinite(row.mean):
            lines.append(
                f"{row.source}, {_comparison_title(first, second)}: mean {row.mean:.4f} ± se "
                f"{row.se:.4f} nats per choice across mice (n={row.n} mice); first model fits "
                f"worse in {row.n_positive}; Wilcoxon p={row.p_wilcoxon:.3g}."
            )
    return lines


def _mouse_lines(sources: Mapping) -> list[str]:
    lines = []
    for row in pipe.per_mouse_table(sources).itertuples():
        nats = ", ".join(f"{MODEL_LABELS[m]} {getattr(row, m):.4f}" for m in sm.MODELS)
        lines.append(
            f"{row.source}, {row.mouse}: held-out nats per choice {nats}; winner "
            f"{MODEL_LABELS[row.winner]}; v={row.v:.2f}, u={row.u:.2f}, Q learning rate "
            f"{row.q_learning_rate:.3f}, Q beta {row.q_beta:.2f}, graded w_m={row.graded_w_m:.2f}, "
            f"p_recent={row.p_recent:.2f}"
            + (f"; choice logit {row.choice_logit:.4f}" if np.isfinite(row.choice_logit) else "")
            + "."
        )
    return lines


def _wins_panel(ax, sources: Mapping, ticks: list[str]) -> None:
    """Mice won by each model, per source (grouped bars)."""
    wins = pipe.win_table(sources)
    width = 0.8 / len(sm.MODELS)
    for k, model in enumerate(sm.MODELS):
        counts = wins[wins["model"] == model].set_index("source")["n_mice"]
        x = np.arange(len(sources)) + (k - (len(sm.MODELS) - 1) / 2) * width
        ax.bar(x, [counts[name] for name in sources], width, color=MODEL_COLORS[model],
               label=MODEL_LABELS[model])  # fmt: skip
    ax.set_xticks(range(len(ticks)), ticks, rotation=20, ha="right")
    ax.set_ylabel("Mice Won", fontsize=12)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False, fontsize=9)
    despine(ax)


def _draw_panel(ax, sources, column, source_column, colors, unnamed, rng, ticks, ylabel, title):
    """One dot-per-mouse panel across sources; *source_column* picks each source's values."""
    labeled, counts, errors, means = set(unnamed), [], [], {}
    for x, (name, s) in enumerate(sources.items()):
        values = source_column(name, s, column).dropna()
        if values.empty:
            continue
        counts.append(len(values))
        errors.append(draw_column(ax, x, values, colors, rng, labeled))
        means[x] = column_mean(values)
    label_means(ax, means)
    style_source_axis(ax, ticks, ylabel)
    set_title(ax, Title(title, source_stats_line(counts, errors)), fontsize=12)
    if ax.get_legend_handles_labels()[1]:
        ax.legend(fontsize=9)
    add_headroom(ax)


def plot_strategy_model_comparison(
    sources: Mapping, save_path: str | Path | None = None, recovery=None
) -> None:
    """Held-out paired differences of the four models, wins per model and fitted parameters.

    Top row: held-out nats per choice of the first model minus the second (negative: the first
    predicts held-out choices better) per mouse, and the mice won by each model. Bottom row: the
    fitted ``v``, ``u``, Q-learner learning rate and beta, and WSLS (Graded) ``w_m`` and
    ``p_recent``.

    Args:
        sources (Mapping): ``{source label: SourceComparison}``, the mice first.
        save_path (str | Path | None): Base path without extension; ``None`` shows the figure.
        recovery (Recovery | None): Model recovery on the mice's schedules; its verdict leads the
            caption. Default ``None``: the caption says recovery was not run.

    Raises:
        ValueError: If there are no sources or a source has no mice.
    """
    _check_sources(sources)
    colors, unnamed = _source_colors(sources)
    ticks, table = _ticks(sources), pipe.per_mouse_table(sources)
    verdict = (
        recovery_statement(recovery)
        if recovery is not None
        else ["Model recovery was not run for this figure; the winners are not validated."]
    )
    caption = (
        *verdict,
        "Models, each fitted on all but one problem and scored on the held-out problem (every "
        "problem in turn): WSLS (Magnitude), whose staying grows with the last payout and whose "
        "switches follow the decayed choice trace; Heuristic+, the same stay rule with switches "
        "that also follow what the alternatives last paid (v) and whether they were ever chosen "
        "(u); the Q-learner (value_agent: learning rate, beta, stickiness, bias to the plus-shaped "
        "ports; several starts). The trace decay is fixed at 0.9 for both heuristics, not fitted, "
        "so Heuristic+ nests WSLS (Magnitude) and a difference is attributable to v and u. "
        "WSLS (Graded): staying grows with the last payout (intercept w0 and slope w_m), and a "
        "switch goes to the more recently chosen of the two alternatives with probability "
        "p_recent, otherwise to the other (it does not nest WSLS (Magnitude)).",
        "Scored: every choice with a previous choice in its segment, for all four models; the "
        "Q-learner still learns from the segment starts. The choice logit's held-out nats (listed "
        "per mouse below) is a descriptive reference, scored on all trials.",
        "Differences: first model minus second, nats per choice pooled over the mouse's held-out "
        "problems; negative: the first model fits better. A source with fewer than "
        f"{ext.MIN_TEST_MICE} mice shows its mice only: no mean, error bar or test.",
        "Dots: mice, colored per mouse; gray: simulated mice not fitted to a mouse. Black bars: "
        "mean ± se across mice, with the mean written at the top of the panel.",
        *_skipped_lines(sources),
        *_summary_lines(sources),
        *_mouse_lines(sources),
    )
    pairs = sm.COMPARISONS

    def differences(name, s, column):
        return s.differences.set_index("mouse")[column]

    def parameter(name, s, column):
        rows = table[table["source"] == name]
        return rows.set_index("mouse")[column]

    n_panels = len(pairs) + 1 + len(PARAMETER_LABELS)
    n_rows = -(-n_panels // PANEL_COLUMNS)
    with new_figure(
        nrows=n_rows, ncols=PANEL_COLUMNS, figsize=(17.0, 4.3 * n_rows), squeeze=False
    ) as (fig, axs):
        rng = jitter_rng()
        flat = axs.ravel()
        # The wins panel ends the first row, so its legend sits at the figure's right edge.
        wins_at = PANEL_COLUMNS - 1
        slots = [i for i in range(n_panels) if i != wins_at]
        for i, (first, second) in zip(slots, pairs, strict=False):
            ylabel = (
                "Held-Out Nats per Choice, First Minus Second" if i % PANEL_COLUMNS == 0 else ""
            )
            _draw_panel(flat[i], sources, sm.comparison_name(first, second), differences,
                        colors, unnamed, rng, ticks, ylabel,
                        _comparison_title(first, second))  # fmt: skip
        _wins_panel(flat[wins_at], sources, ticks)
        for i, name in zip(slots[len(pairs) :], PARAMETER_LABELS, strict=True):
            _draw_panel(flat[i], sources, name, parameter, colors, unnamed, rng, ticks,
                        "Fitted Value", PARAMETER_LABELS[name])  # fmt: skip
        for ax in flat[n_panels:]:
            ax.set_visible(False)
        finish_figure(fig, Title(STRATEGY_MODEL_COMPARISON, None, caption))
    save_or_show(fig, save_path or None)


# ---------------------------------------------------------------------------
# Strategy Model Recovery
# ---------------------------------------------------------------------------


def _confusion_panel(ax, recovery) -> None:
    ax.imshow(recovery.fractions.to_numpy(dtype=float), vmin=0, vmax=1, cmap="Blues")
    for i, generator in enumerate(sm.MODELS):
        for j, winner in enumerate(sm.MODELS):
            fraction = recovery.fractions.loc[generator, winner]
            if np.isfinite(fraction):
                ax.text(j, i, f"{fraction:.0%}\n{recovery.counts.loc[generator, winner]}",
                        ha="center", va="center", fontsize=11,
                        color="white" if fraction > 0.5 else "black")  # fmt: skip
    labels = [MODEL_LABELS[m] for m in sm.MODELS]
    ax.set_xticks(range(len(labels)), labels, rotation=20, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Winning Model (Held-Out)", fontsize=12)
    ax.set_ylabel("Generating Model", fontsize=12)


def _parameter_panel(ax, recovery, model, parameter, colors, labeled) -> None:
    rows = recovery.parameters.query("generator == @model and parameter == @parameter")
    for mouse, group in rows.groupby("mouse"):
        label = None if mouse in labeled else mouse
        labeled.add(mouse)
        ax.scatter(group["true"], group["estimate"], s=22, color=colors[mouse], label=label)
    low, high = (np.nanmin(rows[["true", "estimate"]]), np.nanmax(rows[["true", "estimate"]]))
    ax.plot([low, high], [low, high], color="0.6", lw=0.8, ls=":", zorder=1)
    ax.set_xlabel(f"Generating {_parameter_name(parameter)}", fontsize=12)
    ax.set_ylabel(f"Recovered {_parameter_name(parameter)}", fontsize=12)
    stats = f"per simulated mouse, n={rows['mouse'].nunique()} mice"
    set_title(ax, Title(f"{MODEL_LABELS[model]} {_parameter_name(parameter)}", stats), fontsize=12)
    despine(ax)


def plot_strategy_model_recovery(recovery, save_path: str | Path | None = None) -> None:
    """Confusion matrix of model recovery, and recovery of v, u, the Q rate and p_recent.

    Args:
        recovery (Recovery): :func:`src.pipelines.strategy_comparison.recover_models`.
        save_path (str | Path | None): Base path without extension; ``None`` shows the figure.

    Raises:
        ValueError: If no simulated mouse was compared.
    """
    if recovery.won.empty:
        raise ValueError("no simulated mice to plot")
    colors = mouse_color_map(sorted(recovery.won["mouse"].unique()))
    n_mice = recovery.won["mouse"].nunique()
    caption = (
        f"Each of {n_mice} mice's fitted models played that mouse's own problems "
        f"({recovery.replicates} simulated mouse per model and mouse), and the four models "
        "were compared on those simulated mice by the identical held-out procedure. Left: "
        "fraction of each generating model's simulated mice won by each model (diagonal: "
        "recovered), with counts. Other panels: parameter fitted to the simulated mouse against "
        "the parameter that generated it (dotted line: identity), one dot per simulated mouse "
        "in the color of the real mouse it was fitted to.",
        *recovery_statement(recovery),
    )
    with new_figure(nrows=1, ncols=1 + len(RECOVERED), figsize=(20.0, 4.6), squeeze=False) as (
        fig,
        axs,
    ):
        _confusion_panel(axs[0, 0], recovery)
        stats = f"n={recovery.won.shape[0]} simulated mice"
        set_title(axs[0, 0], Title("Model Recovery", stats), fontsize=12)
        labeled: set = set()
        for ax, (model, parameter) in zip(axs[0, 1:], RECOVERED, strict=True):
            _parameter_panel(ax, recovery, model, parameter, colors, labeled)
        handles, names = axs[0, 1].get_legend_handles_labels()  # the mice, named once
        axs[0, -1].legend(handles, names, loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False,
                         fontsize=9)  # fmt: skip
        finish_figure(fig, Title(STRATEGY_MODEL_RECOVERY, None, caption))
    save_or_show(fig, save_path or None)


def plot_strategy_models(sources: Mapping, out_dir: str | Path, recovery=None) -> None:
    """Save both strategy-model figures into *out_dir* (the recovery one only with *recovery*).

    Args:
        sources (Mapping): ``{source label: SourceComparison}``, the mice first.
        out_dir (str | Path): Folder of the figures.
        recovery (Recovery | None): Model recovery. Default ``None``.
    """
    out_dir = Path(out_dir)
    plot_strategy_model_comparison(sources, out_dir / STRATEGY_MODEL_COMPARISON, recovery)
    if recovery is not None:
        plot_strategy_model_recovery(recovery, out_dir / STRATEGY_MODEL_RECOVERY)


__all__ = [
    "FIGURE_NAMES",
    "STRATEGY_MODEL_COMPARISON",
    "STRATEGY_MODEL_RECOVERY",
    "plot_strategy_model_comparison",
    "plot_strategy_model_recovery",
    "plot_strategy_models",
    "recovery_statement",
]
