"""Figures of the Q-learner with kernel and forgetting: ablations, parameters and kernel decay.

Drawing only: the numbers come from :mod:`src.pipelines.strategy_comparison`. A source with fewer
than ``choice_logit_extensions.MIN_TEST_MICE`` mice shows its mice only, with no mean, error bar
or test. Simulated mice that are not fitted to a mouse (``mice_named=False``) are gray and unnamed.
"""

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from src.agents import kernel_q_agent as kq
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
    OTHER_MICE_COLOR,
    draw_column,
    source_stats_line,
    style_source_axis,
)
from src.behavior_visualization.plot_strategy_models import (
    check_sources,
    draw_panel,
    skipped_lines,
    source_colors,
    source_ticks,
)
from src.behavior_visualization.plot_value_psychometrics import finish_figure, new_figure
from src.pipelines import strategy_comparison as pipe

KERNEL_Q_ABLATIONS = "Kernel Q Ablations"
KERNEL_Q_PARAMETERS = "Kernel Q Parameters"
CHOICE_KERNEL_DECAY = "Choice Kernel Decay"
FIGURE_NAMES = (KERNEL_Q_ABLATIONS, KERNEL_Q_PARAMETERS, CHOICE_KERNEL_DECAY)
ABLATION_LABELS = {
    f"kernel_q_{name}": variant.label for name, variant in kq.VARIANTS.items() if name != "full"
}
PARAMETER_LABELS = {
    "alpha_pos": "Alpha Pos (Better Than Expected)",
    "alpha_neg": "Alpha Neg (Worse Than Expected)",
    "zeta": "Zeta (Retention of Unchosen Values)",
    "alpha_k": "Alpha K (Kernel Rate)",
    "beta": "Beta (Value Weight)",
    "beta_k": "Beta K (Kernel Weight)",
}
PANEL_COLUMNS = 3
# What each ablation removes, for captions.
ABLATION_TEXT = {
    "no_asymmetry": "No Asymmetry: one learning rate (alpha_neg tied to alpha_pos)",
    "no_forgetting": "No Forgetting: zeta held at 1, unchosen values keep all of themselves",
    "no_kernel": "No Kernel: beta_k held at 0, no choice kernel (its rate is then unidentified)",
    "kernel_only": (
        "Kernel Only: one learning rate and no forgetting but a free multi-trial kernel, the "
        "value agent with its one-trial stickiness replaced by the kernel"
    ),
}


def _ablation_lines(sources: Mapping) -> list[str]:
    lines = []
    for row in pipe.ablation_summaries(sources).itertuples():
        ablation, _, _ = row.comparison.partition("_minus_")
        if np.isfinite(row.mean):
            lines.append(
                f"{row.source}, {ABLATION_LABELS[ablation]} minus full: mean {row.mean:.4f} ± se "
                f"{row.se:.4f} nats per choice across mice (n={row.n} mice); the ablation fits "
                f"worse in {row.n_positive}; Wilcoxon p={row.p_wilcoxon:.3g}."
            )
    return lines


def plot_kernel_q_ablations(sources: Mapping, save_path: str | Path | None = None) -> None:
    """Held-out nats of each ablation minus the full kernel Q-learner, per mouse.

    Positive: removing that piece makes held-out choices less predictable (the piece helps).

    Args:
        sources (Mapping): ``{source label: SourceComparison}``, the mice first.
        save_path (str | Path | None): Base path without extension; ``None`` shows the figure.

    Raises:
        ValueError: If there are no sources, a source has no mice or no ablation scores.
    """
    check_sources(sources)
    colors, unnamed = source_colors(sources)
    ticks = source_ticks(sources)
    tables = {name: pipe.ablation_table(s).set_index("mouse") for name, s in sources.items()}
    ablations = sm.ABLATIONS
    caption = (
        "Each ablation is the full Q-learner (Kernel + Forgetting) with one piece removed, "
        "fitted on all but one problem and scored on the held-out one, on the same folds as "
        "the strategy comparison. Held-out nats per choice of the ablation minus the full "
        "model; positive: the removed piece helps.",
        *[f"{ABLATION_TEXT[a.removeprefix('kernel_q_')]}." for a in ablations],
        "Dots: mice, colored per mouse; gray: simulated mice not fitted to a mouse. Black bars: "
        f"mean ± se across mice (from {ext.MIN_TEST_MICE} mice).",
        *skipped_lines(sources),
        *_ablation_lines(sources),
    )

    def column(name, s, col):
        return tables[name][col]

    with new_figure(
        nrows=-(-len(ablations) // 2), ncols=2, figsize=(10.0, 4.3 * -(-len(ablations) // 2)),
        squeeze=False,
    ) as (fig, axs):  # fmt: skip
        rng = jitter_rng()
        for k, (ax, name) in enumerate(zip(axs.ravel(), ablations, strict=False)):
            ylabel = "Held-Out Nats per Choice, Ablation Minus Full" if k % 2 == 0 else ""
            draw_panel(ax, sources, sm.comparison_name(name, sm.KERNEL_Q), column, colors, unnamed,
                       rng, ticks, ylabel, f"{ABLATION_LABELS[name]} Minus Full")  # fmt: skip
        for ax in axs.ravel()[len(ablations) :]:
            ax.set_visible(False)
        finish_figure(fig, Title(KERNEL_Q_ABLATIONS, None, caption))
    save_or_show(fig, save_path or None)


def _parameter_lines(sources: Mapping, tables: Mapping) -> list[str]:
    lines = []
    for name, table in tables.items():
        for row in table.itertuples():
            values = ", ".join(f"{p} {getattr(row, p):.3f}" for p in kq.PARAMETER_NAMES)
            bound = f"; at a bound: {row.at_bounds}" if row.at_bounds else ""
            lines.append(f"{name}, {row.mouse}: {values}{bound}.")
    return lines


def plot_kernel_q_parameters(sources: Mapping, save_path: str | Path | None = None) -> None:
    """The fitted parameters of the kernel Q-learner, one dot per mouse.

    Args:
        sources (Mapping): ``{source label: SourceComparison}``, the mice first.
        save_path (str | Path | None): Base path without extension; ``None`` shows the figure.

    Raises:
        ValueError: If there are no sources or a source has no mice.
    """
    check_sources(sources)
    colors, unnamed = source_colors(sources)
    ticks = source_ticks(sources)
    tables = {name: pipe.kernel_parameters_table(s) for name, s in sources.items()}
    indexed = {name: t.set_index("mouse") for name, t in tables.items()}
    caption = (
        "Maximum-likelihood parameters of the Q-learner (Kernel + Forgetting) fitted to all of "
        "each mouse's trials (several starts, L-BFGS-B, exact gradient): learning rates for "
        "outcomes better (alpha_pos) and worse (alpha_neg) than expected, retention of the "
        "unchosen values (zeta; 1 is no forgetting), kernel rate (alpha_k; 1 is a one-trial "
        "kernel), value weight (beta), kernel weight (beta_k). Bounds: rates 0.001-1, zeta 0-1, "
        "beta 0.001-20, beta_k -20-20; a parameter at a bound is listed below.",
        "Dots: mice, colored per mouse; gray: simulated mice not fitted to a mouse. Black bars: "
        f"mean ± se across mice (from {ext.MIN_TEST_MICE} mice).",
        *skipped_lines(sources),
        *_parameter_lines(sources, tables),
    )

    def column(name, s, col):
        return indexed[name][col]

    n_rows = -(-len(PARAMETER_LABELS) // PANEL_COLUMNS)
    with new_figure(
        nrows=n_rows, ncols=PANEL_COLUMNS, figsize=(13.0, 4.3 * n_rows), squeeze=False
    ) as (fig, axs):
        rng = jitter_rng()
        for k, (ax, (name, label)) in enumerate(
            zip(axs.ravel(), PARAMETER_LABELS.items(), strict=True)
        ):
            ylabel = "Fitted Value" if k % PANEL_COLUMNS == 0 else ""
            draw_panel(ax, sources, name, column, colors, unnamed, rng, ticks, ylabel, label)
        finish_figure(fig, Title(KERNEL_Q_PARAMETERS, None, caption))
    save_or_show(fig, save_path or None)


# ---------------------------------------------------------------------------
# Choice Kernel Decay
# ---------------------------------------------------------------------------

DECAY_LABELS = {
    "kq_decay": "Q-Learner (Kernel + Forgetting)\n1 − Alpha K",
    "habit_decay": "WSLS (Habit)\n1 − Alpha K",
    "long_decay": "Choice Logit\nGeometric Decay",
}


def _profile_matrix(profile: pd.DataFrame) -> pd.DataFrame:
    """Mice by decay: held-out nats per choice minus that at the default decay of 0.9."""
    wide = profile.pivot(index="mouse", columns="decay", values="nats_per_choice")
    return wide.sub(wide[0.9], axis=0)


def _interpolate(x: float, decays: np.ndarray, values: np.ndarray) -> float:
    return float(np.interp(x, decays, values))


def _decay_lines(profile: pd.DataFrame, timescales: pd.DataFrame) -> list[str]:
    wide = profile.pivot(index="mouse", columns="decay", values="nats_per_choice")
    grid = wide.drop(columns=sm.NO_TRACE)
    best = grid.idxmin(axis=1)
    lines = [
        "Best decay of the grid per mouse: "
        + ", ".join(f"{m} {d:g}" for m, d in best.items())
        + f". Held-out nats per choice at the fixed 0.9 minus the best of the grid: "
        f"{(grid[0.9] - grid.min(axis=1)).mean():.4f} on average across mice.",
        "No-trace reference (uniform switches) minus 0.9: "
        f"{(wide[sm.NO_TRACE] - wide[0.9]).mean():.4f} on average across mice.",
    ]
    for row in timescales.itertuples():
        lines.append(
            f"{row.mouse}: kernel Q alpha_k {row.kq_alpha_k:.3f} (decay {row.kq_decay:.3f}); "
            f"WSLS (Habit) alpha_k {row.habit_alpha_k:.3f}; choice-logit decay "
            + (f"{row.long_decay:.3f} (timescale {row.long_tau:.1f} trials)."
               if np.isfinite(row.long_decay) else "not available.")
        )  # fmt: skip
    return lines


def plot_choice_kernel_decay(
    profile: pd.DataFrame,
    timescales: pd.DataFrame,
    save_path: str | Path | None = None,
    *,
    mice_named: bool = True,
) -> None:
    """Held-out nats of WSLS (Magnitude) against its trace decay, and the kernels' decays.

    Left: per mouse, held-out nats per choice minus that at the fixed decay of 0.9, against the
    trace decay (0: no trace, uniform switches); a dot marks the mouse's fitted ``1 - alpha_k`` of
    the kernel Q-learner on its curve. Right: each mouse's decay from three estimates, the
    kernel Q-learner and WSLS (Habit) fits and the geometric decay of the choice logit's lag
    weights.

    Args:
        profile (pd.DataFrame): :func:`src.pipelines.strategy_comparison.decay_profile_table`.
        timescales (pd.DataFrame): :func:`src.pipelines.strategy_comparison.timescale_table`.
        save_path (str | Path | None): Base path without extension; ``None`` shows the figure.
        mice_named (bool): Color the mice individually; ``False`` for simulated mice that are not
            fitted to a mouse (gray, unnamed). Default True.

    Raises:
        ValueError: If the profile or the timescales are empty.
    """
    if profile.empty or timescales.empty:
        raise ValueError("no decay profile or timescales to plot")
    matrix = _profile_matrix(profile)
    mice = sorted(matrix.index)
    colors = mouse_color_map(mice) if mice_named else dict.fromkeys(mice, OTHER_MICE_COLOR)
    decays = matrix.columns.to_numpy(dtype=float)
    caption = (
        "Left: WSLS (Magnitude) with its choice-trace decay fixed at each value on the x axis, "
        "fitted on all but one problem and scored on the held-out one (the folds of the strategy "
        "comparison); held-out nats per choice minus the value at the decay of 0.9 that the "
        "comparison uses, so lower than 0 means a decay fits better than 0.9. Decay 0 is the "
        "model with no trace (uniform switches). Dots: the mouse's fitted 1 - alpha_k of the "
        "Q-learner (Kernel + Forgetting), placed on its curve by linear interpolation.",
        "Right: the decay per trial of a mouse's choice kernel from three estimates: "
        "1 - alpha_k of the Q-learner (Kernel + Forgetting), 1 - alpha_k of WSLS (Habit), and the "
        "geometric decay fitted to the choice logit's long-lag weights. Black bars: mean ± se "
        f"across mice (from {ext.MIN_TEST_MICE} mice).",
        *_decay_lines(profile, timescales),
    )
    with new_figure(ncols=2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1.1, 1]}) as (
        fig,
        axs,
    ):
        ax = axs[0]
        for mouse, values in matrix.iterrows():
            ax.plot(decays, values, "-o", color=colors[mouse], lw=1.2, ms=3, alpha=0.7,
                    label=mouse if mice_named else None)  # fmt: skip
        fitted = timescales.set_index("mouse")["kq_decay"]
        for mouse, decay in fitted.items():
            if mouse in matrix.index:
                y = _interpolate(decay, decays, matrix.loc[mouse].to_numpy())
                ax.plot([decay], [y], "D", color=colors[mouse], mec="black", ms=7, zorder=5)
        ax.axhline(0, color="0.7", lw=0.8)
        ax.set_xlabel("Trace Decay of WSLS (Magnitude)", fontsize=12)
        ax.set_ylabel("Held-Out Nats per Choice, Minus Decay 0.9", fontsize=12)
        stats = f"per mouse, n={len(mice)} mice"
        set_title(ax, Title("Held-Out Fit by Trace Decay", stats), fontsize=12)
        if mice_named:
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False, fontsize=9)
        despine(ax)
        ax = axs[1]
        rng, labeled, errors, counts = jitter_rng(), set(), [], []
        for x, column in enumerate(DECAY_LABELS):
            values = timescales.set_index("mouse")[column].dropna()
            counts.append(len(values))
            errors.append(draw_column(ax, x, values, colors, rng, labeled))
        ticks = [f"{label}\nn={n}" for label, n in zip(DECAY_LABELS.values(), counts, strict=True)]
        style_source_axis(ax, ticks,
                          "Decay per Trial")  # fmt: skip
        ax.axhline(0.9, color="0.4", lw=0.8, ls=":")
        ax.set_ylim(0, 1.02)
        set_title(ax, Title("Kernel Decay by Estimate", source_stats_line(counts, errors)),
                  fontsize=12)  # fmt: skip
        despine(ax)
        finish_figure(fig, Title(CHOICE_KERNEL_DECAY, None, caption))
    save_or_show(fig, save_path or None)


def plot_kernel_q_figures(
    sources: Mapping, profile: pd.DataFrame, timescales: pd.DataFrame, out_dir: str | Path,
    mice_named: bool = True,
) -> None:  # fmt: skip
    """Save the three kernel Q figures into *out_dir*.

    Args:
        sources (Mapping): ``{source label: SourceComparison}``, the mice first.
        profile (pd.DataFrame): The WSLS decay profile of the first source's mice.
        timescales (pd.DataFrame): The kernel timescales of the first source's mice.
        out_dir (str | Path): Folder of the figures.
        mice_named (bool): As :func:`plot_choice_kernel_decay`. Default True.
    """
    out_dir = Path(out_dir)
    plot_kernel_q_ablations(sources, out_dir / KERNEL_Q_ABLATIONS)
    plot_kernel_q_parameters(sources, out_dir / KERNEL_Q_PARAMETERS)
    plot_choice_kernel_decay(profile, timescales, out_dir / CHOICE_KERNEL_DECAY,
                             mice_named=mice_named)  # fmt: skip


__all__ = [
    "CHOICE_KERNEL_DECAY",
    "FIGURE_NAMES",
    "KERNEL_Q_ABLATIONS",
    "KERNEL_Q_PARAMETERS",
    "plot_choice_kernel_decay",
    "plot_kernel_q_ablations",
    "plot_kernel_q_figures",
    "plot_kernel_q_parameters",
]
