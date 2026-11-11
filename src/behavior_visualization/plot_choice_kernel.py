"""Figures of persistence: the choice kernel to a long lag, and P(stay) by run length.

Drawing only: the numbers come from :mod:`src.pipelines.choice_kernel_long` and
:mod:`src.behavior_analysis.stay_by_run_length`. Both figures show the mice (one thin line each,
the mean ± se in black once ``choice_logit_extensions.MIN_TEST_MICE`` mice are there) against the
agents (one mean line each). A source with fewer mice shows its mice only, with no mean or error.
``mice_named=False`` is for simulated "mice" that are not fitted to any mouse: gray, unnamed.
"""

from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.behavior_analysis import choice_kernel_timescale as ts
from src.behavior_analysis import choice_logit as cl
from src.behavior_analysis import choice_logit_extensions as ext
from src.behavior_analysis import stay_by_run_length as rl
from src.behavior_analysis.stats import nan_mean, nan_standard_error
from src.behavior_visualization.plot_choice_logit import kernel_matrix, mice_stats
from src.behavior_visualization.plot_helpers import (
    Title,
    despine,
    error_drawn,
    mouse_color_map,
    save_or_show,
    set_title,
)
from src.behavior_visualization.plot_null_agents import source_colors
from src.behavior_visualization.plot_regression_extensions import OTHER_MICE_COLOR
from src.behavior_visualization.plot_value_psychometrics import finish_figure, new_figure

CHOICE_KERNEL_LONG = "Choice Kernel Long"
STAY_BY_RUN_LENGTH = "Stay by Run Length"
FIGURE_NAMES = (CHOICE_KERNEL_LONG, STAY_BY_RUN_LENGTH)
PAYOUT_HEADINGS = {"zero": "After Zero Payout", "non-zero": "After Non-Zero Payout"}


def _check(sources: Mapping, empty) -> None:
    if not sources:
        raise ValueError("no sources to plot")
    missing = [name for name, s in sources.items() if empty(s)]
    if missing:
        raise ValueError(f"no data for {missing}")


def draw_source_curves(
    ax: plt.Axes,
    x: np.ndarray,
    matrices: Mapping[str, pd.DataFrame],
    *,
    mice_named: bool = True,
) -> list[np.ndarray]:
    """Draw each source's curves over *x* and return the standard errors drawn.

    The first source is the mice: one thin line per mouse. Every source with at least
    ``MIN_TEST_MICE`` mice also gets a mean line with its se band (black for the mice, a palette
    color for the others); a source with fewer shows thin lines only.

    Args:
        ax (plt.Axes): The panel.
        x (np.ndarray): Positions of the columns of every matrix.
        matrices (Mapping[str, pd.DataFrame]): ``{source: mice-by-x values}``.
        mice_named (bool): Color the first source's mice individually (else gray). Default True.

    Returns:
        list[np.ndarray]: The se of every source with a mean drawn.
    """
    palette = source_colors(list(matrices))
    errors = []
    for k, (name, matrix) in enumerate(matrices.items()):
        thin = (
            mouse_color_map(sorted(matrix.index))
            if (k == 0 and mice_named)
            else dict.fromkeys(matrix.index, OTHER_MICE_COLOR if k == 0 else palette[name])
        )
        for mouse, values in matrix.iterrows():
            ax.plot(x, values, "-o", color=thin[mouse], lw=1, ms=2.5, alpha=0.4, zorder=2)
        label = f"{name} (n={len(matrix)})"
        if len(matrix) < ext.MIN_TEST_MICE:
            ax.plot([], [], "-o", color=thin[matrix.index[0]], lw=1, ms=2.5, label=label)
            continue
        mean, se = nan_mean(matrix.to_numpy(), axis=0), nan_standard_error(matrix.to_numpy())
        color = "black" if k == 0 else palette[name]
        if error_drawn(se):
            ax.fill_between(x, mean - se, mean + se, color=color, alpha=0.2, lw=0, zorder=3)
        ax.plot(x, mean, "-o", color=color, lw=2.2, ms=4.5, label=label, zorder=5)
        errors.append(se)
    return errors


def _frameless_legend(ax: plt.Axes) -> None:
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False, fontsize=9)


# ---------------------------------------------------------------------------
# Choice Kernel Long
# ---------------------------------------------------------------------------


def _kernel_lines(sources: Mapping) -> list[str]:
    """Each source's kernel at lags 1, 5 and the last, its decay and what the tail adds."""
    lines = []
    for name, s in sources.items():
        matrix = kernel_matrix(s.weights, cl.CHOICE)
        lags = matrix.columns.to_numpy(dtype=int)
        mean = matrix.mean(axis=0)
        at = ", ".join(f"lag {k} {mean[k]:.2f}" for k in (1, 5, int(lags.max())) if k in lags)
        decay = ts.fit_decay(s.weights)
        wide = s.comparison.pivot(index="mouse", columns="model", values="nats_per_choice")
        gain = (wide["full"] - wide["full + choice tail"]).dropna()
        se = gain.std(ddof=1) / np.sqrt(len(gain)) if len(gain) > 1 else np.nan
        error = f" ± se {se:.4f}" if np.isfinite(se) else ""
        lines.append(
            f"{name} (n={len(matrix)} mice): mean choice weight {at}; geometric decay "
            f"{decay['decay'].mean():.2f}, timescale {decay['tau'].median():.1f} trials (median); "
            f"held-out nats per choice without the tail {wide['full'].mean():.4f}, with it "
            f"{wide['full + choice tail'].mean():.4f}; gain from the tail {gain.mean():.4f}{error} "
            f"across mice (positive: the tail helps), helps in {int((gain > 0).sum())} of "
            f"{len(gain)}."
        )
    return lines


def _recovery_panel(ax: plt.Axes, recovery: pd.DataFrame) -> None:
    colors = source_colors([f"{a:g}" for a in sorted(recovery["alpha_k"].unique())])
    for alpha_k, rows in recovery.groupby("alpha_k"):
        color = colors[f"{alpha_k:g}"]
        low = rows["fitted_mean"] - rows["fitted_se"]
        high = rows["fitted_mean"] + rows["fitted_se"]
        ax.fill_between(rows["lag"], low, high, color=color, alpha=0.2, lw=0)
        ax.plot(rows["lag"], rows["fitted_mean"], "-o", color=color, lw=2, ms=4,
                label=f"Alpha K {alpha_k:g}, Fitted")  # fmt: skip
        ax.plot(
            rows["lag"], rows["true"], "--", color=color, lw=1.5, label=f"Alpha K {alpha_k:g}, True"
        )
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set_xticks(sorted(recovery["lag"].unique()))  # lags are whole trials
    ax.set_xlabel("Lag (Trials Back)", fontsize=12)
    ax.set_ylabel("Choice Weight (Logit)", fontsize=12)
    n_mice = int(recovery["n_mice"].max())
    set_title(ax, Title("Known-Kernel Recovery", mice_stats(n_mice, recovery["fitted_se"])),
              fontsize=12)  # fmt: skip
    _frameless_legend(ax)
    despine(ax)


def plot_choice_kernel_long(
    sources: Mapping,
    save_path: str | Path | None = None,
    recovery: pd.DataFrame | None = None,
    *,
    mice_named: bool = True,
) -> None:
    """The choice kernel to a long lag for the mice and the agents, and a known-kernel recovery.

    Args:
        sources (Mapping): ``{label: LongKernel}``, the mice first, then the agents fitted to the
            same mice (each analyzed with the identical long-lag design).
        save_path (str | Path | None): Base path without extension; ``None`` shows the figure.
        recovery (pd.DataFrame | None): :func:`src.pipelines.choice_kernel_long.
            recover_choice_kernel`; drawn as a second panel. Default ``None``.
        mice_named (bool): Color the first source's mice individually; ``False`` for simulated
            mice that are not fitted to a mouse. Default True.

    Raises:
        ValueError: If there are no sources or a source has no weights.
    """
    _check(sources, lambda s: s.weights.empty)
    first = next(iter(sources.values()))
    matrices = {name: kernel_matrix(s.weights, cl.CHOICE) for name, s in sources.items()}
    lags = matrices[next(iter(matrices))].columns.to_numpy(dtype=float)
    caption = (
        f"Weights of the choice logit with {first.n_lags} reward lags and {first.n_lags} choice "
        f"lags (L2 {first.l2:g}), fitted to all of each source's trials; the weight on an arm's "
        "logit if it was chosen k trials back. Thin lines: single mice; black: mean ± se across "
        "mice (shown from "
        f"{ext.MIN_TEST_MICE} mice); colored lines: the mean of each agent fitted to the same "
        "mice and analyzed with the same design.",
        f"A second model adds one regressor for the choices older than the lags (an exponentially "
        f"weighted choice indicator, decay {first.choice_tail_decay:g}, half-life about "
        f"{np.log(0.5) / np.log(first.choice_tail_decay):.0f} trials); both models are scored "
        "on held-out problems.",
        "Geometric decay: amplitude * decay**(k - 1) fitted to a mouse's weights; the timescale "
        "is -1 / ln(decay) trials. A flat kernel has a decay near 1.",
        "Recovery (right panel, if shown): agents that choose by a choice kernel alone (known "
        "rate alpha_k, weight beta_k) played the mice's own schedules and went through the "
        "same fit; dashed: the weights the kernel implies, solid: those recovered.",
        *_kernel_lines(sources),
    )
    n_panels = 1 + (recovery is not None)
    with new_figure(ncols=n_panels, figsize=(5.4 * n_panels + 2.0, 4.4), squeeze=False) as (
        fig,
        axs,
    ):
        ax = axs[0, 0]
        errors = draw_source_curves(ax, lags, matrices, mice_named=mice_named)
        ax.axhline(0, color="0.7", lw=0.8)
        ax.set_xticks(lags)
        ax.set_xlabel("Lag (Trials Back)", fontsize=12)
        ax.set_ylabel("Choice Weight (Logit)", fontsize=12)
        n_mice = max(len(m) for m in matrices.values())
        set_title(ax, Title(f"Choice Kernel to Lag {int(lags.max())}", mice_stats(n_mice, *errors)),
                  fontsize=12)  # fmt: skip
        _frameless_legend(ax)
        despine(ax)
        if recovery is not None:
            _recovery_panel(axs[0, 1], recovery)
        finish_figure(fig, Title(CHOICE_KERNEL_LONG, None, caption))
    save_or_show(fig, save_path or None)


# ---------------------------------------------------------------------------
# Stay by Run Length
# ---------------------------------------------------------------------------


def _stay_matrix(table: pd.DataFrame, payout: str) -> pd.DataFrame:
    rows = table[table["payout"] == payout]
    return rows.pivot(index="mouse", columns="bin", values="p_stay")[list(rl.BIN_NAMES)]


def _slope_lines(sources: Mapping) -> list[str]:
    lines = []
    for name, table in sources.items():
        slopes = rl.run_length_slopes(table)
        for payout, heading in PAYOUT_HEADINGS.items():
            values = slopes.loc[slopes["payout"] == payout, "slope"].dropna()
            if values.empty:
                lines.append(f"{name}, {heading.lower()}: no slope (too few bins).")
                continue
            se = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else np.nan
            error = f" ± se {se:.3f}" if np.isfinite(se) else ""
            lines.append(
                f"{name}, {heading.lower()}: slope of P(stay) per bin {values.mean():.3f}{error} "
                f"across mice (n={len(values)} mice)."
            )
    return lines


def plot_stay_by_run_length(
    sources: Mapping[str, pd.DataFrame],
    save_path: str | Path | None = None,
    *,
    mice_named: bool = True,
) -> None:
    """P(stay) against run length after a zero and after a non-zero payout, mice and agents.

    Args:
        sources (Mapping[str, pd.DataFrame]): ``{label: stay_by_run_length table of that source's
            mice, concatenated}``, the mice first, then the agents replayed on the same task.
        save_path (str | Path | None): Base path without extension; ``None`` shows the figure.
        mice_named (bool): Color the first source's mice individually; ``False`` for simulated
            mice that are not fitted to a mouse. Default True.

    Raises:
        ValueError: If there are no sources or a source has no rows.
    """
    _check(sources, lambda t: t.empty)
    caption = (
        "A run is the consecutive trials at one port within a session; the run length is that "
        "of the run ending at the previous trial, and a decision is a stay if the next choice "
        f"repeats the port. Bins: {', '.join(rl.BIN_NAMES)} trials. A bin with fewer than "
        f"{rl.MIN_PAIRS} decisions of a mouse is left out. Non-zero payouts are pooled over "
        "magnitudes, so the magnitude mix of long runs (a port is kept longer when it pays "
        "well) is in the mice and the agents alike, which replay the same task.",
        "Thin lines: single mice; black: mean ± se across mice (from "
        f"{ext.MIN_TEST_MICE} mice); colored lines: the mean of each agent fitted to the "
        "same mice.",
        "Slope: least-squares change of P(stay) per bin step, per mouse.",
        *_slope_lines(sources),
    )
    x = np.arange(len(rl.BIN_NAMES))
    with new_figure(ncols=2, figsize=(11.0, 4.4), sharey=True) as (fig, axs):
        for ax, (payout, heading) in zip(axs, PAYOUT_HEADINGS.items(), strict=True):
            matrices = {name: _stay_matrix(t, payout) for name, t in sources.items()}
            errors = draw_source_curves(ax, x, matrices, mice_named=mice_named)
            ax.set_xticks(x, rl.BIN_NAMES)
            ax.set_xlabel("Run Length (Trials at the Port)", fontsize=12)
            n_mice = max(len(m) for m in matrices.values())
            set_title(ax, Title(heading, mice_stats(n_mice, *errors)), fontsize=12)
            despine(ax)
        axs[0].set_ylabel("P(Stay)", fontsize=12)
        _frameless_legend(axs[1])
        finish_figure(fig, Title(STAY_BY_RUN_LENGTH, None, caption))
    save_or_show(fig, save_path or None)


def plot_choice_kernel_figures(
    long_sources: Mapping,
    stay_sources: Mapping[str, pd.DataFrame],
    out_dir: str | Path,
    recovery: pd.DataFrame | None = None,
    mice_named: bool = True,
) -> None:
    """Save both persistence figures into *out_dir*.

    Args:
        long_sources (Mapping): As :func:`plot_choice_kernel_long`.
        stay_sources (Mapping[str, pd.DataFrame]): As :func:`plot_stay_by_run_length`.
        out_dir (str | Path): Folder of the figures.
        recovery (pd.DataFrame | None): The known-kernel recovery. Default ``None``.
        mice_named (bool): As the two figures. Default True.
    """
    out_dir = Path(out_dir)
    plot_choice_kernel_long(long_sources, out_dir / CHOICE_KERNEL_LONG, recovery,
                            mice_named=mice_named)  # fmt: skip
    plot_stay_by_run_length(stay_sources, out_dir / STAY_BY_RUN_LENGTH, mice_named=mice_named)


__all__ = [
    "CHOICE_KERNEL_LONG",
    "FIGURE_NAMES",
    "STAY_BY_RUN_LENGTH",
    "draw_source_curves",
    "plot_choice_kernel_figures",
    "plot_choice_kernel_long",
    "plot_stay_by_run_length",
]
