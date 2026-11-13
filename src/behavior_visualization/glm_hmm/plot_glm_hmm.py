"""Figures of the GLM-HMM: model selection, state weights, occupancy, recovery, kernel by state.

Drawing only: the numbers come from :mod:`src.pipelines.glm_hmm.glm_hmm` (a :class:`FigureData`
bundles
the mice's results, the replays' null and the simulations' recoveries). One dot is one mouse in its
own color; an agent fitted to a mouse has that mouse's color. A black bar is the mean ± se across
mice, drawn from ``MIN_TEST_MICE`` mice; with fewer, dots only. States are numbered in the order
of their value weight (State 1 has the largest), not by an interpretation. Each figure is saved as
pdf, png and svg with a ``.caption.txt``; the long text (definitions, per-mouse numbers, the
calibrated z and p) is in the caption.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

from src.agents import glm_hmm_agent as gha
from src.behavior_analysis.choice_logit import choice_logit as cl
from src.behavior_analysis.choice_logit import choice_logit_extensions as ext
from src.behavior_analysis.data.stats import nan_standard_error
from src.behavior_analysis.glm_hmm import glm_hmm_kernel as ker
from src.behavior_analysis.strategy_models import strategy_models as sm
from src.behavior_visualization.choice_logit.plot_regression_extensions import (
    colors_of,
    source_stats_line,
)
from src.behavior_visualization.common.plot_helpers import (
    Title,
    despine,
    jitter_rng,
    jittered,
    save_or_show,
    set_title,
)
from src.behavior_visualization.value.plot_value_psychometrics import finish_figure, new_figure
from src.pipelines.glm_hmm import glm_hmm as pipe

MODEL_SELECTION = "GLM-HMM Model Selection"
STATE_WEIGHTS = "GLM-HMM State Weights"
STATE_OCCUPANCY = "GLM-HMM State Occupancy"
SIMULATED_RECOVERY = "GLM-HMM Simulated Recovery"
KERNEL_BY_STATE = "Choice Kernel by State"
FIGURE_NAMES = (
    MODEL_SELECTION,
    STATE_WEIGHTS,
    STATE_OCCUPANCY,
    SIMULATED_RECOVERY,
    KERNEL_BY_STATE,
)

# The Okabe-Ito colors that are not used for mice first: one per state number.
STATE_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
MEAN_BAR = dict(color="black", lw=2, zorder=5)
FEATURE_LABELS = {
    "q": "Q Value",
    "habit": "Choice Kernel",
    "stay_win": "Stay After Reward",
    "stay_magnitude": "Stay Per Extra Pulse",
    "stay_zero": "Stay After Zero",
    "switch_trace": "Switch To Recent",
    "plus_bias": "Plus-Port Bias",
}
SCENARIO_LABELS = {
    "switch_slow": "Switching, Slow",
    "switch_fast": "Switching, Fast",
    "static_kernel_q": "Static Q-Learner",
    "static_wsls_habit": "Static WSLS (Habit)",
    "static_blend": "Static Blend",
}
READOUT_LABELS = {
    "gain_k2": "2 States",
    "gain_k3": "3 States",
    "best_gain": "Best of 2 or 3",
}


class FigureData(NamedTuple):
    """Everything the figures draw.

    Attributes:
        mice (Mapping[str, pipe.MouseResult]): ``{mouse: result}``.
        null (pd.DataFrame | None): ``pipeline.replay_source`` output, or ``None``.
        calibration (pipe.Calibration | None): The mice's gains against the null, or ``None``.
        recovery (Sequence[pipe.Recovery]): The simulated agents' analyses (may be empty).
        replicates (int): Replays per mouse behind *null*.
        colors (Mapping[str, str] | None): ``{mouse: color}``; default one per mouse of *mice*.
    """

    mice: Mapping
    null: pd.DataFrame | None = None
    calibration: pipe.Calibration | None = None
    recovery: Sequence = ()
    replicates: int = 0
    colors: Mapping | None = None


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def _colors(data: FigureData) -> dict[str, str]:
    if data.colors is not None:
        return dict(data.colors)
    return colors_of([pd.DataFrame({"mouse": sorted(data.mice)})])


def _se(values) -> float:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    return float(nan_standard_error(values[:, None])[0]) if len(values) > 1 else np.nan


def _mean_bar(ax, x: float, values) -> float:
    """Black mean ± se of *values* at *x* (nothing below ``MIN_TEST_MICE`` values); the se."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) < ext.MIN_TEST_MICE:
        return np.nan
    se = _se(values)
    ax.errorbar([x], [values.mean()], yerr=[se], fmt="_", markersize=14, capsize=4, **MEAN_BAR)
    return se


def _dots(ax, x: float, values: pd.Series, colors: dict, rng, hollow=()) -> float:
    """One dot per mouse of *values* (indexed by mouse) around *x*, and the mean bar; the se."""
    values = values.dropna()
    xs = jittered(np.full(len(values), float(x)), 0.14, rng)
    for xi, (mouse, y) in zip(xs, values.items(), strict=True):
        filled = mouse not in hollow
        ax.scatter([xi], [y], s=34, zorder=3, facecolor=colors[mouse] if filled else "white",
                   edgecolor="none" if filled else colors[mouse], linewidth=1.3)  # fmt: skip
    return _mean_bar(ax, x, values.to_numpy())


def _stats(n_by_column: Sequence[int], errors: Sequence[float]) -> str:
    return source_stats_line(list(n_by_column), list(errors))


def _mice_legend(ax, colors: dict, mice) -> None:
    handles = [Line2D([], [], marker="o", ls="", color=colors[m], label=m) for m in sorted(mice)]
    ax.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False, fontsize=9
    )


def _state_legend(ax, n_states: int, title: str | None = None) -> None:
    handles = [
        Line2D([], [], color=STATE_COLORS[k], lw=2.4, label=f"State {k + 1}")
        for k in range(n_states)
    ]
    ax.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False, fontsize=9,
        title=title,
    )  # fmt: skip


def _state_curve(ax, curves: dict[int, pd.DataFrame], color: str, label: str) -> bool:
    """Mean ± se across mice (columns of *curves*[state]) of one state's curve; True if se drawn."""
    wide = curves
    drawn = wide.shape[1] >= ext.MIN_TEST_MICE
    mean = wide.mean(axis=1)
    if drawn:
        se = wide.apply(lambda row: _se(row.dropna().to_numpy()), axis=1)
        ax.fill_between(wide.index, mean - se, mean + se, color=color, alpha=0.2, lw=0)
    ax.plot(wide.index, mean, color=color, lw=2, label=label)
    return drawn


def _per_mouse(n_mice: int) -> str:
    return f"per mouse, n={n_mice} mice"


def _line_stats(n_mice: int, errors_drawn: bool) -> str:
    return f"mean ± se across mice, n={n_mice} mice" if errors_drawn else _per_mouse(n_mice)


def _state_wide(frames: Mapping[str, pd.DataFrame], column: str, value: str) -> dict:
    """``{state: DataFrame (index column, one column per mouse)}`` from per-mouse long tables."""
    out: dict = {}
    for mouse, frame in frames.items():
        for state, group in frame.groupby("state"):
            out.setdefault(int(state), {})[mouse] = group.set_index(column)[value]
    return {s: pd.DataFrame(cols).sort_index() for s, cols in sorted(out.items())}


# ---------------------------------------------------------------------------
# Captions
# ---------------------------------------------------------------------------


def _mouse_line(mouse: str, r: pipe.MouseResult) -> str:
    gains = ", ".join(f"{k.removeprefix('gain_k')} states {v:+.4f}" for k, v in r.readouts.items()
                      if k.startswith("gain_k"))  # fmt: skip
    static = ", ".join(
        f"{pipe.MODEL_LABELS[row.model]} {row.nats:.4f}" for row in r.static_levels.itertuples()
    )
    one = r.table.set_index("n_states").loc[1, "nats"]
    return (
        f"{mouse}: {r.n_trials} trials in {r.n_sessions} sessions; selected {r.selected_k} "
        f"state(s); held-out gain over 1 state (nats per trial): {gains}; 1 state {one:.4f} "
        f"nats per trial, {static}."
    )


def _calibration_lines(data: FigureData) -> list[str]:
    if data.calibration is None:
        return []
    lines = [
        f"Null: {data.replicates} replays per mouse of the kernel + forgetting Q-learner fitted to "
        "that mouse, played on its own problems and analyzed identically (the same signals, "
        "folds and state counts). z = (observed gain - replay mean) / replay sd; p = two-sided "
        "empirical (1 + #extreme) / (n + 1); positive z means more held-out gain from states than "
        "a static agent produces."
    ]
    for row in data.calibration.summary.itertuples():
        label = READOUT_LABELS[row.readout]
        stats = (
            f"z mean {row.mean:.2f} ± se {row.se:.2f}, Wilcoxon p={row.p_wilcoxon:.3g}"
            if np.isfinite(row.mean)
            else "z not summarized (fewer than 6 mice)"
        )
        lines.append(f"Gain with {label}: {stats}; mice with p < 0.05 and z > 0 ({row.n_pass}): "
                     f"{row.passing or 'none'}.")  # fmt: skip
    return lines


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

STATIC_ORDER = (sm.KERNEL_Q, sm.WSLS_HABIT)


def _gain_table(data: FigureData) -> pd.DataFrame:
    """``mouse, column, gain``: gain over one state of each state count and static model."""
    rows = []
    for mouse, r in data.mice.items():
        table = r.table.set_index("n_states")
        rows += [dict(mouse=mouse, column=f"states_{k}", gain=g) for k, g in table["gain"].items()]
        static = r.static_levels.set_index("model")["nats"]
        rows += [
            dict(mouse=mouse, column=m, gain=table.loc[1, "nats"] - static[m]) for m in STATIC_ORDER
        ]
    return pd.DataFrame(rows)


def _null_band(data: FigureData, k: int) -> tuple[float, float] | None:
    """Mean over mice of the replays' 2.5th and 97.5th percentile of the gain of *k* states."""
    if data.calibration is None:
        return None
    table = data.calibration.table
    rows = table[table["readout"] == f"gain_k{k}"]
    if rows.empty:
        return None
    return float(rows["null_low"].mean()), float(rows["null_high"].mean())


def _selection_panel(ax, data: FigureData, colors: dict, rng) -> list[float]:
    gains = _gain_table(data)
    ks = sorted({int(c[7:]) for c in gains["column"] if c.startswith("states_")})
    positions = {f"states_{k}": float(k) for k in ks} | {
        m: len(ks) + 1.4 + i for i, m in enumerate(STATIC_ORDER)
    }
    for k in ks:
        band = _null_band(data, k)
        if band is not None:
            ax.fill_between([k - 0.4, k + 0.4], *band, color="0.85", lw=0, zorder=1)
    wide = gains.pivot(index="mouse", columns="column", values="gain")
    for mouse, row in wide.iterrows():
        xs = [positions[f"states_{k}"] for k in ks]
        ax.plot(
            xs, row[[f"states_{k}" for k in ks]], color=colors[mouse], lw=0.8, alpha=0.5, zorder=2
        )
    errors = []
    for column, x in positions.items():
        errors.append(_dots(ax, x, wide[column], colors, rng))
    ax.axhline(0, color="0.7", lw=0.8, zorder=0)
    ticks = [f"{k} State{'s' if k > 1 else ''}" for k in ks] + [
        pipe.MODEL_LABELS[m] for m in STATIC_ORDER
    ]
    ax.set_xticks(list(positions.values()), ticks, rotation=25, ha="right", fontsize=9)
    ax.set(ylabel="Held-Out Gain Over 1 State (Nats per Trial)")
    despine(ax)
    return errors


def _z_panel(ax, data: FigureData, colors: dict, rng) -> list[float] | None:
    if data.calibration is None:
        return None
    table = data.calibration.table
    errors = []
    for x, readout in enumerate(READOUT_LABELS):
        z = table[table["readout"] == readout].set_index("mouse")["z"]
        errors.append(_dots(ax, x, z, colors, rng))
    ax.axhline(0, color="0.7", lw=0.8, zorder=0)
    ax.axhline(1.96, color="0.7", lw=0.8, ls="--", zorder=0)
    ax.set_xticks(range(len(READOUT_LABELS)), list(READOUT_LABELS.values()), fontsize=9)
    ax.set(ylabel="z of Gain Against Replays")
    despine(ax)
    return errors


def _selected_panel(ax, data: FigureData, colors: dict) -> None:
    counts: dict[int, int] = {}
    for mouse, r in sorted(data.mice.items()):
        level = counts.get(r.selected_k, 0)
        counts[r.selected_k] = level + 1
        ax.scatter([r.selected_k], [level + 0.5], s=70, color=colors[mouse], zorder=3)
    ks = sorted({k for r in data.mice.values() for k in r.table["n_states"]})
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.set_xlim(min(ks) - 0.6, max(ks) + 0.6)
    ax.set_ylim(0, max(counts.values()) + 0.5)
    ax.set(xlabel="Selected States", ylabel="Mice")
    despine(ax)


def plot_model_selection(data: FigureData, save_path=None) -> None:
    """Held-out gain by state count with the replay null, its z, and the selected counts."""
    colors, rng, n = _colors(data), jitter_rng(), len(data.mice)
    caption = (
        "GLM-HMMs with 1 to 4 states, each state a conditional logit on the agent signals (Q "
        "value, choice kernel, stay after reward, stay per extra pulse, stay after zero, switch "
        "to the recently chosen, plus-port bias), scored on whole held-out sessions (5 folds, the "
        "sessions of each problem dealt across them) by the forward-filter predictive log "
        "probability of each trial with a previous choice. Left: gain = held-out nats per trial "
        "of 1 state minus those of the model, positive when it predicts better; the gray band is "
        f"the mean over mice of the 2.5th to 97.5th percentile of the gain in {data.replicates} "
        "replays per mouse of its fitted static kernel + forgetting Q-learner; the last two "
        "columns are the static kernel Q-learner and WSLS (Habit), refitted on the same folds "
        "(their nats are compared with those of 1 state: positive means they predict better than "
        "the signals with no states). Middle: z of each mouse's gain against its own replays "
        "(dashed: 1.96). Right: the state count chosen by the smallest-within-one-standard-error "
        "rule, one dot per mouse.",
        *_calibration_lines(data),
        *[_mouse_line(m, r) for m, r in sorted(data.mice.items())],
    )
    with new_figure(nrows=1, ncols=3, figsize=(16.5, 4.8)) as (fig, axs):
        errors = _selection_panel(axs[0], data, colors, rng)
        set_title(axs[0], Title("Gain by States", _stats([n], errors)), fontsize=11)
        z_errors = _z_panel(axs[1], data, colors, rng)
        if z_errors is None:
            axs[1].set_visible(False)
        else:
            set_title(axs[1], Title("Gain Against Replays", _stats([n], z_errors)), fontsize=11)
        _selected_panel(axs[2], data, colors)
        set_title(axs[2], Title("Selected States", _per_mouse(n)), fontsize=11)
        _mice_legend(axs[2], colors, data.mice)
        finish_figure(fig, Title(MODEL_SELECTION, None, caption))
    save_or_show(fig, save_path or None)


# ---------------------------------------------------------------------------
# State weights
# ---------------------------------------------------------------------------


def _summary_weights(data: FigureData) -> pd.DataFrame:
    """``mouse, state, feature, weight, selected``: the weights at each mouse's summary count."""
    frames = []
    for mouse, r in data.mice.items():
        w = r.weights[r.weights["n_states"] == r.summary_k]
        frames.append(w.assign(mouse=mouse, selected=r.selected_k >= 2))
    return pd.concat(frames, ignore_index=True)


def plot_state_weights(data: FigureData, save_path=None) -> None:
    """The weight of each signal in each state, one panel per signal, one dot per mouse."""
    colors, rng = _colors(data), jitter_rng()
    weights = _summary_weights(data)
    n_states = int(weights["state"].max()) + 1
    hollow = {m for m, r in data.mice.items() if r.selected_k < 2}
    caption = (
        "Weights of the state-specific conditional logit at each mouse's chosen number of states "
        "(at least 2: a mouse whose held-out score chose 1 state is drawn hollow, its 2-state fit "
        "being descriptive only). States are numbered by their weight on the Q value, State 1 the "
        "largest, so State 1 is the most value-like state of each mouse; a state number is not a "
        "shared identity across mice. A weight is a logit change per unit of the signal: Q value "
        "(pulses), choice kernel (0 to 1), stay signals (0/1, or pulses beyond the first), the "
        "centered choice trace of the two other arms, and the plus-port indicator. "
        "A value-like state weights Q value; a WSLS-like state weights the stay signals and the "
        "switch trace; a habit/biased state weights the choice kernel and the plus-port bias.",
        *[
            f"{m}: {r.summary_k} states, occupancy "
            + ", ".join(f"{o:.2f}" for o in r.occupancy["occupancy"])
            + (" (hollow: 1 state was selected)" if r.selected_k < 2 else "")
            for m, r in sorted(data.mice.items())
        ],
    )
    n_panels = len(gha.FEATURES)
    n_columns = 4
    n_rows = -(-n_panels // n_columns)
    with new_figure(nrows=n_rows, ncols=n_columns, figsize=(4.2 * n_columns + 1.5, 4.2 * n_rows),
                    squeeze=False) as (fig, axs):  # fmt: skip
        for ax, feature in zip(axs.flat, gha.FEATURES, strict=False):
            counts, errors = [], []
            for k in range(n_states):
                values = weights[(weights["feature"] == feature) & (weights["state"] == k)]
                series = values.set_index("mouse")["weight"]
                errors.append(_dots(ax, k, series, colors, rng, hollow))
                counts.append(len(series))
            ax.axhline(0, color="0.8", lw=0.8, zorder=0)
            ax.set_xticks(range(n_states), [f"State {k + 1}\nn={c}" for k, c in enumerate(counts)])
            ax.set(ylabel="Weight (Logits per Unit)")
            despine(ax)
            set_title(ax, Title(FEATURE_LABELS[feature], _stats(counts, errors)), fontsize=11)
        for ax in axs.flat[n_panels:]:
            ax.set_visible(False)
        _mice_legend(axs[0, -1], colors, data.mice)
        finish_figure(fig, Title(STATE_WEIGHTS, None, caption))
    save_or_show(fig, save_path or None)


# ---------------------------------------------------------------------------
# Occupancy
# ---------------------------------------------------------------------------


def _occupancy_panel(ax, data: FigureData, colors: dict, rng, column: str):
    counts, errors = [], []
    states = sorted({int(s) for r in data.mice.values() for s in r.occupancy["state"]})
    for k in states:
        series = pd.Series(
            {m: r.occupancy.set_index("state")[column].get(k, np.nan) for m, r in data.mice.items()}
        )
        errors.append(_dots(ax, k, series, colors, rng))
        counts.append(int(series.notna().sum()))
    ax.set_xticks(states, [f"State {k + 1}\nn={c}" for k, c in zip(states, counts, strict=True)])
    despine(ax)
    return counts, errors


def _curve_panel(ax, frames: dict, column: str, n_mice: int) -> str:
    """Per-state mean ± se across mice of the mice's posterior curves; the panel's stats line."""
    wide = _state_wide(frames, column, "probability")
    drawn = [_state_curve(ax, w, STATE_COLORS[k], f"State {k + 1}") for k, w in wide.items()]
    despine(ax)
    ax.set(ylim=(0, 1))
    return _line_stats(n_mice, any(drawn))


def plot_state_occupancy(data: FigureData, save_path=None) -> None:
    """Occupancy, dwell times, and state probability by trial in session, problem and block."""
    colors, rng, n = _colors(data), jitter_rng(), len(data.mice)
    caption = (
        "Posterior state probabilities of the GLM-HMM at each mouse's chosen number of states (at "
        "least 2). Occupancy: mean posterior of the state over the mouse's trials. Dwell time: "
        "1 / (1 - the state's self-transition probability), in trials, under a sticky Dirichlet "
        "prior (concentration 2, plus 20 on the diagonal); states restart from the initial "
        "distribution at each session start, so a dwell longer than a session is carried by "
        "that distribution. Trial in session: mean posterior in bins of 10 trials. Problem: the "
        "mouse's problems in order. Block change: trials relative to the first trial of a new "
        "block within a session. Lines: mean ± se across the mice that have the state, States "
        "numbered by their weight on the Q value.",
        *[
            f"{m}: selected {r.selected_k} state(s), shown with {r.summary_k}; dwell (trials) "
            + ", ".join(f"{d:.0f}" for d in r.occupancy["dwell"])
            for m, r in sorted(data.mice.items())
        ],
    )
    results = data.mice
    with new_figure(nrows=2, ncols=3, figsize=(16.5, 9.2)) as (fig, axs):
        counts, errors = _occupancy_panel(axs[0, 0], data, colors, rng, "occupancy")
        axs[0, 0].set(ylabel="Fraction of Trials", ylim=(0, 1))
        set_title(axs[0, 0], Title("Occupancy", _stats(counts, errors)), fontsize=11)
        counts, errors = _occupancy_panel(axs[0, 1], data, colors, rng, "dwell")
        axs[0, 1].set(ylabel="Dwell Time (Trials)", yscale="log")
        set_title(axs[0, 1], Title("Dwell Time", _stats(counts, errors)), fontsize=11)
        stats = _curve_panel(axs[0, 2], {m: r.by_trial_in_session for m, r in results.items()},
                             "trial_in_session", n)  # fmt: skip
        axs[0, 2].set(xlabel="Trial in Session", ylabel="State Probability")
        set_title(axs[0, 2], Title("State by Trial in Session", stats), fontsize=11)
        stats = _curve_panel(axs[1, 0], {m: r.by_problem for m, r in results.items()},
                             "problem_number", n)  # fmt: skip
        axs[1, 0].set(xlabel="Problem", ylabel="State Probability")
        set_title(axs[1, 0], Title("State by Problem", stats), fontsize=11)
        stats = _curve_panel(axs[1, 1], {m: r.block_aligned for m, r in results.items()},
                             "offset", n)  # fmt: skip
        axs[1, 1].axvline(0, color="0.8", lw=0.8, zorder=0)
        axs[1, 1].set(xlabel="Trial From Block Change", ylabel="State Probability")
        set_title(axs[1, 1], Title("State Around Block Changes", stats), fontsize=11)
        axs[1, 2].set_visible(False)
        n_states = max(r.summary_k for r in results.values())
        _state_legend(axs[1, 1], n_states)
        finish_figure(fig, Title(STATE_OCCUPANCY, None, caption))
    save_or_show(fig, save_path or None)


# ---------------------------------------------------------------------------
# Simulated recovery
# ---------------------------------------------------------------------------


def _recovery_columns(data: FigureData) -> list[str]:
    present = {r.scenario for r in data.recovery}
    return [s for s in pipe.SCENARIOS if s in present]


def _scenario_dots(ax, data: FigureData, colors: dict, rng, value) -> list:
    """Dots of ``value(recovery)`` per scenario; returns (n, se) per scenario."""
    out = []
    for x, scenario in enumerate(_recovery_columns(data)):
        series = pd.Series({r.mouse: value(r) for r in data.recovery if r.scenario == scenario})
        series = series.dropna()
        out.append((len(series), _dots(ax, x, series, colors, rng)))
    labels = [
        f"{SCENARIO_LABELS[s]}\nn={c}"
        for s, (c, _) in zip(_recovery_columns(data), out, strict=True)
    ]
    ax.set_xticks(range(len(out)), labels, rotation=20, ha="right", fontsize=9)
    despine(ax)
    return out


def _identity_scatter(ax, table: pd.DataFrame, colors: dict, label: str) -> None:
    for mouse, group in table.groupby("mouse"):
        ax.scatter(
            group["true"], group["recovered"], s=22, color=colors[mouse], alpha=0.8, zorder=3
        )
    lim = [table[["true", "recovered"]].min().min(), table[["true", "recovered"]].max().max()]
    pad = 0.05 * (lim[1] - lim[0] or 1.0)
    ax.plot([lim[0] - pad, lim[1] + pad], [lim[0] - pad, lim[1] + pad], color="0.6", lw=1, zorder=1)
    ax.set(xlabel=f"True {label}", ylabel=f"Recovered {label}")
    despine(ax)


def plot_simulated_recovery(data: FigureData, save_path=None) -> None:
    """State count, gain, weights, state accuracy and dwell times recovered from simulations."""
    if not data.recovery:
        raise ValueError("no simulated recoveries to draw")
    colors, rng = _colors(data), jitter_rng()
    n = len({r.mouse for r in data.recovery})
    weights = pd.concat([r.weights.assign(mouse=r.mouse) for r in data.recovery if len(r.weights)])
    dwell = pd.concat([r.dwell.assign(mouse=r.mouse) for r in data.recovery if len(r.dwell)])
    caption = (
        "Agents simulated on the problems of the mice named in the legend (their fitted kernel Q "
        "rates make the signals) and analyzed exactly like a mouse. Switching: three states, a "
        "value learner (the mouse's fitted rate and inverse temperature), WSLS (Magnitude) (its "
        "fitted stay and switch weights) and a habit/biased strategy (choice kernel weight "
        f"{gha.HABIT_WEIGHT}, plus-port bias {gha.HABIT_PLUS_BIAS}); sticky transitions with a "
        f"mean dwell of {pipe.SLOW_DWELL} (slow) or {pipe.FAST_DWELL} (fast) trials, the state "
        "redrawn uniformly at every session start. No switching: the mouse's fitted kernel Q "
        "agent, its fitted WSLS (Habit) agent, and one agent whose weights are the mean of the "
        "three strategies on every trial. Selected states: smallest count within one standard "
        "error of the best held-out score. Gain: best held-out gain over 1 state among 2 and 3 "
        "states (nats per trial). Weights, accuracy and dwell: the 3-state fit, its states "
        "matched one to one to the true states by weight distance. State accuracy: share of "
        "trials whose most probable fitted state is the matched true state.",
        *[
            f"{r.mouse}, {SCENARIO_LABELS[r.scenario]}: selected {r.readouts['selected_k']}, "
            f"best gain {r.readouts['best_gain']:+.4f}"
            + (f", state accuracy {r.accuracy:.2f}" if np.isfinite(r.accuracy) else "")
            + "."
            for r in data.recovery
        ],
    )
    with new_figure(nrows=2, ncols=3, figsize=(16.5, 9.4)) as (fig, axs):
        out = _scenario_dots(axs[0, 0], data, colors, rng, lambda r: r.readouts["selected_k"])
        axs[0, 0].set(ylabel="Selected States")
        axs[0, 0].set_yticks(sorted({r.readouts["selected_k"] for r in data.recovery}))
        set_title(
            axs[0, 0],
            Title("Selected States", _stats([c for c, _ in out], [e for _, e in out])),
            fontsize=11,
        )
        out = _scenario_dots(axs[0, 1], data, colors, rng, lambda r: r.readouts["best_gain"])
        axs[0, 1].axhline(0, color="0.7", lw=0.8, zorder=0)
        axs[0, 1].set(ylabel="Held-Out Gain Over 1 State (Nats per Trial)")
        set_title(
            axs[0, 1],
            Title("Gain by Scenario", _stats([c for c, _ in out], [e for _, e in out])),
            fontsize=11,
        )
        _identity_scatter(axs[0, 2], weights, colors, "Weight")
        set_title(axs[0, 2], Title("Weights, 3-State Fit", _per_mouse(n)), fontsize=11)
        out = _scenario_dots(axs[1, 0], data, colors, rng, lambda r: r.accuracy)
        axs[1, 0].set(ylabel="State Accuracy", ylim=(0, 1.05))
        set_title(
            axs[1, 0],
            Title("Posterior State Accuracy", _stats([c for c, _ in out], [e for _, e in out])),
            fontsize=11,
        )
        _identity_scatter(axs[1, 1], dwell, colors, "Dwell (Trials)")
        axs[1, 1].set(xscale="log", yscale="log")
        for axis in (axs[1, 1].xaxis, axs[1, 1].yaxis):
            axis.set_minor_formatter(NullFormatter())  # overlapping minor labels
        set_title(axs[1, 1], Title("Dwell Times, 3-State Fit", _per_mouse(n)), fontsize=11)
        axs[1, 2].set_visible(False)
        _mice_legend(axs[1, 1], colors, {r.mouse for r in data.recovery})
        finish_figure(fig, Title(SIMULATED_RECOVERY, None, caption))
    save_or_show(fig, save_path or None)


# ---------------------------------------------------------------------------
# Choice kernel by state
# ---------------------------------------------------------------------------


def _kernel_curves(tables: Mapping[str, pd.DataFrame]) -> dict:
    """``{group: DataFrame (lag x mouse)}`` for ``"all"`` and each state number."""
    out: dict = {}
    for mouse, table in tables.items():
        for state, group in table.groupby("state"):
            if group["weight"].notna().any():
                out.setdefault(state, {})[mouse] = group.set_index("lag")["weight"]
    order = sorted(out, key=lambda s: -1 if s == "all" else s)
    return {s: pd.DataFrame(out[s]).sort_index() for s in order}


def _replay_kernels(data: FigureData) -> dict[str, pd.DataFrame]:
    """``{mouse: kernel table averaged over the replays that have one}``."""
    if data.null is None or "kernel" not in data.null.columns:
        return {}
    out = {}
    for mouse, group in data.null.groupby("mouse"):
        tables = [t for t in group["kernel"] if isinstance(t, pd.DataFrame)]
        if tables:
            out[mouse] = (
                pd.concat(tables).groupby(["state", "lag"], as_index=False)["weight"].mean()
            )
    return out


def _kernel_panel(ax, curves: dict, n_mice: int) -> str:
    drawn = []
    for state, wide in curves.items():
        color = "black" if state == "all" else STATE_COLORS[int(state)]
        label = "All Trials" if state == "all" else f"State {int(state) + 1}"
        drawn.append(_state_curve(ax, wide, color, label))
    ax.axhline(0, color="0.8", lw=0.8, zorder=0)
    ax.set(xlabel="Lag (Trials Back)", ylabel="Choice Weight (Logits)")
    ax.set_xticks(range(1, cl.DEFAULT_LAGS + 1))
    despine(ax)
    return _line_stats(n_mice, any(drawn))


def _flatness_panel(ax, mice: Mapping, replays: Mapping, colors: dict, rng) -> tuple:
    columns = [("Mice", mice), ("Agents", replays)]
    groups = ["all", 0, 1]
    ticks, counts, errors, x = [], [], [], 0
    for source, tables in columns:
        for group in groups:
            values = {}
            for mouse, table in tables.items():
                weights = table[table["state"] == group].sort_values("lag")["weight"].to_numpy()
                if len(weights) and np.isfinite(weights).all():
                    values[mouse] = ker.flatness(weights)
            series = pd.Series(values, dtype=float)
            errors.append(_dots(ax, x, series, colors, rng))
            counts.append(int(series.notna().sum()))
            name = "All" if group == "all" else f"State {group + 1}"
            ticks.append(f"{source}: {name}\nn={counts[-1]}")
            x += 1
        x += 0.6
    positions = [i + 0.6 * (i // len(groups)) for i in range(len(ticks))]
    ax.axhline(1.0, color="0.8", lw=0.8, zorder=0)
    ax.axhline(0.0, color="0.8", lw=0.8, zorder=0)
    ax.set_xticks(positions, ticks, rotation=30, ha="right", fontsize=8)
    # A ratio explodes when the lag-1 weight is near zero, so a symmetric log axis (linear
    # within +-1) keeps the typical values readable next to the outliers.
    ax.set(ylabel="Kernel Flatness (Lags 3-5 / Lag 1)", yscale="symlog")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_yticks([0, 1, 10], ["0", "1", "10"])
    ax.set_ylim(bottom=-0.5)
    despine(ax)
    return counts, errors


def plot_choice_kernel_by_state(data: FigureData, save_path=None) -> None:
    """The choice-history kernel within each state, for mice and for static-agent replays."""
    colors, rng, n = _colors(data), jitter_rng(), len(data.mice)
    mice_kernels = {m: r.kernel for m, r in data.mice.items()}
    replay_kernels = _replay_kernels(data)
    caption = (
        "The mice's choice logit (5 reward lags, 5 choice lags, tower biases; ridge) refitted "
        "within the trials of each GLM-HMM state (posterior above 0.8; a state with fewer than "
        f"{ker.MIN_TRIALS} such trials has no fit), at each mouse's chosen number of states (at "
        "least 2). Panels show the five choice-lag weights: a kernel that decays after lag 1 is "
        "sharp; a flat one persists. Flatness is the mean of the lag 3 to 5 weights over the lag 1 "
        "weight (1 flat, 0 only the last choice; undefined if lag 1 is not positive). Agents: the "
        "mouse's fitted static kernel + forgetting Q-learner replayed on its problems and given "
        "the same states-by-posterior analysis (the first replays only), a reference for what a "
        "static agent's kernel looks like inside the states of a GLM-HMM. If the flat kernel of "
        "the pooled trials came from a mixture of sharp kernels, the state curves would be sharper "
        "than the pooled one.",
    )
    with new_figure(nrows=1, ncols=3, figsize=(16.5, 4.8)) as (fig, axs):
        stats = _kernel_panel(axs[0], _kernel_curves(mice_kernels), n)
        set_title(axs[0], Title("Kernel of Mice", stats), fontsize=11)
        if replay_kernels:
            stats = _kernel_panel(axs[1], _kernel_curves(replay_kernels), len(replay_kernels))
            set_title(axs[1], Title("Kernel of Static Agents", stats), fontsize=11)
        else:
            axs[1].set_visible(False)
        counts, errors = _flatness_panel(axs[2], mice_kernels, replay_kernels, colors, rng)
        set_title(axs[2], Title("Flatness", _stats(counts, errors)), fontsize=11)
        _state_legend(axs[0], max(r.summary_k for r in data.mice.values()))
        _mice_legend(axs[2], colors, data.mice)
        finish_figure(fig, Title(KERNEL_BY_STATE, None, caption))
    save_or_show(fig, save_path or None)


def plot_all(data: FigureData, out_dir: str | Path) -> None:
    """Save every figure of the GLM-HMM analysis under *out_dir*.

    Args:
        data (FigureData): Results of the mice, the null, the calibration and the simulations.
        out_dir (str | Path): Folder for the figures (created if needed).
    """
    out_dir = Path(out_dir)
    plots = [
        (MODEL_SELECTION, plot_model_selection),
        (STATE_WEIGHTS, plot_state_weights),
        (STATE_OCCUPANCY, plot_state_occupancy),
        (KERNEL_BY_STATE, plot_choice_kernel_by_state),
    ]
    if data.recovery:
        plots.append((SIMULATED_RECOVERY, plot_simulated_recovery))
    for name, plot in plots:
        plot(data, out_dir / name)
