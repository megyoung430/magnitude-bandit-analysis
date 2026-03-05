"""Visualise non-choice long-poke counts across sessions and subjects.

A "long poke" is an exploratory poke into a port that is not the animal's
final choice for that trial.  This module provides a session-by-session bar
chart with per-mouse overlaid lines.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from src.behavior_import.sort_by_session import sort_sessions_chronologically, get_short_session_label
from src.behavior_visualization.plot_style import MOUSE_COLORS


def plot_num_pokes_across_mice(avg_num_pokes_across_subjects, y_text, save_path=None):
    """Plot mean non-choice long-poke counts per session with per-mouse overlays.

    Draws a bar for each session (mean ± SE across mice) and overlays
    individual mouse lines with jitter.  Sessions are sorted by numeric
    ``ses-XX`` index; x-tick labels show the short ``ses-XX`` form.

    Args:
        avg_num_pokes_across_subjects: Nested dict
            ``{mouse: {session_key: mean_pokes_float}}``, optionally including
            an ``"average_across_subjects"`` key which is excluded from the
            per-mouse overlay.
        y_text: Y-coordinate for bar-value annotations and the upper y-limit
            of the axis.
        save_path: Base path (without extension) for saving ``.pdf`` and
            ``.png`` output.  If ``None``, the figure is shown interactively.
    """

    mice = sorted([m for m in avg_num_pokes_across_subjects.keys()
               if m != "average_across_subjects"])
    if len(mice) == 0:
        raise ValueError("avg_num_pokes_across_subjects has no mice.")

    # union of all session keys across mice
    all_sessions = set()
    for m in mice:
        all_sessions.update(avg_num_pokes_across_subjects.get(m, {}).keys())

    if len(all_sessions) == 0:
        raise ValueError("No sessions found in avg_num_pokes_across_subjects.")

    # sort sessions by ses-XX numeric index
    sessions = sorted(all_sessions, key=sort_sessions_chronologically)

    # per-session mean and SE across mice
    per_session_mean = []
    per_session_se = []
    per_session_n = []

    for s in sessions:
        vals = []
        for m in mice:
            if s in avg_num_pokes_across_subjects.get(m, {}):
                v = avg_num_pokes_across_subjects[m][s]
                if v is not None and np.isfinite(v):
                    vals.append(float(v))

        vals = np.asarray(vals, dtype=float)
        n = len(vals)
        per_session_n.append(n)

        if n == 0:
            per_session_mean.append(np.nan)
            per_session_se.append(np.nan)
        else:
            per_session_mean.append(np.mean(vals))
            if n >= 2:
                per_session_se.append(np.std(vals, ddof=1) / np.sqrt(n))
            else:
                per_session_se.append(0.0)

    per_session_mean = np.asarray(per_session_mean, dtype=float)
    per_session_se = np.asarray(per_session_se, dtype=float)
    per_session_n = np.asarray(per_session_n, dtype=int)

    per_mouse_series = {}
    for m in mice:
        y = np.full(len(sessions), np.nan, dtype=float)
        md = avg_num_pokes_across_subjects.get(m, {})
        for i, s in enumerate(sessions):
            if s in md and md[s] is not None and np.isfinite(md[s]):
                y[i] = float(md[s])
        per_mouse_series[m] = y

    x = np.arange(len(sessions))
    fig, ax = plt.subplots(figsize=(12, 5.5))

    ax.bar(x, per_session_mean, yerr=per_session_se, capsize=6, color="#999999", alpha=0.55, edgecolor="black", linewidth=1.5, zorder=1)

    for xi, m in zip(x, per_session_mean):
        if np.isfinite(m):
            ax.text(xi, y_text, f"{m:.2f}", ha="center", va="bottom", fontsize=12, color="black", zorder=10)

    mouse_to_color = {m: MOUSE_COLORS[i % len(MOUSE_COLORS)] for i, m in enumerate(mice)}

    legend_handles = []
    rng = np.random.default_rng()

    for m in mice:
        c = mouse_to_color[m]
        y = per_mouse_series[m]

        mask = np.isfinite(y)
        if not np.any(mask):
            continue

        xs = x[mask].astype(float)
        ys = y[mask].astype(float)

        xs = xs + rng.uniform(-0.03, 0.03, size=len(xs))

        ax.plot(xs, ys, color=c, linewidth=2.5, alpha=0.9, marker="o", markersize=6, zorder=4)

        legend_handles.append(Line2D([0], [0], color=c, lw=2.5, marker="o", markersize=6, label=m))

    ax.set_xlabel("Session")
    ax.set_ylabel("Non-Choice Long Pokes")

    ax.set_xticks(x)
    ax.set_xticklabels([get_short_session_label(s) for s in sessions], fontsize=12, rotation=45)

    ax.tick_params(axis="y", labelsize=13)
    ax.set_ylim(0, y_text)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    title = f"Mean Non-Choice Long Pokes Across Sessions\n (mean ± se across subjects | n={len(mice)} subjects)"
    ax.set_title(title, fontsize=16, pad=25)

    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right", frameon=True, fontsize=10)

    plt.tight_layout()

    if save_path:
        fig.savefig(str(save_path) + ".pdf", bbox_inches="tight")
        fig.savefig(str(save_path) + ".png", dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close(fig)
