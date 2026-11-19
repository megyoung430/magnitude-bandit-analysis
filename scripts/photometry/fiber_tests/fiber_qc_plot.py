"""Summary figures for the fiber/rotary-joint tests: the four measures that decide between
combinations, and the mouse traces on a common scale so the steps can be seen directly."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fiber_qc import DATA, load

OUT = Path("/home/claude/fiber/out")
ORDER = ["R2 F0", "R2 F1", "R2 F2", "R2 F3", "R2 F4", "R2 F5", "R1 F4"]
MOUSE_COLORS = {"RT_176": "#0072B2", "RT_65": "#D55E00"}
MOUSE_LABELS = {"RT_176": "RT_176_L", "RT_65": "RT_65"}
POOR_COUPLING_PCT = 10  # brain fraction below this means the attachment barely reached the brain


def style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
        "svg.fonttype": "none", "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 110, "savefig.dpi": 200,
    })


def per_mouse_points(ax, mice: pd.DataFrame, column: str, ylabel: str, title: str,
                     log: bool = False) -> None:
    """One point per mouse recording per combination; hollow where coupling was poor."""
    for mouse, color in MOUSE_COLORS.items():
        rows = mice[mice["kind"] == mouse]
        x = [ORDER.index(c) + (-0.12 if mouse == "RT_176" else 0.12) for c in rows["combo"]]
        poor = (100 * rows["signal_brain_fraction"] < POOR_COUPLING_PCT).to_numpy()
        values = rows[column].to_numpy()
        ax.scatter(np.array(x)[~poor], values[~poor], s=36, color=color,
                   label=MOUSE_LABELS[mouse], zorder=3)
        ax.scatter(np.array(x)[poor], values[poor], s=36, facecolor="white", edgecolor=color,
                   zorder=3)
    ax.axvline(len(ORDER) - 1.5, color="0.8", lw=0.8)
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels(ORDER, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log:
        ax.set_yscale("log")


def summary_figure(table: pd.DataFrame) -> None:
    mice = table[table["kind"].isin(MOUSE_COLORS)].copy()
    mice["brain_pct"] = 100 * mice["signal_brain_fraction"]
    mice["brain_mv_per_ma"] = 1000 * mice["signal_brain_v_per_ma"]
    plottable = mice[100 * mice["signal_brain_fraction"] >= POOR_COUPLING_PCT]

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4))
    per_mouse_points(axes[0], mice, "brain_mv_per_ma", "Brain Signal (mV Per mA LED)",
                     "Throughput", log=True)
    per_mouse_points(axes[1], mice, "brain_pct", "Brain-Derived Share Of Light (%)",
                     "Brain Fraction")
    per_mouse_points(axes[2], plottable, "corrected_sd_pct_of_brain",
                     "Residual SD (% Of Brain Light)", "Noise After Motion Correction")
    per_mouse_points(axes[3], mice, "signal_largest_step_pct", "Largest Step (% Of Signal)",
                     "Connection Steps")
    axes[1].set_ylim(0, None)
    axes[2].set_ylim(0, None)
    axes[3].set_ylim(0, None)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=MOUSE_LABELS[m])
               for m, c in MOUSE_COLORS.items()]
    handles.append(plt.Line2D([], [], marker="o", ls="", markerfacecolor="white",
                              markeredgecolor="0.3", label="Poor Coupling"))
    axes[-1].legend(handles=handles, frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.suptitle("Fiber And Rotary Joint Comparison\none point per mouse recording, n=2 mice "
                 "per combination", y=1.05)
    fig.tight_layout()
    save(fig, "fiber_qc_summary")
    (OUT / "fiber_qc_summary.caption.txt").write_text(
        "Comparison of rotary-joint (R) and patch-cord (F) combinations tested on 13 Nov 2026, "
        "each recorded with the tip in air, with paper touched to the tip, and on two implanted "
        "mice (RT_176_L and RT_65; F5 used RT_65R and F4 on R2 is labeled RT_65). R2 ran at "
        "100/100 mA and R1 at 16/20 mA. The gray line separates the two rotary joints; R1 was "
        "tested with F4 only, so F4 is the one cord seen through both.\n"
        "Background is the tip-in-air level, i.e. light generated in the patch cord and joint "
        "rather than in the brain. Because the cord bleaches by 1.3-2.8% per minute, background at "
        "the time of each mouse recording is interpolated between the tip-in-air measurements "
        "either side of it where both exist, and taken from the nearest otherwise.\n"
        "Throughput: brain-derived signal (mouse minus background) per mA of LED current, log "
        "axis. Brain fraction: the share of recorded 465 light that comes from the animal. Noise "
        "after motion correction: SD of the 465 channel after a quadratic detrend and regression "
        "on the 405 channel, as a percentage of brain-derived light, which is the ΔF/F noise floor "
        "once background is subtracted. Connection steps: the largest change across a 1 s window "
        "as a percentage of the total signal.\n"
        f"Hollow points are recordings where under {POOR_COUPLING_PCT}% of the light came from "
        "the brain (R2 F2 and R2 F3 on RT_176_L), which looks like a poorly seated connection on "
        "that attachment rather than a property of the cord; they are left out of the noise "
        "panel, where dividing by a near-zero brain signal is meaningless.", encoding="utf-8")


def traces_figure() -> None:
    """Every mouse recording as % change from its median, on one shared scale."""
    recordings = [load(p) for p in sorted(DATA.glob("*.ppd"))]
    mice = [r for r in recordings if r["kind"] in MOUSE_COLORS]
    fig, axes = plt.subplots(len(ORDER), 2, figsize=(13, 1.55 * len(ORDER)), sharey=True,
                             squeeze=False)
    for rec in mice:
        row = ORDER.index(rec["combo"])
        col = 0 if rec["kind"] == "RT_176" else 1
        ax = axes[row, col]
        t = np.arange(len(rec["signal"])) / rec["fs"]
        for channel, color in (("control", "#999999"), ("signal", "#009E73")):
            x = rec[channel]
            ax.plot(t, 100 * (x / np.median(x) - 1), lw=0.5, color=color,
                    label="405 Control" if channel == "control" else "465 Signal")
        ax.set_title(f"{rec['combo']}  {rec['label']}", fontsize=9, loc="left")
        ax.set_xlim(0, 150)
    for ax in axes[:, 0]:
        ax.set_ylabel("Change (%)")
    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")
    axes[0, 0].set_ylim(-9, 9)
    axes[0, -1].legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle("Mouse Recordings On A Common Scale\nraw signal as % change from each "
                 "recording's median, n=2 mice per combination", y=1.01)
    fig.tight_layout()
    save(fig, "fiber_qc_traces")
    (OUT / "fiber_qc_traces.caption.txt").write_text(
        "Every mouse recording from the 13 Nov 2026 fiber tests, shown as the raw 465 signal "
        "(green) and 405 control (gray) expressed as percentage change from that recording's own "
        "median, on one shared y-axis and the first 150 s, so the size of steps and drift can be "
        "compared directly across combinations. Rows are rotary joint (R) and patch cord (F); "
        "the left column is RT_176_L and the right RT_65 (RT_65R for F5). The percentages are of "
        "total light including the patch-cord background, so they understate changes relative to "
        "brain-derived light by roughly the brain fraction.", encoding="utf-8")


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png", "svg"):
        fig.savefig(OUT / f"{name}.{extension}", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    style()
    table = pd.read_csv("fiber_qc_per_recording.csv")
    summary_figure(table)
    traces_figure()


if __name__ == "__main__":
    main()
