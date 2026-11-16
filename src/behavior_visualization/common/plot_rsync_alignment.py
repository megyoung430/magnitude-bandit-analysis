"""Diagnostic figure for an rsync alignment between two recording systems."""

from collections.abc import Sequence

import matplotlib.pyplot as plt

from src.sync.rsync import RsyncAligner

_MATCH_COLOR = "tab:blue"
_NON_MATCH_COLOR = "tab:red"


def _new_axes() -> tuple[plt.Axes, plt.Axes, plt.Axes]:
    """Axes of a new figure: two small panels above one wide panel."""
    fig = plt.figure(figsize=(7, 9), layout="constrained")
    grid = fig.add_gridspec(3, 3)
    return (
        fig.add_subplot(grid[0, :2]),
        fig.add_subplot(grid[0, 2]),
        fig.add_subplot(grid[1:, :]),
    )


def plot_rsync_alignment(
    aligner: RsyncAligner, axes: Sequence[plt.Axes] | None = None
) -> Sequence[plt.Axes]:
    """Plot how well the pulse trains of *aligner* matched.

    Three panels: the histogram of log mean squared error for chunks labelled match and
    non-match (the two should separate cleanly), the histogram of inter-pulse interval
    discrepancies between the systems, and the matched pulse times of B against those of A
    (a straight line, with gaps where pulses are missing).

    Args:
        aligner (RsyncAligner): A fitted aligner.
        axes (Sequence[plt.Axes] | None): Three axes to draw into (error histogram,
            discrepancy histogram, matched times). Default None, meaning a new figure.

    Returns:
        Sequence[plt.Axes]: The three axes drawn into.

    Raises:
        ValueError: If *axes* does not hold exactly three axes.
    """
    if axes is None:
        axes = _new_axes()
    if len(axes) != 3:
        raise ValueError(f"plot_rsync_alignment needs three axes, got {len(axes)}.")
    ax_mse, ax_discrepancy, ax_times = axes

    log_mse, matched = aligner.log_mse, aligner.valid_matches
    ax_mse.hist(log_mse[matched], 20, color=_MATCH_COLOR, label="Match")
    ax_mse.hist(log_mse[~matched], 20, color=_NON_MATCH_COLOR, label="Non-Match")
    ax_mse.legend(loc="upper center")
    ax_mse.set(xlabel="Log mean squared error", ylabel="# chunks")

    ax_discrepancy.hist(aligner.interval_discrepancy_ms(), 100)
    ax_discrepancy.set_yscale("log", nonpositive="clip")
    ax_discrepancy.set(xlabel="Inter-pulse interval\ndiscrepancy (ms)", ylabel="# pulses")

    ax_times.plot(aligner.pulse_times_A, aligner.cor_times_B, ".", markersize=2)
    ax_times.set_xlim(aligner.pulse_times_A[0], aligner.pulse_times_A[-1])
    ax_times.set(xlabel="pulse times A", ylabel="pulse times B")
    return axes
