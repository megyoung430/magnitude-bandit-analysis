"""Gaussian Process (GP) bandit task simulations.

Functions here generate synthetic 3-armed bandit task sessions where arm
reward values are sampled from Gaussian Processes, with block-swap dynamics
that mimic the experimental paradigm.

The public entry points are thin compositions of small pure helpers:

- :func:`sample_two_gp_bandits` — two GP reward streams (integer and raw).
- :func:`simulate_3armed_blockswaps_gp` — streams -> 3-armed session -> metrics.
- :func:`sweep_gp_metrics` — metric heatmaps over (max_block_len, sig).
- :func:`plot_gp_heatmaps_pair_1` — gap-metric heatmaps from a sweep.
- :func:`plot_gp_heatmaps_pair_2` — run-length heatmaps from a sweep.
- :func:`plot_example_session_gp` — simulate one session and draw it.
- :func:`collect_gp_run_stats` — mean and std of best-run length per sig.

Simulations consume the random generator in a fixed order, so a seeded
generator reproduces a session exactly.
"""

from __future__ import annotations

import operator
from collections.abc import Iterable
from functools import partial, reduce

import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    "collect_gp_run_stats",
    "plot_example_session_gp",
    "plot_gp_heatmaps_pair_1",
    "plot_gp_heatmaps_pair_2",
    "sample_two_gp_bandits",
    "simulate_3armed_blockswaps_gp",
    "sweep_gp_metrics",
]

# Default sweep grids, shared module-level arrays that are never mutated.
DEFAULT_SIG_VALUES = np.arange(5, 30, 1)
DEFAULT_MAX_BLOCK_LENS = np.arange(20, 81, 1)

# ---------------------------------------------------------------------------
# Two GP reward streams
# ---------------------------------------------------------------------------


def sample_two_gp_bandits(
    T: int,
    sig: float,
    scale: float = 2.0,
    shift: float = 2.5,
    clip: tuple[int, int] = (1, 5),
    jitter: float = 1e-6,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample two GP-driven reward streams of length *T*.

    Args:
        T: Number of time steps.
        sig: GP length-scale parameter (controls temporal smoothness).
        scale: Amplitude scaling of the GP samples (default: 2.0).
        shift: Mean shift applied to GP samples (default: 2.5).
        clip: ``(min, max)`` clipping range for integer reward values
            (default: ``(1, 5)``).
        jitter: Small diagonal regularisation added to the GP kernel for
            numerical stability (default: 1e-6).
        rng: Optional NumPy random generator.  Created fresh if not provided.

    Returns:
        ``(gp_int, gp_raw)`` — integer (clipped) and continuous (raw) reward
        arrays of shape ``(T, 2)``.
    """
    if rng is None:
        rng = np.random.default_rng()

    xs = rng.normal(0, 1, (T, 2))
    ts = np.arange(T)

    K = np.exp(-((ts[:, None] - ts[None, :]) ** 2) / (2 * sig**2)) + np.eye(T) * jitter
    L = np.linalg.cholesky(K)

    gp_raw = scale * (L @ xs) + shift
    gp_int = np.clip(np.round(gp_raw), clip[0], clip[1]).astype(int)
    return gp_int, gp_raw


# ---------------------------------------------------------------------------
# Three-armed block-swap session
# ---------------------------------------------------------------------------


def draw_block_boundaries(
    rng: np.random.Generator,
    *,
    T: int,
    min_block_len: int,
    max_block_len: int,
    n_blocks_draw: int,
) -> np.ndarray:
    """Trial indices at which blocks end, drawn from random block lengths.

    Block lengths are uniform on ``[min_block_len, max_block_len]``; boundaries at
    or beyond *T* are dropped. All *n_blocks_draw* lengths come from one
    ``rng.integers`` call, so the generator advances identically however many
    boundaries survive.
    """
    block_lengths = rng.integers(min_block_len, max_block_len + 1, size=n_blocks_draw)
    boundaries = np.cumsum(block_lengths)
    return boundaries[boundaries < T]


def first_other_active_arm(best_arm: int, *, active_arms: list[int], old_inactive: int) -> int:
    """First active arm that is not *best_arm*.

    If every active entry is *best_arm* (a duplicated entry, reachable only after
    an inactive arm won the argmax), the first arm other than *best_arm* and
    *old_inactive* is used instead; one always exists among three arms.
    """
    candidates = [a for a in active_arms if a != best_arm]
    if not candidates:
        candidates = [a for a in range(3) if a != best_arm and a != old_inactive]
    return candidates[0]


def swap_arm_roles(
    row: np.ndarray,
    inactive_arm: int,
    active_arms: list[int],
    arm_to_stream: dict[int, int],
) -> tuple[int, list[int], dict[int, int]]:
    """Arm roles after a block boundary.

    The arm that is best in *row* becomes inactive. The previously inactive arm
    takes over the best arm's stream, and the first remaining active arm keeps
    its own stream.

    Returns:
        ``(inactive_arm, active_arms, arm_to_stream)`` for the next block.
    """
    best_arm = int(np.argmax(row))
    old_inactive = inactive_arm
    # Known defect: with non-positive values the inactive arm (reward 0) can win the
    # argmax. It then has no stream entry (so it inherits stream 0 here), is listed
    # as active again, and two arms share a stream; e.g.
    # swap_arm_roles(np.array([-3., -1., 0.]), 2, [0, 1], {0: 0, 1: 1})
    # returns (2, [2, 0], {0: 0, 2: 0}). Later swaps can then raise KeyError.
    # Behaviour is kept so seeded results do not change.
    inherited_stream = arm_to_stream.get(best_arm, 0)
    other_active = first_other_active_arm(
        best_arm, active_arms=active_arms, old_inactive=old_inactive
    )
    return (
        best_arm,
        [old_inactive, other_active],
        {other_active: arm_to_stream[other_active], old_inactive: inherited_stream},
    )


def build_arm_values(streams: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    """Per-trial reward of the three arms, shape ``(T, 3)``.

    Two arms are active and read the two *streams*; the third is inactive and
    pays zero. At each boundary the best arm turns inactive (see
    :func:`swap_arm_roles`), so only the roles change, not the streams.
    """
    T = len(streams)
    vals3 = np.zeros((T, 3), dtype=float)
    inactive_arm, active_arms, arm_to_stream = 2, [0, 1], {0: 0, 1: 1}

    # A swap follows trial b - 1 for each distinct boundary b >= 1.
    swap_trials = np.unique(boundaries[boundaries >= 1])
    edges = np.r_[0, swap_trials, T]
    for start, stop in zip(edges[:-1], edges[1:], strict=True):
        for arm in active_arms:
            vals3[start:stop, arm] = streams[start:stop, arm_to_stream[arm]]
        if stop < T:
            inactive_arm, active_arms, arm_to_stream = swap_arm_roles(
                vals3[stop - 1], inactive_arm, active_arms, arm_to_stream
            )
    return vals3


def gap_best_minus_second(vals3: np.ndarray) -> np.ndarray:
    """Per-trial value of the best arm minus the second-best arm."""
    sorted_vals = np.sort(vals3, axis=1)
    return sorted_vals[:, -1] - sorted_vals[:, -2]


def best_run_lengths(best_arm_series: np.ndarray) -> np.ndarray:
    """Lengths of consecutive stretches in which the same arm is best."""
    changes = np.where(best_arm_series[1:] != best_arm_series[:-1])[0] + 1
    return np.diff(np.r_[0, changes, len(best_arm_series)]).astype(float)


def session_metrics(
    *, best_minus_second: np.ndarray, best_arm_series: np.ndarray
) -> tuple[float, float, float, float]:
    """Summary of a session.

    Returns:
        ``(mean gap, fraction of trials with gap >= 2, mean best-run length,
        std of best-run length)``.
    """
    run_lengths = best_run_lengths(best_arm_series)
    return (
        float(best_minus_second.mean()),
        float((best_minus_second >= 2).mean()),
        float(run_lengths.mean()),
        float(run_lengths.std(ddof=0)),
    )


def build_session(
    *,
    vals3: np.ndarray,
    boundaries: np.ndarray,
    best_arm_series: np.ndarray,
    best_minus_second: np.ndarray,
    gp_int: np.ndarray,
    gp_raw: np.ndarray,
) -> dict:
    """Session dict returned by :func:`simulate_3armed_blockswaps_gp`."""
    return {
        "vals3": vals3,
        "boundaries": boundaries,
        "best_arm_series": best_arm_series,
        "best_minus_second": best_minus_second,
        "gp_int": gp_int,
        "gp_raw": gp_raw,
    }


def simulate_3armed_blockswaps_gp(
    T: int,
    sig: float,
    max_block_len: int,
    min_block_len: int | None = None,
    n_blocks_draw: int = 1000,
    rng: np.random.Generator | None = None,
    scale: float = 2.0,
    shift: float = 2.5,
    clip: tuple[int, int] = (1, 5),
    jitter: float = 1e-6,
    use_raw: bool = False,
    return_session: bool = False,
):
    """Simulate a 3-armed GP bandit session with block-swap dynamics.

    Two arms are active at any time; at each block boundary the current best
    arm becomes inactive and a previously inactive arm re-enters.

    Args:
        T: Session length in trials.
        sig: GP length-scale.
        max_block_len: Maximum block length in trials.
        min_block_len: Minimum block length (defaults to
            ``max_block_len - 10``).
        n_blocks_draw: Number of block lengths to pre-sample (default: 1000).
        rng: Optional random generator.
        scale: GP amplitude scaling (default: 2.0).
        shift: GP mean shift (default: 2.5).
        clip: Integer clipping range (default: ``(1, 5)``).
        jitter: GP kernel jitter (default: 1e-6).
        use_raw: If ``True``, use continuous GP values instead of integers
            for metric computation (default: False).
        return_session: If ``True``, also return a session dict with
            ``vals3``, ``boundaries``, etc. (default: False).

    Returns:
        ``(mean_best_minus_second, frac_gap_ge2, mean_best_run_length,
        std_best_run_length)`` and optionally a *session* dict when
        *return_session* is ``True``.
    """
    if rng is None:
        rng = np.random.default_rng()
    if min_block_len is None:
        min_block_len = max_block_len - 10

    # Seeded results depend on this draw order: GP noise first, then block lengths.
    gp_int, gp_raw = sample_two_gp_bandits(
        T=T, sig=sig, scale=scale, shift=shift, clip=clip, jitter=jitter, rng=rng
    )
    streams = gp_raw if use_raw else gp_int  # (T, 2)
    boundaries = draw_block_boundaries(
        rng,
        T=T,
        min_block_len=min_block_len,
        max_block_len=max_block_len,
        n_blocks_draw=n_blocks_draw,
    )
    vals3 = build_arm_values(streams, boundaries)

    best_minus_second = gap_best_minus_second(vals3)
    best_arm_series = np.argmax(vals3, axis=1)
    metrics = session_metrics(best_minus_second=best_minus_second, best_arm_series=best_arm_series)
    if not return_session:
        return metrics

    session = build_session(
        vals3=vals3,
        boundaries=boundaries,
        best_arm_series=best_arm_series,
        best_minus_second=best_minus_second,
        gp_int=gp_int,
        gp_raw=gp_raw,
    )
    return (*metrics, session)


# ---------------------------------------------------------------------------
# Parameter sweep
# ---------------------------------------------------------------------------


def mean_left_to_right(values: Iterable[float], *, count: int | None = None) -> float:
    """Mean of *values*, added one at a time in order.

    Python's built-in ``sum`` compensates rounding error from version 3.12 on,
    which would change the last bit of a heatmap cell between interpreters.
    Plain left-to-right addition gives the same result everywhere.

    Args:
        values: Numbers to average.
        count: Divisor; defaults to the number of *values*. The sweep passes its
            ``n_sims`` so that a non-positive ``n_sims`` divides exactly as the
            accumulating loop always did (0 raises, a negative count gives -0.0).
    """
    values = list(values)
    return reduce(operator.add, values, 0.0) / (len(values) if count is None else count)


def summarize_sweep_cell(results: list[tuple], *, n_sims: int) -> tuple[float, float, float, float]:
    """Average the simulations of one (max_block_len, sig) cell.

    Args:
        results: One :func:`simulate_3armed_blockswaps_gp` return per simulation.
        n_sims: Number of simulations requested for the cell (the divisor).

    Returns:
        ``(gap_mean, gap_ge2, run_mean, run_std)``.
    """
    gap_mean, gap_ge2, run_mean, run_std = (
        mean_left_to_right((r[k] for r in results), count=n_sims) for k in range(4)
    )
    return gap_mean, gap_ge2, run_mean, run_std


def sweep_gp_metrics(
    sig_values: np.ndarray = DEFAULT_SIG_VALUES,
    max_block_lens: np.ndarray = DEFAULT_MAX_BLOCK_LENS,
    T: int = 150,
    n_sims: int = 1000,
    seed: int = 0,
    scale: float = 2.0,
    shift: float = 2.5,
    clip: tuple[int, int] = (1, 5),
    jitter: float = 1e-6,
    use_raw: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sweep over GP parameter combinations and compute summary metrics.

    Args:
        sig_values: Array of GP length-scale values to sweep.
        max_block_lens: Array of maximum block lengths to sweep.
        T: Session length (default: 150).
        n_sims: Number of simulations per parameter combination (default: 1000).
        seed: Random seed (default: 0).
        scale: GP amplitude (default: 2.0).
        shift: GP mean shift (default: 2.5).
        clip: Integer reward range (default: ``(1, 5)``).
        jitter: GP kernel jitter (default: 1e-6).
        use_raw: Use continuous GP values (default: False).

    Returns:
        Four 2-D arrays ``(H_gap_mean, H_gap_ge2, H_run_mean, H_run_std)``
        of shape ``(len(max_block_lens), len(sig_values))``.
    """
    rng = np.random.default_rng(seed)
    simulate = partial(
        simulate_3armed_blockswaps_gp,
        T=T,
        rng=rng,
        scale=scale,
        shift=shift,
        clip=clip,
        jitter=jitter,
        use_raw=use_raw,
        return_session=False,
    )
    shape = (len(max_block_lens), len(sig_values))
    heatmaps = [np.zeros(shape) for _ in range(4)]

    # One shared rng drawn in row-major cell order: the cells are not independent.
    # min_block_len = max_block_len - 10 is non-positive for max_block_len <= 10, which
    # yields zero or negative block lengths; such rows are not validated.
    for i, mbl in enumerate(max_block_lens):
        for j, sig in enumerate(sig_values):
            results = [
                simulate(sig=float(sig), max_block_len=int(mbl), min_block_len=int(mbl) - 10)
                for _ in range(n_sims)
            ]
            cell = summarize_sweep_cell(results, n_sims=n_sims)
            for heatmap, value in zip(heatmaps, cell, strict=True):
                heatmap[i, j] = value
            print(f"Done sig={sig}, max_block_len={mbl}")

    return tuple(heatmaps)


# ---------------------------------------------------------------------------
# Heatmaps and example session plots
# ---------------------------------------------------------------------------


def plot_gp_heatmaps_pair_1(
    H_gap_mean: np.ndarray,
    H_gap_ge2: np.ndarray,
    sig_values: np.ndarray,
    max_block_lens: np.ndarray,
) -> None:
    """Plot heatmaps of gap-metric sweeps for the GP bandit.

    Args:
        H_gap_mean: Mean (best − second-best) heatmap.
        H_gap_ge2: Fraction of trials with gap ≥ 2 heatmap.
        sig_values: GP length-scale values used in the sweep.
        max_block_lens: Maximum block-length values used in the sweep.
    """
    extent = [sig_values[0], sig_values[-1], max_block_lens[0], max_block_lens[-1]]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    im0 = axes[0].imshow(H_gap_mean, origin="lower", aspect="auto", extent=extent)
    axes[0].set_title("Avg (best − second best)")
    axes[0].set_xlabel("sig")
    axes[0].set_ylabel("max_block_len")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(H_gap_ge2, origin="lower", aspect="auto", extent=extent, vmin=0, vmax=1)
    axes[1].set_title("Frac trials (best − second ≥ 2)")
    axes[1].set_xlabel("sig")
    axes[1].set_ylabel("max_block_len")
    fig.colorbar(im1, ax=axes[1])

    plt.show()


def plot_gp_heatmaps_pair_2(
    H_run_mean: np.ndarray,
    H_run_std: np.ndarray,
    sig_values: np.ndarray,
    max_block_lens: np.ndarray,
) -> None:
    """Plot heatmaps of run-length-metric sweeps for the GP bandit.

    Args:
        H_run_mean: Mean best-run length heatmap.
        H_run_std: Std dev of best-run length heatmap.
        sig_values: GP length-scale values.
        max_block_lens: Maximum block-length values.
    """
    extent = [sig_values[0], sig_values[-1], max_block_lens[0], max_block_lens[-1]]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    im0 = axes[0].imshow(H_run_mean, origin="lower", aspect="auto", extent=extent)
    axes[0].set_title("Avg run length (best arm stays best)")
    axes[0].set_xlabel("sig")
    axes[0].set_ylabel("max_block_len")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(H_run_std, origin="lower", aspect="auto", extent=extent)
    axes[1].set_title("Std dev of best-run length")
    axes[1].set_xlabel("sig")
    axes[1].set_ylabel("max_block_len")
    fig.colorbar(im1, ax=axes[1])

    plt.show()


def simulate_example_session(
    *,
    sig: float,
    max_block_len: int,
    T: int,
    seed: int,
    scale: float,
    shift: float,
    clip: tuple[int, int],
    jitter: float,
    use_raw: bool,
) -> tuple[tuple[float, ...], dict]:
    """Simulate one seeded session.

    Returns:
        ``((gap_mean, gap_ge2, run_mean, run_std), session)`` as returned by
        :func:`simulate_3armed_blockswaps_gp` with ``return_session=True``.
    """
    *metrics, session = simulate_3armed_blockswaps_gp(
        T=T,
        sig=float(sig),
        max_block_len=int(max_block_len),
        min_block_len=int(max_block_len) - 10,
        rng=np.random.default_rng(seed),
        scale=scale,
        shift=shift,
        clip=clip,
        jitter=jitter,
        use_raw=use_raw,
        return_session=True,
    )
    return tuple(metrics), session


def format_session_title(
    metrics: tuple[float, ...], *, sig: float, max_block_len: int, title: str | None = None
) -> str:
    """*title* if given, otherwise a summary of the parameters and session metrics."""
    gap_mean, gap_ge2, run_mean, run_std = metrics
    return title or (
        f"Example GP session | sig={sig:.2f}, max_block_len={max_block_len} "
        f"| gap_mean={gap_mean:.2f}, frac≥2={gap_ge2:.2f}, "
        f"run_mean={run_mean:.2f}, run_std={run_std:.2f}"
    )


def y_axis_max(vals3: np.ndarray, *, clip: tuple[int, int], use_raw: bool) -> float:
    """Largest reward the y-axis must show: the clip maximum, or the data maximum if raw."""
    return np.nanmax(vals3) if use_raw else max(clip)


def plot_session_panel(
    vals3: np.ndarray,
    boundaries: np.ndarray,
    *,
    y_max: float,
    choices: np.ndarray | None = None,
    ax: plt.Axes | None = None,
    legend: bool = True,
) -> plt.Axes:
    """Draw the three arms' rewards over trials, with block boundaries and choices.

    The y-label is "Reward Magnitude"; a caller plotting latent values relabels it.
    The caller sets the title, so the panel can sit in a larger figure. A caller
    that must set the title first passes ``legend=False`` and adds the legend itself.

    Args:
        vals3: ``(T, 3)`` reward of each arm per trial.
        boundaries: Trial indices where blocks end.
        y_max: Largest reward value; sets the y-limit.
        choices: Optional ``(T,)`` int array of chosen arm indices (0–2).
        ax: Axes to draw into; a new figure is created if ``None``.
        legend: Add the legend (default: True).

    Returns:
        The axes drawn into.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    T = len(vals3)
    x = np.arange(T)

    for arm_idx in range(3):
        ax.plot(x, vals3[:, arm_idx], linewidth=2.5, label=f"Arm {arm_idx}")

    if choices is not None:
        ch = np.asarray(choices, dtype=int)
        ax.scatter(x, vals3[x, ch], s=25, color="black", zorder=5, label="Choice")

    # Only the first boundary carries the "Block" legend label.
    for k, b in enumerate(boundaries):
        ax.axvline(
            b - 0.5, linestyle="--", color="k", linewidth=2.0, label="Block" if k == 0 else None
        )

    ax.set_xlim(0, T - 1)
    ax.set_ylim(-0.5, y_max + 0.5)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Reward Magnitude")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if legend:
        ax.legend(loc="upper left")
    return ax


def plot_example_session_gp(
    sig: float,
    max_block_len: int,
    T: int = 150,
    seed: int = 0,
    scale: float = 2.0,
    shift: float = 2.5,
    clip: tuple[int, int] = (1, 5),
    jitter: float = 1e-6,
    use_raw: bool = False,
    choices: np.ndarray | None = None,
    title: str | None = None,
) -> dict:
    """Plot a single example GP bandit session.

    Args:
        sig: GP length-scale.
        max_block_len: Maximum block length.
        T: Session length (default: 150).
        seed: Random seed (default: 0).
        scale: GP amplitude (default: 2.0).
        shift: GP mean shift (default: 2.5).
        clip: Integer reward range (default: ``(1, 5)``).
        jitter: GP kernel jitter (default: 1e-6).
        use_raw: Plot continuous GP values (default: False).
        choices: Optional ``(T,)`` int array of chosen arm indices (0–2).
        title: Custom figure title.

    Returns:
        The session dict from :func:`simulate_3armed_blockswaps_gp`.
    """
    metrics, session = simulate_example_session(
        sig=sig,
        max_block_len=max_block_len,
        T=T,
        seed=seed,
        scale=scale,
        shift=shift,
        clip=clip,
        jitter=jitter,
        use_raw=use_raw,
    )
    vals3 = session["vals3"]
    ax = plot_session_panel(
        vals3,
        session["boundaries"],
        y_max=y_axis_max(vals3, clip=clip, use_raw=use_raw),
        choices=choices,
        legend=False,
    )
    if use_raw:
        ax.set_ylabel("Latent value")
    # Formatting the title validates sig, so it comes before the legend: a bad sig
    # fails without a legend on the figure.
    ax.set_title(format_session_title(metrics, sig=sig, max_block_len=max_block_len, title=title))
    ax.legend(loc="upper left")
    plt.show()
    return session


def collect_gp_run_stats(
    sig_values: np.ndarray,
    T: int = 150,
    n_sims: int = 1000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect mean and std dev of best-arm run lengths for each sig value.

    Args:
        sig_values: Array of GP length-scale values.
        T: Session length (default: 150).
        n_sims: Simulations per sig value (default: 1000).
        seed: Random seed (default: 0).

    Returns:
        ``(mean_runs, std_runs)`` arrays of shape ``(len(sig_values),)``.
    """
    rng = np.random.default_rng(seed)
    mean_runs, std_runs = [], []

    for sig in sig_values:
        run_mean_acc = 0.0
        run_std_acc = 0.0
        for _ in range(n_sims):
            _, _, run_mean, run_std = simulate_3armed_blockswaps_gp(
                T=T,
                sig=sig,
                max_block_len=80,
                min_block_len=70,
                rng=rng,
            )
            run_mean_acc += run_mean
            run_std_acc += run_std
        mean_runs.append(run_mean_acc / n_sims)
        std_runs.append(run_std_acc / n_sims)
        print(f"GP done sig={sig}")

    return np.array(mean_runs), np.array(std_runs)
