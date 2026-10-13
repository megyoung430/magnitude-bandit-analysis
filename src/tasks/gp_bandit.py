"""Gaussian Process (GP) bandit task simulations.

Functions here generate synthetic 3-armed bandit task sessions where arm
reward values are sampled from Gaussian Processes, with block-swap dynamics
that mimic the experimental paradigm.
"""

from __future__ import annotations

from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np

# Default sweep grids, shared module-level arrays that are never mutated.
DEFAULT_SIG_VALUES = np.arange(5, 30, 1)
DEFAULT_MAX_BLOCK_LENS = np.arange(20, 81, 1)


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


class ArmState(NamedTuple):
    """Which arm is inactive, which two are active, and which GP stream each follows."""

    inactive_arm: int
    active_arms: tuple[int, ...]
    arm_to_gp: dict[int, int]


def initial_arm_state() -> ArmState:
    """Arms 0 and 1 follow GP streams 0 and 1; arm 2 starts inactive."""
    return ArmState(inactive_arm=2, active_arms=(0, 1), arm_to_gp={0: 0, 1: 1})


def draw_block_boundaries(
    rng: np.random.Generator,
    min_block_len: int,
    max_block_len: int,
    n_blocks_draw: int,
    T: int,
) -> np.ndarray:
    """Cumulative block ends (in trials) that fall strictly inside a session of *T*.

    Draws all *n_blocks_draw* block lengths in one ``rng.integers`` call, so the
    RNG advances identically regardless of how many boundaries survive.
    """
    block_lengths = rng.integers(min_block_len, max_block_len + 1, size=n_blocks_draw)
    boundaries = np.cumsum(block_lengths)
    return boundaries[boundaries < T]


def trial_arm_values(state: ArmState, stream_row: np.ndarray) -> np.ndarray:
    """Reward value of each of the three arms on one trial.

    The inactive arm is 0; each active arm takes the value of the GP stream it follows.
    """
    row = np.zeros(3, dtype=float)
    for arm in state.active_arms:
        row[arm] = stream_row[state.arm_to_gp[arm]]
    return row


def swap_arms(state: ArmState, vals_row: np.ndarray) -> ArmState:
    """Block swap: the best arm goes inactive and the old inactive arm re-enters.

    The re-entering arm inherits the GP stream of the arm that just left; the
    other active arm keeps its stream.
    """
    best_arm = int(np.argmax(vals_row))
    old_inactive = state.inactive_arm
    inherited_gp = state.arm_to_gp.get(best_arm, 0)
    remaining = [a for a in state.active_arms if a != best_arm]
    if not remaining:
        remaining = [a for a in [0, 1, 2] if a != best_arm and a != old_inactive][:1]
    other_active = remaining[0]
    other_gp = state.arm_to_gp[other_active]
    return ArmState(
        inactive_arm=best_arm,
        active_arms=(old_inactive, other_active),
        arm_to_gp={other_active: other_gp, old_inactive: inherited_gp},
    )


def build_arm_values(streams: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    """``(T, 3)`` arm values for a session, swapping arms after each boundary trial.

    A boundary *b* means the swap takes effect after trial index ``b - 1``.
    """
    boundary_set = set(boundaries.tolist())
    state = initial_arm_state()
    vals3 = np.zeros((len(streams), 3), dtype=float)
    for t, stream_row in enumerate(streams):
        vals3[t] = trial_arm_values(state, stream_row)
        if (t + 1) in boundary_set:
            state = swap_arms(state, vals3[t])
    return vals3


def gap_best_minus_second(vals3: np.ndarray) -> np.ndarray:
    """Per-trial difference between the best and second-best arm value."""
    sorted_vals = np.sort(vals3, axis=1)
    return sorted_vals[:, -1] - sorted_vals[:, -2]


def best_run_lengths(best_arm_series: np.ndarray) -> np.ndarray:
    """Lengths of consecutive stretches in which the same arm is best."""
    changes = np.where(best_arm_series[1:] != best_arm_series[:-1])[0] + 1
    return np.diff(np.r_[0, changes, len(best_arm_series)]).astype(float)


def summarize_session(
    vals3: np.ndarray, boundaries: np.ndarray, gp_int: np.ndarray, gp_raw: np.ndarray
) -> tuple[tuple[float, float, float, float], dict]:
    """Gap and run-length metrics plus the session dict for simulated arm values.

    Returns:
        ``((mean_gap, frac_gap_ge2, mean_run_length, std_run_length), session)``.
    """
    best_minus_second = gap_best_minus_second(vals3)
    best_arm_series = np.argmax(vals3, axis=1)
    run_lengths = best_run_lengths(best_arm_series)
    metrics = (
        float(best_minus_second.mean()),
        float((best_minus_second >= 2).mean()),
        float(run_lengths.mean()),
        float(run_lengths.std(ddof=0)),
    )
    session = {
        "vals3": vals3,
        "boundaries": boundaries,
        "best_arm_series": best_arm_series,
        "best_minus_second": best_minus_second,
        "gp_int": gp_int,
        "gp_raw": gp_raw,
    }
    return metrics, session


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

    # RNG order matters for seeded reproducibility: GP noise first, then block lengths.
    gp_int, gp_raw = sample_two_gp_bandits(T, sig, scale, shift, clip, jitter, rng)
    streams = gp_raw if use_raw else gp_int  # (T, 2)
    boundaries = draw_block_boundaries(rng, min_block_len, max_block_len, n_blocks_draw, T)
    vals3 = build_arm_values(streams, boundaries)

    metrics, session = summarize_session(vals3, boundaries, gp_int, gp_raw)
    return (*metrics, session) if return_session else metrics


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

    shape = (len(max_block_lens), len(sig_values))
    H_gap_mean = np.zeros(shape)
    H_gap_ge2 = np.zeros(shape)
    H_run_mean = np.zeros(shape)
    H_run_std = np.zeros(shape)

    for i, mbl in enumerate(max_block_lens):
        for j, sig in enumerate(sig_values):
            acc = [0.0, 0.0, 0.0, 0.0]
            for _ in range(n_sims):
                gm, gg, rm, rs = simulate_3armed_blockswaps_gp(
                    T=T,
                    sig=float(sig),
                    max_block_len=int(mbl),
                    min_block_len=int(mbl) - 10,
                    rng=rng,
                    scale=scale,
                    shift=shift,
                    clip=clip,
                    jitter=jitter,
                    use_raw=use_raw,
                    return_session=False,
                )
                acc[0] += gm
                acc[1] += gg
                acc[2] += rm
                acc[3] += rs

            H_gap_mean[i, j] = acc[0] / n_sims
            H_gap_ge2[i, j] = acc[1] / n_sims
            H_run_mean[i, j] = acc[2] / n_sims
            H_run_std[i, j] = acc[3] / n_sims

            print(f"Done sig={sig}, max_block_len={mbl}")

    return H_gap_mean, H_gap_ge2, H_run_mean, H_run_std


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


def format_session_title(
    sig: float, max_block_len: int, metrics: tuple[float, float, float, float]
) -> str:
    """Default title: session parameters plus ``(gap_mean, frac_gap_ge2, run_mean, run_std)``."""
    gap_mean, gap_ge2, run_mean, run_std = metrics
    return (
        f"Example GP session | sig={sig:.2f}, max_block_len={max_block_len} "
        f"| gap_mean={gap_mean:.2f}, frac≥2={gap_ge2:.2f}, "
        f"run_mean={run_mean:.2f}, run_std={run_std:.2f}"
    )


def plot_session_panel(
    ax: plt.Axes,
    vals3: np.ndarray,
    boundaries: np.ndarray,
    clip: tuple[int, int],
    use_raw: bool,
    choices: np.ndarray | None = None,
) -> plt.Axes:
    """Draw the three arm-value traces, block boundaries and optional choices on *ax*.

    Args:
        vals3: ``(T, 3)`` arm values.
        boundaries: Block ends in trials; a dashed line is drawn between trial
            ``b - 1`` and ``b``.
        clip: Integer reward range, used for the y-limit unless *use_raw*.
        use_raw: Label the y-axis as a latent value and fit the y-limit to the data.
        choices: Optional ``(T,)`` chosen arm indices, drawn as black dots.
    """
    T = len(vals3)
    x = np.arange(T)
    for arm_idx in range(3):
        ax.plot(x, vals3[:, arm_idx], linewidth=2.5, label=f"Arm {arm_idx}")

    if choices is not None:
        ch = np.asarray(choices, dtype=int)
        ax.scatter(x, vals3[x, ch], s=25, color="black", zorder=5, label="Choice")

    # Only the first boundary carries the "Block" legend label.
    for i, b in enumerate(boundaries):
        ax.axvline(
            b - 0.5, linestyle="--", color="k", linewidth=2.0, label="Block" if i == 0 else None
        )

    ax.set_xlim(0, T - 1)
    ax.set_ylim(-0.5, (max(clip) if not use_raw else np.nanmax(vals3)) + 0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Reward Magnitude" if not use_raw else "Latent value")
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
    rng = np.random.default_rng(seed)
    gap_mean, gap_ge2, run_mean, run_std, session = simulate_3armed_blockswaps_gp(
        T=T,
        sig=float(sig),
        max_block_len=int(max_block_len),
        min_block_len=int(max_block_len) - 10,
        rng=rng,
        scale=scale,
        shift=shift,
        clip=clip,
        jitter=jitter,
        use_raw=use_raw,
        return_session=True,
    )

    _, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    plot_session_panel(ax, session["vals3"], session["boundaries"], clip, use_raw, choices)
    # `or`: an empty title falls back to the generated one.
    ax.set_title(
        title or format_session_title(sig, max_block_len, (gap_mean, gap_ge2, run_mean, run_std))
    )
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
