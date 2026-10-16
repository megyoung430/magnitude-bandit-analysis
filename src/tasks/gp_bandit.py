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

Helpers whose names do not start with an underscore are importable for tests but
are not stable API; only the names in ``__all__`` are.
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
        T (int): Number of time steps.
        sig (float): GP length-scale parameter (controls temporal smoothness).
        scale (float): Amplitude scaling of the GP samples. Default 2.0.
        shift (float): Mean shift applied to GP samples. Default 2.5.
        clip (tuple[int, int]): ``(min, max)`` clipping range for integer reward values. Default
            ``(1, 5)``.
        jitter (float): Small diagonal regularisation added to the GP kernel for numerical
            stability. Default 1e-6.
        rng (np.random.Generator | None): Random generator; None creates a fresh unseeded one.
            Default None.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(gp_int, gp_raw)``: integer (rounded and clipped) and
            continuous (raw) reward arrays, each of shape ``(T, 2)``.

    Raises:
        numpy.linalg.LinAlgError: If the kernel is not positive definite (for example *jitter* too
            small for a large *sig*).
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


def check_min_block_len(min_block_len: int, max_block_len: int) -> None:
    """Raise ``ValueError`` unless ``1 <= min_block_len <= max_block_len``.

    Args:
        min_block_len (int): Shortest block length, in trials.
        max_block_len (int): Longest block length, in trials.

    Raises:
        ValueError: If *min_block_len* is below 1 or above *max_block_len*.
    """
    if min_block_len < 1:
        raise ValueError(
            f"min_block_len must be >= 1, got {min_block_len} "
            f"(max_block_len={max_block_len}; the default minimum is max_block_len - 10, "
            "so pass min_block_len explicitly or use max_block_len > 10)"
        )
    if min_block_len > max_block_len:
        raise ValueError(f"min_block_len ({min_block_len}) exceeds max_block_len ({max_block_len})")


def check_finite(**values) -> None:
    """Raise ``ValueError`` if any named number (or sequence of numbers) is not finite.

    Non-finite rewards make the gap metrics ``nan`` without any other symptom.

    Args:
        **values: Named numbers or sequences to check; the keyword is used in the error message.

    Raises:
        ValueError: If any value contains NaN or infinity.
    """
    for name, value in values.items():
        if not np.all(np.isfinite(np.asarray(value, dtype=float))):
            raise ValueError(f"{name} must be finite, got {value!r}")


def check_sweep_max_block_lens(max_block_lens: np.ndarray) -> None:
    """Raise ``ValueError`` if a sweep row would get a minimum block length below 1.

    A sweep derives ``min_block_len = max_block_len - 10`` for every row, so each entry of
    *max_block_lens* must exceed 10.

    Args:
        max_block_lens (np.ndarray): Maximum block lengths to be swept.

    Raises:
        ValueError: If any entry is 10 or less.
    """
    too_short = [int(m) for m in max_block_lens if int(m) <= 10]
    if too_short:
        raise ValueError(
            f"every max_block_lens entry must be > 10 (the sweep uses "
            f"min_block_len = max_block_len - 10); got {too_short}"
        )


def draw_block_boundaries(
    rng: np.random.Generator,
    *,
    T: int,
    min_block_len: int,
    max_block_len: int,
    n_blocks_draw: int,
) -> np.ndarray:
    """Trial indices at which blocks end, drawn from random block lengths.

    Block lengths are uniform on ``[min_block_len, max_block_len]``; boundaries at or beyond *T* are
    dropped. *min_block_len* must be at least 1 so the boundaries are strictly increasing
    (:func:`simulate_3armed_blockswaps_gp` checks this).

    All *n_blocks_draw* lengths come from one ``rng.integers`` call, so the generator advances
    identically however many boundaries survive.

    Args:
        rng (np.random.Generator): Random generator.
        T (int): Session length in trials (keyword-only).
        min_block_len (int): Shortest block, in trials (keyword-only).
        max_block_len (int): Longest block, in trials (keyword-only).
        n_blocks_draw (int): Number of block lengths to draw (keyword-only).

    Returns:
        np.ndarray: 1-D int array of cumulative block ends below *T*; entry ``b`` means a swap
            follows trial ``b - 1`` (0-based).
    """
    block_lengths = rng.integers(min_block_len, max_block_len + 1, size=n_blocks_draw)
    boundaries = np.cumsum(block_lengths)
    return boundaries[boundaries < T]


def remaining_active_arm(best_arm: int, *, active_arms: list[int]) -> int:
    """First active arm that is not *best_arm*; it stays active across the swap.

    Args:
        best_arm (int): Index of the arm that becomes inactive.
        active_arms (list[int]): Indices of the two currently active arms (keyword-only).

    Returns:
        int: The active arm other than *best_arm*.
    """
    return next(a for a in active_arms if a != best_arm)


def swap_arm_roles(
    row: np.ndarray,
    inactive_arm: int,
    active_arms: list[int],
    arm_to_stream: dict[int, int],
) -> tuple[int, list[int], dict[int, int]]:
    """Arm roles after a block boundary.

    The active arm that is best in *row* becomes inactive. The previously inactive arm takes over
    the best arm's stream, and the first remaining active arm keeps its own stream.

    Only active arms compete: the inactive arm pays 0, which would beat negative active values and
    leave two arms sharing a stream. Ties go to the lowest arm index, whatever the order of
    *active_arms*.

    Args:
        row (np.ndarray): ``(3,)`` reward of each arm on the last trial of the block.
        inactive_arm (int): Index of the currently inactive arm.
        active_arms (list[int]): Indices of the two currently active arms.
        arm_to_stream (dict[int, int]): For each active arm, the column of the reward streams it
            reads.

    Returns:
        tuple[int, list[int], dict[int, int]]: ``(inactive_arm, active_arms, arm_to_stream)`` for
            the next block.
    """
    best_arm = max(sorted(active_arms), key=row.__getitem__)
    other_active = remaining_active_arm(best_arm, active_arms=active_arms)
    return (
        best_arm,
        [inactive_arm, other_active],
        {other_active: arm_to_stream[other_active], inactive_arm: arm_to_stream[best_arm]},
    )


def build_arm_roles(streams: np.ndarray, boundaries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-trial reward of the three arms and the inactive arm of each trial.

    Two arms are active and read the two *streams*; the third is inactive and pays zero. At each
    boundary the best active arm turns inactive (see :func:`swap_arm_roles`), so only the roles
    change, not the streams.

    *boundaries* come from :func:`draw_block_boundaries`: strictly increasing and inside ``[1,
    len(streams))``. Repeated boundaries, boundaries below 1 and boundaries at or beyond
    ``len(streams)`` are ignored for direct callers.

    Args:
        streams (np.ndarray): ``(T, 2)`` rewards of the two reward streams.
        boundaries (np.ndarray): 1-D int array of trial indices, as from ``draw_block_boundaries``:
            ``b`` means the block ends before trial ``b`` (0-based), so a swap follows trial ``b -
            1``.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(vals3, inactive_arms)`` of shapes ``(T, 3)`` (float
            rewards) and ``(T,)`` (int index of the inactive arm).
    """
    T = len(streams)
    vals3 = np.zeros((T, 3), dtype=float)
    inactive_arms = np.zeros(T, dtype=int)
    inactive_arm, active_arms, arm_to_stream = 2, [0, 1], {0: 0, 1: 1}

    # A swap follows trial b - 1 for each distinct boundary b >= 1.
    swap_trials = np.unique(boundaries[boundaries >= 1])
    edges = np.r_[0, swap_trials, T]
    for start, stop in zip(edges[:-1], edges[1:], strict=True):
        for arm in active_arms:
            vals3[start:stop, arm] = streams[start:stop, arm_to_stream[arm]]
        inactive_arms[start:stop] = inactive_arm
        if stop < T:
            inactive_arm, active_arms, arm_to_stream = swap_arm_roles(
                vals3[stop - 1], inactive_arm, active_arms, arm_to_stream
            )
    return vals3, inactive_arms


def build_arm_values(streams: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    """Per-trial reward of the three arms, shape ``(T, 3)``; see :func:`build_arm_roles`.

    Args:
        streams (np.ndarray): ``(T, 2)`` rewards of the two reward streams.
        boundaries (np.ndarray): 1-D int array of trial indices, as from ``draw_block_boundaries``:
            ``b`` means the block ends before trial ``b`` (0-based), so a swap follows trial ``b -
            1``.

    Returns:
        np.ndarray: ``(T, 3)`` float reward of each arm per trial.
    """
    return build_arm_roles(streams, boundaries)[0]


def active_arm_values(vals3: np.ndarray, inactive_arms: np.ndarray) -> np.ndarray:
    """Copy of *vals3* with each trial's inactive arm set to ``-inf``.

    The inactive arm pays 0 whatever the active values are, so it must not compete for best or
    second-best when active values are zero or negative.

    Args:
        vals3 (np.ndarray): ``(T, 3)`` reward of each arm per trial.
        inactive_arms (np.ndarray): ``(T,)`` int index of the inactive arm on each trial.

    Returns:
        np.ndarray: ``(T, 3)`` float copy of *vals3* with ``-inf`` at each trial's inactive arm.
    """
    masked = vals3.copy()
    masked[np.arange(len(vals3)), inactive_arms] = -np.inf
    return masked


def gap_best_minus_second(vals3: np.ndarray) -> np.ndarray:
    """Per-trial value of the best arm minus the second-best arm.

    Args:
        vals3 (np.ndarray): ``(T, 3)`` reward of each arm per trial; use ``active_arm_values``
            output so the inactive arm is excluded.

    Returns:
        np.ndarray: ``(T,)`` gap on each trial.
    """
    sorted_vals = np.sort(vals3, axis=1)
    return sorted_vals[:, -1] - sorted_vals[:, -2]


def best_run_lengths(best_arm_series: np.ndarray) -> np.ndarray:
    """Lengths of consecutive stretches in which the same arm is best.

    Args:
        best_arm_series (np.ndarray): ``(T,)`` int index of the best arm on each trial.

    Returns:
        np.ndarray: ``(n_runs,)`` float run lengths in trials; they sum to ``T``.
    """
    changes = np.where(best_arm_series[1:] != best_arm_series[:-1])[0] + 1
    return np.diff(np.r_[0, changes, len(best_arm_series)]).astype(float)


def session_metrics(
    *, best_minus_second: np.ndarray, best_arm_series: np.ndarray
) -> tuple[float, float, float, float]:
    """Summary of a session.

    Args:
        best_minus_second (np.ndarray): ``(T,)`` per-trial gap between the best and second-best arm
            (keyword-only).
        best_arm_series (np.ndarray): ``(T,)`` int index of the best arm on each trial
            (keyword-only).

    Returns:
        tuple[float, float, float, float]: ``(mean gap, fraction of trials with gap >= 2, mean
            best-run length, std of best-run length)``.
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
    """Session dict returned by :func:`simulate_3armed_blockswaps_gp`.

    Args:
        vals3 (np.ndarray): ``(T, 3)`` reward of each arm per trial.
        boundaries (np.ndarray): 1-D int array of trial indices, as from ``draw_block_boundaries``:
            ``b`` means the block ends before trial ``b`` (0-based), so a swap follows trial ``b -
            1``.
        best_arm_series (np.ndarray): ``(T,)`` int index of the best active arm on each trial
            (keyword-only).
        best_minus_second (np.ndarray): ``(T,)`` gap between the best and second-best active arm
            (keyword-only).
        gp_int (np.ndarray): ``(T, 2)`` integer GP reward streams (keyword-only).
        gp_raw (np.ndarray): ``(T, 2)`` continuous GP reward streams (keyword-only).

    Returns:
        dict: The arguments gathered under the keys ``vals3``, ``boundaries``, ``best_arm_series``,
            ``best_minus_second``, ``gp_int`` and ``gp_raw``.
    """
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
) -> tuple[float, float, float, float] | tuple[float, float, float, float, dict]:
    """Simulate a 3-armed GP bandit session with block-swap dynamics.

    Two arms are active at any time; at each block boundary the current best arm becomes inactive
    and a previously inactive arm re-enters.

    Args:
        T (int): Session length in trials.
        sig (float): GP length-scale.
        max_block_len (int): Maximum block length in trials.
        min_block_len (int | None): Minimum block length (defaults to ``max_block_len - 10``). A
            value below 1 (including a default derived from ``max_block_len <= 10``) raises
            ``ValueError``.
        n_blocks_draw (int): Number of block lengths to pre-sample. Default 1000.
        rng (np.random.Generator | None): Random generator; None creates a fresh unseeded one.
            Default None.
        scale (float): GP amplitude scaling. Default 2.0.
        shift (float): GP mean shift. Default 2.5.
        clip (tuple[int, int]): Integer clipping range. Default ``(1, 5)``.
        jitter (float): GP kernel jitter. Default 1e-6.
        use_raw (bool): If True, use continuous GP values instead of integers for metric
            computation. Default False.
        return_session (bool): If True, also return a session dict. Default False.

    Returns:
        tuple[float, float, float, float] | tuple[float, float, float, float, dict]:
            ``(mean_best_minus_second, frac_gap_ge2, mean_best_run_length, std_best_run_length)``,
            followed by a *session* dict when *return_session* is True. The dict has keys ``vals3``
            (``(T, 3)`` arm rewards), ``boundaries``, ``best_arm_series`` (``(T,)``),
            ``best_minus_second`` (``(T,)``), ``gp_int`` and ``gp_raw`` (``(T, 2)`` each). The
            metrics compare the two active arms only.

    Raises:
        ValueError: If the minimum block length is below 1 or above *max_block_len*, or *sig*,
            *scale*, *shift* or *jitter* is not finite.
    """
    if rng is None:
        rng = np.random.default_rng()
    if min_block_len is None:
        min_block_len = max_block_len - 10
    check_min_block_len(min_block_len, max_block_len)
    check_finite(sig=sig, scale=scale, shift=shift, jitter=jitter)

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
    vals3, inactive_arms = build_arm_roles(streams, boundaries)

    # Metrics compare the two active arms only (see active_arm_values).
    active_vals = active_arm_values(vals3, inactive_arms)
    best_minus_second = gap_best_minus_second(active_vals)
    best_arm_series = np.argmax(active_vals, axis=1)
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

    Python's built-in ``sum`` compensates rounding error from version 3.12 on, which would change
    the last bit of a heatmap cell between interpreters. Plain left-to-right addition gives the same
    result everywhere.

    Args:
        values (Iterable[float]): Numbers to average.
        count (int | None): Divisor; defaults to the number of *values*. When given, it is the
            divisor whatever the number of *values*: 0 raises ``ZeroDivisionError`` and a negative
            value gives -0.0 for an empty input. Keyword-only. Default None.

    Returns:
        float: The sum of *values* divided by the divisor.

    Raises:
        ZeroDivisionError: If the divisor is 0.
    """
    values = list(values)
    return reduce(operator.add, values, 0.0) / (len(values) if count is None else count)


def summarize_sweep_cell(results: list[tuple], *, n_sims: int) -> tuple[float, float, float, float]:
    """Average the simulations of one (max_block_len, sig) cell.

    Args:
        results (list[tuple]): One :func:`simulate_3armed_blockswaps_gp` return per simulation (only
            the first four entries are used).
        n_sims (int): Number of simulations requested for the cell. It is the divisor: 0 raises
            ``ZeroDivisionError`` and a negative value gives -0.0 (keyword-only).

    Returns:
        tuple[float, float, float, float]: ``(gap_mean, gap_ge2, run_mean, run_std)``, each averaged
            over the simulations.

    Raises:
        ZeroDivisionError: If *n_sims* is 0.
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

    Every cell uses ``min_block_len = max_block_len - 10`` and shares one random generator, drawn in
    row-major cell order, so cells are not independent. Progress is printed for each cell.

    Args:
        sig_values (np.ndarray): GP length-scale values to sweep (columns). Default
            ``DEFAULT_SIG_VALUES``.
        max_block_lens (np.ndarray): Maximum block lengths to sweep (rows); each must exceed 10.
            Default ``DEFAULT_MAX_BLOCK_LENS``.
        T (int): Session length. Default 150.
        n_sims (int): Number of simulations per parameter combination. Default 1000.
        seed (int): Random seed. Default 0.
        scale (float): GP amplitude. Default 2.0.
        shift (float): GP mean shift. Default 2.5.
        clip (tuple[int, int]): Integer reward range. Default ``(1, 5)``.
        jitter (float): GP kernel jitter. Default 1e-6.
        use_raw (bool): Use continuous GP values. Default False.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Four 2-D arrays ``(H_gap_mean,
            H_gap_ge2, H_run_mean, H_run_std)`` of shape ``(len(max_block_lens), len(sig_values))``.

    Raises:
        ValueError: If any entry of *max_block_lens* is 10 or less.
    """
    check_sweep_max_block_lens(max_block_lens)
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

    Draws a new two-panel figure and shows it.

    Args:
        H_gap_mean (np.ndarray): Mean (best - second-best) heatmap, shape ``(len(max_block_lens),
            len(sig_values))``.
        H_gap_ge2 (np.ndarray): Fraction of trials with gap >= 2 heatmap, same shape.
        sig_values (np.ndarray): GP length-scale values used in the sweep (x axis).
        max_block_lens (np.ndarray): Maximum block-length values used in the sweep (y axis).
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

    Draws a new two-panel figure and shows it.

    Args:
        H_run_mean (np.ndarray): Mean best-run length heatmap, shape ``(len(max_block_lens),
            len(sig_values))``.
        H_run_std (np.ndarray): Std dev of best-run length heatmap, same shape.
        sig_values (np.ndarray): GP length-scale values (x axis).
        max_block_lens (np.ndarray): Maximum block-length values (y axis).
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
) -> tuple[tuple[float, float, float, float], dict]:
    """Simulate one seeded session.

    All arguments are keyword-only. The block lengths are ``max_block_len - 10`` to
    ``max_block_len``.

    Args:
        sig (float): GP length-scale.
        max_block_len (int): Maximum block length in trials.
        T (int): Session length in trials.
        seed (int): Seed of the random generator.
        scale (float): GP amplitude scaling.
        shift (float): GP mean shift.
        clip (tuple[int, int]): Integer clipping range.
        jitter (float): GP kernel jitter.
        use_raw (bool): Use continuous GP values for the metrics.

    Returns:
        tuple[tuple[float, float, float, float], dict]: ``((gap_mean, gap_ge2, run_mean, run_std),
            session)`` as returned by :func:`simulate_3armed_blockswaps_gp` with
            ``return_session=True``.
    """
    # The generator comes first so an invalid seed is reported before an invalid
    # sig or max_block_len.
    rng = np.random.default_rng(seed)
    *metrics, session = simulate_3armed_blockswaps_gp(
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
    return tuple(metrics), session


def format_session_title(
    metrics: tuple[float, float, float, float],
    *,
    sig: float,
    max_block_len: int,
    title: str | None = None,
) -> str:
    """*title* if given, otherwise a summary of the parameters and session metrics.

    Args:
        metrics (tuple[float, float, float, float]): ``(gap_mean, gap_ge2, run_mean, run_std)`` of
            the session.
        sig (float): GP length-scale (keyword-only).
        max_block_len (int): Maximum block length (keyword-only).
        title (str | None): Custom title that replaces the summary. Default None.

    Returns:
        str: The title text.
    """
    gap_mean, gap_ge2, run_mean, run_std = metrics
    return title or (
        f"Example GP session | sig={sig:.2f}, max_block_len={max_block_len} "
        f"| gap_mean={gap_mean:.2f}, frac≥2={gap_ge2:.2f}, "
        f"run_mean={run_mean:.2f}, run_std={run_std:.2f}"
    )


def y_axis_max(vals3: np.ndarray, *, clip: tuple[int, int], use_raw: bool) -> float | int:
    """Largest reward the y-axis must show: the clip maximum, or the data maximum if raw.

    The clip maximum keeps its type (an ``int`` for the default clip); the data maximum is a float.

    Args:
        vals3 (np.ndarray): ``(T, 3)`` reward of each arm per trial.
        clip (tuple[int, int]): Integer clipping range (keyword-only).
        use_raw (bool): Whether the rewards are continuous GP values (keyword-only).

    Returns:
        float | int: ``np.nanmax(vals3)`` if *use_raw*, else ``max(clip)``.
    """
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

    The y-label is "Reward Magnitude"; a caller plotting latent values relabels it. The caller sets
    the title, so the panel can sit in a larger figure. A caller that must set the title first
    passes ``legend=False`` and adds the legend itself.

    Args:
        vals3 (np.ndarray): ``(T, 3)`` reward of each arm per trial.
        boundaries (np.ndarray): 1-D array of trial indices where blocks end (a dashed line is drawn
            before each).
        y_max (float): Largest reward value; sets the y-limit (keyword-only).
        choices (np.ndarray | None): ``(T,)`` int array of chosen arm indices (0-2), drawn as black
            dots; None draws none. Default None.
        ax (plt.Axes | None): Axes to draw into; a new figure is created if None. Default None.
        legend (bool): Add the legend. Default True.

    Returns:
        plt.Axes: The axes drawn into.
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

    Simulates the session, draws it in a new figure and shows it.

    Args:
        sig (float): GP length-scale.
        max_block_len (int): Maximum block length.
        T (int): Session length. Default 150.
        seed (int): Random seed. Default 0.
        scale (float): GP amplitude. Default 2.0.
        shift (float): GP mean shift. Default 2.5.
        clip (tuple[int, int]): Integer reward range. Default ``(1, 5)``.
        jitter (float): GP kernel jitter. Default 1e-6.
        use_raw (bool): Plot continuous GP values. Default False.
        choices (np.ndarray | None): ``(T,)`` int array of chosen arm indices (0-2) to overlay.
            Default None.
        title (str | None): Custom figure title; None builds one from the parameters and metrics.
            Default None.

    Returns:
        dict: The session dict from :func:`simulate_3armed_blockswaps_gp`: ``vals3``,
            ``boundaries``, ``best_arm_series``, ``best_minus_second``, ``gp_int`` and ``gp_raw``.
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

    Sessions use ``max_block_len=80`` and ``min_block_len=70`` with the default GP settings, and one
    shared random generator across all *sig_values*. Progress is printed for each value.

    Args:
        sig_values (np.ndarray): GP length-scale values.
        T (int): Session length. Default 150.
        n_sims (int): Simulations per sig value. Default 1000.
        seed (int): Random seed. Default 0.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(mean_runs, std_runs)``, each of shape
            ``(len(sig_values),)``: the per-sig average over simulations of each session's mean and
            std of best-run length.
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
