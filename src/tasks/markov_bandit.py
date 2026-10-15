"""Markov chain bandit task simulations.

Functions here generate synthetic 3-armed bandit sessions where arm reward
values evolve as discrete Markov chains, with block-swap dynamics that mimic
the experimental paradigm.

The public entry points are thin compositions of small pure helpers:

- :func:`sample_two_markov_bandits` — two Markov reward streams.
- :func:`simulate_3armed_blockswaps_markov` — streams -> 3-armed session -> metrics.
- :func:`sweep_markov_metrics` — metric heatmaps over (max_block_len, pud).
- :func:`count_best_second_switches` — best/second reversals within each block.
- :func:`plot_heatmaps_pair_1` — gap-metric heatmaps from a sweep.
- :func:`plot_heatmaps_pair_2` — run-length heatmaps from a sweep.
- :func:`plot_switch_heatmap` — best/second switch heatmap from a sweep.
- :func:`plot_example_session_markov` — simulate one session and draw it.
- :func:`collect_markov_run_stats` — mean and std of best-run length per pud.

Simulations consume the random generator in a fixed order, so a seeded
generator reproduces a session exactly.
"""

from __future__ import annotations

import operator
from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial, reduce

import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    "collect_markov_run_stats",
    "count_best_second_switches",
    "plot_example_session_markov",
    "plot_heatmaps_pair_1",
    "plot_heatmaps_pair_2",
    "plot_switch_heatmap",
    "sample_two_markov_bandits",
    "simulate_3armed_blockswaps_markov",
    "sweep_markov_metrics",
]

# ---------------------------------------------------------------------------
# Two Markov reward streams
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChainSpec:
    """Parameters of the two-arm Markov chain.

    Attributes:
        n_states: Number of discrete reward states.
        pud: Probability of moving down (and, separately, up) by one step.
        min_hold: Steps an arm must stay in a state before it may change.
        enforce_distinct: Keep the two arms in different states.

    Raises:
        ValueError: If distinctness cannot be satisfied with *n_states*, or if
            *pud* exceeds 0.5 (the stay probability would be negative).
    """

    n_states: int
    pud: float
    min_hold: int
    enforce_distinct: bool

    def __post_init__(self) -> None:
        if self.enforce_distinct and self.n_states < 2:
            raise ValueError("Need at least 2 states in val_list to enforce distinctness.")
        if self.pstay < 0:
            raise ValueError("pud must be ≤ 0.5")

    @property
    def pstay(self) -> float:
        """Probability of staying in the current state, ``1 - 2 * pud``."""
        return 1 - 2 * self.pud


def draw_initial_states(
    rng: np.random.Generator, n_states: int, enforce_distinct: bool
) -> np.ndarray:
    """Draw the two arms' starting state indices, distinct if requested."""
    if enforce_distinct:
        return rng.choice(n_states, size=2, replace=False)
    return rng.integers(0, n_states, size=2)


def sample_direction(rng: np.random.Generator, *, pud: float, pstay: float) -> int:
    """Sample -1 (down), 0 (stay), or +1 (up) using the Markov transition probabilities."""
    r = rng.random()
    if r < pud:
        return -1
    if r < pud + pstay:
        return 0
    return 1


def _first_free_state(
    *, cur: int, direction: int, other_state: int, chain: ChainSpec
) -> int | None:
    """First state reached walking from *cur* along *direction* that the other arm does not occupy.

    Returns ``None`` if the walk leaves the state range first.
    """
    s = cur
    while 0 <= s + direction < chain.n_states:
        s += direction
        if not (chain.enforce_distinct and s == other_state):
            return s
    return None


def step_in_direction(*, cur: int, direction: int, other_state: int, chain: ChainSpec) -> int:
    """Move *cur* in *direction*, skipping *other_state* if needed.

    If the move runs off the end of the state range, the opposite direction is
    tried.

    Raises:
        RuntimeError: If neither direction reaches a free state.
    """
    if direction == 0:
        return cur
    for step in (direction, -direction):
        s = _first_free_state(cur=cur, direction=step, other_state=other_state, chain=chain)
        if s is not None:
            return s
    raise RuntimeError("Could not resolve collision; val_list may be too small.")


def advance_arms(
    rng: np.random.Generator, *, cur: np.ndarray, hold: np.ndarray, chain: ChainSpec
) -> np.ndarray:
    """Next state of both arms, visiting them in random order.

    An arm whose dwell count is below ``chain.min_hold`` stays put and consumes
    no random draw. The second arm sees the first arm's new state.
    """
    nxt = cur.copy()
    for arm in rng.permutation(2):
        if hold[arm] < chain.min_hold:
            continue
        other_state = nxt[1 - arm]
        direction = sample_direction(rng, pud=chain.pud, pstay=chain.pstay)
        nxt[arm] = step_in_direction(
            cur=cur[arm], direction=direction, other_state=other_state, chain=chain
        )
    return nxt


def resolve_collision(nxt: np.ndarray, n_states: int) -> np.ndarray:
    """Return a copy of two equal states with arm 0 nudged up (or down at the top).

    Raises:
        RuntimeError: If neither neighbouring state exists.
    """
    fixed = nxt.copy()
    if fixed[0] + 1 < n_states and fixed[0] + 1 != fixed[1]:
        fixed[0] += 1
    elif fixed[0] - 1 >= 0 and fixed[0] - 1 != fixed[1]:
        fixed[0] -= 1
    else:
        raise RuntimeError("Distinctness enforcement failed unexpectedly.")
    return fixed


def update_hold(*, hold: np.ndarray, prev: np.ndarray, new: np.ndarray) -> np.ndarray:
    """Dwell counts after a step: reset to 1 for arms that changed, else +1."""
    return np.where(new != prev, 1, hold + 1)


def sample_two_markov_bandits(
    T: int,
    pud: float,
    val_list: tuple | list,
    rng: np.random.Generator | None = None,
    min_hold: int = 5,
    enforce_distinct: bool = True,
) -> np.ndarray:
    """Sample two Markov-chain reward streams of length *T*.

    Each arm independently transitions between discrete states in *val_list*
    with probability *pud* of moving up or down at each step.  When
    *enforce_distinct* is ``True``, the two arms are prevented from sharing the
    same state.

    Args:
        T: Number of time steps.
        pud: Probability of moving up or down by one state per step.
            Must satisfy ``2 * pud <= 1``.
        val_list: Ordered list of reward values (states).
        rng: Optional random generator.
        min_hold: Minimum number of steps an arm must stay in a state before
            it can change (default: 5).
        enforce_distinct: If ``True``, prevent the two arms from having the
            same reward value at any time step (default: True).

    Returns:
        Array of shape ``(T, 2)`` with reward values from *val_list*.

    Raises:
        ValueError: If *pud* is too large or *val_list* is too short.
    """
    if rng is None:
        rng = np.random.default_rng()

    val_list = np.array(val_list)
    n_states = len(val_list)
    chain = ChainSpec(n_states, pud, min_hold, enforce_distinct)

    state_vals = np.zeros((T, 2), dtype=int)
    state_vals[0] = draw_initial_states(rng, n_states, enforce_distinct)
    hold = np.ones(2, dtype=int)

    # Sequential: each step reads the previous state and the dwell counts.
    for t in range(T - 1):
        nxt = advance_arms(rng, cur=state_vals[t], hold=hold, chain=chain)
        if enforce_distinct and nxt[0] == nxt[1]:
            nxt = resolve_collision(nxt, n_states)
        state_vals[t + 1] = nxt
        hold = update_hold(hold=hold, prev=state_vals[t], new=nxt)

    return val_list[state_vals]


# ---------------------------------------------------------------------------
# Three-armed block-swap session
# ---------------------------------------------------------------------------


def count_best_second_switches(vals3: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    """Count strict best-arm reversals within each observed block.

    Ignore tied trials, comparing successive unique winners across ties.
    Reset at each boundary; include zero-switch and final partial blocks.
    *vals3* should hold ``-inf`` for each trial's inactive arm (see
    :func:`active_arm_values`) so it never wins, whatever the active values.
    """
    edges = np.r_[0, boundaries, len(vals3)]
    counts = []
    for start, stop in zip(edges[:-1], edges[1:], strict=True):
        block = vals3[start:stop]
        maxima = block.max(axis=1, keepdims=True)
        unique = (block == maxima).sum(axis=1) == 1
        winners = block[unique].argmax(axis=1)
        counts.append(np.count_nonzero(winners[1:] != winners[:-1]))
    return np.asarray(counts, dtype=int)


def check_min_block_len(min_block_len: int, max_block_len: int) -> None:
    """Raise ``ValueError`` unless ``1 <= min_block_len <= max_block_len``."""
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
    """
    for name, value in values.items():
        if not np.all(np.isfinite(np.asarray(value, dtype=float))):
            raise ValueError(f"{name} must be finite, got {value!r}")


def check_sweep_max_block_lens(max_block_lens: np.ndarray) -> None:
    """Raise ``ValueError`` if a sweep row would get a minimum block length below 1.

    A sweep derives ``min_block_len = max_block_len - 10`` for every row, so each
    entry of *max_block_lens* must exceed 10.
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

    Block lengths are uniform on ``[min_block_len, max_block_len]``; boundaries at
    or beyond *T* are dropped.
    """
    block_lengths = rng.integers(min_block_len, max_block_len + 1, size=n_blocks_draw)
    boundaries = np.cumsum(block_lengths)
    return boundaries[boundaries < T]


def remaining_active_arm(best_arm: int, *, active_arms: list[int]) -> int:
    """First active arm that is not *best_arm*; it stays active across the swap."""
    return next(a for a in active_arms if a != best_arm)


def swap_arm_roles(
    row: np.ndarray,
    inactive_arm: int,
    active_arms: list[int],
    arm_to_stream: dict[int, int],
) -> tuple[int, list[int], dict[int, int]]:
    """Arm roles after a block boundary.

    The active arm that is best in *row* becomes inactive. The previously
    inactive arm takes over the best arm's stream, and the first remaining active
    arm keeps its own stream.

    Only active arms compete: the inactive arm pays 0, which would beat negative
    active values and leave two arms sharing a stream. Ties go to the lowest arm
    index, whatever the order of *active_arms*.

    Returns:
        ``(inactive_arm, active_arms, arm_to_stream)`` for the next block.
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

    Two arms are active and read the two *streams*; the third is inactive and
    pays zero. At each boundary the best active arm turns inactive (see
    :func:`swap_arm_roles`), so only the roles change, not the streams.

    *boundaries* come from :func:`draw_block_boundaries`: strictly increasing and
    inside ``[1, len(streams))``. Repeated boundaries, boundaries below 1 and
    boundaries at or beyond ``len(streams)`` are ignored for direct callers.

    Returns:
        ``(vals3, inactive_arms)`` of shapes ``(T, 3)`` and ``(T,)``.
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
    """Per-trial reward of the three arms, shape ``(T, 3)``; see :func:`build_arm_roles`."""
    return build_arm_roles(streams, boundaries)[0]


def active_arm_values(vals3: np.ndarray, inactive_arms: np.ndarray) -> np.ndarray:
    """Copy of *vals3* with each trial's inactive arm set to ``-inf``.

    The inactive arm pays 0 whatever the active values are, so it must not compete
    for best or second-best when active values are zero or negative.
    """
    masked = vals3.copy()
    masked[np.arange(len(vals3)), inactive_arms] = -np.inf
    return masked


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
    active_vals: np.ndarray,
    boundaries: np.ndarray,
    best_arm_series: np.ndarray,
    best_minus_second: np.ndarray,
) -> dict:
    """Session dict returned by :func:`simulate_3armed_blockswaps_markov`."""
    return {
        "vals3": vals3,
        "boundaries": boundaries,
        "best_arm_series": best_arm_series,
        "best_minus_second": best_minus_second,
        "best_second_switches_per_block": count_best_second_switches(active_vals, boundaries),
    }


def simulate_3armed_blockswaps_markov(
    T: int,
    pud: float,
    max_block_len: int,
    min_block_len: int | None = None,
    n_blocks_draw: int = 1000,
    val_list: tuple | list = (1, 2, 3, 4),
    rng: np.random.Generator | None = None,
    min_hold: int = 5,
    enforce_distinct: bool = True,
    return_session: bool = False,
):
    """Simulate a 3-armed Markov bandit session with block-swap dynamics.

    Args:
        T: Session length in trials.
        pud: State-transition probability.
        max_block_len: Maximum block length.
        min_block_len: Minimum block length (defaults to
            ``max_block_len - 10``). A value below 1 (including a default derived
            from ``max_block_len <= 10``) raises ``ValueError``.
        n_blocks_draw: Pre-sampled block count (default: 1000).
        val_list: Discrete reward state values (default: ``(1, 2, 3, 4)``).
        rng: Optional random generator.
        min_hold: Minimum dwell time per state (default: 5).
        enforce_distinct: Keep arm values distinct (default: True).
        return_session: Also return a session dict (default: False).

    Returns:
        ``(mean_best_minus_second, frac_gap_ge2, mean_best_run_length,
        std_best_run_length)`` and optionally a *session* dict.
    """
    if rng is None:
        rng = np.random.default_rng()
    if min_block_len is None:
        min_block_len = max_block_len - 10
    check_min_block_len(min_block_len, max_block_len)
    check_finite(val_list=val_list)

    # Seeded results depend on this draw order: streams first, then block lengths.
    streams = sample_two_markov_bandits(
        T=T,
        pud=pud,
        val_list=val_list,
        rng=rng,
        min_hold=min_hold,
        enforce_distinct=enforce_distinct,
    )
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
        active_vals=active_vals,
        boundaries=boundaries,
        best_arm_series=best_arm_series,
        best_minus_second=best_minus_second,
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
            ``n_sims``, so 0 raises ZeroDivisionError and a negative value gives -0.0.
    """
    values = list(values)
    return reduce(operator.add, values, 0.0) / (len(values) if count is None else count)


def summarize_sweep_cell(
    results: list[tuple], *, n_sims: int, include_switches: bool
) -> tuple[float, ...]:
    """Average the simulations of one (max_block_len, pud) cell.

    Args:
        results: One :func:`simulate_3armed_blockswaps_markov` return per
            simulation; with a session dict last when *include_switches*.
        n_sims: Number of simulations requested for the cell; the divisor of the
            four means. 0 raises ZeroDivisionError and a negative value gives -0.0.
        include_switches: Also pool the best/second switches per observed block.

    Returns:
        ``(gap_mean, gap_ge2, run_mean, run_std)``, followed by ``switch_mean``
        when *include_switches* is True.
    """
    means = tuple(mean_left_to_right((r[k] for r in results), count=n_sims) for k in range(4))
    if not include_switches:
        return means
    counts = [r[4]["best_second_switches_per_block"] for r in results]
    return (*means, sum(int(c.sum()) for c in counts) / sum(len(c) for c in counts))


DEFAULT_PUD_VALUES = np.linspace(0.01, 0.25, 25)
DEFAULT_MAX_BLOCK_LENS = np.arange(20, 81, 1)


def sweep_markov_metrics(
    pud_values: np.ndarray = DEFAULT_PUD_VALUES,
    max_block_lens: np.ndarray = DEFAULT_MAX_BLOCK_LENS,
    T: int = 150,
    n_sims: int = 1000,
    seed: int = 0,
    val_list: tuple | list = (1, 2, 3, 4),
    min_hold: int = 5,
    enforce_distinct: bool = True,
    include_switches: bool = False,
) -> tuple[np.ndarray, ...]:
    """Sweep over Markov parameter combinations and compute summary metrics.

    Args:
        pud_values: Array of transition probabilities to sweep.
        max_block_lens: Array of maximum block lengths to sweep.
        T: Session length (default: 150).
        n_sims: Simulations per parameter combination (default: 1000).
        seed: Random seed (default: 0).
        val_list: Discrete reward values (default: ``(1, 2, 3, 4)``).
        min_hold: Minimum dwell time (default: 5).
        enforce_distinct: Enforce distinctness (default: True).
        include_switches: Append mean best/second switches per observed block.
            Ties are ignored; block-boundary changes are excluded.

    Returns:
        Four 2-D arrays ``(H_gap_mean, H_gap_ge2, H_run_mean, H_run_std)``
        of shape ``(len(max_block_lens), len(pud_values))``. When
        ``include_switches=True``, append ``H_switch_mean``, pooling all observed
        blocks (including final partial blocks) across simulations.
    """
    check_sweep_max_block_lens(max_block_lens)
    rng = np.random.default_rng(seed)
    simulate = partial(
        simulate_3armed_blockswaps_markov,
        T=T,
        val_list=val_list,
        rng=rng,
        min_hold=min_hold,
        enforce_distinct=enforce_distinct,
        return_session=include_switches,
    )
    shape = (len(max_block_lens), len(pud_values))
    heatmaps = [np.zeros(shape) for _ in range(5 if include_switches else 4)]

    # One shared rng drawn in row-major cell order: the cells are not independent.
    for i, mbl in enumerate(max_block_lens):
        for j, pud in enumerate(pud_values):
            results = [
                simulate(pud=float(pud), max_block_len=int(mbl), min_block_len=int(mbl) - 10)
                for _ in range(n_sims)
            ]
            cell = summarize_sweep_cell(results, n_sims=n_sims, include_switches=include_switches)
            for heatmap, value in zip(heatmaps, cell, strict=True):
                heatmap[i, j] = value
            print(f"Done pud={pud:.4f}, max_block_len={mbl}")

    return tuple(heatmaps)


# ---------------------------------------------------------------------------
# Heatmaps and example session plots
# ---------------------------------------------------------------------------


def plot_heatmaps_pair_1(
    H_gap_mean: np.ndarray,
    H_gap_ge2: np.ndarray,
    pud_values: np.ndarray,
    max_block_lens: np.ndarray,
) -> None:
    """Plot gap-metric heatmaps for the Markov bandit sweep.

    Args:
        H_gap_mean: Mean (best − second-best) heatmap.
        H_gap_ge2: Fraction of trials with gap ≥ 2 heatmap.
        pud_values: Transition probability values.
        max_block_lens: Maximum block-length values.
    """
    extent = [pud_values[0], pud_values[-1], max_block_lens[0], max_block_lens[-1]]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    im0 = axes[0].imshow(H_gap_mean, origin="lower", aspect="auto", extent=extent)
    axes[0].set_title("Avg (best − second best)")
    axes[0].set_xlabel("p_ud")
    axes[0].set_ylabel("max_block_len")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(H_gap_ge2, origin="lower", aspect="auto", extent=extent, vmin=0, vmax=1)
    axes[1].set_title("Frac trials (best − second ≥ 2)")
    axes[1].set_xlabel("p_ud")
    axes[1].set_ylabel("max_block_len")
    fig.colorbar(im1, ax=axes[1])

    plt.show()


def plot_heatmaps_pair_2(
    H_run_mean: np.ndarray,
    H_run_std: np.ndarray,
    pud_values: np.ndarray,
    max_block_lens: np.ndarray,
) -> None:
    """Plot run-length-metric heatmaps for the Markov bandit sweep.

    Args:
        H_run_mean: Mean best-run length heatmap.
        H_run_std: Std dev of best-run length heatmap.
        pud_values: Transition probability values.
        max_block_lens: Maximum block-length values.
    """
    extent = [pud_values[0], pud_values[-1], max_block_lens[0], max_block_lens[-1]]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    im0 = axes[0].imshow(H_run_mean, origin="lower", aspect="auto", extent=extent)
    axes[0].set_title("Avg run length (best arm stays best)")
    axes[0].set_xlabel("p_ud")
    axes[0].set_ylabel("max_block_len")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(H_run_std, origin="lower", aspect="auto", extent=extent)
    axes[1].set_title("Std dev of best-run length")
    axes[1].set_xlabel("p_ud")
    axes[1].set_ylabel("max_block_len")
    fig.colorbar(im1, ax=axes[1])

    plt.show()


def plot_switch_heatmap(
    H_switch_mean: np.ndarray,
    pud_values: np.ndarray,
    max_block_lens: np.ndarray,
) -> None:
    """Plot the mean count of best/second reversals per observed block."""
    extent = [pud_values[0], pud_values[-1], max_block_lens[0], max_block_lens[-1]]
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    im = ax.imshow(H_switch_mean, origin="lower", aspect="auto", extent=extent, vmin=0)
    ax.set_title("Avg best/second switches within a block")
    ax.set_xlabel("p_ud")
    ax.set_ylabel("max_block_len")
    fig.colorbar(im, ax=ax, label="Switches per observed block")
    plt.show()


def simulate_example_session(
    *,
    pud: float,
    max_block_len: int,
    val_list: tuple | list,
    T: int,
    seed: int,
) -> tuple[tuple[float, ...], dict]:
    """Simulate one seeded session.

    Returns:
        ``((gap_mean, gap_ge2, run_mean, run_std), session)`` as returned by
        :func:`simulate_3armed_blockswaps_markov` with ``return_session=True``.
    """
    *metrics, session = simulate_3armed_blockswaps_markov(
        T=T,
        pud=float(pud),
        max_block_len=int(max_block_len),
        min_block_len=int(max_block_len) - 10,
        val_list=val_list,
        min_hold=5,
        rng=np.random.default_rng(seed),
        return_session=True,
    )
    return tuple(metrics), session


def format_session_title(
    metrics: tuple[float, ...], *, pud: float, max_block_len: int, title: str | None = None
) -> str:
    """*title* if given, otherwise a summary of the parameters and session metrics."""
    gap_mean, gap_ge2, run_mean, run_std = metrics
    return title or (
        f"Example session | pud={pud:.3f}, max_block_len={max_block_len} "
        f"| gap_mean={gap_mean:.2f}, frac≥2={gap_ge2:.2f}, "
        f"run_mean={run_mean:.2f}, run_std={run_std:.2f}"
    )


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


def plot_example_session_markov(
    pud: float,
    max_block_len: int,
    val_list: tuple | list = (1, 2, 3, 4),
    T: int = 150,
    seed: int = 0,
    choices: np.ndarray | None = None,
    title: str | None = None,
) -> dict:
    """Plot a single example Markov bandit session.

    Args:
        pud: State-transition probability.
        max_block_len: Maximum block length.
        val_list: Discrete reward values (default: ``(1, 2, 3, 4)``).
        T: Session length (default: 150).
        seed: Random seed (default: 0).
        choices: Optional ``(T,)`` int array of chosen arm indices (0–2).
        title: Custom figure title.

    Returns:
        The session dict from :func:`simulate_3armed_blockswaps_markov`.
    """
    metrics, session = simulate_example_session(
        pud=pud, max_block_len=max_block_len, val_list=val_list, T=T, seed=seed
    )
    ax = plot_session_panel(
        session["vals3"],
        session["boundaries"],
        y_max=max(val_list),
        choices=choices,
        legend=False,
    )
    # Formatting the title validates pud, so it comes before the legend: a bad pud
    # fails without a legend on the figure.
    ax.set_title(format_session_title(metrics, pud=pud, max_block_len=max_block_len, title=title))
    ax.legend(loc="upper left")
    plt.show()
    return session


def collect_markov_run_stats(
    pud_values: np.ndarray,
    T: int = 150,
    n_sims: int = 1000,
    seed: int = 0,
    val_list: tuple | list = (1, 2, 3, 4, 5),
) -> tuple[np.ndarray, np.ndarray]:
    """Collect mean and std dev of best-arm run lengths for each pud value.

    Args:
        pud_values: Array of transition probabilities.
        T: Session length (default: 150).
        n_sims: Simulations per pud value (default: 1000).
        seed: Random seed (default: 0).
        val_list: Discrete reward values (default: ``(1, 2, 3, 4, 5)``).

    Returns:
        ``(mean_runs, std_runs)`` arrays of shape ``(len(pud_values),)``.
    """
    rng = np.random.default_rng(seed)
    mean_runs, std_runs = [], []

    for pud in pud_values:
        run_mean_acc = 0.0
        run_std_acc = 0.0
        for _ in range(n_sims):
            _, _, run_mean, run_std = simulate_3armed_blockswaps_markov(
                T=T, pud=pud, max_block_len=80, min_block_len=70, val_list=val_list, rng=rng
            )
            run_mean_acc += run_mean
            run_std_acc += run_std
        mean_runs.append(run_mean_acc / n_sims)
        std_runs.append(run_std_acc / n_sims)
        print(f"Markov done pud={pud}")

    return np.array(mean_runs), np.array(std_runs)
