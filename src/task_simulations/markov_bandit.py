"""Markov chain bandit task simulations.

Functions here generate synthetic 3-armed bandit sessions where arm reward
values evolve as discrete Markov chains, with block-swap dynamics that mimic
the experimental paradigm.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


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
    if enforce_distinct and n_states < 2:
        raise ValueError("Need at least 2 states in val_list to enforce distinctness.")

    pstay = 1 - 2 * pud
    if pstay < 0:
        raise ValueError("pud must be ≤ 0.5")

    state_vals = np.zeros((T, 2), dtype=int)
    if enforce_distinct:
        state_vals[0] = rng.choice(n_states, size=2, replace=False)
    else:
        state_vals[0] = rng.integers(0, n_states, size=2)

    hold = np.ones(2, dtype=int)

    def _step_in_direction(cur: int, direction: int, other_state: int) -> int:
        """Move *cur* in *direction*, skipping *other_state* if needed."""
        if direction == 0:
            return cur
        s = cur
        while True:
            s_next = s + direction
            if s_next < 0 or s_next >= n_states:
                break
            s = s_next
            if not (enforce_distinct and s == other_state):
                return s
        # fallback: try opposite direction
        direction2 = -direction
        s = cur
        while True:
            s_next = s + direction2
            if s_next < 0 or s_next >= n_states:
                break
            s = s_next
            if not (enforce_distinct and s == other_state):
                return s
        raise RuntimeError("Could not resolve collision; val_list may be too small.")

    def _sample_direction() -> int:
        """Sample -1 (down), 0 (stay), or +1 (up) using the Markov transition probabilities."""
        r = rng.random()
        if r < pud:
            return -1
        elif r < pud + pstay:
            return 0
        return +1

    for t in range(T - 1):
        cur = state_vals[t].copy()
        order = rng.permutation(2)
        nxt = cur.copy()

        for arm in order:
            other = 1 - arm
            other_state = nxt[other]
            if hold[arm] < min_hold:
                nxt[arm] = cur[arm]
                continue
            nxt[arm] = _step_in_direction(cur[arm], _sample_direction(), other_state)

        if enforce_distinct and nxt[0] == nxt[1]:
            if nxt[0] + 1 < n_states and nxt[0] + 1 != nxt[1]:
                nxt[0] += 1
            elif nxt[0] - 1 >= 0 and nxt[0] - 1 != nxt[1]:
                nxt[0] -= 1
            else:
                raise RuntimeError("Distinctness enforcement failed unexpectedly.")

        state_vals[t + 1] = nxt
        for arm in range(2):
            hold[arm] = (
                1 if state_vals[t + 1, arm] != state_vals[t, arm] else hold[arm] + 1
            )

    return val_list[state_vals]


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
            ``max_block_len - 10``).
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

    streams = sample_two_markov_bandits(
        T=T, pud=pud, val_list=val_list, rng=rng,
        min_hold=min_hold, enforce_distinct=enforce_distinct,
    )

    block_lengths = rng.integers(min_block_len, max_block_len + 1, size=n_blocks_draw)
    boundaries = np.cumsum(block_lengths)
    boundaries = boundaries[boundaries < T]
    boundary_set = set(boundaries.tolist())

    inactive_arm = 2
    active_arms = [0, 1]
    arm_to_stream = {0: 0, 1: 1}

    vals3 = np.zeros((T, 3), dtype=float)

    for t in range(T):
        vals3[t, inactive_arm] = 0.0
        for a in active_arms:
            vals3[t, a] = streams[t, arm_to_stream[a]]

        if (t + 1) in boundary_set:
            best_arm = int(np.argmax(vals3[t]))
            old_inactive = inactive_arm
            inherited_stream = arm_to_stream.get(best_arm, 0)
            other_active = [a for a in active_arms if a != best_arm]
            if not other_active:
                other_active = [a for a in [0, 1, 2] if a != best_arm and a != old_inactive][:1]
            other_active = other_active[0]
            inactive_arm = best_arm
            active_arms = [old_inactive, other_active]
            other_stream = arm_to_stream[other_active]
            arm_to_stream = {other_active: other_stream, old_inactive: inherited_stream}

    # Gap metrics
    sorted_vals = np.sort(vals3, axis=1)
    best_minus_second = sorted_vals[:, -1] - sorted_vals[:, -2]
    mean_best_minus_second = float(best_minus_second.mean())
    frac_gap_ge2 = float((best_minus_second >= 2).mean())

    # Run-length metrics
    best_arm_series = np.argmax(vals3, axis=1)
    changes = np.where(best_arm_series[1:] != best_arm_series[:-1])[0] + 1
    run_lengths = np.diff(np.r_[0, changes, T]).astype(float)
    mean_best_run_length = float(run_lengths.mean())
    std_best_run_length = float(run_lengths.std(ddof=0))

    if return_session:
        session = {
            "vals3": vals3,
            "boundaries": boundaries,
            "best_arm_series": best_arm_series,
            "best_minus_second": best_minus_second,
        }
        return (mean_best_minus_second, frac_gap_ge2,
                mean_best_run_length, std_best_run_length, session)

    return mean_best_minus_second, frac_gap_ge2, mean_best_run_length, std_best_run_length


def sweep_markov_metrics(
    pud_values: np.ndarray = np.linspace(0.01, 0.25, 25),
    max_block_lens: np.ndarray = np.arange(20, 81, 1),
    T: int = 150,
    n_sims: int = 1000,
    seed: int = 0,
    val_list: tuple | list = (1, 2, 3, 4),
    min_hold: int = 5,
    enforce_distinct: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

    Returns:
        Four 2-D arrays ``(H_gap_mean, H_gap_ge2, H_run_mean, H_run_std)``
        of shape ``(len(max_block_lens), len(pud_values))``.
    """
    rng = np.random.default_rng(seed)
    shape = (len(max_block_lens), len(pud_values))
    H_gap_mean = np.zeros(shape)
    H_gap_ge2 = np.zeros(shape)
    H_run_mean = np.zeros(shape)
    H_run_std = np.zeros(shape)

    for i, mbl in enumerate(max_block_lens):
        for j, pud in enumerate(pud_values):
            acc = [0.0, 0.0, 0.0, 0.0]
            for _ in range(n_sims):
                gm, gg, rm, rs = simulate_3armed_blockswaps_markov(
                    T=T, pud=float(pud), max_block_len=int(mbl),
                    min_block_len=int(mbl) - 10, val_list=val_list,
                    rng=rng, min_hold=min_hold,
                    enforce_distinct=enforce_distinct, return_session=False,
                )
                acc[0] += gm; acc[1] += gg; acc[2] += rm; acc[3] += rs

            H_gap_mean[i, j] = acc[0] / n_sims
            H_gap_ge2[i, j] = acc[1] / n_sims
            H_run_mean[i, j] = acc[2] / n_sims
            H_run_std[i, j] = acc[3] / n_sims

            print(f"Done pud={pud:.4f}, max_block_len={mbl}")

    return H_gap_mean, H_gap_ge2, H_run_mean, H_run_std


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

    im1 = axes[1].imshow(H_gap_ge2, origin="lower", aspect="auto",
                         extent=extent, vmin=0, vmax=1)
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
    rng = np.random.default_rng(seed)
    gap_mean, gap_ge2, run_mean, run_std, session = simulate_3armed_blockswaps_markov(
        T=T, pud=float(pud), max_block_len=int(max_block_len),
        min_block_len=int(max_block_len) - 10, val_list=val_list,
        min_hold=5, rng=rng, return_session=True,
    )

    vals3 = session["vals3"]
    boundaries = session["boundaries"]
    x = np.arange(T)

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    for arm_idx in range(3):
        ax.plot(x, vals3[:, arm_idx], linewidth=2.5, label=f"Arm {arm_idx}")

    if choices is not None:
        ch = np.asarray(choices, dtype=int)
        ax.scatter(x, vals3[np.arange(T), ch], s=25, color="black",
                   zorder=5, label="Choice")

    first = True
    for b in boundaries:
        ax.axvline(b - 0.5, linestyle="--", color="k", linewidth=2.0,
                   label="Block" if first else None)
        first = False

    ax.set_xlim(0, T - 1)
    ax.set_ylim(-0.5, max(val_list) + 0.5)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Reward Magnitude")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(
        title or (
            f"Example session | pud={pud:.3f}, max_block_len={max_block_len} "
            f"| gap_mean={gap_mean:.2f}, frac≥2={gap_ge2:.2f}, "
            f"run_mean={run_mean:.2f}, run_std={run_std:.2f}"
        )
    )
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
                T=T, pud=pud, max_block_len=80, min_block_len=70,
                val_list=val_list, rng=rng,
            )
            run_mean_acc += run_mean
            run_std_acc += run_std
        mean_runs.append(run_mean_acc / n_sims)
        std_runs.append(run_std_acc / n_sims)
        print(f"Markov done pud={pud}")

    return np.array(mean_runs), np.array(std_runs)
