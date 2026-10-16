"""Tiny deterministic bandit sessions for plotting tests."""

TOWERS = ("A1", "B1", "C1")
REWARDS = (4, 2, 1)  # best, second, third
RANK_NAMES = ("best", "second", "third")


def make_session(n_trials=24, block_len=8, pick=0, counters=True):
    """A session whose reward values rotate between towers every *block_len* trials.

    Every block change moves the best tower. The animal always chooses the tower of rank
    *pick* (0 best, 1 second, 2 third). With *counters* the session is counter format (it
    logs the good and bad reversal counters, marking every block change a good reversal);
    without, it is blocks-only, as the drifting task records it.
    """
    blocks = [1 + t // block_len for t in range(n_trials)]
    rewards = {
        tower: [REWARDS[(i + blocks[t]) % 3] for t in range(n_trials)]
        for i, tower in enumerate(TOWERS)
    }
    choices_by_tower = {tower: [0] * n_trials for tower in TOWERS}
    choices_by_rank = {rank: [0] * n_trials for rank in RANK_NAMES}
    for t in range(n_trials):
        chosen = next(tw for tw in TOWERS if rewards[tw][t] == REWARDS[pick])
        choices_by_tower[chosen][t] = 1
        choices_by_rank[RANK_NAMES[pick]][t] = 1
    good = [blocks[t] - 1 for t in range(n_trials)]
    counter_series = dict(good_reversals=good, bad_reversals=[0] * n_trials) if counters else {}
    return dict(
        trial=list(range(1, n_trials + 1)),
        blocks=blocks,
        trials_in_block=[t % block_len + 1 for t in range(n_trials)],
        reward_magnitudes_by_tower=rewards,
        choices_by_tower=choices_by_tower,
        choices_by_rank=choices_by_rank,
        has_good=counters,
        has_bad=counters,
        **counter_series,
    )


def session_key(number, day):
    return f"ses-{number}_date-202609{day:02d}"


def make_mouse(sessions, *, pick=0, n_trials=24, first_day=1, counters=True):
    """*sessions* sessions on every other day, starting at day *first_day* of September 2026."""
    return {
        session_key(i + 1, first_day + 2 * i): make_session(n_trials, pick=pick, counters=counters)
        for i in range(sessions)
    }
