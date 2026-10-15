"""Tiny deterministic bandit sessions for plotting tests."""

TOWERS = ("A1", "B1", "C1")
REWARDS = (4, 2, 1)  # best, second, third
RANK_NAMES = ("best", "second", "third")


def make_session(n_trials=24, block_len=8, pick=0):
    """A session whose reward values rotate between towers every *block_len* trials.

    Every block change is a good reversal (the best tower changes). The animal
    always chooses the tower of rank *pick* (0 best, 1 second, 2 third).
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
    return dict(
        trial=list(range(1, n_trials + 1)),
        blocks=blocks,
        trials_in_block=[t % block_len + 1 for t in range(n_trials)],
        reward_magnitudes_by_tower=rewards,
        choices_by_tower=choices_by_tower,
        choices_by_rank=choices_by_rank,
        good_reversals=good,
        bad_reversals=[0] * n_trials,
        has_good=True,
        has_bad=True,
    )


def session_key(number, day):
    return f"ses-{number}_date-202609{day:02d}"


def make_mouse(sessions, *, pick=0, n_trials=24, first_day=1):
    """*sessions* sessions on every other day, starting at day *first_day* of September 2026."""
    return {
        session_key(i + 1, first_day + 2 * i): make_session(n_trials, pick=pick)
        for i in range(sessions)
    }
