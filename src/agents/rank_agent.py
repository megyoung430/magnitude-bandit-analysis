"""A toy ranked-value agent: the known answer for the rank-signature tests.

The agent keeps the last payout of each arm (its believed value) and sits on the arm it
believes is best. After every payout it compares the arm it is on with the best of the other
two: while the arm is not below that alternative it stays, the moment its payout falls below it
the agent moves to that alternative. Nothing else enters: the arm it believes is worst is never
compared with, so it is chosen only when the arm above it drops below it (promotion), and the
step at a believed crossing is abrupt. A lapse (*epsilon*) makes a uniformly random choice with
that probability, so the agent is not deterministic. Before every arm has been visited it
visits the unvisited ones.

It is the contrast to a softmax value learner (:class:`src.agents.drifting_simulation.ValuePolicy`
with learning rate 1 is the value learner of the same believed values), whose stay probability
falls smoothly with both alternatives' values.
"""

import numpy as np

from src.agents import drifting_simulation as sim

DEFAULT_EPSILON = 0.05


class RankPolicy:
    """The ranked-value toy agent as a :class:`~src.agents.drifting_simulation.Policy`.

    Args:
        layout (sim.MouseLayout): Gives the ports (the agent needs only how many).
        epsilon (float): Probability of a uniformly random choice, in [0, 1).

    Raises:
        ValueError: If *epsilon* is outside [0, 1).
    """

    def __init__(self, layout: "sim.MouseLayout", epsilon: float = DEFAULT_EPSILON):
        if not 0 <= epsilon < 1:
            raise ValueError(f"epsilon must be in [0, 1), got {epsilon}")
        self.epsilon = epsilon
        self.believed = np.full(len(layout.ports), np.nan)
        self.current: int | None = None

    def start_session(self) -> None:
        self.current = None

    def _target(self) -> np.ndarray:
        """Probability of each arm without the lapse: all mass on the arm it means to choose."""
        n = len(self.believed)
        unvisited = np.isnan(self.believed)
        if unvisited.any():
            return unvisited / unvisited.sum()
        best = int(np.argmax(self.believed))
        target = best if self.current is None else self.current
        others = [a for a in range(n) if a != target]
        best_other = max(others, key=lambda a: self.believed[a])
        if self.believed[target] < self.believed[best_other]:
            target = best_other
        probabilities = np.zeros(n)
        probabilities[target] = 1.0
        return probabilities

    def probabilities(self) -> np.ndarray:
        n = len(self.believed)
        return (1 - self.epsilon) * self._target() + self.epsilon / n

    def update(self, action: int, magnitude: float) -> None:
        self.believed[action] = magnitude
        self.current = action
