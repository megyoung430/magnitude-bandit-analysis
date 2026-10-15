import numpy as np
import random

# Bump when a change alters what this model computes; comments and refactors don't need a bump.
# Runs that use this model record the version in their saved-run identity.
MODEL_VERSION = 1
class DriftingBanditTask():

    """
    This class implements a multi-armed bandit task, similar to the drifting reward bandit.

    swap_mode: "random" chooses a nonzero arm uniformly at each block boundary;
    "most_chosen" chooses the nonzero arm with the most choices in that block.
    Ties are broken uniformly at random. Counts reset after each swap and persist
    across simulate calls when a block is only partially completed.
    """

    def __init__(self, arms, reward_values=[1,2,3,4], drift_rate=0.1, min_block_length=30, max_block_length=35, swap_mode="random"):
        if swap_mode not in ("random", "most_chosen"):
            raise ValueError("swap_mode must be 'random' or 'most_chosen'.")
        self.swap_mode = swap_mode
        self.block_choice_counts = np.zeros(len(arms), dtype=int)
        self.arms = arms
        self.num_arms = len(arms)
        self.reward_values = reward_values
        self.drift_rate = drift_rate
        self.min_block_length = min_block_length
        self.max_block_length = max_block_length
        self.current_block_length = random.randint(self.min_block_length, self.max_block_length)

    def get_reward(self, arm):
        return self.arms[arm]
    
    def drift_arms(self):
        for i in range(self.num_arms):
            current = self.arms[i]

            if current == 0:
                continue

            if random.random() >= self.drift_rate:
                continue

            direction = 1 if random.random() < 0.5 else -1
            occupied = {value for j, value in enumerate(self.arms) if j != i}

            candidate = current + direction

            while self.reward_values[0] <= candidate <= self.reward_values[-1] and candidate in occupied:
                candidate += direction

            if self.reward_values[0] <= candidate <= self.reward_values[-1]:
                self.arms[i] = candidate
    
    def simulate(self, agent, num_trials):
        previous_action = None
        actions = []
        rewards = []
        q_values = []
        # Available rewards before each choice, aligned with actions and rewards.
        self.arm_history = []
        for i in range(num_trials):
            self.arm_history.append(self.arms.copy())
            current_action = agent.choose_action(previous_action)
            actions.append(current_action)
            self.block_choice_counts[current_action] += 1
            
            reward = self.get_reward(current_action)
            rewards.append(reward)

            agent.update_q_values(current_action, reward)
            q_values.append(agent.q_values.copy())
            previous_action = current_action

            self.drift_arms()

            self.current_block_length -= 1
            if self.current_block_length <= 0:
                self.current_block_length = random.randint(self.min_block_length, self.max_block_length)
                zero_arm_index = self.arms.index(0)
                candidates = [i for i, value in enumerate(self.arms) if value != 0]
                if self.swap_mode == "most_chosen":
                    max_choices = max(self.block_choice_counts[i] for i in candidates)
                    candidates = [i for i in candidates
                                  if self.block_choice_counts[i] == max_choices]
                other_index = random.choice(candidates)
                self.arms[zero_arm_index], self.arms[other_index] = (self.arms[other_index], self.arms[zero_arm_index],)      
                self.block_choice_counts.fill(0)
        return actions, rewards, q_values
