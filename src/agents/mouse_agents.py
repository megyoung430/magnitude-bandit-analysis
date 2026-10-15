from abc import ABC, abstractmethod
import numpy as np

# Bump when a change alters what this model computes; comments and refactors don't need a bump.
# Runs that use this model record the version in their saved-run identity.
MODEL_VERSION = 1
class BaseMouse(ABC):
    """Abstract base class for all mouse agents."""
    
    def __init__(self):
        self.curr_opto = False
    
    @abstractmethod
    def choose_action(self, previous_action):
        pass

    def update_q_values(self, action, reward):
        pass

    def update_q_states(self, action, reward):
        pass

    def infer_state(self, bandit_task, action, reward):
        pass

class QLearningMouse():

    """
    This class implements a standard Q-learning agent.
    """

    def __init__(self, num_arms, learning_rate, beta, alpha):
        self.q_values = np.zeros(num_arms)
        self.learning_rate = learning_rate
        self.beta = beta
        self.alpha = alpha
    
    def choose_action(self, previous_action):
        
        stickiness = np.zeros(len(self.q_values))
        if previous_action is not None:
            stickiness[previous_action] += self.alpha
        
        exponent = np.exp(self.beta * (self.q_values + stickiness))
        probabilities = exponent / np.sum(exponent)
        return np.random.choice(range(len(exponent)), p=probabilities)
    
    def update_q_values(self, action, reward):
        self.q_values[action] += self.learning_rate * (reward - self.q_values[action])