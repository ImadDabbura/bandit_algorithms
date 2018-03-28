"""
Implementing epsilon-Greedy algorithm in both standard and annealing forms.
"""

import numpy as np


class EpsilonGreedy:
    """Implementing standard epsilon-Greedy algorithm.

    Parameters
    ----------
    epsilon : float
        probability of exploration and trying random arms.
    counts : list or array-like
        number of times each arm was played, shape (num_arms,).
    values : list or array-like
        estimated value (mean) of rewards of each arm, shape (num_arms,).

    Attributes
    ----------
    select_arm : int
        select the best arm with probability (1 - epsilon) or randomly select
        any arm available with probability (1 / N) for each arm.
    """

    def __init__(self, epsilon, counts=None, values=None):
        self.epsilon = epsilon
        self.counts = counts
        self.values = values

    def initialize(self, n_arms):
        """Initialize counts and values array with zeros."""
        self.counts = np.zeros(n_arms, dtype=int)
        self.values = np.zeros(n_arms, dtype=float)

    def select_arm(self):
        z = np.random.random()
        if z > self.epsilon:
            # Pick the best arm
            return np.argmax(self.values)
        # Randomly pick any arm with prob 1 / len(self.counts)
        return np.random.randint(0, len(self.values))

    def update(self, chosen_arm, reward):
        """Update counts and estimated value of rewards for the chosen arm."""
        # Increment chosen arm's count by one
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]

        # Recompute the estimated value of chosen arm using new reward
        value = self.values[chosen_arm]
        new_value = value * ((n - 1) / n) + reward / n
        self.values[chosen_arm] = new_value


class AnnealingEpsilonGreedy(EpsilonGreedy):
    """Implementing annealing epsilon-Greedy algorithm.

    Using this method, the algorithm would explore a lot at the beginning of
    the experiment because the epsilon would be close to inf and then start
    decaying with time where it will start exploiting more.
    """
    def __init__(self, counts=None, values=None):
        self.counts = counts
        self.values = values

    def select_arm(self):
        # Epsilon decay schedule
        t = np.sum(self.counts) + 1
        epsilon = 1 / np.log(t + 0.0000001)

        z = np.random.random()
        if z > epsilon:
            # Pick the best arm
            return np.argmax(self.values)
        # Randomly pick any arm with prob 1 / len(self.counts)
        return np.random.randint(0, len(self.values))
