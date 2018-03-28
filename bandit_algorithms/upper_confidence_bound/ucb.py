"""
Implementing Upper Confidence Bound algorithm.
"""

import numpy as np


class UCB:
    """Implementing `UCB` algorithm.

    Parameters
    ----------
    counts : list or array-like
        number of times each arm was played, shape (num_arms,).
    values : list or array-like
        estimated value (mean) of rewards of each arm, shape (num_arms,).

    Attributes
    ----------
    select_arm : int
        select the arm based on the knowledge we know about each one. All arms
        will be rescaled based on how many times they were selected to avoid
        neglecting arms that are good overall but the algorithm has had a
        negative initial interactions.
    """
    def __init__(self, counts=None, values=None):
        self.counts = counts
        self.values = values

    def initialize(self, n_arms):
        """Initialize counts and values array with zeros."""
        self.counts = np.zeros(n_arms, dtype=int)
        self.values = np.zeros(n_arms, dtype=float)

    def select_arm(self):
        n_arms = len(self.counts)

        # Make sure to visit each arm at least once at the beginning
        for arm in range(n_arms):
            if self.counts[arm] == 0:
                return arm

        # Compute estimated value using original values and bonus
        ucb_values = np.zeros(n_arms)
        n = np.sum(self.counts)
        for arm in range(n_arms):
            # Rescale based on total counts and arm_specific count
            bonus = np.sqrt((2 * np.log(n)) / self.counts[arm])
            ucb_values[arm] = self.values[arm] + bonus

        return np.argmax(ucb_values)

    def update(self, chosen_arm, reward):
        """Update counts and estimated value of rewards for the chosen arm."""
        # Increment counts
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]

        # Update the expected values of chosen arm
        value = self.values[chosen_arm]
        new_value = value * ((n - 1) / n) + reward / n
        self.values[chosen_arm] = new_value
