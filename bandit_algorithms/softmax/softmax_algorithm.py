"""
Implementing Softmax Algorithm in both standard and annealing forms.
"""

import numpy as np


class Softmax:
    """Implementing standard `Softmax` algorithm.

    Parameters
    ----------
    temperature : float
        controls the level of randomness in selecting next arm.
    counts : list or array-like
        number of times each arm was played, shape (num_arms,).
    values : list or array-like
        estimated value (mean) of rewards of each arm, shape (num_arms,).

    Attributes
    ----------
    select_arm : int
        select the arm using categorical distribution.
    """
    def __init__(self, temperature, counts=None, values=None):
        self.temperature = temperature
        self.counts = counts
        self.values = values

    def initialize(self, n_arms):
        """Initialize counts and values array with zeros."""
        self.counts = np.zeros(n_arms, dtype=int)
        self.values = np.zeros(n_arms, dtype=float)

    def select_arm(self):
        # Compute softmax probs
        z = self.values / self.temperature
        probs = np.exp(z) / np.sum(np.exp(z))
        return Softmax.categorical_draw(probs)

    @staticmethod
    def categorical_draw(probs):
        """Use categorical distribution to return the index of the next arm.

        Parameters
        ----------
        probs : list or array-like
            softmax probability from `select_arms`.

        Returns
        -------
        index : int
            index of selected arm using one experiment of multinomial
            distribution.
        """
        preds = np.random.multinomial(1, probs, 1)
        return np.argmax(preds)

    def update(self, chosen_arm, reward):
        """Update counts and estimated value of rewards for the chosen arm."""
        # Increment counts
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]

        # Update the expected values of chosen arm
        value = self.values[chosen_arm]
        new_value = value * ((n - 1) / n) + reward / n
        self.values[chosen_arm] = new_value


class AnnealingSoftmax(Softmax):
    """Implementing annealing Softmax algorithm.

    Using this method, the algorithm would explore a lot at the beginning of
    the experiment because the temperature would be close to inf and then start
    decaying with time where it will start exploiting more.
    """
    def __init__(self, counts=None, values=None):
        self.counts = counts
        self.values = values

    def select_arm(self):
        # Increment t each time select_arm is called
        t = np.sum(self.counts) + 1
        # Decay temperature with time
        temperature = 1 / np.log(t + 0.0000001)

        # Compute softmax probs
        z = self.values / temperature
        probs = np.exp(z) / np.sum(np.exp(z))
        return Softmax.categorical_draw(probs)
