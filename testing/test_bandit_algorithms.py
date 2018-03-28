"""
Use Monte Carlo simulations to test bandit algorithms.
"""

import numpy as np


class BernoulliArm:
    """Draw an arm's reward from a Bernoulli distribution.
    Parameters
    ----------
    p : float
            probability of getting reward for a specific arm.

    Attributes
    ----------
    draw : float
        rewards drawn using Bernoulli probability distribution.
    """
    def __init__(self, p):
        self.p = p

    def draw(self):
        z = np.random.random()
        if z > self.p:
            return 0.0
        return 1.0


def test_algorithm(algo, arms, num_simulations, horizon):
    """
    Run the bandit algorithm `algo` for `num_simulations` over time horizon.

    Parameters
    ----------
    algo : class instance
        bandit algorithm class instance.
    arms : list
        list of Bernoulli distribution class instance with different
        probability of rewards.
    num_simulations : int
        number of simulations to run the bandit algorithm.

    horizon : int
        play period to run the bandit algorithm for each simulations.

    Returns
    -------
    chosen_arms : array
        2d-array of chosen arms, shape: (num_simulations x horizon).
    average_rewards : array
        1d-array of average rewards per time horizon, shape: (horizon,).
    cumulative_rewards : array
        1d-array of cumulative awards over time horizon, shape: (horizon,).
    """
    # Initialize rewards and chosen_arms with zero 2d arrays
    chosen_arms = np.zeros((num_simulations, horizon))
    rewards = np.zeros((num_simulations, horizon))

    # Loop over all simulations
    for sim in range(num_simulations):
        # Re-initialize algorithm's counts and values arrays
        algo.initialize(len(arms))

        # Loop over all time horizon
        for t in range(horizon):
            # Select arm
            chosen_arm = algo.select_arm()
            chosen_arms[sim, t] = chosen_arm

            # Draw from Bernoulli distribution to get rewards
            reward = arms[chosen_arm].draw()
            rewards[sim, t] = reward

            # Update the algorithms' count and estimated values
            algo.update(chosen_arm, reward)

    # Average rewards across all sims and compute cumulative rewards
    average_rewards = np.mean(rewards, axis=0)
    cumulative_rewards = np.cumsum(average_rewards)

    return chosen_arms, average_rewards, cumulative_rewards
