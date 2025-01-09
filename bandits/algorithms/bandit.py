"""
This module contains the abstract class Bandit, which is the base class for all bandit algorithms.
"""
import numpy as np
from abc import ABC, abstractmethod


class Bandit(ABC):

    def __init__(self, n_arms, seed=None):
        n_arms = n_arms if isinstance(n_arms, int) else len(n_arms)
        self.counts = [1 for col in range(n_arms)]
        self.values = [1e-9 for col in range(n_arms)]
        self.chosen_arms = []
        self.regret = []
        self.accuracy = []
        if seed is not None:
            np.random.seed(seed)

    @abstractmethod
    def select_arm(self):
        pass

    def update(self, chosen_arm, reward, max_reward):
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        self.values[chosen_arm] = ((n - 1) / float(n)) * self.values[chosen_arm] + (1 / float(n)) * reward
        self.chosen_arms.append(chosen_arm)

    def calc_regret(self, max_value, reward):
        self.regret.append(max_value - reward)
        self.accuracy.append(int(reward == max_value))
