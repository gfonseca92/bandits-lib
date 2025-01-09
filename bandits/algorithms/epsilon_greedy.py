from .bandit import Bandit
import random


class EpsilonGreedy(Bandit):

    def __init__(self, n_arms, epsilon=0.1, **kwargs):
        super().__init__(n_arms, seed=kwargs.get('seed'))
        self.epsilon = epsilon
        self.epsilon_start = epsilon

    @staticmethod
    def ind_max(x):
        m = max(x)
        return x.index(m)

    def select_arm(self):
        if random.random() > self.epsilon:
            return self.ind_max(self.values)
        else:
            return random.randrange(len(self.values))
