from multi_armed_bandit.algorithms.bernoulli_dist.min_dsw_ts import MinDSWTS
from .bandit import Bandit


class CavenaghiMinDSWTS(Bandit):

    _algo: MinDSWTS

    def __init__(self, n_arms: int, gamma: float = 0.9, n: int = 30, store_estimates:bool=True):
        self._algo = MinDSWTS(n_arms=n_arms, gamma=gamma, n=n, store_estimates=store_estimates)
        super().__init__(n_arms, self._algo)

    def update(self, chosen_arm, reward, max_reward):
        self._algo.update_estimates(chosen_arm, reward)
