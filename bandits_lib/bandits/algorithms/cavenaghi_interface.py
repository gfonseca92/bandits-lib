from multi_armed_bandit.algorithms.bernoulli_dist.min_dsw_ts import MinDSWTS
from multi_armed_bandit.algorithms.bernoulli_dist.max_dsw_ts import MaxDSWTS
from multi_armed_bandit.algorithms.bernoulli_dist.mean_dsw_ts import MeanDSWTS
from typing import Literal

from .bandit import Bandit


class CavenaghiFDSWTS(Bandit):

        def __init__(
                self,
                f: Literal['min', 'max', 'mean'],
                n_arms: int,
                gamma: float = 0.9,
                n: int = 30,
                seed: int = 42,
                store_estimates: bool = True
        ):
            if f == 'min':
                self._algo = MinDSWTS(n_arms=n_arms, gamma=gamma, n=n, store_estimates=store_estimates)
            elif f == 'max':
                self._algo = MaxDSWTS(n_arms=n_arms, gamma=gamma, n=n, store_estimates=store_estimates)
            elif f == 'mean':
                self._algo = MeanDSWTS(n_arms=n_arms, gamma=gamma, n=n, store_estimates=store_estimates)
            else:
                raise ValueError('f must be one of "min", "max", "mean"')
            super().__init__(n_arms, seed=seed)

        def update(self, chosen_arm, reward, max_reward):
            if reward == max_reward:
                reward = 1
            else:
                reward = 0
            self._algo.update_estimates(int(chosen_arm), int(reward))

        def select_arm(self):
            return self._algo.select_action()
