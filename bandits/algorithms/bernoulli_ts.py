from .bandit import Bandit
import numpy as np
from numpy import ndarray
from typing import Union


class ThompsonSampling(Bandit):

    def __init__(self, n_arms, **kwargs):
        super(ThompsonSampling, self).__init__(n_arms, seed=kwargs.get("seed"))
        self.beta_distributions = self.refresh_beta_distributions(n_arms)

    @staticmethod
    def refresh_beta_distributions(n_arms: Union[int, ndarray]):
        if isinstance(n_arms, int):
            return {
                arm: {
                    "alpha": 1,
                    "beta": 1,
                    "samples": [],
                    "rewards": np.array([0, 1]),
                }
                for arm in range(n_arms)
            }
        return {
            i: {
                "alpha": 1,
                "beta": 1,
                "samples": [],
                "rewards": np.array([0, 1]),
            }
            for i in range(len(n_arms))
        }

    def __store_beta__(self):
        for value in self.beta_distributions.values():
            value["samples"].append((value["alpha"], value["beta"]))

    def sample(self):
        samples_list = [
            np.random.beta(
                a=v["alpha"],
                b=v["beta"]
            )
            for v in self.beta_distributions.values()
        ]
        return samples_list

    def update(self, chosen_arm, reward, max_reward):
        self.counts[chosen_arm] += 1
        if reward == max_reward:
            self.beta_distributions[chosen_arm]["alpha"] += 1
        else:
            self.beta_distributions[chosen_arm]["beta"] += 1
        self.chosen_arms.append(chosen_arm)
        return chosen_arm, reward, max_reward

    def select_arm(self):
        self.__store_beta__()
        return np.argmax(self.sample())
