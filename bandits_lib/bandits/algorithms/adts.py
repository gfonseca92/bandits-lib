import numpy as np
from numpy import ndarray
from typing import Union
from .beta_discounted_bernoulli_ts import BetaDiscountedThompsonSampling
from itertools import product, combinations


class AdaptiveDiscountedThompsonSampling(BetaDiscountedThompsonSampling):

    def __init__(self, n_arms, gamma, f, w, **kwargs):
        super().__init__(n_arms, gamma, seed=kwargs.get("seed"))
        self.w = w
        self.f = f
        self.w_beta_distributions = self.refresh_w_beta_distributions(n_arms)

    @staticmethod
    def refresh_w_beta_distributions(n_arms: Union[int, ndarray]):
        if isinstance(n_arms, int):
            return {
                arm: {
                    "rewards": np.array([0, 1]),
                }
                for arm in range(n_arms)
            }
        return {
            i: {
                "rewards": np.array([0, 1]),
            }
            for i in range(len(n_arms))
        }

    def sample_windowed(self):
        samples_list = [
            np.random.beta(
                a=max(1, len(np.where(v["rewards"][-self.w:] == 1)[0])),
                b=max(1, len(np.where(v["rewards"][-self.w:] == 0)[0])),
            )
            for v in self.w_beta_distributions.values()
        ]
        return samples_list

    def select_arm(self):
        self.__store_beta__()
        historical_sample_list = self.sample()
        windowed_sample_list = self.sample_windowed()
        samples_list = [
            getattr(np, self.f)([historical_sample_list[i], windowed_sample_list[i]])
            for i in range(len(self.beta_distributions))
        ]
        return np.argmax(samples_list)

    def update(self, chosen_arm, reward, max_reward):
        binary_max_reward_multiplier = int(reward == max_reward)
        self.w_beta_distributions[chosen_arm]["rewards"] = np.append(
            self.w_beta_distributions[chosen_arm]["rewards"],
            binary_max_reward_multiplier
        )
        self.beta_distributions[chosen_arm]["alpha"] = (
                self.beta_distributions[chosen_arm]["alpha"] +
                self.gamma + binary_max_reward_multiplier
        )
        self.beta_distributions[chosen_arm]["beta"] = (
                self.beta_distributions[chosen_arm]["beta"] +
                self.gamma + (1 - binary_max_reward_multiplier)
        )
        self.chosen_arms.append(chosen_arm)


def generate_swaps(lst):
    swaps = []
    for i, j in combinations(range(len(lst)), 2):
        swap = lst.copy()
        swap[i], swap[j] = swap[j], swap[i]
        swaps.append(swap)
    return swaps


class CombinatorialAdaptiveDiscountedThompsonSampling(AdaptiveDiscountedThompsonSampling):

    def __init__(self, n_arms, gamma, f, w, **kwargs):
        all_weights = np.array(list(product(np.linspace(0., 1., 2*n_arms+1)[1:-1], repeat=n_arms)))
        self.max_weight = all_weights.max()
        combs = all_weights[np.where(all_weights.sum(axis=1) == 1.)[0], :]
        final_list = []
        for i in range(len(combs)):
            all_swaps = generate_swaps(combs[i, :])
            for swap in all_swaps:
                final_list.append(swap)
        combs = np.unique(final_list, axis=0)
        super().__init__(combs, gamma, f, w, seed=kwargs.get('seed'))

    @staticmethod
    def refresh_beta_distributions(combs: ndarray):
        return {
            i: {
                "w": combs[i, :],
                "alpha": 1,
                "beta": 1,
                "samples": [],
            }
            for i in range(len(combs))
        }

    @staticmethod
    def refresh_w_beta_distributions(n_arms: Union[int, ndarray]):
        return {
            i: {
                "rewards": np.array([0, 1]),
            }
            for i in range(len(n_arms))
        }

    def update(self, chosen_arm, reward_array: ndarray):
        all_weights = np.array([v["w"] for v in self.beta_distributions.values()])
        max_reward = np.dot(all_weights,  reward_array).max()
        reward = np.dot(self.beta_distributions[chosen_arm]["w"],  reward_array)
        binary_max_reward_multiplier = int(reward == max_reward)
        self.w_beta_distributions[chosen_arm]["rewards"] = np.append(
            self.w_beta_distributions[chosen_arm]["rewards"],
            binary_max_reward_multiplier
        )
        self.beta_distributions[chosen_arm]["alpha"] = (
                self.beta_distributions[chosen_arm]["alpha"] +
                self.gamma + binary_max_reward_multiplier
        )
        self.beta_distributions[chosen_arm]["beta"] = (
                self.beta_distributions[chosen_arm]["beta"] +
                self.gamma + (1 - binary_max_reward_multiplier)
        )
        self.chosen_arms.append(chosen_arm)
        return self.beta_distributions[chosen_arm]["w"], binary_max_reward_multiplier, 1
