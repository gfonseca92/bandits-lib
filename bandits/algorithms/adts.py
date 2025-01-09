import numpy as np
from numpy import ndarray
from typing import Union
from .beta_discounted_bernoulli_ts import BetaDiscountedThompsonSampling


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
        for arm, value in self.beta_distributions.items():
            binary_selected_arm_multiplier = int(arm == chosen_arm)
            binary_max_reward_multiplier = int(reward == max_reward)
            if binary_selected_arm_multiplier == 1:
                self.w_beta_distributions[arm]["rewards"] = np.append(
                    self.w_beta_distributions[arm]["rewards"],
                    binary_max_reward_multiplier
                )
                self.beta_distributions[arm]["alpha"] = max(
                    1.,
                    self.beta_distributions[arm]["alpha"] * self.gamma +
                    binary_selected_arm_multiplier * binary_max_reward_multiplier
                )
                self.beta_distributions[arm]["beta"] = max(
                    1.,
                    self.beta_distributions[arm]["beta"] * self.gamma +
                    binary_selected_arm_multiplier * (1 - binary_max_reward_multiplier)
                )
        self.chosen_arms.append(chosen_arm)
