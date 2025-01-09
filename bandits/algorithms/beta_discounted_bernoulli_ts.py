from bandits.algorithms.bernoulli_ts import ThompsonSampling


class BetaDiscountedThompsonSampling(ThompsonSampling):

    def __init__(self, n_arms, gamma, **kwargs):
        super().__init__(n_arms, seed=kwargs.get("seed"))
        self.gamma = gamma

    def update(self, chosen_arm, reward, max_reward):
        for arm, value in self.beta_distributions.items():
            binary_selected_arm_multiplier = int(arm == chosen_arm)
            binary_max_reward_multiplier = int(reward == max_reward)
            # if binary_selected_arm_multiplier == 1:
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