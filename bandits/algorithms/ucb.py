import numpy as np
from bandits.bandit import Bandit


class UCB1(Bandit):
    def __init__(self, n_arms, **kwargs):
        super().__init__(n_arms, seed=kwargs.get('seed'))
        n_arms = n_arms if isinstance(n_arms, int) else len(n_arms)
        self.N = np.zeros(n_arms)
        self.Q = np.zeros(n_arms)
        self.t = 0

    def select_arm(self):
        self.t += 1
        if 0 in self.N:
            idx_list = np.where(self.N == 0)[0]
            return np.random.choice(idx_list)
        else:
            return np.argmax(self.Q + np.sqrt(2 * np.log(self.t) / self.N))

    def update(self, chosen_arm, reward, max_reward):
        self.N[chosen_arm] += 1
        reward = int(reward == max_reward)
        n = self.N[chosen_arm]
        self.Q[chosen_arm] = ((n - 1) / float(n)) * self.Q[chosen_arm] + (1 / float(n)) * reward
        self.chosen_arms.append(chosen_arm)


class DiscountedUCB1(UCB1):
    def __init__(self, n_arms, gamma, **kwargs):
        super().__init__(n_arms, seed=kwargs.get('seed'))
        self.gamma = gamma
        self.N = np.zeros(n_arms)
        self.N_star = np.ones(n_arms)

    def update(self, chosen_arm, reward, max_reward):
        self.N_star[chosen_arm] *= self.gamma
        self.N[chosen_arm] += self.N_star[chosen_arm]
        reward = int(reward == max_reward)
        self.Q[chosen_arm] = (1 - self.gamma) * self.Q[chosen_arm] + self.gamma * reward
        self.chosen_arms.append(chosen_arm)


class SlidingWindowUCB1(UCB1):
    def __init__(self, n_arms, w, **kwargs):
        super().__init__(n_arms, seed=kwargs.get('seed'))
        self.window_size = w
        self.rewards = {i: [] for i in range(n_arms)}

    def update(self, chosen_arm, reward, max_reward):
        self.N[chosen_arm] += 1
        reward = int(reward == max_reward)
        self.rewards[chosen_arm].append(reward)
        self.Q[chosen_arm] = np.mean(self.rewards[chosen_arm][-self.window_size:])
        self.chosen_arms.append(chosen_arm)
