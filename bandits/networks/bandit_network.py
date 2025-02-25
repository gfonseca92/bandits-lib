from pandas import DataFrame
from collections import Counter
from bandits.algorithms import available_bandits
import numpy as np
from typing import Dict, List, Tuple


class BanditNetwork:

    def __init__(self,
                 portfolio: np.ndarray,
                 policy_name: str,
                 policy_args: Dict,
                 final_layer_policy_name: str,
                 final_layer_policy_args: Dict,
                 n_partitions: int,
                 portfolio_size: int,
                 arm,
                 final_layer_arm):
        self.arms = list(map(lambda x: arm, range(portfolio.shape[1])))
        self.final_layer_arms = list(map(lambda x: final_layer_arm, range(portfolio.shape[1])))
        self.portfolio_columns = list(map(lambda x: x, range(portfolio.shape[1])))
        self.sliced_portfolio_size = portfolio.shape[1] // n_partitions
        self.n_partitions = n_partitions
        self.portfolio_size = portfolio_size
        self.portfolio = portfolio
        self.network_contract = {
            "parallel_layer": {
                "policy_name": policy_name,
                "args": policy_args.copy(),
            },
            "final_layer": {
                "policy_name": final_layer_policy_name,
                "args": final_layer_policy_args.copy(),
            },
        }
        self.network_contract["parallel_layer"]["args"]["n_arms"] = self.sliced_portfolio_size
        self.network_contract["final_layer"]["args"]["n_arms"] = self.n_partitions
        self.network = self.build_network()

    def build_network(self):
        parallel_layer = self.network_contract["parallel_layer"]
        parallel_special_args = parallel_layer["args"].copy()
        mod = len(self.portfolio_columns) % self.n_partitions
        parallel_special_args["n_arms"] += mod

        sequential_layer = self.network_contract["final_layer"]
        sequential_layer["args"]["portfolio_size"] = self.portfolio_size
        return {
            "parallel_layer": {
                "policies": [available_bandits[parallel_layer["policy_name"]](**parallel_layer["args"])
                             if i != self.n_partitions - 1 else
                             available_bandits[parallel_layer["policy_name"]](**parallel_special_args)
                             for i in range(self.n_partitions)],
                "portfolio": [
                    self.portfolio[:, i * self.sliced_portfolio_size: (i + 1) * self.sliced_portfolio_size]
                    if i != self.n_partitions - 1 else
                    self.portfolio[:, i * self.sliced_portfolio_size: (i + 1) * self.sliced_portfolio_size + mod]
                    for i in range(self.n_partitions)
                    ],
                "portfolio_columns": [
                    self.portfolio_columns[i * self.sliced_portfolio_size: (i + 1) * self.sliced_portfolio_size]
                    if i != self.n_partitions - 1 else
                    self.portfolio_columns[i * self.sliced_portfolio_size: (i + 1) * self.sliced_portfolio_size + mod]
                    for i in range(self.n_partitions)
                ],
                "portfolio_columns_indexes": [
                    list(range(i * self.sliced_portfolio_size, (i + 1) * self.sliced_portfolio_size))
                    if i != self.n_partitions - 1 else
                    list(range(i * self.sliced_portfolio_size, (i + 1) * self.sliced_portfolio_size + mod))
                    for i in range(self.n_partitions)
                ],
            },
            "final_layer": {
                "policy": available_bandits[sequential_layer["policy_name"]](**sequential_layer["args"]),
                "portfolio": self.portfolio,
                "portfolio_columns": self.portfolio_columns,
            },
        }

    def update_parallel_layer(self, parallel_layer, t):
        policies = parallel_layer["policies"]
        # portfolio = parallel_layer["portfolio"]
        portfolio_columns = parallel_layer["portfolio_columns"]
        portfolio_columns_indexes = parallel_layer["portfolio_columns_indexes"]
        chosen_arms = [int(p.select_arm()) for p in policies]
        chosen_columns_index = [
            partitioned_portfolio[chosen_arm]
            for partitioned_portfolio, chosen_arm in zip(portfolio_columns, chosen_arms)
        ]
        rewards = [self.arms[i].draw(self.portfolio[:t, i]) for i in chosen_columns_index]
        max_rewards = [
            np.max([
                self.arms[i].draw(self.portfolio[:t, j])
                for j in portfolio_columns_indexes[i]
            ])
            for i in range(len(portfolio_columns_indexes))
        ]

        [
            policies[i].update(chosen_arms[i], rewards[i], max_rewards[i])
            for i in range(len(policies))
        ]
        [
            policies[i].calc_regret(max_value=max_rewards[i], reward=rewards[i])
            for i in range(len(policies))
        ]
        return chosen_columns_index

    def update_sequential_layer(self, sequential_layer, chosen_columns_index, t):
        policy = sequential_layer["policy"]
        chosen_arm = policy.select_arm()
        reward = self.final_layer_arms[chosen_columns_index[chosen_arm]].draw(self.portfolio[:t, chosen_columns_index[chosen_arm]])
        max_reward = np.max([
            self.final_layer_arms[c].draw(self.portfolio[:t, c])
            for c in chosen_columns_index
        ])
        policy.update(chosen_arm, reward, max_reward)
        policy.calc_regret(max_value=max_reward, reward=reward)

        # Create weights based on the counter of frequency of each arm
        chosen_arms = policy.chosen_arms
        counter = dict(Counter(chosen_arms))
        weights = np.zeros(len(chosen_columns_index))
        for arm, count in counter.items():
            weights[arm] = count / len(chosen_arms)

        reward_array = np.array([self.portfolio[t, c] for c in chosen_columns_index])
        portfolio_reward = np.dot(reward_array, weights)

        return portfolio_reward, weights

    def forward_propagation(self, t):
        parallel_layer = self.network["parallel_layer"]
        chosen_columns_index = self.update_parallel_layer(parallel_layer, t)
        sequential_layer = self.network["final_layer"]
        reward, weights = self.update_sequential_layer(sequential_layer, chosen_columns_index, t)
        return reward, chosen_columns_index, 1, weights


class CombinatorialBanditNetwork(BanditNetwork):

    def __init__(self,
                 portfolio: np.ndarray,
                 policy_name: str,
                 policy_args: Dict,
                 final_layer_policy_name: str,
                 final_layer_policy_args: Dict,
                 n_partitions: int,
                 portfolio_size: int,
                 arm,
                 final_layer_arm):
        self.arms = list(map(lambda x: arm, range(portfolio.shape[1])))
        self.final_layer_arms = list(map(lambda x: final_layer_arm, range(portfolio.shape[1])))
        self.portfolio_columns = list(map(lambda x: x, range(portfolio.shape[1])))
        self.sliced_portfolio_size = portfolio.shape[1] // n_partitions
        self.n_partitions = n_partitions
        self.portfolio_size = portfolio_size
        self.portfolio = portfolio
        self.network_contract = {
            "parallel_layer": {
                "policy_name": policy_name,
                "args": policy_args.copy(),
            },
            "final_layer": {
                "policy_name": final_layer_policy_name,
                "args": final_layer_policy_args.copy(),
            },
        }
        self.network_contract["parallel_layer"]["args"]["n_arms"] = self.sliced_portfolio_size
        self.network_contract["final_layer"]["args"]["n_arms"] = self.n_partitions
        self.network = self.build_network()

    def build_network(self):
        parallel_layer = self.network_contract["parallel_layer"]
        parallel_special_args = parallel_layer["args"].copy()
        mod = len(self.portfolio_columns) % self.n_partitions
        parallel_special_args["n_arms"] += mod

        combinatorial_layer = self.network_contract["final_layer"]
        combinatorial_layer["args"]["portfolio_size"] = self.portfolio_size
        return {
            "parallel_layer": {
                "policies": [available_bandits[parallel_layer["policy_name"]](**parallel_layer["args"])
                             if i != self.n_partitions - 1 else
                             available_bandits[parallel_layer["policy_name"]](**parallel_special_args)
                             for i in range(self.n_partitions)],
                "portfolio": [
                    self.portfolio[:, i * self.sliced_portfolio_size: (i + 1) * self.sliced_portfolio_size]
                    if i != self.n_partitions - 1 else
                    self.portfolio[:, i * self.sliced_portfolio_size: (i + 1) * self.sliced_portfolio_size + mod]
                    for i in range(self.n_partitions)
                    ],
                "portfolio_columns": [
                    self.portfolio_columns[i * self.sliced_portfolio_size: (i + 1) * self.sliced_portfolio_size]
                    if i != self.n_partitions - 1 else
                    self.portfolio_columns[i * self.sliced_portfolio_size: (i + 1) * self.sliced_portfolio_size + mod]
                    for i in range(self.n_partitions)
                ],
                "portfolio_columns_indexes": [
                    list(range(i * self.sliced_portfolio_size, (i + 1) * self.sliced_portfolio_size))
                    if i != self.n_partitions - 1 else
                    list(range(i * self.sliced_portfolio_size, (i + 1) * self.sliced_portfolio_size + mod))
                    for i in range(self.n_partitions)
                ],
            },
            "final_layer": {
                "policy": available_bandits[combinatorial_layer["policy_name"]](**combinatorial_layer["args"]),
                "portfolio": self.portfolio,
                "portfolio_columns": self.portfolio_columns,
            },
        }

    def update_combinatorial_layer(self, combinatorial_layer, chosen_columns_index, t):
        policy = combinatorial_layer["policy"]
        chosen_super_arm = policy.select_arm()
        reward_array = np.array([
            self.final_layer_arms[i].draw(self.portfolio[:t, i])
            for i in chosen_columns_index
        ])
        weights, reward, max_reward = policy.update(chosen_super_arm, reward_array)
        policy.calc_regret(max_value=max_reward, reward=reward)
        columns = self.portfolio_columns
        top_columns = [columns[i] for i in chosen_columns_index]
        real_reward = np.dot(
            weights,
            np.array([self.portfolio[t, c] for c in top_columns])
        )
        return real_reward, top_columns, chosen_super_arm, weights

    def forward_propagation(self, t):
        parallel_layer = self.network["parallel_layer"]
        chosen_columns_index = self.update_parallel_layer(parallel_layer, t)
        combinatorial_layer = self.network["final_layer"]
        reward, top_columns, chosen_super_arm, weights = self.update_combinatorial_layer(
            combinatorial_layer, chosen_columns_index, t
        )
        return reward, top_columns, 1, weights
