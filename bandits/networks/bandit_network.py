from pandas import DataFrame
from bandits.algorithms import available_bandits
import numpy as np
from typing import Dict, List, Tuple


class BanditNetwork:

    def __init__(self,
                 portfolio: DataFrame,
                 policy_name: str,
                 policy_args: Dict,
                 sequential_policy_name: str,
                 sequential_policy_args: Dict,
                 n_partitions: int,
                 portfolio_size: int,
                 extended_parallel_top_arms: int,
                 arm,
                 sequential_arm):
        self.arms = list(map(lambda x: arm, portfolio.columns))
        self.sequential_arms = list(map(lambda x: sequential_arm, portfolio.columns))
        self.sliced_portfolio_size = len(portfolio.columns) // n_partitions
        self.n_partitions = n_partitions
        self.portfolio_size = portfolio_size
        self.portfolio = portfolio
        self.extended_parallel_top_arms = extended_parallel_top_arms
        mod = len(portfolio.columns) % n_partitions
        policy_args["mod"] = mod
        self.network_contract = {
            "parallel_layer": {
                "policy_name": policy_name,
                "args": policy_args.copy(),
            },
            "sequential_layer": {
                "policy_name": sequential_policy_name,
                "args": sequential_policy_args.copy(),
            },
        }
        self.network_contract["parallel_layer"]["args"]["n_arms"] = self.sliced_portfolio_size
        self.network_contract["sequential_layer"]["args"]["n_arms"] = self.n_partitions*self.extended_parallel_top_arms
        self.network = self.build_network()

    def build_network(self):
        parallel_layer = self.network_contract["parallel_layer"]
        parallel_special_args = parallel_layer["args"].copy()
        mod = parallel_special_args["mod"]
        parallel_special_args["n_arms"] += mod

        sequential_layer = self.network_contract["sequential_layer"]
        sequential_layer["args"]["portfolio_size"] = self.portfolio_size
        return {
            "parallel_layer": {
                "policies": [available_bandits[parallel_layer["policy_name"]](**parallel_layer["args"])
                             if i != self.n_partitions - 1 else
                             available_bandits[parallel_layer["policy_name"]](**parallel_special_args)
                             for i in range(self.n_partitions)],
                "portfolio": [
                    self.portfolio.iloc[:, i * self.sliced_portfolio_size: (i + 1) * self.sliced_portfolio_size]
                    if i != self.n_partitions - 1 else
                    self.portfolio.iloc[:, i * self.sliced_portfolio_size: (i + 1) * self.sliced_portfolio_size + mod]
                    for i in range(self.n_partitions)
                    ],
                "portfolio_columns": [
                    self.portfolio.columns[i * self.sliced_portfolio_size: (i + 1) * self.sliced_portfolio_size]
                    if i != self.n_partitions - 1 else
                    self.portfolio.columns[i * self.sliced_portfolio_size: (i + 1) * self.sliced_portfolio_size + mod]
                    for i in range(self.n_partitions)
                ],
                "portfolio_columns_indexes": [
                    list(range(i * self.sliced_portfolio_size, (i + 1) * self.sliced_portfolio_size))
                    if i != self.n_partitions - 1 else
                    list(range(i * self.sliced_portfolio_size, (i + 1) * self.sliced_portfolio_size + mod))
                    for i in range(self.n_partitions)
                ],
            },
            "sequential_layer": {
                "policy": available_bandits[sequential_layer["policy_name"]](**sequential_layer["args"]),
                "portfolio": self.portfolio,
                "portfolio_columns": self.portfolio.columns,
            },
        }

    def update_parallel_layer(self, parallel_layer, t):
        policies = parallel_layer["policies"]
        portfolio = parallel_layer["portfolio"]
        portfolio_columns_indexes = parallel_layer["portfolio_columns_indexes"]
        chosen_arms = [p.select_arm() for p in policies]
        chosen_columns = [portfolio[i].columns[chosen_arms[i]] for i in range(len(portfolio))]
        chosen_columns_index = [list(self.portfolio.columns).index(c) for c in chosen_columns]
        rewards = [self.arms[i].draw(self.portfolio.iloc[:t, i]) for i in chosen_columns_index]
        max_rewards = [
            np.max([
                self.arms[i].draw(self.portfolio.iloc[:t, j])
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
        reward = self.sequential_arms[chosen_columns_index[chosen_arm]].draw(self.portfolio.iloc[:t, chosen_columns_index[chosen_arm]])
        max_reward = np.max([
            self.sequential_arms[c].draw(self.portfolio.iloc[:t, c])
            for c in chosen_columns_index
        ])
        policy.update(chosen_arm, reward, max_reward)
        policy.calc_regret(max_value=max_reward, reward=reward)
        historical_sample_list = policy.sample()
        windowed_sample_list = policy.sample_windowed()
        samples_list = [
            getattr(np, policy.f)([historical_sample_list[i], windowed_sample_list[i]])
            for i in range(len(policy.beta_distributions))
        ]
        weights = [w/sum(samples_list) for w in samples_list]
        reward_array = np.array([self.portfolio.iloc[t, c] for c in chosen_columns_index])
        portfolio_reward = np.dot(reward_array, weights)

        return portfolio_reward, weights

    def update_combinatorial_layer(self, combinatorial_layer, chosen_columns_index, top_arms, t):
        policy = combinatorial_layer["policy"]
        chosen_super_arm = policy.select_arm()
        reward_array = np.array([
            self.combinatorial_arms[i].draw(self.portfolio.iloc[:t, i])
            for i in top_arms
        ])
        weights, reward, max_reward = policy.update(chosen_super_arm, reward_array)
        policy.calc_regret(max_value=max_reward, reward=reward)
        top_columns = [i for i in top_arms]
        real_reward = np.dot(
            weights,
            np.array([self.portfolio.iloc[t, c] for c in top_columns])
        )
        return real_reward, top_columns, chosen_super_arm, weights

    def forward_propagation(self, t):
        parallel_layer = self.network["parallel_layer"]
        chosen_columns_index = self.update_parallel_layer(parallel_layer, t)
        sequential_layer = self.network["sequential_layer"]
        reward, weights = self.update_sequential_layer(sequential_layer, chosen_columns_index, t)
        return reward, chosen_columns_index, 1, weights
