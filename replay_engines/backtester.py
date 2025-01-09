import numpy as np
import pandas as pd
from bandits.algorithms import available_bandits, Bandit
from bandits.bandit_network import BanditNetwork
from typing import Dict, Union, Tuple, List
import collections


def run_backtest(
        policy_name: str,
        policy_args: Dict,
        portfolio: pd.DataFrame,
        arm) -> Tuple[List, Bandit]:
    arms = list(map(lambda x: arm, portfolio.columns))
    rewards = []

    policy = available_bandits[policy_name](**policy_args)
    columns = portfolio.columns
    for t in range(len(portfolio)):
        chosen_arm = policy.select_arm()
        reward = arms[chosen_arm].draw(portfolio.iloc[:t, chosen_arm])
        max_reward = np.max([arms[c].draw(portfolio.iloc[:t, c]) for c in range(portfolio.shape[1])])
        idx_max = [np.argmax([arms[c].draw(portfolio.iloc[:t, c]) for c in range(portfolio.shape[1])])]
        # print(f"The selected arm is {[columns[i] for i in range(len(columns)) if i in chosen_arm]} "
        #       f"and the best arm is {[columns[i] for i in range(len(columns)) if i in idx_max]}")
        rewards.append(np.mean(portfolio.iloc[t, chosen_arm]))
        policy.update(chosen_arm, reward, max_reward)
        policy.calc_regret(max_value=max_reward,
                           reward=reward)
    return rewards, policy


def run_combinatorial_backtest(
        policy_name: str,
        policy_args: Dict,
        portfolio: pd.DataFrame,
        arm,
        shift=0) -> Tuple[List, CombinatorialBandit]:
    arms = list(map(lambda x: arm, portfolio.columns))
    rewards = []

    policy = available_bandits[policy_name](**policy_args)
    for t in range(shift, len(portfolio)):
        chosen_arm = policy.select_arm()
        reward_array = np.array([arms[c].draw(portfolio.iloc[:t, c]) for c in range(len(portfolio.columns))])
        weights, reward, max_reward = policy.update(chosen_arm, reward_array)
        policy.calc_regret(max_value=max_reward,
                           reward=reward)
        real_reward = np.dot(weights, np.array(portfolio.iloc[t, :]))
        rewards.append(real_reward)
    return rewards, policy


def run_bandit_network_backtest(
        combinatorial_policy_name: str,
        combinatorial_policy_args: Dict,
        policy_name: str,
        policy_args: Dict,
        portfolio: pd.DataFrame,
        arm,
        combinatorial_arm,
        shift=0) -> Tuple[List, Dict, CombinatorialBandit, Bandit]:
    n_super_arms = combinatorial_policy_args.get("n_arms")
    policy = available_bandits[policy_name](**policy_args)
    combinatorial_policy = available_bandits[combinatorial_policy_name](**combinatorial_policy_args)
    arms = list(map(lambda x: arm, portfolio.columns))
    combinatorial_arms = list(map(lambda x: combinatorial_arm, portfolio.columns))

    rewards = []
    chosen_superarms_dict = {
        "bandit_arms": [],
        "combinatorial_arms": [],
        "weights": []
    }

    for t in range(shift, len(portfolio)):
        chosen_arm = policy.select_arm()
        reward = arms[chosen_arm].draw(portfolio.iloc[:t, chosen_arm])
        max_reward = np.max([arms[c].draw(portfolio.iloc[:t, c]) for c in range(portfolio.shape[1])])
        policy.update(chosen_arm, reward, max_reward)
        policy.calc_regret(max_value=max_reward,
                           reward=reward)

        chosen_super_arm = combinatorial_policy.select_arm()
        chosen_arms_counter = collections.Counter(policy.chosen_arms)
        top_arms = [
            x[0] for x in chosen_arms_counter.most_common(n_super_arms)] \
            if len(chosen_arms_counter) > n_super_arms and chosen_arms_counter.most_common(n_super_arms)[-1][1] > 1\
            else np.random.randint(0, len(portfolio.columns), n_super_arms)
        reward_array = np.array([combinatorial_arms[c].draw(portfolio.iloc[:t, c]) for c in top_arms])
        weights, reward, max_reward = combinatorial_policy.update(chosen_super_arm, reward_array)
        combinatorial_policy.calc_regret(max_value=max_reward,
                                         reward=reward)
        real_reward = np.dot(weights, np.array(portfolio.iloc[t, top_arms]))
        rewards.append(real_reward)
        chosen_superarms_dict["bandit_arms"].append(top_arms)
        chosen_superarms_dict["combinatorial_arms"].append(chosen_super_arm)
        chosen_superarms_dict["weights"].append(weights)
    return rewards, chosen_superarms_dict, combinatorial_policy, policy


def run_three_stage_bandit_network_backtest(
        combinatorial_policy_name: str,
        combinatorial_policy_args: Dict,
        sequential_policy_name: str,
        sequential_policy_args: Dict,
        policy_name: str,
        policy_args: Dict,
        portfolio: pd.DataFrame,
        arm,
        sequential_arm,
        shift=0,
        n_partitions=10,
        extended_parallel_top_arms=1) -> Tuple[List, Dict, List, Bandit]:

    bn = BanditNetwork(
        portfolio=portfolio,
        policy_name=policy_name,
        policy_args=policy_args,
        combinatorial_policy_name=combinatorial_policy_name,
        combinatorial_policy_args=combinatorial_policy_args,
        sequential_policy_name=sequential_policy_name,
        sequential_policy_args=sequential_policy_args,
        n_partitions=n_partitions,
        portfolio_size=combinatorial_policy_args.get("n_arms"),
        extended_parallel_top_arms=extended_parallel_top_arms,
        arm=arm,
        sequential_arm=sequential_arm
    )
    rewards = []
    chosen_superarms_dict = {
        "bandit_arms": [],
        "combinatorial_arms": [],
        "weights": []
    }

    for t in range(shift, len(portfolio)):
        real_reward, top_columns, chosen_super_arm, weights = bn.forward_propagation(t)
        rewards.append(real_reward)
        chosen_superarms_dict["bandit_arms"].append(top_columns)
        chosen_superarms_dict["combinatorial_arms"].append(chosen_super_arm)
        chosen_superarms_dict["weights"].append(weights)
    return rewards, chosen_superarms_dict, bn.network["sequential_layer"]["policy"], bn.network["sequential_layer"]["policy"]
