import numpy as np
import pandas as pd
from bandits.algorithms import available_bandits, Bandit
from typing import Dict, Tuple, List
from bandits.networks import available_bandit_networks
from numba import njit, prange, jit


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


def run_numpy_backtest(
        policy_name: str,
        policy_args: Dict,
        numpy_portfolio: np.array,
        arm) -> Tuple[List, Bandit]:
    arms = list(map(lambda x: arm, range(numpy_portfolio.shape[1])))
    rewards = []

    policy = available_bandits[policy_name](**policy_args)
    for t in prange(len(numpy_portfolio)):
        chosen_arm = policy.select_arm()
        reward = arms[chosen_arm].draw(numpy_portfolio[:t, chosen_arm])
        max_reward = np.max([arms[c].draw(numpy_portfolio[:t, c]) for c in range(numpy_portfolio.shape[1])])
        idx_max = [np.argmax([arms[c].draw(numpy_portfolio[:t, c]) for c in range(numpy_portfolio.shape[1])])
                   ]
        # print(f"The selected arm is {[columns[i] for i in range(len(columns)) if i in chosen_arm]} "
        #       f"and the best arm is {[columns[i] for i in range(len(columns)) if i in idx_max]}")
        rewards.append(float(numpy_portfolio[t, chosen_arm]))
        policy.update(chosen_arm, reward, max_reward)
        policy.calc_regret(max_value=max_reward,
                           reward=reward)
    return rewards, policy


# Using Numba to speed up the backtest
@njit(parallel=True)
def run_numba_backtest(
        policy_name: str,
        policy_args: Dict,
        numpy_portfolio: np.array,
        arm) -> Tuple[List, Bandit]:
    arms = list(map(lambda x: arm, range(numpy_portfolio.shape[1])))
    rewards = []

    policy = available_bandits[policy_name](**policy_args)
    for t in prange(len(numpy_portfolio)):
        chosen_arm = policy.select_arm()
        reward = arm.draw(numpy_portfolio[:t, chosen_arm])
        max_reward = np.max([arm.draw(numpy_portfolio[:t, c]) for c in range(numpy_portfolio.shape[1])])
        idx_max = [np.argmax([arm.draw(numpy_portfolio[:t, c]) for c in range(numpy_portfolio.shape[1])])
                   ]
        # print(f"The selected arm is {[columns[i] for i in range(len(columns)) if i in chosen_arm]} "
        #       f"and the best arm is {[columns[i] for i in range(len(columns)) if i in idx_max]}")
        rewards.append(float(numpy_portfolio[t, chosen_arm]))
        policy.update(chosen_arm, reward, max_reward)
        policy.calc_regret(max_value=max_reward,
                           reward=reward)
    return rewards, policy


def run_bandit_network_backtest(
        network_name: str,
        **kwargs) -> Tuple[List, Dict, List, Bandit]:

    bn = available_bandit_networks[network_name](**kwargs)
    rewards = []
    chosen_superarms_dict = {
        "bandit_arms": [],
        "weights": []
    }
    shift = kwargs.get("shift", 0)
    numpy_portfolio = kwargs.get("portfolio")
    for t in range(shift, len(numpy_portfolio)):
        real_reward, top_columns, chosen_super_arm, weights = bn.forward_propagation(t)
        rewards.append(real_reward)
        chosen_superarms_dict["bandit_arms"].append(top_columns)
        chosen_superarms_dict["weights"].append(weights)
    return rewards, chosen_superarms_dict, bn.network["final_layer"]["policy"], bn.network["final_layer"]["policy"]
