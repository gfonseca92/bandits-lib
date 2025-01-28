from bandits.reward_functions import available_reward_functions
from replay_engines.backtester import run_bandit_network_backtest, run_numpy_backtest
import numpy as np
from pandas import DataFrame
from tqdm import tqdm
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import matplotlib.colors as mpl_colors
import os


def interpolate_colors(color1, color2, n_size):
    """
    Interpolate between two colors dynamically.

    Parameters:
        color1 (str): Hex code or name of the first color.
        color2 (str): Hex code or name of the second color.
        n_size (int): Number of interpolated colors to generate.

    Returns:
        list: List of interpolated colors in hex format.
    """
    # Convert colors to RGB format
    rgb1 = mpl_colors.to_rgb(color1)
    rgb2 = mpl_colors.to_rgb(color2)

    # Linearly interpolate between the two colors
    interpolated = [
        mpl_colors.to_hex((1 - t) * np.array(rgb1) + t * np.array(rgb2))
        for t in np.linspace(0, 1, n_size)
    ]
    return interpolated


class BacktestInterface:
    """
    Interface to run backtests on different bandit policies in parallel
    """

    def __init__(self,
                 bandit_policies: Dict,
                 n_simulations: int,
                 reward_function: str,
                 reward_function_args: Dict,
                 portfolio: DataFrame):
        self.n_simulations = n_simulations
        self.reward_function = available_reward_functions[reward_function](**reward_function_args)
        self.portfolio = portfolio
        self.bandit_policies = bandit_policies.copy()

    def run_backtest(self, policy_nickname: str, args: Dict) -> Dict:
        rewards_matrix = np.zeros(shape=(self.portfolio.shape[0], self.n_simulations))
        regret_matrix = np.zeros(shape=(self.portfolio.shape[0], self.n_simulations))
        seeds = list(range(self.n_simulations))
        policy_results = {
            policy_nickname: {
                "agents": [],
            }
        }
        for i in range(self.n_simulations):
            kwargs = args["args"].copy()
            kwargs["seed"] = seeds[i]
            rewards, policy = run_numpy_backtest(
                policy_name=args["policy_name"],
                policy_args=kwargs,
                numpy_portfolio=np.array(self.portfolio),
                arm=self.reward_function
            )
            rewards_matrix[:, i] = np.array(rewards)
            regret_matrix[:, i] = np.array(policy.regret)
            policy_results[policy_nickname]["agents"].append(policy)
        policy_results[policy_nickname]["rewards_matrix"] = rewards_matrix
        policy_results[policy_nickname]["regret_matrix"] = regret_matrix
        policy_results[policy_nickname]["std_reward"] = np.std(rewards_matrix, axis=1)
        policy_results[policy_nickname]["mean_reward"] = np.mean(rewards_matrix, axis=1)
        policy_results[policy_nickname]["std_regret"] = np.std(regret_matrix, axis=1)
        policy_results[policy_nickname]["mean_regret"] = np.mean(regret_matrix, axis=1)
        return policy_results

    def run(self, n_threads: int = None):
        # retrieve the machine max number of threads
        max_threads = os.cpu_count()
        if n_threads is not None:
            max_threads = n_threads

        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            futures = {
                executor.submit(self.run_backtest, policy_name, args): policy_name
                for policy_name, args in tqdm(self.bandit_policies.items())
            }
            for future in as_completed(futures):
                policy_results = future.result()
                self.bandit_policies.update(policy_results)

    def plot_regrets(self,
                     save_fig: bool = False,
                     figure_name: str = None,
                     plot_log: bool = False,
                     personalized_colors: List = None,
                     color_start: str = "#1f77b4",
                     color_end: str = "#ff7f0e"):

        if save_fig and figure_name is None:
            raise ValueError("Please provide a figure name to save the plot.")

        # Default Matplotlib blue and orange
        color_vector_size = len(self.bandit_policies)
        colors = interpolate_colors(color_start, color_end, color_vector_size)
        if personalized_colors is not None:
            colors = personalized_colors

        with plt.style.context(['science', 'ieee', 'no-latex']):
            plt.subplots(figsize=(6, 3))
            i = 0
            for policy, output in self.bandit_policies.items():
                plt.plot(np.cumsum(output['mean_regret']), color=colors[i], linestyle='-', label=policy)
                confidence_interval = 1.96 * output["std_regret"] / np.sqrt(self.n_simulations)
                upper = np.cumsum(output["mean_regret"] + confidence_interval)
                lower = np.cumsum(output["mean_regret"] - confidence_interval)
                plt.fill_between(list(range(len(output['mean_regret']))), upper, lower, color=colors[i], alpha=0.15,
                                 label='_nolegend_')
                i += 1
            plt.legend()
            plt.xlabel('t', fontsize=12)
            plt.ylabel('Cumulative Regret', fontsize=12)
            if plot_log:
                plt.yscale('log')
            if save_fig:
                plt.savefig(figure_name, dpi=600)

    def plot_rewards(self,
                     save_fig: bool = False,
                     figure_name: str = None,
                     oracle: np.ndarray = None,
                     oracle_name: str = "Oracle",
                     personalized_colors: List = None,
                     color_start: str = "#1f77b4",
                     color_end: str = "#ff7f0e"):

            if save_fig and figure_name is None:
                raise ValueError("Please provide a figure name to save the plot.")

            # Default Matplotlib blue and orange
            color_vector_size = len(self.bandit_policies)
            colors = interpolate_colors(color_start, color_end, color_vector_size)
            if personalized_colors is not None:
                colors = personalized_colors

            with plt.style.context(['science', 'ieee', 'no-latex']):
                plt.subplots(figsize=(6, 3))

                if oracle is not None:
                    plt.plot(self.portfolio.index, np.cumsum(oracle), color='black', linestyle='-', label=oracle_name)

                i = 0
                for policy, output in self.bandit_policies.items():
                    plt.plot(self.portfolio.index, np.cumsum(output['mean_reward']), color=colors[i], linestyle='-', label=policy)
                    i += 1
                plt.legend()
                plt.xlabel('t', fontsize=12)
                plt.ylabel('y(t)', fontsize=12)
                if save_fig:
                    plt.savefig(figure_name, dpi=600)


class BanditNetworkBacktestInterface(BacktestInterface):

    def __init__(self,
                 bandit_policies: Dict,
                 portfolio_size: int,
                 n_simulations: int,
                 reward_function: str,
                 reward_function_args: Dict,
                 portfolio: DataFrame):
        super().__init__(bandit_policies, n_simulations, reward_function, reward_function_args, portfolio)
        self.portfolio_size = portfolio_size

    def run_backtest(self, policy_nickname: str, args: Dict) -> Dict:
        rewards_matrix = np.zeros(shape=(self.portfolio.shape[0], self.n_simulations))
        regret_matrix = np.zeros(shape=(self.portfolio.shape[0], self.n_simulations))
        seeds = list(range(self.n_simulations))
        policy_results = {
            policy_nickname: {
                "agents": [],
                "combinatorial_agents": [],
                "chosen_superarms": [],
            }
        }
        for i in range(self.n_simulations):
            policy_args = args["args"].copy()
            policy_args["seed"] = seeds[i]
            rewards, chosen_superarms_dict, combinatorial_policy, policy = (
                run_bandit_network_backtest(
                    portfolio_size=self.portfolio_size,
                    sequential_policy_name=args["sequential_policy_name"],
                    sequential_policy_args=args["sequential_policy_args"],
                    policy_name=args["policy_name"],
                    policy_args=policy_args,
                    numpy_portfolio=np.array(self.portfolio),
                    sequential_arm=self.reward_function,
                    arm=self.reward_function,
                    n_partitions=self.portfolio_size
                )
            )
            rewards_matrix[:, i] = np.array(rewards)
            regret_matrix[:, i] = np.array(policy.regret)
            policy_results[policy_nickname]["agents"].append(policy)
            policy_results[policy_nickname]["combinatorial_agents"].append(combinatorial_policy)
            policy_results[policy_nickname]["chosen_superarms"].append(chosen_superarms_dict)
        policy_results[policy_nickname]["rewards_matrix"] = rewards_matrix
        policy_results[policy_nickname]["regret_matrix"] = regret_matrix
        policy_results[policy_nickname]["std_reward"] = np.std(rewards_matrix, axis=1)
        policy_results[policy_nickname]["mean_reward"] = np.mean(rewards_matrix, axis=1)
        policy_results[policy_nickname]["std_regret"] = np.std(regret_matrix, axis=1)
        policy_results[policy_nickname]["mean_regret"] = np.mean(regret_matrix, axis=1)
        return policy_results


# import pandas as pd
# portfolio = pd.read_csv('/Users/gfonseca/Desktop/Doutorado - EEC-I/Pesquisa/bandits-lib/computational-economics-paper/experiments/06_cryptocurrencies/crypto_market_data.csv')
# portfolio = portfolio.fillna(0.)
# portfolio["Date"] = pd.to_datetime(portfolio["Date"], utc=True).dt.date
# portfolio = portfolio.set_index("Date")
# portfolio = portfolio.astype(float)
# portfolio = portfolio.drop(columns=['TIA-USD', 'OP-USD', 'UNI-USD', 'ARB-USD'], axis=1)
# portfolio = portfolio[portfolio.mean().sort_values(ascending=False).index[:60]]
#
# portfolio_size = 10
#
# bandit_net_policies = {
#     f"Two Stage ADTS | (n={portfolio_size})": {
#         "sequential_policy_name": "AdaptiveDiscountedThompsonSampling",
#         "sequential_policy_args": {"n_arms": portfolio.shape[1], "gamma": 0.5, "f": "mean", "w": 160},
#         "policy_name": "AdaptiveDiscountedThompsonSampling",
#         "args": {"n_arms": portfolio.shape[1], "gamma": 0.5, "f": "min", "w": 160},
#         "combinatorial_agents": [],
#         "agents": [],
#         "chosen_superarms": []
#     },
#     f"Two ADTS + UCB1 | (n={portfolio_size})": {
#         "sequential_policy_name": "AdaptiveDiscountedThompsonSampling",
#         "sequential_policy_args": {"n_arms": portfolio.shape[1], "gamma": 0.5, "f": "mean", "w": 160},
#         "policy_name": "UCB1",
#         "args": {},
#         "combinatorial_agents": [],
#         "agents": [],
#         "chosen_superarms": []
#     },
#     f"Two ADTS + f-DSW TS | (n={portfolio_size})": {
#         "sequential_policy_name": "AdaptiveDiscountedThompsonSampling",
#         "sequential_policy_args": {"n_arms": portfolio.shape[1], "gamma": 0.5, "f": "mean", "w": 160},
#         "policy_name": "CavenaghiFDSWTS",
#         "args": {"n_arms": portfolio.shape[1], "gamma": 0.5, "f": "min", "n": 160},
#         "combinatorial_agents": [],
#         "agents": [],
#         "chosen_superarms": []
#     },
# }
#
# bandit_backtester = BanditNetworkBacktestInterface(
#     bandit_policies=bandit_net_policies,
#     portfolio_size=portfolio_size,
#     n_simulations=1,
#     reward_function="WindowedReturnArm",
#     reward_function_args=dict(window=60),
#     portfolio=portfolio,
# )
# bandit_backtester.run()