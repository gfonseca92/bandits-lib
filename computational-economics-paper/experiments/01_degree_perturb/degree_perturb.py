import sys
import os
from pathlib import Path

BASE_PATH = Path(os.path.abspath('')).parent.parent.parent
sys.path.append(str(BASE_PATH))

from bandits.algorithms import *
from bandits.reward_functions import *
from replay_engines.backtester import run_backtest
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import datetime

import matplotlib.pyplot as plt
import scienceplots


x = np.zeros(1000)


portfolio = pd.DataFrame(
    {
        "Class 1": x + 1.,
        "Class 2": x + 2.,
        "Class 3": np.array([x[i] + 5.0 if 300 <= i <= 700 else x[i] + 0.5 for i in range(len(x))]),
    }
)

n_sim = 30
shift = 0
polices = {
    "ADTS (min)": {
        "policy_name": "AdaptiveDiscountedThompsonSampling",
        "args": {"n_arms": portfolio.shape[1], "gamma": 0.99, "f": "min", "w": 30},
        "agents": [],
    },
    "F-DSW TS (min)": {
        "policy_name": "CavenaghiFDSWTS",
        "args": {"n_arms": portfolio.shape[1], "gamma": 0.99, "f": "min", "n": 30},
        "agents": [],
    },
}
seeds = list(range(n_sim))
for policy_name, args in polices.items():
    rewards_mxt = np.zeros(shape=(portfolio.shape[0], n_sim))
    regret_mxt = np.zeros(shape=(portfolio.shape[0], n_sim))
    chosen_arms_mxt = np.zeros(shape=(portfolio.shape[0], n_sim))
    for i in tqdm(range(n_sim)):
        kwargs = args["args"].copy()
        kwargs["seed"] = seeds[i]
        rewards, policy = run_backtest(
            policy_name=args["policy_name"],
            policy_args=kwargs,
            portfolio=portfolio,
            arm=LastReturnArm()
        )
        rewards_mxt[:, i] = np.array(rewards)
        regret_mxt[:, i] = np.array(policy.regret)
        polices[policy_name]["agents"].append(policy)
    polices[policy_name]["rewards_mxt"] = rewards_mxt
    polices[policy_name]["regret_mxt"] = regret_mxt
    polices[policy_name]["std_reward"] = np.std(rewards_mxt, axis=1)
    polices[policy_name]["mean_reward"] = np.mean(rewards_mxt, axis=1)
    polices[policy_name]["std_regret"] = np.std(regret_mxt, axis=1)
    polices[policy_name]["mean_regret"] = np.mean(regret_mxt, axis=1)