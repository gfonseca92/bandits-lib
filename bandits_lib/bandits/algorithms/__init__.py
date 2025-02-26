__all__ = ["Bandit", "available_bandits"]

from .bandit import Bandit
from .epsilon_greedy import EpsilonGreedy
from .bernoulli_ts import ThompsonSampling
from .beta_discounted_bernoulli_ts import BetaDiscountedThompsonSampling
from .adts import (AdaptiveDiscountedThompsonSampling, CombinatorialAdaptiveDiscountedThompsonSampling)
from .cavenaghi_interface import CavenaghiFDSWTS
from .ucb import (
    UCB1,
    DiscountedUCB1,
    SlidingWindowUCB1,
)


available_bandits = {
    "EpsilonGreedy": EpsilonGreedy,
    "ThompsonSampling": ThompsonSampling,
    "BetaDiscountedThompsonSampling": BetaDiscountedThompsonSampling,
    "UCB1": UCB1,
    "DiscountedUCB1": DiscountedUCB1,
    "SlidingWindowUCB1": SlidingWindowUCB1,
    "AdaptiveDiscountedThompsonSampling": AdaptiveDiscountedThompsonSampling,
    "CombinatorialAdaptiveDiscountedThompsonSampling": CombinatorialAdaptiveDiscountedThompsonSampling,
    "CavenaghiFDSWTS": CavenaghiFDSWTS,
}
