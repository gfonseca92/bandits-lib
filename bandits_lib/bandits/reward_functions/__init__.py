__all__ = [
    "available_reward_functions",
    "BaseLabeler",
    "BernoulliArm",
    "BinaryArm",
    "ContinuousArm",
    "DiscreteArm",
    "SharpeArm",
    "ReturnArm",
    "DiscountedReturns",
    "WindowedReturnArm",
    "WindowedSharpeArm",
    "LastReturnArm",
    "MeanReturnArm",
    "WindowedMeanReturnArm",
    "WindowedNegativeReturnArm",
]

from .labeler import BaseLabeler
from .bernoulli import BernoulliArm
from .binary import BinaryArm
from .continuous import ContinuousArm
from .discrete import DiscreteArm
from .sharpe import SharpeArm, WindowedSharpeArm
from .returns import (
    ReturnArm,
    DiscountedReturns,
    WindowedReturnArm,
    LastReturnArm,
    MeanReturnArm,
    WindowedNegativeReturnArm,
    WindowedMeanReturnArm
)

available_reward_functions = {
    "BernoulliArm": BernoulliArm,
    "BinaryArm": BinaryArm,
    "ContinuousArm": ContinuousArm,
    "DiscreteArm": DiscreteArm,
    "SharpeArm": SharpeArm,
    "ReturnArm": ReturnArm,
    "DiscountedReturns": DiscountedReturns,
    "WindowedReturnArm": WindowedReturnArm,
    "WindowedSharpeArm": WindowedSharpeArm,
    "LastReturnArm": LastReturnArm,
    "MeanReturnArm": MeanReturnArm,
    "WindowedMeanReturnArm": WindowedMeanReturnArm,
    "WindowedNegativeReturnArm": WindowedNegativeReturnArm,
}
