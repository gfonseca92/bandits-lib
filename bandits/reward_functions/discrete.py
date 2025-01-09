import numpy as np
from numpy import ndarray
from .labeler import BaseLabeler


class DiscreteArm(BaseLabeler):

    def draw(self, arm_return):
        if len(arm_return) != 0:
            arm_return = arm_return.sum()
            grid_returns = np.linspace(-10, 10, 10000)
            return np.argmin(np.abs(arm_return - grid_returns))
        return 0.
