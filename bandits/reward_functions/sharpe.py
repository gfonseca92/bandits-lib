from .labeler import BaseLabeler
from numpy import ndarray
import math


class SharpeArm(BaseLabeler):

    def draw(self, arm_return: ndarray):
        # if arm_return < 0:
        #     return 0.
        if math.isnan(arm_return.std()):
            return 0.0
        return arm_return.mean() / (arm_return.std() + 1e-6)


class WindowedSharpeArm(BaseLabeler):

    def __init__(self, window=30):
        self.window = window

    def draw(self, arm_return: ndarray):
        if len(arm_return) < self.window:
            if math.isnan(arm_return.sum()):
                return 0.0
            return arm_return.sum()
        else:
            return arm_return[-self.window:].mean() / (arm_return[-self.window:].std() + 1e-6)
