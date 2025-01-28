from .labeler import BaseLabeler
from numpy import ndarray
import numpy as np
import math


class ReturnArm(BaseLabeler):

    def draw(self, arm_return: ndarray):
        if len(arm_return) == 0:
            return 0.0
        return arm_return.sum()


class MeanReturnArm(BaseLabeler):

    def draw(self, arm_return: ndarray):
        if len(arm_return) == 0:
            return 0.0
        return arm_return.mean()


class LastReturnArm(BaseLabeler):

    def draw(self, arm_return: ndarray):
        if len(arm_return) == 0:
            return 0.0
        return arm_return.iloc[-1]


def exponentially_decaying_weights(len_series):
    # Generate exponentially decaying weights
    weights = np.exp(np.linspace(0, -1, len_series))
    weights /= weights.sum()
    return weights


class DiscountedReturns(BaseLabeler):

    def draw(self, arm_return: ndarray):
        weights = exponentially_decaying_weights(len(arm_return))
        return np.dot(arm_return, weights)


class WindowedReturnArm(BaseLabeler):

    def __init__(self, window=30):
        self.window = window

    def draw(self, arm_return: ndarray):
        if len(arm_return) < self.window:
            if math.isnan(arm_return.sum()):
                return 0.0
            return arm_return.sum()
        else:
            return arm_return[-self.window:].sum()


class WindowedNegativeReturnArm(BaseLabeler):

    def __init__(self, window=30):
        self.window = window

    def draw(self, arm_return: ndarray):
        if len(arm_return) < self.window:
            if math.isnan(arm_return.sum()):
                return 0.0
            return arm_return.sum()
        else:
            return -arm_return[-self.window:].sum()


class WindowedMeanReturnArm(BaseLabeler):

    def __init__(self, window=30):
        self.window = window

    def draw(self, arm_return: ndarray):
        if len(arm_return) < self.window:
            if math.isnan(arm_return.mean()):
                return 0.0
            return arm_return.mean()
        else:
            return arm_return[-self.window:].mean()
