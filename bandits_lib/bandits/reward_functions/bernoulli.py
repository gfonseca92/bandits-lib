import random
from .labeler import BaseLabeler


class BernoulliArm(BaseLabeler):

    def __init__(self, p):
        self.p = p

    def draw(self):
        if random.random() > self.p:
            return 0.0
        else:
            return 1.0