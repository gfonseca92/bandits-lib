from .labeler import BaseLabeler


class BinaryArm(BaseLabeler):

    def draw(self, arm_return):
        if arm_return.sum() > 0.:
            return 1.0
        return 0.
