from .labeler import BaseLabeler


class ContinuousArm(BaseLabeler):

    def draw(self, arm_return):
        # if arm_return < 0:
        #     return 0.
        return arm_return