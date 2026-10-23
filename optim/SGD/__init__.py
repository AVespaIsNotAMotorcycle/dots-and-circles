import numpy as np

from optim.Optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, learn_rate):
        super().__init__(learn_rate)

    def delta(self, grads):
        delta = {}

        for key in grads.keys():
            if isinstance(grads[key], np.ndarray):
                delta[key] = grads[key] * self.learn_rate
            else:
                delta[key] = self.delta(grads[key])

        return delta
