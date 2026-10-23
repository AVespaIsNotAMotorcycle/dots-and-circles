import numpy as np

from optim.Optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, learn_rate, max_norm=None):
        '''
        max_norm: if set, gradients are scaled down together whenever their
        combined norm (over every parameter) is larger than this, so one bad
        batch can't take an oversized step.
        '''
        super().__init__(learn_rate)
        self.max_norm = max_norm

    def squared_norm(self, grads):
        total = 0
        for value in grads.values():
            if isinstance(value, np.ndarray):
                total += np.sum(value ** 2)
            else:
                total += self.squared_norm(value)
        return total

    def scale(self, grads, factor):
        delta = {}

        for key in grads.keys():
            if isinstance(grads[key], np.ndarray):
                delta[key] = grads[key] * factor
            else:
                delta[key] = self.scale(grads[key], factor)

        return delta

    def delta(self, grads):
        factor = self.learn_rate
        if self.max_norm is not None:
            norm = np.sqrt(self.squared_norm(grads))
            if norm > self.max_norm:
                factor *= self.max_norm / norm
        return self.scale(grads, factor)
