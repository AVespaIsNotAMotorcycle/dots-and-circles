import numpy as np

from optim.Optimizer import Optimizer


class Adam(Optimizer):
    def __init__(self, learn_rate, beta_one=0.9, beta_two=0.999, keep_history=False):
        '''
        max_norm: if set, gradients are scaled down together whenever their
        combined norm (over every parameter) is larger than this, so one bad
        batch can't take an oversized step.
        '''
        super().__init__(learn_rate, keep_history)
        self.beta_one = beta_one
        self.beta_two = beta_two
        self.m = None
        self.v = None
        self.t = 0
        self.epsilon = 10**-8

    def delta(self, grads):
        self.t += 1
        if self.m == None:
            self.m = self.scale(grads, 0)
        if self.v == None:
            self.v = self.scale(grads, 0)
        
        self.m = self.add(self.scale(self.m, self.beta_one),
                          self.scale(grads, 1 - self.beta_one))
        m = self.scale(self.m, 1 / (1 - self.beta_one**self.t))

        g2 = self.power(grads, 2)
        self.v = self.add(self.scale(self.v, self.beta_two),
                          self.scale(g2, 1 - self.beta_two))
        v = self.scale(self.v, 1 / (1 - self.beta_two**self.t))
        v = self.add(v, self.epsilon)

        delta = self.scale(self.divide(m, self.power(v, 0.5)),
                           self.learn_rate)
        return delta
