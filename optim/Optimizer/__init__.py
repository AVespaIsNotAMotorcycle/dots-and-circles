import numpy as np


class Optimizer:
    def __init__(self, learn_rate, keep_history=False):
        '''
        keep_history: if True, every call records its gradients and delta.
        Off by default, since each entry is as big as the model.
        '''
        self.learn_rate = learn_rate
        self.keep_history = keep_history
        self.grads_history = []
        self.delta_history = []

    def operate(self, a, b, operation):
        valid_ops = ['scale', 'divide', 'power', 'add']
        if operation not in valid_ops:
            raise ValueError

        out = {}
        for key in a.keys():
            try:
                b_k = b[key]
            except:
                b_k = b
            if not isinstance(a[key], np.ndarray):
                out[key] = self.operate(a[key], b_k, operation)
            else:
                if operation == 'scale':
                    out[key] = a[key] * b_k
                if operation == 'divide':
                    out[key] = a[key] / b_k
                if operation == 'power':
                    out[key] = a[key] ** b_k
                if operation == 'add':
                    out[key] = a[key] + b_k
        return out

    def scale(self, a, b):
        return self.operate(a, b, 'scale')

    def power(self, a, b):
        return self.operate(a, b, 'power')

    def add(self, a, b):
        return self.operate(a, b, 'add')

    def divide(self, a, b):
        return self.operate(a, b, 'divide')

    def delta(self, grads):
        # Returns the change to be made to each parameter
        raise NotImplementedError

    def __call__(self, grads):
        delta = self.delta(grads)
        if self.keep_history:
            self.grads_history.append(grads)
            self.delta_history.append(delta)
        return delta

    def set_rate(self, learn_rate):
        self.learn_rate = learn_rate
