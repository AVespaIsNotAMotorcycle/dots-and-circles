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
