class Optimizer:
    def __init__(self, learn_rate):
        self.learn_rate = learn_rate
        self.grads_history = []
        self.delta_history = []

    def delta(self, grads):
        # Returns the change to be made to each parameter
        raise NotImplementedError

    def __call__(self, grads):
        self.grads_history.append(grads)
        delta = self.delta(grads)
        self.delta_history.append(delta)
        return delta

    def set_rate(self, learn_rate):
        self.learn_rate = learn_rate
