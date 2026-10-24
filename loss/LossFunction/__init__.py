import copy


class LossFunction:
    def __init__(self, module, keep_history=False):
        '''
        keep_history: if True, every call records (loss, gradients) for
        history(). Off by default: each entry is a full copy of the
        gradients, so a long training run would keep gigabytes of them.
        '''
        self.module = module
        self.keep_history = keep_history
        self._history = []

    def loss(self, x, y, label):
        # Returns the scalar loss of output y given input x and label
        raise NotImplementedError

    def loss_gradient(self, x, y, label):
        # Returns dL/dy in the shape the module's backward expects
        raise NotImplementedError

    def __call__(self, x, y, label):
        loss = self.loss(x, y, label)

        dLdy = self.loss_gradient(x, y, label)
        self.module.backward(dLdy)
        gradient = self.module.gradients()
        if self.keep_history:
            # Modules reuse their grads dict, so copy it to keep history
            # intact
            gradient = copy.deepcopy(gradient)
            self._history.append((loss, gradient))
        return loss, gradient

    def history(self):
        return list(self._history)
