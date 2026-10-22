import copy


class LossFunction:
    def __init__(self, module):
        self.module = module
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
        # Modules reuse their grads dict, so copy it to keep history intact
        gradient = copy.deepcopy(self.module.gradients())

        self._history.append((loss, gradient))
        return loss, gradient

    def history(self):
        return list(self._history)
