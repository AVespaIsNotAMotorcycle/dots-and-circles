import numpy as np

from nn.Module import Module


class TakeFirst(Module):
    # Keeps only the first count tokens of each item
    def __init__(self, count):
        super().__init__()
        self.count = count

    def forward(self, x):
        self.last_shape = np.shape(x)
        return x[:, :self.count]

    def backward(self, dLdy):
        # Discarded tokens didn't affect the loss, so their gradient is 0
        dLdx = np.zeros(self.last_shape)
        dLdx[:, :self.count] = dLdy
        return dLdx
