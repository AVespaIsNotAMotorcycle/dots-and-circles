import numpy as np

from nn.Module import Module


class Dropout(Module):
    def __init__(self, drop_rate):
        super().__init__()

        assert drop_rate < 1, \
            f"Dropout cannot accept a drop rate >= 1" \
            f"(drop rate was {drop_rate})"
        assert drop_rate >= 0, \
            f"Dropout expects a positive drop rate" \
            f"(drop_rate was {drop_rate})"

        self.drop_rate = drop_rate

    def forward(self, x):
        '''
        Inverted dropout: zeroes exactly int(size * drop_rate) elements, and
        scales the rest up so each element keeps its expected value. Only
        active while training; in evaluation mode it passes x through.
        '''
        shape = np.shape(x)
        size = np.size(x)
        drop_count = int(size * self.drop_rate)
        self.last_dropped = self.training and drop_count > 0
        if not self.last_dropped:
            return x

        mask = np.ones(size)
        mask[:drop_count] = 0
        np.random.shuffle(mask)
        # Each element survives with probability (size - drop_count) / size,
        # so dividing by that keeps E[y] = x. This is 1 / (1 - drop_rate),
        # up to int() rounding the drop count.
        mask = mask.reshape(shape) * (size / (size - drop_count))

        self.last_mask = mask
        return x * mask

    def backward(self, dLdy):
        if not self.last_dropped:
            return dLdy

        # y = mask * x, where mask is fixed (0 or the scale), so
        # dL/dx = mask * dL/dy
        return dLdy * self.last_mask
