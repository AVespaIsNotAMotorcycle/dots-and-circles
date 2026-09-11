import numpy as np
import random

from nn.Module import Module

class Dropout(Module):
    def __init__(self, size, drop_rate):
        super().__init__(size, size)

        assert drop_rate < 1, \
            f"Dropout cannot accept a drop rate >= 1 (drop rate was {drop_rate})"
        assert drop_rate >= 0, \
            "Dropout expects a positive drop rate (drop_rate was {drop_rate})"

        self.drop_rate = drop_rate
        self.eps = 1e-7

    def forward(self, x):
        size = self.size_in
        drop_count = int(size * self.drop_rate)
        keep_count = size - drop_count
        drop = [0] * drop_count
        keep = [1] * keep_count
        mask = drop + keep
        random.shuffle(mask)
        mask = np.array(mask)

        y = np.multiply(x, mask)
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        scale = x_mean / (y_mean + self.eps)
        y = y * scale
        return y
