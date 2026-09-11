import numpy as np

from nn.Module import Module

class LayerNorm(Module):
    def __init__(self, size):
        super().__init__(size, size)
        self.eps = 1e-7

        self.scale = np.ones(size)
        self.shift = np.zeros(size)

    def forward(self, x):
        mean = np.mean(x)
        var = np.var(x)

        numerator = x - mean
        denominator = np.sqrt(var + self.eps) # Add epsilon to avoid sqrt(0)
        norm_x = numerator / denominator

        return self.scale * norm_x + self.shift
