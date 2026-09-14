import numpy as np

from nn.Module import Module

class GELU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        a = 0.5
        b = x
        c = 1 + np.tanh(((2 / np.pi) ** .5) * (x + 0.044715 * (x**3)))
        y = a * b * c
        return y
