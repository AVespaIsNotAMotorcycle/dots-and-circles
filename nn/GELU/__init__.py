import numpy as np

from nn.Module import Module

class GELU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        self.last_x = x
        a = 0.5
        b = x
        c = 1 + np.tanh(((2 / np.pi) ** .5) * (x + 0.044715 * (x**3)))
        y = a * b * c
        return y

    def backward(self, dLdy):
        a = 0.5 * np.tanh((0.0356774 * self.last_x**3) \
            + (0.797885 * self.last_x))
        b = ((0.0535161 * self.last_x**3) + (0.398942 * self.last_x)) \
            / np.cosh((0.0356774 * self.last_x**3) + (0.797885 * self.last_x))**2
        c = 0.5
        dydx = a + b + c
        dLdx = dydx * dLdy
        return dLdx
