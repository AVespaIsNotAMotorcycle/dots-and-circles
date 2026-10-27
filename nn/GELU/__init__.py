import numpy as np

from nn.Module import Module


class GELU(Module):
    def __init__(self):
        super().__init__()
        self.k = np.sqrt(2 / np.pi)

    def forward(self, x):
        self.last_x = x
        self.last_tanh = np.tanh(0.7978845 * (x + (0.044715 * x**3)))
        return 0.5 * x * (1 + self.last_tanh)

    def backward(self, dLdy):
        x, t = self.last_x, self.last_tanh
        dudx = self.k * (1 + 3 * 0.044715 * x * x)
        dydx = 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * dudx
        return dydx * dLdy
