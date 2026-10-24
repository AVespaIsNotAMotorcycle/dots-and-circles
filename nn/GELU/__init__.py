import numpy as np

from nn.Module import Module


class GELU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        self.last_x = x
        self.last_tanh = np.tanh(0.7978845 * (x + (0.044715 * x**3)))
        return 0.5 * x * (1 + self.last_tanh)

    def cosh(self, x):
        # np.cosh will overflow if values are too high
        cutoff = 100
        x = np.clip(x, a_min=cutoff*-1, a_max=cutoff)
        return np.cosh(x)

    def backward(self, dLdy):
        lx = self.last_x
        lx3 = lx * lx * lx

        a = 0.5 * self.last_tanh

        b_1 = ((0.0535161 * lx3) + (0.398942 * lx))
        b_2 = self.cosh((0.0356774 * lx3) + (0.797885 * lx))
        b = b_1 / b_2**2

        dydx = a + b + 0.5
        dLdx = dydx * dLdy

        assert np.shape(dLdx) == np.shape(self.last_x), \
            f"GELU expects dLdy to have the shape of the last x, " \
            f"{np.shape(self.last_x)}, but it was {np.shape(dLdy)}."

        return dLdx
