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

    def cosh(self, x):
        # np.cosh will overflow if values are too high
        cutoff = 100
        x = np.clip(x, a_min=cutoff*-1, a_max=cutoff)
        return np.cosh(x)

    def backward(self, dLdy):
        a = 0.5 \
            * np.tanh((0.0356774 * self.last_x**3) + (0.797885 * self.last_x))

        lx = self.last_x
        b_1 = ((0.0535161 * lx**3) + (0.398942 * lx))
        b_2 = self.cosh((0.0356774 * lx**3) + (0.797885 * lx))
        b = b_1 / b_2**2

        c = 0.5
        dydx = a + b + c
        dLdx = dydx * dLdy

        assert np.shape(dLdx) == np.shape(self.last_x), \
            f"GELU expects dLdy to have the shape of the last x, " \
            f"{np.shape(self.last_x)}, but it was {np.shape(dLdy)}."

        return dLdx
