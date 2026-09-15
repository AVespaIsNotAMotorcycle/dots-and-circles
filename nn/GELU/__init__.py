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
        a = 0.5 * np.tanh((0.0356774 * self.last_x**3) \
            + (0.797885 * self.last_x))
        b = ((0.0535161 * self.last_x**3) + (0.398942 * self.last_x)) \
            / self.cosh((0.0356774 * self.last_x**3) + (0.797885 * self.last_x))**2
        c = 0.5
        dydx = a + b + c
        dLdx = dydx * dLdy
        while sum(np.shape(dLdx)) > np.shape(self.last_x)[-1]:
            dLdx = np.mean(dLdx, axis=0)
        dLdx = dLdx.reshape(1, np.shape(self.last_x)[-1])

        assert np.shape(dLdx) == (1, np.shape(self.last_x)[-1]), \
            f"GELU expects dLdx to have the (1, z), where z is the final" \
            f"dimension of the last x, "\
            f"(z = {np.shape(self.last_x)[-1]}), but it was {np.shape(dLdx)}."
        
        return dLdx
