import numpy as np
from numpy.random import randn

class Layer():
    def __init__(self, size_in, size_out, normalize=False):
        self.weight = randn(size_out, size_in) / 1000
        self.bias = np.zeros((size_out, 1))
        self.history = []
        self.normalize = normalize

    def get(self):
        return self.weight, self.bias

    def set_property(self, value, key):
        if type(value) == str and value == "default": return
        if key == 'weight': current_value = self.weight
        if key == 'bias':   current_value = self.bias
        if not isinstance(value, np.ndarray):
            raise ValueError(f"Layer.set_property expects value to be a numpy "
                             f"array, got {type(value)} instead.")
        assert np.shape(value) == np.shape(current_value), \
            f"Expected value to be of shape {np.shape(current_value)}, but they were {np.shape(value)}"
        if key == 'weight': self.weight = value
        if key == 'bias': self.bias = value

    def set(self, weight="default", bias="default"):
        self.set_property(weight, "weight")
        self.set_property(bias, "bias")

    def forward(self, x):
        y = np.tanh(self.weight @ x + self.bias)
        self.history.append((x, y))
        return y

    def backprop(self, dLdy, learn_rate):
        assert np.shape(dLdy) == np.shape(self.bias), \
            (f"Layer.backprop expects dLdy to have the same shape as the layer's output "
             f"{np.shape(self.bias)}, but it was instead of shape {np.shape(dLdy)}")

        w = self.weight
        b = self.bias
        u = w @ self.history[-1][0]
        z = u + b
        y = np.tanh(z)

        '''
        In dydz, z is clipped so all values are between -7 and 7, since for values beyond
        that range 1 / cosh(z) ** 2 will be zero, and as z increases there's a good chance
        of getting warnings about an overflow.
        '''
        dydz = 1 / np.cosh(np.clip(z, -7, 7))**2
        dzdu = np.ones(np.shape(w))
        dudw = self.history[-1][0]
        dzdb = np.ones(np.shape(b))
        dudx = w

        dLdz = dydz * dLdy
        dLdu = dzdu.T * dLdz.T
        dLdw = dudw * dLdu
        dLdw = dLdw.T
        
        assert np.shape(dLdw) == np.shape(self.weight), \
            (f"dLdw {np.shape(dLdw)} must have the same shape as self.weight {np.shape(self.weight)}.")
        dLdb = dzdb * dydz * dLdy
        dLdx = dudx * dzdu * dydz * dLdy

        w -= dLdw * learn_rate
        b -= dLdb * learn_rate

        return dLdx
