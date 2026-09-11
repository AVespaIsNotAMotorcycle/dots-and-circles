import numpy as np

from nn.Module import Module

class Linear(Module):
    def __init__(self, size_in, size_out,
                 weights=None, bias=None, use_bias=True):
        super().__init__(size_in, size_out)

        if weights is not None:
            assert np.shape(weights) == (size_in, size_out), \
                f"Expected weights to be of shape {(size_in, size_out)}," \
                f"but it was {np.shape(weights)}"
            self.weights = weights
        else:
            self.weights = np.random.rand(size_in, size_out)

        if bias is not None:
            assert np.shape(bias) == (1, size_out), \
                f"Expected bias to be of shape {(1, size_out)}," \
                f"but it was {np.shape(bias)}"
            self.bias = bias
        else:
            self.bias = np.zeros((1, size_out))

        self.use_bias = use_bias

    def forward(self, x):
        y = x @ self.weights
        if self.use_bias:
            return y + self.bias
        else:
            return y
