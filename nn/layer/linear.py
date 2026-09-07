import numpy as np

from layer import Layer

def Linear(Layer):
    def __init__(self, size_in, size_out,
                 weights=None, bias=None, use_bias=True):
        super().__init__(size_in, size_out)

        self.weights = weights \
            if weights != None \
            else np.random.rand(size_in, size_out)
        self.bias = bias \
            if bias != None \
            else np.zeros((size_in, size_out))

    def forward(self, x):
        y = x @ self.weights
        z = y + self.bias
        return z
