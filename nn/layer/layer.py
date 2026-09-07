import numpy as np

class Layer:
    def __init__(self, size_in, size_out):
        self.size_in = size_in
        self.size_out = size_out

    def forward(self, x):
        return np.zeros((1, self.size_out))

    def __call__(self, x):
        assert np.shape(x) == (1, self.size_in), \
                f"Expected x to be of shape {(1, self.size_in)}, but it was" \
                f"{np.shape(x)}"
        y = self.forward(x)
        assert np.shape(y) == (1, self.size_out), \
            f"Expected y to be of shape {(1, self.size_out)}, but it was" \
            f"{np.shape(y)}"
        return y
