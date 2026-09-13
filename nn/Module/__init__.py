import numpy as np

class Module:
    def __init__(self, size_in=None, size_out=None):
        self.size_in = size_in
        self.size_out = size_out

    def forward(self, x):
        shape = (1, self.size_out) if size_out != None else np.shape(x)
        return np.zeros(shape)

    def __call__(self, x):
        x = np.array(x)
        if self.size_in != None:
            assert np.shape(x) == (1, self.size_in), \
                f"Expected x to be of shape {(1, self.size_in)}, but it was" \
                f"{np.shape(x)}"
        y = self.forward(x)
        if self.size_out != None:
            assert np.shape(y) == (1, self.size_out), \
                f"Expected y to be of shape {(1, self.size_out)}, but it was" \
                f"{np.shape(y)}"
        return y
