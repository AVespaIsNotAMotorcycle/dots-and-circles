import numpy as np

class Module:
    def __init__(self, size_in=None, size_out=None):
        self.size_in = size_in
        self.size_out = size_out

        self.params = {}
        self.grads = {}

    def forward(self, x):
        shape = (1, self.size_out) if size_out != None else np.shape(x)
        return np.zeros(shape)

    def __call__(self, x):
        x = np.array(x)
        if self.size_in != None:
            assert np.shape(x)[-1] == self.size_in, \
                f"Expected x to be of shape {(1, self.size_in)}, but it was" \
                f"{np.shape(x)}"
        y = self.forward(x)
        if self.size_out != None:
            assert np.shape(y)[-1] == self.size_out, \
                f"Expected y to be of shape {(1, self.size_out)}, but it was" \
                f"{np.shape(y)}"
        return y

    def parameters(self, only_shape=False):
        # Returns dict of params and modules
        # If item in dict is param, it's an np ndarray
        # If item in dict is module m, it's m.parameters()
        out = {}
        for key, value in self.params.items():
            print(type(value))
            if isinstance(value, np.ndarray):
                if only_shape:
                    out[key] = np.shape(value)
                else:
                    out[key] = value
            else:
                out[key] = value.parameters(only_shape=only_shape)
        return out

    def gradients(self):
        # self.gradients is same shape as self.parameters
        return self.grads

    def backward(self, dL_dy):
        # Calculate gradients
        # Set gradients
        # Calculate dL_dx
        return dL_dx

    def descend(self, delta):
        for key in self.params.keys():
            self.params[key] -= delta[key]
