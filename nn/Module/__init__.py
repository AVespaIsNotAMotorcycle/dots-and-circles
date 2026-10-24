import numpy as np


class Module:
    def __init__(self, size_in=None, size_out=None):
        self.size_in = size_in
        self.size_out = size_out

        self.params = {}
        self.grads = {}

    def forward(self, x):
        shape = (1, self.size_out) if size_out is not None else np.shape(x)
        return np.zeros(shape)

    def __call__(self, x):
        x = np.array(x)
        if self.size_in is not None:
            assert np.shape(x)[-1] == self.size_in, \
                f"Expected x to be of shape {(1, self.size_in)}, but it was" \
                f"{np.shape(x)}"
        y = self.forward(x)
        if self.size_out is not None:
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
            if isinstance(value, np.ndarray):
                if only_shape:
                    out[key] = np.shape(value)
                else:
                    out[key] = value
            else:
                out[key] = value.parameters(only_shape=only_shape)
        return out

    def flat_parameters(self, prefix=""):
        # Returns {path: array} for every param, e.g.
        # {"layers.0_Linear.weights": ...}. The arrays are the live params,
        # not copies, so writing into them changes the model.
        out = {}
        for key, value in self.params.items():
            if isinstance(value, np.ndarray):
                out[prefix + key] = value
            else:
                out.update(value.flat_parameters(prefix + key + "."))
        return out

    def save_weights(self, path):
        # Saves every param to an .npz file, keyed by its path
        np.savez(path, **self.flat_parameters())

    def load_weights(self, path):
        # Loads params saved by save_weights into this model, which must
        # have the same structure
        params = self.flat_parameters()
        with np.load(path) as saved:
            assert set(saved.files) == set(params), \
                f"Saved params don't match the model's. Missing: " \
                f"{sorted(set(params) - set(saved.files))}, unexpected: " \
                f"{sorted(set(saved.files) - set(params))}"
            for key, value in params.items():
                assert np.shape(saved[key]) == np.shape(value), \
                    f"{key} has shape {np.shape(value)}, but the saved one " \
                    f"is {np.shape(saved[key])}"
                # Copy in place, so modules keep their own arrays
                value[...] = saved[key]

    def gradients(self):
        # self.gradients is same shape as self.parameters
        return self.grads

    def backward(self, dL_dy):
        # Calculate gradients
        # Set gradients
        # Calculate dL_dx
        return dL_dx

    def set_lengths(self, lengths):
        # Passes each batch item's number of real (non-padding) tokens down
        # to every submodule; modules that need it override this
        for value in self.params.values():
            if isinstance(value, Module):
                value.set_lengths(lengths)

    def descend(self, delta):
        for key in self.params.keys():
            if isinstance(self.params[key], np.ndarray):
                self.params[key] -= delta[key]
            else:
                self.params[key].descend(delta[key])
