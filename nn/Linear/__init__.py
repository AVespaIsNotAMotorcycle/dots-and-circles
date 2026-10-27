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
            self.params['weights'] = weights
        else:
            # Zero-centred and scaled by 1/sqrt(size_in), so outputs keep
            # roughly the same scale as inputs instead of growing each layer
            self.params['weights'] = (np.random.randn(size_in, size_out)
                                      / np.sqrt(size_in))

        if bias is not None:
            assert np.shape(bias) == (1, size_out), \
                f"Expected bias to be of shape {(1, size_out)}," \
                f"but it was {np.shape(bias)}"
            self.params['bias'] = bias
        else:
            self.params['bias'] = np.zeros((1, size_out))

        self.use_bias = use_bias

    def forward(self, x):
        self.last_x = x

        y = x @ self.params['weights']
        if self.use_bias:
            y = y + self.params['bias']
        return y

    def backward(self, dLdy):
        assert np.shape(dLdy)[:-1] == np.shape(self.last_x)[:-1] \
            and np.shape(dLdy)[-1] == self.size_out, \
            f"Linear expects dLdy to match the shape of its last output, " \
            f"{np.shape(self.last_x)[:-1] + (self.size_out,)}, " \
            f"but it was {np.shape(dLdy)}"
        '''
        y = x @ w + b

        Every token's output uses the same w and b, so their gradients are
        summed over all tokens (and batches).
        '''
        x = self.last_x.reshape(-1, self.size_in)
        dLdy_flat = np.reshape(dLdy, (-1, self.size_out))

        self.grads['weights'] = x.T @ dLdy_flat
        if self.use_bias:
            self.grads['bias'] = np.sum(dLdy_flat, axis=0, keepdims=True)
        else:
            self.grads['bias'] = np.zeros(np.shape(self.params['bias']))

        dLdx = dLdy @ self.params['weights'].T
        return dLdx
