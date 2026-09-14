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
            self.params['weights'] = np.random.rand(size_in, size_out)

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
            return y + self.params['bias']
        else:
            return y

    def backward(self, dLdy):
        '''
        w = self.params['weights']
        b = self.params['bias']
        x = self.last_x
        u = x @ w
        y = u + b
        '''
        dydu = np.ones(np.shape(self.params['weights']))
        dudw = self.last_x
        dydb = np.ones(np.shape(self.params['bias']))
        dudx = self.params['weights']

        dLdu = dydu.T * dLdy.T
        dLdw = dudw * dLdu
        dLdw = dLdw.T
        assert np.shape(dLdw) == np.shape(self.params['weights']), \
            f"dLdw {np.shape(dLdw)} must have the same shape as" \
            f"self.params['weights'] {np.shape(self.params['weights'])}."

        dLdb = dydb * dLdy
        dLdx = dudx * dydu * dLdy

        self.grads['weights'] = dLdw
        self.grads['bias'] = dLdb

        return dLdx
