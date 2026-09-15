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
        while np.shape(self.last_x) != (1, self.size_in):
            self.last_x = np.mean(self.last_x, axis=0)
            if sum(np.shape(self.last_x)) == self.size_in:
                self.last_x = self.last_x.reshape(1, self.size_in)

        y = x @ self.params['weights']
        if self.use_bias:
            y += self.params['bias']

        self.last_y = y
        while np.shape(self.last_y) != (1, self.size_out):
            self.last_y = np.mean(self.last_y, axis=0)
            if sum(np.shape(self.last_y)) == self.size_out:
                self.last_y = self.last_y.reshape(1, self.size_out)
        return y

    def backward(self, dLdy):
        assert np.shape(dLdy) == (1, self.size_out), \
            f"Linear expects dLdy to have the shape (1, self.size_out), " \
            f"{(1, self.size_out)}, but it was {np.shape(dLdy)}"
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
        '''
        while len(np.shape(dLdw)) > len(np.shape(self.params['weights'])):
            dLdw = np.mean(dLdw, axis=-1)
        '''
        assert np.shape(dLdw) == np.shape(self.params['weights']), \
            f"dLdw {np.shape(dLdw)} must have the same shape as " \
            f"self.params['weights'] {np.shape(self.params['weights'])}."

        dLdb = dydb * dLdy
        while len(np.shape(dLdb)) > len(np.shape(self.params['bias'])):
            dLdb = np.mean(dLdb, axis=0)
        assert np.shape(dLdb) == np.shape(self.params['bias']), \
            f"dLdb {np.shape(dLdb)} must have the same shape as " \
            f"self.params['bias'] {np.shape(self.params['bias'])}."
        dLdx = dudx * dLdy
        dLdx = np.mean(dLdx, axis=-1, keepdims=True).T
        dLdx = dLdx.reshape(np.shape(self.last_x)[-2:])

        self.grads['weights'] = dLdw
        self.grads['bias'] = dLdb

        assert np.shape(dLdx) == (1, self.size_in), \
            f"Linear expects dLdx to have the shape of (1, self.size_in), " \
            f"{(1, self.size_in)}, but it was {np.shape(dLdx)}"
        return dLdx
