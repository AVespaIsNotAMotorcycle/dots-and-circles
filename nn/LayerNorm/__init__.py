import numpy as np

from nn.Module import Module


class LayerNorm(Module):
    def __init__(self, size):
        super().__init__(size, size)
        self.eps = 1e-7

        self.params['scale'] = np.ones(size)
        self.params['shift'] = np.zeros(size)

    def forward(self, x):
        # Each token is normalised over its own features
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)

        numerator = x - mean
        denominator = np.sqrt(var + self.eps)   # Add epsilon to avoid sqrt(0)
        norm_x = numerator / denominator

        self.last_norm = norm_x
        self.last_denominator = denominator
        return self.params['scale'] * norm_x + self.params['shift']

    def backward(self, dLdy):
        '''
        a = scale
        b = shift
        n = (x - mean) / sqrt(var + eps)
        y = a * n + b
        '''
        n = self.last_norm
        flat_dLdy = np.reshape(dLdy, (-1, self.size_in))
        self.grads['scale'] = np.sum(flat_dLdy * n.reshape(-1, self.size_in),
                                     axis=0)
        self.grads['shift'] = np.sum(flat_dLdy, axis=0)

        # Every element of n depends on the whole token through the mean and
        # variance, which gives the two subtracted terms
        dLdn = dLdy * self.params['scale']
        dLdx = (dLdn
                - np.mean(dLdn, axis=-1, keepdims=True)
                - n * np.mean(dLdn * n, axis=-1, keepdims=True))
        dLdx = dLdx / self.last_denominator
        return dLdx
