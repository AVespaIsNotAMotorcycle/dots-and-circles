import numpy as np

from nn.Module import Module


class Dropout(Module):
    def __init__(self, drop_rate):
        super().__init__()

        assert drop_rate < 1, \
            f"Dropout cannot accept a drop rate >= 1" \
            f"(drop rate was {drop_rate})"
        assert drop_rate >= 0, \
            f"Dropout expects a positive drop rate" \
            f"(drop_rate was {drop_rate})"

        self.drop_rate = drop_rate
        self.eps = 1e-7

    def forward(self, x):
        shape = np.shape(x)
        x = x.flatten()

        size = np.size(x)
        drop_count = int(size * self.drop_rate)
        self.last_dropped = drop_count > 0
        if not self.last_dropped:
            return x.reshape(shape)

        mask = np.ones(size)
        mask[:drop_count] = 0
        np.random.shuffle(mask)

        y = np.multiply(x, mask)
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        scale = x_mean / (y_mean + self.eps)
        y = y * scale

        self.last_x = x
        self.last_mask = mask
        self.last_scale = scale
        self.last_x_mean = x_mean
        self.last_y_mean = y_mean

        y = y.reshape(shape)
        return y

    def backward(self, dLdy):
        if not self.last_dropped:
            return dLdy

        '''
        m = mask
        s = mean(x) / (mean(m * x) + eps)
        y = s * m * x

        s depends on every element of x, so each element's gradient also
        gets a share of dL/ds.
        '''
        shape = np.shape(dLdy)
        dLdy = np.reshape(dLdy, -1)
        m = self.last_mask
        size = np.size(m)
        denominator = self.last_y_mean + self.eps

        dLds = np.sum(dLdy * m * self.last_x)
        dsdx = (1 / denominator
                - self.last_x_mean * m / denominator**2) / size
        dLdx = self.last_scale * m * dLdy + dLds * dsdx
        return dLdx.reshape(shape)
