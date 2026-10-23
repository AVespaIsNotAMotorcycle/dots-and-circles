import numpy as np

from nn.Linear import Linear
from optim.SGD import SGD
from loss.MSELoss import MSELoss


class TestSGD:
    def test_on_linear(self):
        size_in = 5
        size_out = 3
        np.random.seed(123)
        learn_rate = 0.01

        x = np.random.rand(1, size_in)
        label = np.zeros((1, size_out))
        label[0][0] = 1.

        layer = Linear(size_in, size_out)

        mse = MSELoss(layer)
        sgd = SGD(learn_rate)

        last_loss = None
        for _ in range(5):
            y = layer(x)

            loss, grads = mse(x, y, label)
            delta = sgd(grads)
            layer.descend(delta)

            if last_loss is not None:
                assert loss < last_loss
            last_loss = loss
