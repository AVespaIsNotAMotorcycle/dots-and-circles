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

    def test_clipping(self):
        # Nested like a module's grads; combined norm is sqrt(9 + 16) = 5
        grads = {'a': np.array([3., 0.]),
                 'inner': {'b': np.array([[0., 4.]])}}

        # Above max_norm: scaled down together to max_norm, same direction
        delta = SGD(0.5, max_norm=1)(grads)
        assert np.allclose(delta['a'], np.array([3., 0.]) / 5 * 0.5)
        assert np.allclose(delta['inner']['b'], np.array([[0., 4.]]) / 5 * 0.5)

        # At or below max_norm, or with no max_norm: plain SGD
        for sgd in [SGD(0.5, max_norm=5), SGD(0.5, max_norm=10), SGD(0.5)]:
            delta = sgd(grads)
            assert np.allclose(delta['a'], grads['a'] * 0.5)
            assert np.allclose(delta['inner']['b'], grads['inner']['b'] * 0.5)

        # The gradients passed in are left unchanged
        assert np.array_equal(grads['a'], np.array([3., 0.]))
