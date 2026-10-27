import numpy as np
import pytest

from nn.Linear import Linear
from loss.LossFunction import LossFunction


class SquaredError(LossFunction):
    def loss(self, x, y, label):
        return np.mean((y - label) ** 2)

    def loss_gradient(self, x, y, label):
        return 2 * (y - label) / np.size(y)


class TestLossFunction:
    def setup_method(self):
        np.random.seed(123)
        self.size_in = 4
        self.size_out = 3
        self.x = np.random.rand(1, self.size_in)
        self.label = np.random.rand(1, self.size_out)

    def test_not_implemented(self):
        layer = Linear(self.size_in, self.size_out)
        loss_function = LossFunction(layer)
        y = layer(self.x)
        with pytest.raises(NotImplementedError):
            loss_function.loss(self.x, y, self.label)
        with pytest.raises(NotImplementedError):
            loss_function(self.x, y, self.label)

    def test_call(self):
        layer = Linear(self.size_in, self.size_out)
        loss_function = SquaredError(layer)
        y = layer(self.x)
        loss, gradient = loss_function(self.x, y, self.label)

        assert loss == loss_function.loss(self.x, y, self.label)

        # Gradient should match calling backward on the module directly
        expected = Linear(self.size_in, self.size_out,
                          weights=layer.params['weights'].copy())
        expected(self.x)
        expected.backward(loss_function.loss_gradient(self.x, y, self.label))
        for key, value in expected.gradients().items():
            assert np.allclose(gradient[key], value)

    def test_history(self):
        layer = Linear(self.size_in, self.size_out)
        loss_function = SquaredError(layer, keep_history=True)
        assert loss_function.history() == []

        results = []
        for _ in range(5):
            y = layer(self.x)
            loss, gradient = loss_function(self.x, y, self.label)
            layer.descend({key: value * 0.01
                           for key, value in gradient.items()})
            results.append((loss, gradient))

        history = loss_function.history()
        assert len(history) == len(results)
        for (loss, gradient), (h_loss, h_gradient) in zip(results, history):
            assert loss == h_loss
            for key in gradient.keys():
                assert np.array_equal(gradient[key], h_gradient[key])

        # Entries must not share the module's grads dict
        assert history[0][1] is not history[1][1]
        assert history[0][1] is not layer.gradients()
        assert not np.array_equal(history[0][1]['weights'],
                                  history[-1][1]['weights'])

        # Descending on the gradients should have reduced the loss
        assert history[-1][0] < history[0][0]

        # Modifying the returned list must not affect the stored history
        history.clear()
        assert len(loss_function.history()) == len(results)
