import numpy as np

from nn.ReadoutTokens import ReadoutTokens
from nn.TakeFirst import TakeFirst


class TestReadoutTokens:
    def test_forward(self):
        np.random.seed(123)
        readouts = ReadoutTokens(3, 4)
        x = np.random.randn(2, 5, 4)
        y = readouts(x)

        assert np.shape(y) == (2, 8, 4)
        # Readouts come first and are the same for every item
        for item in range(2):
            assert np.array_equal(y[item, :3], readouts.params['tokens'])
        assert np.array_equal(y[:, 3:], x)

    def test_parameters(self):
        readouts = ReadoutTokens(3, 4)
        params = readouts.parameters()
        assert len(params) == 1
        assert np.shape(params['tokens']) == (3, 4)


class TestTakeFirst:
    def test_forward(self):
        np.random.seed(123)
        x = np.random.randn(2, 5, 4)
        y = TakeFirst(3)(x)
        assert np.array_equal(y, x[:, :3])

    def test_backward(self):
        take = TakeFirst(3)
        take(np.zeros((2, 5, 4)))
        dLdx = take.backward(np.ones((2, 3, 4)))
        assert np.shape(dLdx) == (2, 5, 4)
        assert np.all(dLdx[:, :3] == 1)
        assert np.all(dLdx[:, 3:] == 0)
