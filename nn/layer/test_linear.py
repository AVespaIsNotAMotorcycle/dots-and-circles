import numpy as np

from linear import Linear

class TestLinear:
    def test_sizes(self):
        size_in = 5
        size_out = 9
        layer = Linear(size_in, size_out)

        x = np.zeros((1, size_in))
        y = layer(x)
        expected = np.zeros((1, size_out))
        assert np.array_equal(y, expected)
