import numpy as np

from nn.linear import Linear

class TestLinear:
    sizes = [(5, 9),
             (1, 1),
             (1, 5),
             (5, 1),
             (2, 2),
             (2, 3),
             (3, 2),
             (1500, 87),
             (3200, 3200)]

    def test_sizes(self):
        for size_in, size_out in self.sizes:
            layer = Linear(size_in, size_out)
            x = np.zeros((1, size_in))
            y = layer(x)
            expected = np.zeros((1, size_out))
            assert np.shape(y) == np.shape(expected)
            assert np.array_equal(y, expected)

            weights = np.ones((size_in, size_out)) * 2
            layer = Linear(size_in, size_out, weights=weights)
            x = np.ones((1, size_in))
            y = layer(x)
            expected = np.ones((1, size_out)) * 2 * size_in
            assert np.shape(y) == np.shape(expected)
            assert np.array_equal(y, expected)

    def test_bias(self):
        for size_in, size_out in self.sizes:
            weights = np.zeros((size_in, size_out))
            bias = np.ones((1, size_out))

            layer = Linear(size_in, size_out, weights=weights, bias=bias)
            x = np.ones((1, size_in))
            y = layer(x)
            expected = np.ones((1, size_out))
            assert np.array_equal(y, expected)

            layer = Linear(size_in, size_out, weights=weights, use_bias=False)
            x = np.ones((1, size_in))
            y = layer(x)
            expected = np.zeros((1, size_out))
            assert np.array_equal(y, expected)
