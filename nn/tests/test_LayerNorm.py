import numpy as np

from nn.LayerNorm import LayerNorm

class TestLayerNorm:
    def test_forward(self):
        np.random.seed(123)
        for i in range(10):
            size = (i + 1) ** 2
            norm = LayerNorm(size)

            x = np.random.rand(1, size)
            y = norm(x)

            if size == 1:
                assert np.var(y) == 0
            else:
                assert np.isclose(np.var(y), 1)
            assert np.isclose(np.mean(y), 0)
