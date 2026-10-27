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

    def test_per_token(self):
        # Each token is normalised on its own, regardless of the others
        np.random.seed(123)
        norm = LayerNorm(8)
        x = np.random.rand(2, 3, 8)
        x[1] *= 100
        y = norm(x)
        assert np.allclose(np.mean(y, axis=-1), 0)
        assert np.allclose(np.var(y, axis=-1), 1)
        assert np.allclose(norm(x[:1]), y[:1])

    def test_parameters(self):
        np.random.seed(123)
        for i in range(10):
            size = (i + 1) ** 2
            norm = LayerNorm(size)

            params = norm.parameters()

            assert len(params) == 2
            assert np.shape(params['scale']) == (size,)
            assert np.shape(params['shift']) == (size,)
