import numpy as np
import random
import pytest

from nn.Dropout import Dropout


class TestDropout:
    def shapes_match(self, x, y):
        assert np.shape(y) == np.shape(x)

    def kept_scaled(self, x, y, size, drop_rate):
        # Every element is either dropped (0) or scaled by the same factor,
        # size / (size - dropped), which keeps its expected value
        drop_count = int(size * drop_rate)
        scale = size / (size - drop_count)
        kept = y != 0
        assert np.allclose(y[kept], x[kept] * scale)

    def more_or_same_nonzero(self, x, y):
        assert len(np.flatnonzero(x)) >= len(np.flatnonzero(y))

    def sufficient_zeros(self, x, y, size, drop_rate):
        print("size:", size)
        expected_zeros = int(size * drop_rate)
        num_zeros = size - len(np.flatnonzero(y))
        assert num_zeros >= expected_zeros

    def dropout_correct(self, x, y, size, drop_rate):
        self.shapes_match(x, y)
        self.kept_scaled(x, y, size, drop_rate)
        self.more_or_same_nonzero(x, y)
        self.sufficient_zeros(x, y, size, drop_rate)
        return True

    def test_init(self):
        for i in range(40):
            size = (i + 1) ** 2

            with pytest.raises(Exception) as e:
                dropout = Dropout(1.)

            with pytest.raises(Exception) as e:
                dropout = Dropout(-1)

    def test_forward(self):
        np.random.seed(123)
        random.seed(123)
        drop_rates = [0., 0.2, 0.3, 0.5, 0.9]
        for i in range(40):
            size = (i + 1) ** 2
            for drop_rate in drop_rates:
                dropout = Dropout(drop_rate)

                x = np.random.rand(1, size)
                y = dropout(x)

                assert self.dropout_correct(x, y, size, drop_rate)

    def test_multiple_tokens(self):
        emb_dim, drop_rate = 50, 0.4
        np.random.seed(123)

        dropout = Dropout(drop_rate)
        for token_count in [1, 5, 20]:
            x = np.random.rand(token_count, emb_dim)
            y = dropout(x)
            assert np.shape(y) == (token_count, emb_dim)
            assert self.dropout_correct(x, y, emb_dim * token_count, drop_rate)

    def test_multiple_batches(self):
        emb_dim, token_count, drop_rate = 50, 20, 0.4
        np.random.seed(123)

        dropout = Dropout(drop_rate)
        for batch_size in [1, 5, 20]:
            x = np.random.rand(batch_size, token_count, emb_dim)
            y = dropout(x)
            size = batch_size * token_count * emb_dim
            assert self.dropout_correct(x, y, size, drop_rate)

    def test_expected_value(self):
        # Averaged over many masks, each element comes out as itself
        np.random.seed(123)
        dropout = Dropout(0.3)
        x = np.random.rand(4, 10)
        mean = np.mean([dropout(x) for _ in range(5000)], axis=0)
        assert np.allclose(mean, x, rtol=0.05)

    def test_zero_mean_input_stable(self):
        # The scale doesn't depend on the data, so zero-mean inputs (like
        # the residual stream) can't make it blow up or flip sign
        np.random.seed(123)
        dropout = Dropout(0.1)
        for _ in range(100):
            x = np.random.randn(4, 64, 32)
            y = dropout(x)
            kept = y != 0
            assert np.allclose(y[kept] / x[kept], 1 / 0.9, rtol=1e-3)

    def test_evaluation_mode(self):
        np.random.seed(123)
        dropout = Dropout(0.5)
        x = np.random.rand(3, 8)
        dropout.set_training(False)
        assert np.array_equal(dropout(x), x)
        dLdy = np.random.rand(3, 8)
        assert np.array_equal(dropout.backward(dLdy), dLdy)

        dropout.set_training(True)
        assert not np.array_equal(dropout(x), x)

    def test_backward(self):
        np.random.seed(123)
        dropout = Dropout(0.4)
        x = np.random.rand(2, 5, 6)
        y = dropout(x)
        dLdy = np.random.rand(2, 5, 6)
        dLdx = dropout.backward(dLdy)
        # Dropped elements get no gradient; kept ones get the same scale
        assert np.all(dLdx[y == 0] == 0)
        assert np.allclose(dLdx[y != 0], dLdy[y != 0] / 0.6)
