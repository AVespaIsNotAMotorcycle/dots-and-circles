import numpy as np
import random
import pytest

from nn.Dropout import Dropout

class TestDropout:
    def shapes_match(self, x, y):
        assert np.shape(y) == np.shape(x)

    def means_match(self, x, y):
        assert np.isclose(np.mean(y), np.mean(x))

    def more_or_same_nonzero(self, x, y):
        assert len(np.flatnonzero(x)) >= len(np.flatnonzero(y))

    def sufficient_zeros(self, x, y, size, drop_rate):
        print("size:", size)
        expected_zeros = int(size * drop_rate)
        num_zeros = size - len(np.flatnonzero(y))
        assert num_zeros >= expected_zeros

    def dropout_correct(self, x, y, size, drop_rate):
        self.shapes_match(x, y)
        self.means_match(x, y)
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
