import numpy as np
import random
import pytest

from nn.Dropout import Dropout

class TestDropout:
    def test_init(self):
        for i in range(40):
            size = (i + 1) ** 2

            with pytest.raises(Exception) as e:
                dropout = Dropout(size, 1.)

            with pytest.raises(Exception) as e:
                dropout = Dropout(size, -1)

    def test_forward(self):
        np.random.seed(123)
        random.seed(123)
        drop_rates = [0., 0.2, 0.3, 0.5, 0.9]
        for i in range(40):
            size = (i + 1) ** 2
            for drop_rate in drop_rates:
                dropout = Dropout(size, drop_rate)
    
                x = np.random.rand(1, size)
                y = dropout(x)

                assert np.shape(y) == np.shape(x)
                assert np.isclose(np.mean(y), np.mean(x))
                assert len(np.flatnonzero(x)) >= len(np.flatnonzero(y))

                expected_zeros = int(size * drop_rate)
                num_zeros = size - len(np.flatnonzero(y))
                assert num_zeros >= expected_zeros
