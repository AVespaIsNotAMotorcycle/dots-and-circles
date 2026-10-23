import numpy as np
import warnings

from utils.softmax import softmax, log_softmax

class TestSoftmax():
    def test1d(self):
        x = np.array([0, 1, 2, 3])
        y = np.array([0.0320586, 0.08714432, 0.23688282, 0.64391426])
        out = softmax(x)
        assert np.isclose(y, out).all()

    def test2d(self):
        x = np.array([[0, 1, 2, 3],
                      [0, 3, 7, 9]])
        y = np.array([[0.03205860, 0.08714432, 0.23688282, 0.64391426],
                      [0.00010845, 0.00217829, 0.11893034, 0.87878293]])
        out = softmax(x)
        assert np.isclose(y, out).all()

    def test3d(self):
        x = np.array([[[0, 1, 2, 3],
                       [0, 3, 7, 9]]])
        y = np.array([[[0.03205860, 0.08714432, 0.23688282, 0.64391426],
                       [0.00010845, 0.00217829, 0.11893034, 0.87878293]]])
        out = softmax(x)
        assert np.isclose(y, out).all()

    def test1d_overflow(self):
        x = np.array([0, 1, 2, 3])
        x = x * 2**16

        # Assert that no overflow warning is emitted
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = softmax(x)


class TestLogSoftmax():
    def test_matches_softmax(self):
        np.random.seed(123)
        x = np.random.randn(2, 3, 5) * 3
        assert np.allclose(np.exp(log_softmax(x)), softmax(x))

    def test_large_values(self):
        # softmax would round the small probabilities to 0, so its log
        # would be -inf; log_softmax stays finite and exact
        x = np.array([0., 1000., 2000.])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = log_softmax(x)
        assert np.allclose(out, [-2000., -1000., 0.])

    def test_rows_normalised(self):
        np.random.seed(123)
        x = np.random.randn(4, 6) * 50
        assert np.allclose(np.sum(np.exp(log_softmax(x)), axis=-1), 1)
