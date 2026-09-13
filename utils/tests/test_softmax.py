import numpy as np

from utils.softmax import softmax

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
