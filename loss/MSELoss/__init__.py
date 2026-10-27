import numpy as np

from loss.LossFunction import LossFunction
from utils.softmax import softmax


class MSELoss(LossFunction):
    def __init__(self, module, keep_history=False):
        super().__init__(module, keep_history)

    def loss(self, x, y, label):
        mse = np.sum((label - y)**2, keepdims=True) / len(y)
        return mse

    def loss_gradient(self, x, y, label):
        gradient = -2 * (label - y) / len(y)
        return gradient
