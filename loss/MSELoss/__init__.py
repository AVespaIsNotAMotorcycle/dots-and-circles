import numpy as np

from loss.LossFunction import LossFunction
from utils.softmax import softmax


class MSELoss(LossFunction):
    def __init__(self, module):
        super().__init__(module)

    def loss(self, x, y, label):
        mse = np.sum((label - y)**2, keepdims=True) / len(y)
        return mse

    def loss_gradient(self, x, y, label):
        gradient = -2 * (label - y)
        print(gradient)
        # Modules expect dL/dy of shape (1, size_out), so average the rest
        num_classes = np.shape(gradient)[-1]
        gradient = gradient.reshape(-1, num_classes)
        gradient = np.mean(gradient, axis=0, keepdims=True)
        print(gradient)
        return gradient
