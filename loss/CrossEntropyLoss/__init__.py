import numpy as np

from loss.LossFunction import LossFunction
from utils.softmax import softmax


class CrossEntropyLoss(LossFunction):
    def __init__(self, module):
        super().__init__(module)

    def loss(self, x, y, label):
        # y: logits of shape (..., num_classes), label: class indices (...)
        probs = softmax(y)
        p_c = np.take_along_axis(probs, label[..., None], axis=-1)
        cel = np.log(p_c) * -1
        cel = np.mean(cel)
        return cel

    def loss_gradient(self, x, y, label):
        # dL/dy of softmax + cross entropy is probs - one_hot(label)
        probs = softmax(y)
        one_hot = np.zeros(np.shape(probs))
        np.put_along_axis(one_hot, label[..., None], 1, axis=-1)
        gradient = probs - one_hot
        # Modules expect dL/dy of shape (1, size_out), so average the rest
        num_classes = np.shape(gradient)[-1]
        gradient = gradient.reshape(-1, num_classes)
        gradient = np.mean(gradient, axis=0, keepdims=True)
        return gradient
