import numpy as np

from loss.LossFunction import LossFunction
from utils.softmax import softmax


class CrossEntropyLoss(LossFunction):
    # Positions with a label below 0 are ignored, e.g. padding
    def __init__(self, module, keep_history=False):
        super().__init__(module, keep_history)

    def loss(self, x, y, label):
        # y: logits of shape (..., num_classes), label: class indices (...)
        valid = label >= 0
        label = np.where(valid, label, 0)
        probs = softmax(y)
        p_c = np.take_along_axis(probs, label[..., None], axis=-1)[..., 0]
        cel = np.log(p_c) * -1
        cel = np.mean(cel[valid])
        return cel

    def loss_gradient(self, x, y, label):
        # dL/dy of softmax + cross entropy is probs - one_hot(label),
        # divided by the number of positions since the loss is their mean
        valid = label >= 0
        label = np.where(valid, label, 0)
        probs = softmax(y)
        one_hot = np.zeros(np.shape(probs))
        np.put_along_axis(one_hot, label[..., None], 1, axis=-1)
        gradient = (probs - one_hot) * valid[..., None]
        gradient = gradient / np.sum(valid)
        return gradient
