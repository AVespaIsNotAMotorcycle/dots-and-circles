import numpy as np

def softmax(x):
    exp = np.exp(x)
    denom = np.sum(exp, axis=-1, keepdims=True)
    return exp / denom
