import numpy as np

def softmax(x):
    cutoff = 2**7
    x = np.clip(x, a_min=cutoff*-1, a_max=cutoff)
    exp = np.exp(x)
    denom = np.sum(exp, axis=-1, keepdims=True)
    return exp / denom
