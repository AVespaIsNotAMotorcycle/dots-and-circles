import numpy as np

def softmax(x):
    cutoff = 2**7
    x = np.clip(x, a_min=cutoff*-1, a_max=cutoff)
    exp = np.exp(x)
    denom = np.sum(exp, axis=-1, keepdims=True)
    return exp / denom

def log_softmax(x):
    # log(softmax(x)) without clipping or underflow. Subtracting each row's
    # maximum doesn't change the result, but keeps exp from overflowing.
    shifted = x - np.max(x, axis=-1, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
