import copy

import numpy as np

from nn.Dropout import Dropout
from nn.GELU import GELU
from nn.LayerNorm import LayerNorm
from nn.Linear import Linear
from nn.MultiHeadAttention import MultiHeadAttention
from nn.Sequential import Sequential
from nn.Transformer import Transformer


def numerical_gradient(f, array, eps=1e-6):
    # Central differences: nudge each element of array in place
    gradient = np.zeros(np.shape(array))
    for index in np.ndindex(np.shape(array)):
        original = array[index]
        array[index] = original + eps
        plus = f()
        array[index] = original - eps
        minus = f()
        array[index] = original
        gradient[index] = (plus - minus) / (2 * eps)
    return gradient


def leaves(params, grads, prefix=""):
    # Walks nested params and grads together, yielding each array
    for key, value in params.items():
        if isinstance(value, np.ndarray):
            yield prefix + key, value, grads[key]
        else:
            yield from leaves(value.params, grads[key], prefix + key + ".")


def check_gradients(module, x):
    '''
    Uses L = sum(y * R) for a fixed random R, so dL/dy = R. Reseeds before
    every forward pass so dropout masks are the same each time.
    '''
    np.random.seed(123)
    R = np.random.randn(*np.shape(module(x)))

    def f():
        np.random.seed(0)
        return np.sum(module(x) * R)

    f()
    dLdx = module.backward(R)
    grads = copy.deepcopy(module.gradients())

    assert np.shape(dLdx) == np.shape(x)
    assert np.allclose(dLdx, numerical_gradient(f, x),
                       rtol=1e-4, atol=1e-6), "dL/dx"
    for name, param, grad in leaves(module.params, grads):
        assert np.shape(grad) == np.shape(param), name
        assert np.allclose(grad, numerical_gradient(f, param),
                           rtol=1e-4, atol=1e-6), name


class TestGradients:
    def setup_method(self):
        np.random.seed(123)

    def test_linear(self):
        for shape in [(1, 4), (3, 4), (2, 3, 4)]:
            x = np.random.randn(*shape)
            check_gradients(Linear(4, 5), x)
            check_gradients(Linear(4, 5, use_bias=False), x)

    def test_GELU(self):
        check_gradients(GELU(), np.random.randn(2, 3, 4) * 2)

    def test_LayerNorm(self):
        norm = LayerNorm(6)
        norm.params['scale'] = np.random.randn(6)
        norm.params['shift'] = np.random.randn(6)
        check_gradients(norm, np.random.randn(2, 3, 6))

    def test_dropout(self):
        for drop_rate in [0., 0.3]:
            check_gradients(Dropout(drop_rate), np.random.rand(2, 3, 4) + 0.5)

    def test_sequential(self):
        seq = Sequential([Linear(4, 6), GELU(), Linear(6, 3)])
        check_gradients(seq, np.random.randn(2, 3, 4))

    def test_multi_head_attention(self):
        for drop_rate in [0., 0.2]:
            for causal_mask in [True, False]:
                mha = MultiHeadAttention(size_in=6, size_out=8,
                                         context_length=8,
                                         drop_rate=drop_rate, num_heads=2,
                                         qkv_bias=True,
                                         causal_mask=causal_mask)
                check_gradients(mha, np.random.randn(2, 5, 6))

    def test_transformer(self):
        for drop_rate in [0., 0.2]:
            for causal_mask in [True, False]:
                transformer = Transformer(emb_dim=8, context_length=8,
                                          num_heads=2, drop_rate=drop_rate,
                                          causal_mask=causal_mask)
                check_gradients(transformer, np.random.randn(2, 4, 8))
