import numpy as np

from nn.Transformer import Transformer

def print_grad(grad, indent=0):
    for key, value in grad.items():
        print(' ' * indent * 2, key)
        if isinstance(value, np.ndarray):
            print(' ' * (indent + 1) * 2, np.shape(value))
        else:
            print_grad(value, indent=indent+1)

def compute_delta(grads, learn_rate):
    delta = {}

    for key in grads.keys():
        if isinstance(grads[key], np.ndarray):
            delta[key] = grads[key] * learn_rate
        else:
            delta[key] = compute_delta(grads[key], learn_rate)

    return delta

class TestTransformer():
    def test_SGD(self):
        batch_size = 1
        context_length = 1
        emb_dim = 8

        np.random.seed(123)
        x = np.random.rand(batch_size, context_length, emb_dim)
        y = np.random.rand(batch_size, context_length, emb_dim)

        transformer = Transformer(emb_dim, context_length, num_heads=4)

        loss_history = []
        for _ in range(10):
            out = transformer(x)
            
            mse_loss = np.sum((y - out)**2, keepdims=True) / len(out)
            dLdout = -2 * (1 - out)
            
            transformer.backward(dLdout)
            grads = transformer.gradients()

            learn_rate = 0.01
            delta = compute_delta(grads, learn_rate)
            transformer.descend(delta)

            if len(loss_history) > 0:
                assert mse_loss < loss_history[-1]
            loss_history.append(mse_loss)
        for entry in loss_history:
            print(entry)
        assert False
