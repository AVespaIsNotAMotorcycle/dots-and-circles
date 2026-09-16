import numpy as np

from utils.softmax import softmax
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
    def run_test_SGD(self, batch_size, context_length, emb_dim):
        print('\n', batch_size, context_length, emb_dim)
        print("Total input embeddings:", batch_size * context_length * emb_dim)
        
        np.random.seed(123)
        x = np.random.rand(batch_size, context_length, emb_dim)
        y = np.random.rand(batch_size, context_length, emb_dim)
        y = np.argmax(y, axis=-1)

        transformer = Transformer(emb_dim, context_length, num_heads=4)

        loss_history = []
        iterations = 10
        for _ in range(iterations):
            out = transformer(x)
            probs = softmax(out)
            
            p_c = np.zeros((batch_size, context_length))
            gradient = np.zeros((batch_size, context_length, emb_dim))
            for item in range(batch_size):
                for token in range(context_length):
                    index = np.argmax(probs[item][token])
                    prob = probs[item][token][index]
                    p_c[item][token] = prob
                    gradient[item][token][index] = -1 / prob
            cross_entropy_loss = np.log(p_c) * -1
            cross_entropy_loss = np.mean(cross_entropy_loss)
            gradient = np.mean(gradient, axis=0)
            gradient = np.mean(gradient, axis=0)
            gradient = gradient.reshape(1, emb_dim)

            transformer.backward(gradient)
            grads = transformer.gradients()

            learn_rate = 0.001
            delta = compute_delta(grads, learn_rate)
            transformer.descend(delta)

            loss_history.append(cross_entropy_loss)
        if np.isnan(loss_history[-1]): print("LOSS NAN")
        if loss_history[0] > loss_history[-1]: print("Loss decreased")
        if loss_history[0] < loss_history[-1]: print("Loss INCREASED")
        return loss_history[0] > loss_history[-1]

    def test_SGD(self):
        loss_decreased = []
        for batch_size in [1, 2, 4, 32]:
            for context_length in [1, 4, 16, 64]:
                for emb_dim in [4, 24, 32]:
                    loss_decreased.append(
                        self.run_test_SGD(batch_size, context_length, emb_dim)
                    )
        assert loss_decreased.count(True) > len(loss_decreased) * .67
