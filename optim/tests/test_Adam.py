import numpy as np

from nn.Linear import Linear
from nn.Transformer import Transformer
from optim.Adam import Adam
from loss.MSELoss import MSELoss


class TestAdam:
    def test_on_linear(self):
        size_in = 5
        size_out = 3
        np.random.seed(123)
        learn_rate = 0.01

        x = np.random.rand(1, size_in)
        label = np.zeros((1, size_out))
        label[0][0] = 1.

        layer = Linear(size_in, size_out)

        mse = MSELoss(layer)
        adam = Adam(learn_rate)

        last_loss = None
        for _ in range(5):
            y = layer(x)

            loss, grads = mse(x, y, label)
            delta = adam(grads)
            layer.descend(delta)

            if last_loss is not None:
                assert loss < last_loss
            last_loss = loss

    def test_on_transformer(self):
        batch_size = 4
        emb_dim = 128
        context_length = 64
        np.random.seed(123)
        learn_rate = 0.01

        x = np.random.rand(batch_size, context_length, emb_dim)
        label = np.zeros((batch_size, 1, emb_dim))
        label[0][0] = 1.

        layer = Transformer(emb_dim, context_length, num_heads=4)

        mse = MSELoss(layer)
        adam = Adam(learn_rate)

        last_loss = None
        first_loss = None
        for _ in range(5):
            y = layer(x)

            loss, grads = mse(x, y, label)
            delta = adam(grads)
            layer.descend(delta)

            if _ == 0:
                first_loss = loss
            last_loss = loss
        assert last_loss < first_loss
