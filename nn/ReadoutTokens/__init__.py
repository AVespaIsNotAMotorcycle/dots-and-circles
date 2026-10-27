import numpy as np

from nn.Module import Module


class ReadoutTokens(Module):
    '''
    Puts num_readouts learned tokens in front of every item in the batch.
    They carry no input of their own; through attention they gather what
    they need from the other tokens, and the model reads its output from
    them.
    '''
    def __init__(self, num_readouts, emb_dim):
        super().__init__(emb_dim, emb_dim)
        self.num_readouts = num_readouts
        # Small, like learned positional embeddings
        self.params['tokens'] = np.random.randn(num_readouts, emb_dim) * 0.02

    def forward(self, x):
        batch_size = np.shape(x)[0]
        tokens = np.broadcast_to(self.params['tokens'],
                                 (batch_size,) + np.shape(self.params['tokens']))
        return np.concatenate([tokens, x], axis=1)

    def backward(self, dLdy):
        # Every item shares the same readout tokens, so their gradient is
        # summed over the batch; the rest belongs to x
        self.grads['tokens'] = np.sum(dLdy[:, :self.num_readouts], axis=0)
        return dLdy[:, self.num_readouts:]
