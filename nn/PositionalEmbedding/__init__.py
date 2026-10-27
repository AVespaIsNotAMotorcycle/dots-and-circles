import numpy as np

from nn.Module import Module


class PositionalEmbedding(Module):
    '''
    Adds a vector to each token that depends on its position, so attention
    can tell tokens apart by where they are.

    learned=False: fixed sine and cosine waves at different frequencies
        (as in "Attention Is All You Need"). Defined for any number of
        tokens, with nothing to train.
    learned=True: a trained (context_length, emb_dim) table. Only covers
        the first context_length positions.
    '''
    def __init__(self, emb_dim, context_length=None, learned=False):
        super().__init__(emb_dim, emb_dim)
        self.learned = learned

        if learned:
            assert context_length is not None, \
                "Learned positional embeddings need a context_length"
            # Small, so positions don't swamp the token embeddings at first
            self.params['embeddings'] = \
                np.random.randn(context_length, emb_dim) * 0.02

    def sinusoidal(self, num_tokens):
        position = np.arange(num_tokens)[:, None]
        dimension = np.arange(self.size_in)[None, :]
        # Dimensions 2i and 2i + 1 share a frequency: sin for even, cos for odd
        frequency = 1 / 10000 ** (2 * (dimension // 2) / self.size_in)
        angle = position * frequency
        return np.where(dimension % 2 == 0, np.sin(angle), np.cos(angle))

    def forward(self, x):
        num_tokens = np.shape(x)[-2]
        if self.learned:
            context_length = np.shape(self.params['embeddings'])[0]
            assert num_tokens <= context_length, \
                f"PositionalEmbedding has {context_length} learned " \
                f"positions, but got {num_tokens} tokens"
            embeddings = self.params['embeddings'][:num_tokens]
        else:
            embeddings = self.sinusoidal(num_tokens)

        self.last_num_tokens = num_tokens
        return x + embeddings

    def backward(self, dLdy):
        '''
        y = x + E

        The same E is added to every item in the batch, so its gradient is
        summed over the batch. Unused positions get no gradient.
        '''
        if self.learned:
            num_tokens = self.last_num_tokens
            gradient = np.zeros(np.shape(self.params['embeddings']))
            gradient[:num_tokens] = np.sum(
                np.reshape(dLdy, (-1, num_tokens, self.size_in)), axis=0)
            self.grads['embeddings'] = gradient

        return dLdy
