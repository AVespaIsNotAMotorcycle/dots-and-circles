import numpy as np
import random
import pytest

from nn.MultiHeadAttention import MultiHeadAttention

class TestMultiHeadAttention:
    def test_init_vali(self):
        for i in range(1,10):
            for j in range(1,10):
                size_in = i ** 2
                size_out = j ** 2
                num_heads = j
                dropout = 0
                context_length = 256

                mha = MultiHeadAttention(size_in, size_out, context_length,
                                         dropout, num_heads)

    def test_forward(self):
        '''
        Example taken from Sebastian Raschka's "Build a Large
        Language Model (From Scratch)"
        '''
        batch_size = 1
        tokens = np.array(
              [[0.43, 0.15, 0.89], # Your     (x^1)
               [0.55, 0.87, 0.66], # journey  (x^2)
               [0.57, 0.85, 0.64], # starts   (x^3)
               [0.22, 0.58, 0.33], # with     (x^4)
               [0.77, 0.25, 0.10], # one      (x^5)
               [0.05, 0.80, 0.55]] # step     (x^6)
            )
        emb_dim = 3

        size_in = emb_dim
        num_heads = 4
        size_out = emb_dim * num_heads
        dropout = 0
        context_length = 256

        mha = MultiHeadAttention(size_in, size_out, context_length,
                                 dropout, num_heads)

        x = tokens.reshape(batch_size, 6, emb_dim)

        y = mha(x)
        assert not np.isnan(y).any()
        assert np.shape(y) == (batch_size, 6, size_out)
