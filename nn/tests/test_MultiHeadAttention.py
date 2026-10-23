import numpy as np
import random
import pytest
import json

from nn.MultiHeadAttention import MultiHeadAttention


class TestMultiHeadAttention:
    def test_init_vali(self):
        for i in range(1, 10):
            for j in range(1, 10):
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
              [[0.43, 0.15, 0.89],  # Your     (x^1)
               [0.55, 0.87, 0.66],  # journey  (x^2)
               [0.57, 0.85, 0.64],  # starts   (x^3)
               [0.22, 0.58, 0.33],  # with     (x^4)
               [0.77, 0.25, 0.10],  # one      (x^5)
               [0.05, 0.80, 0.55]]  # step     (x^6)
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

    def test_parameters(self):
        emb_dim = 3
        size_in = emb_dim
        num_heads = 4
        size_out = emb_dim * num_heads
        dropout = 0
        context_length = 256

        mha = MultiHeadAttention(size_in, size_out, context_length,
                                 dropout, num_heads)

        params = mha.parameters(only_shape=True)
        # string = json.dumps(params, indent=4)
        # print(string)
        assert len(params.items()) == 4

    def test_causal_mask(self):
        # With the mask, changing a later token can't affect earlier outputs;
        # without it, it changes every output
        np.random.seed(123)
        num_tokens = 6
        x = np.random.randn(2, num_tokens, 4)
        changed = x.copy()
        changed[:, -1] += 1

        for causal_mask in [True, False]:
            mha = MultiHeadAttention(4, 8, 16, 0, 2,
                                     causal_mask=causal_mask)
            y = mha(x)
            y_changed = mha(changed)
            earlier_unchanged = np.allclose(y[:, :-1], y_changed[:, :-1])
            assert earlier_unchanged == causal_mask
            assert not np.allclose(y[:, -1], y_changed[:, -1])

    def test_padding_mask(self):
        # What's in an item's padding tokens can't affect its real tokens
        np.random.seed(123)
        lengths = [6, 3, 1]
        x = np.random.randn(3, 6, 4)
        changed = x.copy()
        for item, length in enumerate(lengths):
            changed[item, length:] = np.random.randn(6 - length, 4) * 100

        for causal_mask in [True, False]:
            mha = MultiHeadAttention(4, 8, 16, 0, 2,
                                     causal_mask=causal_mask)
            mha.set_lengths(lengths)
            y = mha(x)
            y_changed = mha(changed)
            for item, length in enumerate(lengths):
                assert np.allclose(y[item, :length],
                                   y_changed[item, :length])

            # Without the padding mask, the changed padding does leak in
            mha.set_lengths(None)
            if not causal_mask:
                assert not np.allclose(mha(x)[1, :3], mha(changed)[1, :3])

    def test_full_lengths_match_no_mask(self):
        np.random.seed(123)
        x = np.random.randn(2, 5, 4)
        mha = MultiHeadAttention(4, 8, 16, 0, 2, causal_mask=False)
        y = mha(x)
        mha.set_lengths([5, 5])
        assert np.allclose(mha(x), y)

    def test_lengths_checked(self):
        mha = MultiHeadAttention(4, 8, 16, 0, 2)
        x = np.random.randn(2, 5, 4)
        for lengths in [[5], [5, 0]]:
            mha.set_lengths(lengths)
            with pytest.raises(AssertionError):
                mha(x)

    def test_set_lengths_reaches_nested_attention(self):
        from nn.Sequential import Sequential
        from nn.Transformer import Transformer
        transformers = [Transformer(8, 16, num_heads=2) for _ in range(2)]
        Sequential(transformers).set_lengths([3, 4])
        for transformer in transformers:
            assert transformer.params['attention'].lengths == [3, 4]
