import numpy as np

from utils.softmax import softmax

from nn.Module import Module
from nn.Linear import Linear
from nn.Dropout import Dropout


class MultiHeadAttention(Module):

    def __init__(self, size_in, size_out, context_length,
                 drop_rate, num_heads, qkv_bias=False, causal_mask=True):
        super().__init__(size_in, size_out)
        assert (size_out % num_heads == 0), \
            f"MultiHeadAttention expects size_out to be divisible by" \
            f"num_heads (size_out: {size_out}, num_heads: {num_heads})"

        self.num_heads = num_heads
        self.head_dim = size_out // num_heads
        self.params['W_query'] = Linear(size_in, size_out, use_bias=qkv_bias)
        self.params['W_key'] = Linear(size_in, size_out, use_bias=qkv_bias)
        self.params['W_value'] = Linear(size_in, size_out, use_bias=qkv_bias)
        self.params['out_proj'] = Linear(size_out, size_out)
        self.dropout = Dropout(drop_rate)
        self.causal_mask = causal_mask
        self.mask = np.triu(np.ones((context_length, context_length)), k=1)
        self.lengths = None

    def set_lengths(self, lengths):
        '''
        lengths: number of real tokens in each batch item; the rest are
        padding, which no token attends to. None disables the padding mask.
        '''
        super().set_lengths(lengths)
        self.lengths = lengths

    def forward(self, x):
        batch_size, num_tokens, size_in = np.shape(x)
        keys = self.params['W_key'](x)
        queries = self.params['W_query'](x)
        values = self.params['W_value'](x)

        keys = keys.reshape(batch_size, num_tokens,
                            self.num_heads, self.head_dim)
        values = values.reshape(batch_size, num_tokens,
                                self.num_heads, self.head_dim)
        queries = queries.reshape(batch_size, num_tokens,
                                  self.num_heads, self.head_dim)

        keys = keys.transpose(0, 2, 1, 3)
        queries = queries.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        attn_scores = queries @ keys.transpose(0, 1, 3, 2)

        # Without the causal mask, every token can attend to every other
        if self.causal_mask:
            mask = self.mask[:num_tokens, :num_tokens]
            mask[mask == 1] = np.inf * -1
            np.nan_to_num(mask, copy=False, neginf=np.inf*-1)
            attn_scores = attn_scores + mask

        # Padding mask: no token attends to an item's padding tokens. Only
        # keys are masked; padding tokens' own outputs are ignored by the loss
        if self.lengths is not None:
            lengths = np.asarray(self.lengths)
            assert np.shape(lengths) == (batch_size,), \
                f"Expected {batch_size} lengths, one per batch item, " \
                f"but got {np.shape(lengths)}"
            assert np.all(lengths >= 1), \
                "Every batch item needs at least one real token"
            padding = np.arange(num_tokens)[None, :] >= lengths[:, None]
            attn_scores = np.where(padding[:, None, None, :],
                                   -np.inf, attn_scores)

        attn_weights = softmax(attn_scores / keys.shape[-1]**0.5)
        self.last_softmax = attn_weights
        attn_weights = self.dropout(attn_weights)

        self.last_keys = keys
        self.last_queries = queries
        self.last_values = values
        self.last_attn_weights = attn_weights

        context_vec = (attn_weights @ values).transpose(0, 2, 1, 3)
        context_vec = context_vec.copy().reshape(
            batch_size, num_tokens, self.size_out
        )
        context_vec = self.params['out_proj'](context_vec)

        return context_vec

    def split_heads(self, x):
        # (batch, tokens, size_out) -> (batch, heads, tokens, head_dim)
        batch_size, num_tokens, _ = np.shape(x)
        x = x.reshape(batch_size, num_tokens, self.num_heads, self.head_dim)
        return x.transpose(0, 2, 1, 3)

    def merge_heads(self, x):
        # (batch, heads, tokens, head_dim) -> (batch, tokens, size_out)
        batch_size, _, num_tokens, _ = np.shape(x)
        x = x.transpose(0, 2, 1, 3)
        return x.reshape(batch_size, num_tokens, self.size_out)

    def backward(self, dLdy):
        '''
        S = Q @ K^T / sqrt(head_dim) + mask
        A = dropout(softmax(S))
        C = A @ V
        y = out_proj(C)
        '''
        dLdC = self.params['out_proj'].backward(dLdy)
        self.grads['out_proj'] = self.params['out_proj'].gradients()
        dLdC = self.split_heads(dLdC)

        dLdA = dLdC @ self.last_values.transpose(0, 1, 3, 2)
        dLdV = self.last_attn_weights.transpose(0, 1, 3, 2) @ dLdC

        dLdA = self.dropout.backward(dLdA)
        # Softmax: each row's outputs share one denominator, so the gradient
        # of every score depends on the whole row
        A = self.last_softmax
        dLdS = A * (dLdA - np.sum(dLdA * A, axis=-1, keepdims=True))
        dLdS = dLdS / self.head_dim**0.5

        dLdQ = dLdS @ self.last_keys
        dLdK = dLdS.transpose(0, 1, 3, 2) @ self.last_queries

        # x feeds all three projections, so its gradient is their sum
        dLdx = self.params['W_query'].backward(self.merge_heads(dLdQ))
        self.grads['W_query'] = self.params['W_query'].gradients()

        dLdx = dLdx + self.params['W_key'].backward(self.merge_heads(dLdK))
        self.grads['W_key'] = self.params['W_key'].gradients()

        dLdx = dLdx + self.params['W_value'].backward(self.merge_heads(dLdV))
        self.grads['W_value'] = self.params['W_value'].gradients()

        return dLdx
