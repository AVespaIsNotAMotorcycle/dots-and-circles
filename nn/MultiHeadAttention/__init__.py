import numpy as np

from utils.softmax import softmax

from nn.Module import Module
from nn.Linear import Linear
from nn.Dropout import Dropout

class MultiHeadAttention(Module):
    def __init__(self, size_in, size_out, context_length,
                 drop_rate, num_heads, qkv_bias=False):
        super().__init__(size_in, size_out)
        assert(size_out % num_heads == 0), \
            f"MultiHeadAttention expects size_out to be divisible by" \
            f"num_heads (size_out: {size_out}, num_heads: {num_heads})"
        
        self.num_heads = num_heads
        self.head_dim = size_out // num_heads
        self.params['W_query'] = Linear(size_in, size_out, use_bias=qkv_bias)
        self.params['W_key'] = Linear(size_in, size_out, use_bias=qkv_bias)
        self.params['W_value'] = Linear(size_in, size_out, use_bias=qkv_bias)
        self.params['out_proj'] = Linear(size_out, size_out)
        self.dropout = Dropout(drop_rate)
        self.mask = np.triu(np.ones((context_length, context_length)), k=1)

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

        keys = keys.transpose(0,2,1,3)
        queries = queries.transpose(0,2,1,3)
        values = values.transpose(0,2,1,3)

        attn_scores = queries @ keys.transpose(0,1,3,2)

        mask = self.mask[:num_tokens, :num_tokens]
        mask[mask == 1] = np.inf * -1
        np.nan_to_num(mask, copy=False, neginf=np.inf*-1)
        attn_scores = attn_scores + mask

        attn_weights = softmax(attn_scores / keys.shape[-1]**0.5)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(0, 2, 1, 3)
        context_vec = context_vec.copy().reshape(
            batch_size, num_tokens, self.size_out
        )
        context_vec = self.params['out_proj'](context_vec)

        return context_vec

    def backward(self, dLdy):
        print("MultiHeadAttention.backward is a placeholder - redo it soon!")
        dLdw = self.params['out_proj'].backward(dLdy)
        self.grads['out_proj'] = self.params['out_proj'].gradients()

        dLdx = self.params['W_value'].backward(dLdw)
        self.grads['W_value'] = self.params['W_value'].gradients()

        self.params['W_key'].backward(dLdw)
        self.grads['W_key'] = self.params['W_key'].gradients()

        self.params['W_query'].backward(dLdw)
        self.grads['W_query'] = self.params['W_query'].gradients()

        return dLdx
