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
        self.W_query = Linear(size_in, size_out, use_bias=qkv_bias)
        self.W_key = Linear(size_in, size_out, use_bias=qkv_bias)
        self.W_value = Linear(size_in, size_out, use_bias=qkv_bias)
        self.out_proj = Linear(size_out, size_out)
        self.dropout = Dropout(drop_rate)
        self.mask = np.triu(np.ones((context_length, context_length)), k=1)

    def forward(self, x):
        batch_size, num_tokens, size_in = np.shape(x)
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

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
        context_vec = self.out_proj(context_vec)

        return context_vec
