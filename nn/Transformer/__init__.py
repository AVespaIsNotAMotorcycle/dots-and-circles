from nn.Module import Module
from nn.Linear import Linear
from nn.GELU import GELU
from nn.Dropout import Dropout
from nn.LayerNorm import LayerNorm
from nn.MultiHeadAttention import MultiHeadAttention
from nn.Sequential import Sequential

class FeedForward(Module):
    def __init__(self, emb_dim):
        super().__init__(size_in=emb_dim, size_out=emb_dim)
        self.layers = Sequential([
            Linear(emb_dim, 4 * emb_dim),
            GELU(),
            Linear(4 * emb_dim, emb_dim),
        ])

    def forward(self, x):
        return self.layers(x)

class Transformer(Module):
    def __init__(self, emb_dim, context_length,
                 num_heads=1, drop_rate=0, qkv_bias=False):
        self.attention = MultiHeadAttention(
            size_in=emb_dim,
            size_out=emb_dim,
            context_length=context_length,
            num_heads=num_heads,
            drop_rate=drop_rate,
            qkv_bias=qkv_bias,
        )
        self.ff = FeedForward(emb_dim)
        self.norm1 = LayerNorm(emb_dim)
        self.norm2 = LayerNorm(emb_dim)
        self.drop_shortcut = nn.Dropout(drop_rate)

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x
