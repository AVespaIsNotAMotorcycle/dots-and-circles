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
        self.params['layers'] = Sequential([
            Linear(emb_dim, 4 * emb_dim),
            GELU(),
            Linear(4 * emb_dim, emb_dim),
        ])

    def forward(self, x):
        return self.params['layers'](x)

    def backward(self, dLdy):
        dLdx = self.params['layers'].backward(dLdy)
        self.grads['layers'] = self.params['layers'].gradients()
        return dLdx

class Transformer(Module):
    def __init__(self, emb_dim, context_length,
                 num_heads=1, drop_rate=0, qkv_bias=False):
        super().__init__(size_in=emb_dim, size_out=emb_dim)
        self.params['norm1'] = LayerNorm(emb_dim)
        self.params['attention'] = MultiHeadAttention(
            size_in=emb_dim,
            size_out=emb_dim,
            context_length=context_length,
            num_heads=num_heads,
            drop_rate=drop_rate,
            qkv_bias=qkv_bias,
        )
        self.params['norm2'] = LayerNorm(emb_dim)
        self.params['ff'] = FeedForward(emb_dim)
        self.drop_shortcut = Dropout(drop_rate)

    def forward(self, x):
        shortcut = x
        x = self.params['norm1'](x)
        x = self.params['attention'](x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.params['norm2'](x)
        x = self.params['ff'](x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x

    def backward(self, gradient):
        gradient = self.params['ff'].backward(gradient)
        self.grads['ff'] = self.params['ff'].gradients()

        gradient = self.params['norm2'].backward(gradient)
        self.grads['norm2'] = self.params['norm2'].gradients()

        gradient = self.params['attention'].backward(gradient)
        self.grads['attention'] = self.params['attention'].gradients()

        gradient = self.params['norm1'].backward(gradient)
        self.grads['norm1'] = self.params['norm1'].gradients()

        return gradient
