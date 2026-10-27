from nn.Transformer import Transformer
from nn.Sequential import Sequential
from nn.Linear import Linear
from nn.Module import Module
from nn.PositionalEmbedding import PositionalEmbedding
from nn.ReadoutTokens import ReadoutTokens
from nn.TakeFirst import TakeFirst

class ModelB(Module):
    '''
    ModelA with readout tokens: num_readouts learned tokens go in front of
    the image tokens, and readout token i predicts character i. Image tokens
    only describe their part of the image; their outputs are discarded.

    Readout tokens come first, so with causal_mask=True they couldn't see
    the image.
    '''
    def __init__(self, depth = 2, emb_dim = 32, context_length = 64,
                 num_heads = 4, num_classes = 45, causal_mask = False,
                 token_dim = 128, learned_positions = False,
                 num_readouts = 32):
        super().__init__(size_in=token_dim, size_out=num_classes)
        self.num_readouts = num_readouts

        # Maps each token's pixels to an embedding, then marks its position.
        # Positions are added before the readouts, so image token positions
        # start at 0.
        modules = [Linear(token_dim, emb_dim),
                   PositionalEmbedding(emb_dim, context_length,
                                       learned=learned_positions),
                   ReadoutTokens(num_readouts, emb_dim)]
        for i in range(depth):
            transformer = Transformer(emb_dim, num_readouts + context_length,
                                      num_heads, causal_mask=causal_mask)
            modules.append(transformer)
        # Maps each readout token's embedding to logits over the characters
        modules.append(TakeFirst(num_readouts))
        modules.append(Linear(emb_dim, num_classes))
        self.params['layers'] = Sequential(modules)

    def set_lengths(self, lengths):
        # Readout tokens sit in front of the image tokens and are never
        # padding, so each item's real tokens include them
        if lengths is not None:
            lengths = [length + self.num_readouts for length in lengths]
        super().set_lengths(lengths)

    def forward(self, x):
        return self.params['layers'](x)

    def backward(self, dLdy):
        dLdx = self.params['layers'].backward(dLdy)
        self.grads['layers'] = self.params['layers'].gradients()
        return dLdx
