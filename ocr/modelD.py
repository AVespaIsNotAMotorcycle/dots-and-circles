from nn.Transformer import Transformer
from nn.Sequential import Sequential
from nn.Linear import Linear
from nn.Module import Module
from nn.PositionalEmbedding import PositionalEmbedding

class ModelD(Module):
    '''
    ModelC with dropout: every Transformer drops drop_rate of its attention
    weights and of both shortcut branches' outputs while training. Call
    set_training(False) before evaluating, so nothing is dropped.
    '''
    uses_ctc = True

    def __init__(self, depth = 2, emb_dim = 32, context_length = 64,
                 num_heads = 4, num_classes = 45, causal_mask = False,
                 token_dim = 128, learned_positions = False, blank = None,
                 drop_rate = 0.1):
        super().__init__(size_in=token_dim, size_out=num_classes)
        # The output class used as CTC's blank; the last one by default
        self.blank = num_classes - 1 if blank is None else blank

        # Maps each token's pixels to an embedding, then marks its position
        modules = [Linear(token_dim, emb_dim),
                   PositionalEmbedding(emb_dim, context_length,
                                       learned=learned_positions)]
        for i in range(depth):
            transformer = Transformer(emb_dim, context_length, num_heads,
                                      drop_rate=drop_rate,
                                      causal_mask=causal_mask)
            modules.append(transformer)
        # Maps each token's embedding to logits over the characters
        modules.append(Linear(emb_dim, num_classes))
        self.params['layers'] = Sequential(modules)

    def forward(self, x):
        return self.params['layers'](x)

    def backward(self, dLdy):
        dLdx = self.params['layers'].backward(dLdy)
        self.grads['layers'] = self.params['layers'].gradients()
        return dLdx
