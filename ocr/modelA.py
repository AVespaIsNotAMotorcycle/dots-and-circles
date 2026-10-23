from nn.Transformer import Transformer
from nn.Sequential import Sequential
from nn.Linear import Linear
from nn.Module import Module

class ModelA(Module):
    def __init__(self, depth = 2, emb_dim = 32, context_length = 64,
                 num_heads = 4, num_classes = 45, causal_mask = False,
                 token_dim = 128):
        super().__init__(size_in=token_dim, size_out=num_classes)

        # Maps each token's pixels to an embedding
        modules = [Linear(token_dim, emb_dim)]
        for i in range(depth):
            transformer = Transformer(emb_dim, context_length, num_heads,
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
