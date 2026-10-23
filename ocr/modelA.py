from nn.Transformer import Transformer
from nn.Sequential import Sequential
from nn.Module import Module

class ModelA(Module):
    def __init__(self, depth = 2, emb_dim = 32,
                 context_length = 256, num_heads = 4):
        super().__init__(size_in=emb_dim, size_out=emb_dim)

        modules = []
        for i in range(depth):
            transformer = Transformer(emb_dim, context_length, num_heads)
            modules.append(transformer)
        self.params['layers'] = Sequential(modules)

    def forward(self, x):
        return self.params['layers'](x)

    def backward(self, dLdy):
        dLdx = self.params['layers'].backward(dLdy)
        self.grads['layers'] = self.params['layers'].gradients()
        return dLdx
