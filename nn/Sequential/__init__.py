import numpy as np

from nn.Module import Module

class Sequential(Module):
    def __init__(self, module_list):
        super().__init__()
        for index, item in enumerate(module_list):
            self.params[f"{index}_{type(item).__name__}"] = item

    def forward(self, x):
        for key, module in self.params.items():
            x = module(x)
        return x

    def backward(self, dLdy):
        dLdx = dLdy
        for key, module in list(self.params.items())[::-1]:
            dLdx = module.backward(dLdx)
            self.grads[key] = module.gradients()
        return dLdx
