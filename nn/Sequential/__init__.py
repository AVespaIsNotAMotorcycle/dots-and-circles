import numpy as np

from nn.Module import Module

class Sequential(Module):
    def __init__(self, module_list):
        super().__init__()
        self.modules = module_list

    def forward(self, x):
        for module in self.modules:
            x = module(x)
        return x
