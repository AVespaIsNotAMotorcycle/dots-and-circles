import numpy as np

from nn.Module import Module

class LayerNorm(Module):
    def __init__(self, size):
        super().__init__(size, size)
        self.eps = 1e-7

        '''
        To do: implement learnable weights
        self.params['scale'] = np.ones(size)
        self.params['shift'] = np.zeros(size)
        '''

    def forward(self, x):
        mean = np.mean(x)
        var = np.var(x)

        numerator = x - mean
        denominator = np.sqrt(var + self.eps) # Add epsilon to avoid sqrt(0)
        norm_x = numerator / denominator

        '''
        return self.params['scale'] * norm_x + self.params['shift']
        '''
        return norm_x

    def backward(self, dLdy):
        # print("LayerNorm.backward is a placeholder - redo it soon!")
        '''
        a = scale
        b = shift
        n = norm
        y = a * n + b

        n = 
        '''
        '''
        self.grads['scale'] = np.zeros(self.size_in)
        self.grads['shift'] = np.zeros(self.size_in)
        '''
        return dLdy
