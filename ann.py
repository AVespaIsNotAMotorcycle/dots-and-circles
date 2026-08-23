import json
import numpy as np

from layer import Layer

class ANN():
    def __init__(self, size_in, size_out, num_layers):
        self.size_in = size_in
        self.size_out = size_out

        self.layers = []
        for index in range(num_layers):
            layer_in = size_in if index == 0 else size_out
            layer_out = size_out
            self.layers.append(Layer(layer_in, layer_out))

    def set(self, parameters):
        assert len(parameters) == len(self.layers)
        for index, layer in enumerate(parameters):
            weight, bias = layer
            self.layers[index].set(weight=weight, bias=bias)

    def get(self):
        parameters = []
        for layer in self.layers:
            weight, bias = layer.get()
            parameters.append([weight, bias])
        return parameters

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backprop(self, x, y, dLdy, learn_rate):
        gradient = dLdy
        for i in range(len(self.layers)):
            index = len(self.layers) - i - 1
            gradient = self.layers[index].backprop(x, y, dLdy, learn_rate)
        return gradient

    def save(self, filename):
        with open(filename, "w") as f:
            array_params = self.get()
            list_params = []

            for weight, bias in array_params:
                list_params.append([weight.tolist(), bias.tolist()])

            f.write(json.dumps(list_params))

    def load(self, filename):
        with open(filename, "r") as f:
            print(filename)
            text = f.read()
            obj = json.loads(text)
            parameters = []
            for weight, bias in obj:
                weight = np.array(weight)
                bias = np.array(bias)
                parameters.append([weight, bias])
            self.set(parameters)
