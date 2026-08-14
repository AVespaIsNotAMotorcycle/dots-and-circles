import numpy as np
from numpy.random import randn
import json

import constants

input_size = constants.INPUT_LAYER_SIZE
output_size = 4

def softmax(array):
    return np.exp(array) / sum(np.exp(array))

class Classifier:
    classes = ['A', 'B', 'C', 'D']

    def __init__(self, hidden_size = 64):
        # Weights
        self.Whh = randn(hidden_size, hidden_size) / 1000
        self.Wxh = randn(hidden_size, input_size) / 1000
        self.Why = randn(output_size, hidden_size) / 1000

        # Biases
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def forward(self, inputs):
        h = np.zeros((self.Whh.shape[0], 1))

        self.last_inputs = inputs
        self.last_hs = { 0: h }

        for i, x in enumerate(inputs):
            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
            self.last_hs[i + 1] = h

        y = self.Why @ h + self.by

        return y, h

    def backprop(self, d_y, learn_rate=2e-2):
        # d_y (dL/dy) has shape (output_size, 1).
        # learn_rate is a float.
        n = len(self.last_inputs)

        # Calculate dL/dWhy and dL/dby.
        d_Why = d_y @ self.last_hs[n].T
        d_by = d_y

        # Initialize dL/dWhh, dL/dWxh, and dL/dbh to zero.
        d_Whh = np.zeros(self.Whh.shape)
        d_Wxh = np.zeros(self.Wxh.shape)
        d_bh = np.zeros(self.bh.shape)

        # Calculate dL/dh for the last h.
        d_h = self.Why.T @ d_y

        # Backpropogate through time.
        for t in reversed(range(n)):
            # An intermediate value: dL/dh * (1 - h^2)
            temp = ((1 - self.last_hs[t + 1] ** 2) * d_h)

            # dL/db = dL/dh * (1 - h^2)
            d_bh += temp

            # dL/dWhh = dL/dh * (1 - h^2) * h_{t-1}
            d_Whh += temp @ self.last_hs[t].T

            # dL/dWxh = dL/dh * (1 - h^2) * x
            d_Wxh += temp @ self.last_inputs[t].T

            # Next dL/dh = dL/dh * (1 - h^2) * Whh
            d_h = self.Whh @ temp

        # Clip to prevent exploding gradients.
        for d in [d_Wxh, d_Whh, d_Why, d_bh, d_by]:
            np.clip(d, -1, 1, out=d)

        # Update weights and biases using gradient descent
        self.Whh -= learn_rate * d_Whh
        self.Wxh -= learn_rate * d_Wxh
        self.Why -= learn_rate * d_Why
        self.bh -= learn_rate * d_bh
        self.by -= learn_rate * d_by

    def predict_lexigraph(self, slices):
        out, _ = self.forward(slices)
        probs = softmax(out)

        return self.classes[np.argmax(probs)]

    def train_on_lexigraph(self, slices, label):
        out, _ = self.forward(slices)
        probs = softmax(out)

        target = self.classes.index(label)
        prediction = self.classes[np.argmax(probs)]
        success = prediction == label
        loss = -1 * np.log(probs[target][0])

        d_L_d_y = probs
        d_L_d_y[target] -= 1

        self.backprop(d_L_d_y)

        return prediction, success, loss

    def save(self, filename):
        saved_values = {}

        # Weights
        saved_values['Whh'] = self.Whh.tolist()
        saved_values['Wxh'] = self.Wxh.tolist()
        saved_values['Why'] = self.Why.tolist()

        # Biases
        saved_values['bh'] = self.bh.tolist()
        saved_values['by'] = self.by.tolist()

        with open(filename, 'w') as file:
            json.dump(saved_values, file)
        return

    def load(self, filename):
        try:
            with open(filename, 'r') as file:
                saved_values = json.load(file)
    
                # Weights
                self.Whh = np.array(saved_values['Whh'])
                self.Wxh = np.array(saved_values['Wxh'])
                self.Why = np.array(saved_values['Why'])

                # Biases
                self.bh = np.array(saved_values['bh'])
                self.by = np.array(saved_values['by'])
        except:
            print('Failed to load {0}'.format(filename))
            return
