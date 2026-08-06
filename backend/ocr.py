import numpy as np
from numpy.random import randn
import math
import json

import constants

def softmax(array):
    return np.exp(array) / sum(np.exp(array))
   
def cross_entropy_loss(output, actual_letter):
    return -np.log(output[actual_letter])

class NeuralNetwork:
    LEARNING_RATE = 0.001
    _use_file = True
    NN_FILE_PATH = "saved_ann.json"
    
    def __init__(self, hidden_layer_size=64):
        self.weight_x_h = randn(hidden_layer_size, constants.INPUT_LAYER_SIZE) / 1000
        self.weight_h_y = randn(constants.OUTPUT_LAYER_SIZE, hidden_layer_size) / 1000

        self.bias_h = np.zeros((hidden_layer_size, 1))
        self.bias_y = np.zeros((constants.OUTPUT_LAYER_SIZE, 1))

    def get_neurons(self):
        return { "weight_x_h": self.weight_x_h,
                 "weight_h_y": self.weight_h_y,
                 "bias_h": self.bias_h,
                 "bias_y": self.bias_y }

    def forward(self, x):
        h = np.tanh(self.weight_x_h @ x + self.bias_h)
        self.last_h = h

        y = self.weight_h_y @ h + self.bias_y
        self.last_y = y

        return softmax(y)

    def backward(self, input, output, label):
        loss = cross_entropy_loss(output, label)
        gradient = -1 / output[label]

        t_exp = np.exp(self.last_y) # t in softmax
        S = np.sum(t_exp)  # denomenators in softmax

        d_o_d_t = (-t_exp[label]
                                * t_exp
                                / (S ** 2))
        d_o_d_t[label] = (t_exp[label]
                                       * (S - t_exp[label])
                                       / (S ** 2))

        # Output Layer

        d_t_d_w_h_y = self.last_h
        d_t_d_b_y = 1
        d_t_d_h = self.weight_h_y

        d_L_d_t = gradient * d_o_d_t

        d_L_d_w_h_y = d_L_d_t @ d_t_d_w_h_y.T
        d_L_d_b_y = d_L_d_t * d_t_d_b_y
        d_L_d_h = d_t_d_h.T @ d_L_d_t

        self.weight_h_y -= self.LEARNING_RATE * d_L_d_w_h_y
        self.bias_y -= self.LEARNING_RATE * d_L_d_b_y

        # Hidden Layer

        d_t_d_w_x_h = input
        d_t_d_b_h = 1
        d_h_d_x = self.weight_x_h

        d_L_d_w_x_h = d_L_d_h @ d_t_d_w_x_h.T
        d_L_d_b_h = d_L_d_h * d_t_d_b_h
        d_L_d_x = d_h_d_x.T @ d_L_d_h

        self.weight_x_h -= self.LEARNING_RATE * d_L_d_w_x_h
        self.bias_h -= self.LEARNING_RATE * d_L_d_b_h

        return

    def save(self, filename):
        saved_values = {}

        saved_values['weight_x_h'] = self.weight_x_h.tolist()
        saved_values['weight_h_y'] = self.weight_h_y.tolist()

        saved_values['bias_h'] = self.bias_h.tolist()
        saved_values['bias_y'] = self.bias_y.tolist()

        with open(filename, 'w') as file:
            json.dump(saved_values, file)
        return

    def load(self, filename):
        try:
            with open(filename, 'r') as file:
                saved_values = json.load(file)

                self.weight_x_h = np.array(saved_values['weight_x_h'])
                self.weight_h_y = np.array(saved_values['weight_h_y'])

                self.bias_h = np.array(saved_values['bias_h'])
                self.bias_y = np.array(saved_values['bias_y'])
        except: return

    def predict_lexigraph(self, slices, row_labels=[], backprop=False):
        predictions = []
        for index in range(len(slices)):
            input = slices[index]
            if backprop: label = constants.ALPHABET.index(row_labels[index])

            output = self.forward(input)
            if backprop: self.backward(input, output, label)

            prediction = list(output).index(max(output))
            confidence = max(output)[0]

            return_value = { "character": prediction, "confidence": confidence }
            if backprop: return_value["actual"] = label
            predictions.append(return_value)
        return predictions

    def train_on_lexigraph(self, slices, row_labels):
        return self.predict_lexigraph(slices, row_labels, True)
