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
        self.weight_x_h1 = randn(hidden_layer_size, constants.INPUT_LAYER_SIZE) / 1000
        self.weight_h1_h2 = randn(hidden_layer_size, hidden_layer_size) / 1000
        self.weight_h2_y = randn(constants.OUTPUT_LAYER_SIZE, hidden_layer_size) / 1000

        self.bias_h1 = np.zeros((hidden_layer_size, 1))
        self.bias_h2 = np.zeros((hidden_layer_size, 1))
        self.bias_y = np.zeros((constants.OUTPUT_LAYER_SIZE, 1))

    def get_neurons(self):
        return { "weight_x_h": self.weight_x_h1,
                 "weight_h_y": self.weight_h1_h2,
                 "weight_h_y": self.weight_h2_y,
                 "bias_h": self.bias_h1,
                 "bias_h": self.bias_h2,
                 "bias_y": self.bias_y }

    def forward(self, x):
        h1 = np.tanh(self.weight_x_h1 @ x + self.bias_h1)
        self.last_h1 = h1

        h2 = np.tanh(self.weight_h1_h2 @ h1 + self.bias_h2)
        self.last_h2 = h2

        y = self.weight_h2_y @ h2 + self.bias_y
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

        d_t_d_w_h2_y = self.last_h2
        d_t_d_b_y = 1
        d_t_d_h = self.weight_h2_y

        d_L_d_t = gradient * d_o_d_t

        d_L_d_w_h2_y = d_L_d_t @ d_t_d_w_h2_y.T
        d_L_d_b_y = d_L_d_t * d_t_d_b_y
        d_L_d_h2 = d_t_d_h.T @ d_L_d_t

        self.weight_h2_y -= self.LEARNING_RATE * d_L_d_w_h2_y
        self.bias_y -= self.LEARNING_RATE * d_L_d_b_y

        # Hidden Layer 2

        d_t_d_w_h1_h2 = self.last_h1
        d_t_d_b_h2 = 1
        d_h2_d_h1 = self.weight_h1_h2

        d_L_d_w_h1_h2 = d_L_d_h2 @ d_t_d_w_h1_h2.T
        d_L_d_b_h2 = d_L_d_h2 * d_t_d_b_h2
        d_L_d_h1 = d_h2_d_h1.T @ d_L_d_h2

        self.weight_h1_h2 -= self.LEARNING_RATE * d_L_d_w_h1_h2
        self.bias_h2 -= self.LEARNING_RATE * d_L_d_b_h2

        # Hidden Layer 1

        d_t_d_w_x_h1 = input
        d_t_d_b_h1 = 1
        d_h1_d_x = self.weight_x_h1

        d_L_d_w_x_h1 = d_L_d_h1 @ d_t_d_w_x_h1.T
        d_L_d_b_h1 = d_L_d_h1 * d_t_d_b_h1
        d_L_d_x = d_h1_d_x.T @ d_L_d_h1

        self.weight_x_h1 -= self.LEARNING_RATE * d_L_d_w_x_h1
        self.bias_h1 -= self.LEARNING_RATE * d_L_d_b_h1

        return

    def save(self, filename):
        saved_values = {}

        saved_values['weight_x_h1'] = self.weight_x_h1.tolist()
        saved_values['weight_h1_h2'] = self.weight_h1_h2.tolist()
        saved_values['weight_h_y'] = self.weight_h2_y.tolist()

        saved_values['bias_h1'] = self.bias_h1.tolist()
        saved_values['bias_h2'] = self.bias_h2.tolist()
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
        except:
            print('Failed to load {0}'.format(filename))
            return

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
