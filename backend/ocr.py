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
    LEARNING_RATE = 0.1
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

        softmax_numerators = np.exp(self.last_y) # numerators in softmax
        softmax_denominator = np.sum(softmax_numerators)  # denomenators in softmax

        d_output_d_numerators = (-softmax_numerators[label]
                                * softmax_numerators
                                / (softmax_denominator ** 2))
        d_output_d_numerators[label] = (softmax_numerators[label]
                                       * (softmax_denominator - softmax_numerators[label])
                                       / (softmax_denominator ** 2))

        d_numerators_d_w_h_y = self.last_h
        d_numerators_d_b_y = 1
        d_numerators_d_h = self.weight_h_y

        d_L_d_numerators = gradient * d_output_d_numerators

        d_L_d_w_h_y = d_L_d_numerators @ d_numerators_d_w_h_y.T
        d_L_d_b_y = d_L_d_numerators * d_numerators_d_b_y
        d_L_d_h = d_numerators_d_h.T @ d_L_d_numerators

        self.weight_h_y -= self.LEARNING_RATE * d_L_d_w_h_y
        self.bias_y -= self.LEARNING_RATE * d_L_d_b_y

        '''
        gradient = np.zeros(constants.OUTPUT_LAYER_SIZE)
        gradient[label] = -1 / output[label]
        '''
        return

    def save(self): return

    def load(self): return

    def train_on_lexigraph(self, slices, row_labels):
        predictions = []
        for index in range(len(slices)):
            input = slices[index]
            label = constants.ALPHABET.index(row_labels[index])

            output = self.forward(input)
            self.backward(input, output, label)

            prediction = list(output).index(max(output))
            predictions.append({ "character": prediction, "actual": label })
        return predictions
