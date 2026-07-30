import numpy as np
import math
import json

class NeuralNetwork:
    LEARNING_RATE = 0.2
    _use_file = True
    NN_FILE_PATH = "saved_ann.json"

    def _rand_initialize_weights(self, size_in, size_out):
        return [((x * 0.12) - 0.06) for x in np.random.rand(size_out, size_in)]
    
    def _sigmoid_scalar(self, z):
        return 1 / (1 + math.e ** -z)
    
    def __init__(self, num_hidden_nodes):
        print("Initializing neural network")
        self.theta1 = self._rand_initialize_weights(400, num_hidden_nodes)
        self.theta2 = self._rand_initialize_weights(num_hidden_nodes, 10)
        self.input_layer_bias = self._rand_initialize_weights(1, num_hidden_nodes)
        self.hidden_layer_bias = self._rand_initialize_weights(1, 10)
    
    def sigmoid(self, matrix):
        sigmoid_to_matrix = np.vectorize(self._sigmoid_scalar)
        new_matrix = sigmoid_to_matrix(matrix)
        return new_matrix

    def forward_propogate(self, pixels):
        y1 = np.dot(np.asmatrix(self.theta1), np.asmatrix(pixels).T)
        y1 = y1 + np.asmatrix(self.input_layer_bias)
        y1 = self.sigmoid(y1)
    
        y2 = np.dot(np.array(self.theta2), y1)
        y2 = y2 + np.asmatrix(self.hidden_layer_bias)
        y2 = self.sigmoid(y2)

        predictions = y2.T.tolist()[0]

        results = {
            "y1": y1,
            "y2": y2,
            "predictions": predictions,
        }
        return results
    
    def back_propogate(self, pixels, results, actual_digit):
        y1 = results['y1']
        y2 = results['y2']

        actual_vals = [0] * 10
        actual_vals[actual_digit] = 1
        output_errors = np.asmatrix(actual_vals).T - np.asmatrix(y2)
        hidden_errors = np.multiply(np.dot(np.asmatrix(self.theta2).T, output_errors), y1)
    
        self.theta1 += self.LEARNING_RATE * np.dot(np.asmatrix(hidden_errors), np.asmatrix(pixels))
        self.theta2 += self.LEARNING_RATE * np.dot(np.asmatrix(output_errors), np.asmatrix(y1).T)
        self.hidden_layer_bias += self.LEARNING_RATE * output_errors
        self.input_layer_bias += self.LEARNING_RATE * hidden_errors

    def save(self):
        if not self._use_file:
            return

        json_neural_network = {
            "theta1":[np_mat.tolist()[0] for np_mat in self.theta1],
            "theta2":[np_mat.tolist()[0] for np_mat in self.theta2],
            "b1":self.input_layer_bias[0].tolist()[0],
            "b2":self.hidden_layer_bias[0].tolist()[0]
        };
        with open(self.NN_FILE_PATH,'w') as nnFile:
            json.dump(json_neural_network, nnFile)

    def _load(self):
        if not self._use_file:
            return

        with open(self.NN_FILE_PATH) as nnFile:
            nn = json.load(nnFile)
        self.theta1 = [np.array(li) for li in nn['theta1']]
        self.theta2 = [np.array(li) for li in nn['theta2']]
        self.input_layer_bias = [np.array(nn['b1'][0])]
        self.hidden_layer_bias = [np.array(nn['b2'][0])]
    
    def predict(self, test):
        predictions = self.forward_propogate(test)['predictions']
    
        highest_confidence = max(predictions)
        confidence_percent = int(highest_confidence * 100)
        prediction = predictions.index(highest_confidence)
        print("Predicting digit is {0} with {1}% confidence".format(prediction, confidence_percent))
        return { "digit": predictions.index(max(predictions)), "confidence": confidence_percent }

    def train_on_example(self, pixels, digit):
        prediction = self.predict(pixels)
        results = self.forward_propogate(pixels)
        self.back_propogate(pixels, results, digit)
        print("Actual digit is {0}".format(digit))
        return { "prediction": prediction['digit'], "actual": digit }

    def train(self, training_data):
        predictions = []
        for example in training_data:
            prediction = self.train_on_example(example['y0'], example['label'])
            predictions.append(prediction)
        return predictions
