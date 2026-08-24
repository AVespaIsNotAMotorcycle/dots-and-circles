from PIL import Image
import numpy as np
from Levenshtein import ratio

from utils import to_abkai, to_norman
from lexigraphy import preprocess_image
from constants import ALPHABET
from ann import ANN

'''
A simple ANN which views the whole image at once.
'''
class Model2():
    def __init__(self):
        self.width = 50
        self.height = 350
        self.size_in = self.width * self.height
        self.size_out = len(ALPHABET) * 35

        self.ann = ANN(self.size_in, self.size_out, num_layers=3)

    def name(self):
        return 'Model 2: Simple ANN'

    def resize_image(self, image):
        width, height = image.size
        factor = self.width / width
        image = image.resize((self.width, int(height * factor)), Image.BICUBIC)
        background = Image.new('RGB', (self.width, self.height), (255, 255, 255, 255))
        background.paste(image)
        return background

    def predict(self, image, output="manchu"):
        image = self.resize_image(image)
        image = preprocess_image(image)

        x = np.array(image).reshape((self.size_in, 1))
        y = self.ann.forward(x)
        y = y.reshape(35, (len(ALPHABET)))
        return y, x

    def softmax(self, y):
        for index, row in enumerate(y):
            y[index] = np.exp(row) / sum(np.exp(row))
        return y

    def cross_entropy_loss(self, probs, labels):
        loss = 0
        for index, row in enumerate(probs):
            loss += -np.log(row[labels[index]])
        return loss

    def label_string_to_array(self, label):
        labels = []
        for char in label:
            index = ALPHABET.index(char)
            labels.append(index)
        while len(labels) < 35:
            labels.append(ALPHABET.index(' '))
        return labels

    def train(self, batch):
        successes = 0
        loss = 0
        for image, label in batch:
            y, x = self.predict(image)
            probs = self.softmax(y)

            predictions = np.argmax(y, axis=1)
            word = ''
            for char in predictions:
                word += ALPHABET[char]
            successes += ratio(word, label)

            labels = self.label_string_to_array(label)
            loss += self.cross_entropy_loss(probs, labels)

            gradient = np.zeros(self.size_out)
            for i, j in enumerate(labels):
                gradient[(i * len(ALPHABET)) + j] = -1 / probs[i][j]
            gradient = gradient.reshape((self.size_out, 1))
            self.ann.backprop(x, y.reshape((self.size_out, 1)), gradient, 0.001)
        return successes / len(batch), loss

    def save(self):
        return
