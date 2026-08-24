from PIL import Image
import numpy as np

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
        return y

    def train(self, batch):
        for image, label in batch:
            print(label)
            self.predict(image)
        return 0, 0

    def save(self):
        return
