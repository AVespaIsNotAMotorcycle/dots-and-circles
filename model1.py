from PIL import Image
import numpy as np

from utils import to_abkai, to_norman
from lexigraphy import image_to_array
from constants import ALPHABET

from primary_ocr import NeuralNetwork
from secondary_ocr import CTCNeuralNetwork
from classifier import Classifier

class Model1():
    def __init__(self):
        CLASSIFIER_FILENAME = './saved_nns/classifier.json'
        PRIMARY_FILENAME = './saved_nns/primary.json'
        SECONDARY_FILENAME = './saved_nns/secondary.json'
        
        num_hidden_nodes = 256
        
        self.l_classes = ["A", "B", "C", "D"]
        self.primary_ocr = {}
        self.secondary_ocr = {}

        for l_class in self.l_classes:
            self.primary_ocr[l_class] = NeuralNetwork(num_hidden_nodes)
            self.primary_ocr[l_class].load(self.get_filename("primary", l_class))
        
            self.secondary_ocr[l_class] = CTCNeuralNetwork()
            self.secondary_ocr[l_class].load(self.get_filename("secondary", l_class))
        
        self.classifier = Classifier()
        self.classifier.load(CLASSIFIER_FILENAME)

    def get_filename(self, rank, l_class):
        valid_ranks = ["primary", "secondary"]
        assert rank in valid_ranks, \
            f"get_filename expected rank to be one of {valid_ranks}, was {rank} instead"
    
        assert l_class in self.l_classes, \
            f"get_filename expected l_class to be one of {self.l_classes}, was {l_class} instead"
    
        return "./saved_nns/{0}_{1}.json".format(rank, l_class)


    def name(self):
        return 'Model 1: Basic CTC'

    def normalize_image(self, image):
        desired_width = 50
        width, height = image.size
        factor = desired_width / width

        normalized = image.resize((desired_width, int(height * factor)), Image.BICUBIC)
        width, height = normalized.size
        if width != desired_width:
            raise ValueError(f"Width should be {desired_width}, but was {width}.")

        return normalized

    def slice_image(self, image):
        slices = []
        width, height = image.size

        padding = 10
        start_padding = np.zeros((padding, width))
        array = image_to_array(image)
        array = np.concat((start_padding, array))

        number_of_rows = (padding * 2) + 1
        for i in range(height):
            start = i + 10
            end = start + number_of_rows
            slice = array[start:end]
            slices.append(slice.reshape((number_of_rows * width, 1)))

        return slices

    def construct_secondary_inputs(self, primary_predictions):
        secondary_inputs = []
        for index, primary_prediction in enumerate(primary_predictions):
            secondary_input = [0] * 21
            for delta in range(21):
                pred_index = index + delta - 10
                if pred_index < 0: continue
                if pred_index >= len(primary_predictions): continue
                secondary_input[delta] = primary_predictions[pred_index]['character']
            secondary_input = self.secondary_ocr['A'].digits_array_to_x(secondary_input)
            secondary_inputs.append(secondary_input)
        return secondary_inputs

    def parse_ctc(self, results):
        characters = [ALPHABET[row['character']] for row in results]
        reducedCharacters = []

        for index, char in enumerate(characters):
            if index == 0 or index == len(characters) - 1: continue
            if len(reducedCharacters) > 0 and reducedCharacters[-1] == char: continue
            if characters[index + 1] != char or characters[index - 1] != char: continue
            reducedCharacters.append(char)

        while '*' in reducedCharacters: reducedCharacters.remove('*')
        string = ''.join(reducedCharacters)
        return string

    def predict(self, image, output="manchu"):
        valid_output_types = ['manchu', 'norman', 'abkai']
        if output not in valid_output_types:
            raise ValueError(
                f"Model1.predict expects 'output' to be one of {valid_output_types}, "
                f"but it was {output}.")

        image = self.normalize_image(image)
        slices = self.slice_image(image)
        l_class = self.classifier.predict_lexigraph(slices)

        primary_predictions = self.primary_ocr[l_class].predict_lexigraph(slices)
        secondary_inputs = self.construct_secondary_inputs(primary_predictions)
        secondary_predictions = self.secondary_ocr[l_class].predict_tokens(secondary_inputs)

        manchu = self.parse_ctc(secondary_predictions)
        if output == 'abkai': return to_abkai(manchu)
        if output == 'norman': return to_norman(manchu)
        return manchu
