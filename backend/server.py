from flask import Flask, send_file, request
from flask_cors import CORS
import io

import numpy as np

import corpus
import lexigraphy
import constants
from ocr import NeuralNetwork
from ctc import CTCNeuralNetwork
from classifier import Classifier

app = Flask(__name__)
CORS(app)
    
classifier = Classifier()

def get_filename(stage, l_class):
    valid_stages = ['primary', 'secondary']
    if stage not in valid_stages:
        message = 'get_filename expects stage to be one of {0}, but it was {1}'.format(valid_stages, stage)
        raise ValueError(message)
    valid_classes = ['A', 'B', 'C', 'D']
    if l_class.upper() not in valid_classes:
        message = 'get_filename expects l_class to be one of {0}, but it was {1}'.format(valid_classes, stage)
        raise ValueError(message)

    if stage == 'primary': return 'ocr_class_{0}.json'.format(l_class).lower()
    if stage == 'secondary': return 'ctc_class_{0}.json'.format(l_class).lower()

num_hidden_nodes = 256
OCR_class_A = NeuralNetwork(num_hidden_nodes)
OCR_class_B = NeuralNetwork(num_hidden_nodes)
OCR_class_C = NeuralNetwork(num_hidden_nodes)
OCR_class_D = NeuralNetwork(num_hidden_nodes)

primary_ocr = { 'A': OCR_class_A, 
                'B': OCR_class_B,
                'C': OCR_class_C,
                'D': OCR_class_D }

CTC_class_A = CTCNeuralNetwork()
CTC_class_B = CTCNeuralNetwork()
CTC_class_C = CTCNeuralNetwork()
CTC_class_D = CTCNeuralNetwork()

secondary_ocr = { 'A': CTC_class_A, 
                  'B': CTC_class_B,
                  'C': CTC_class_C,
                  'D': CTC_class_D }

for l_class in ['A', 'B', 'C', 'D']:
    primary_ocr[l_class].load(get_filename('primary', l_class))
    secondary_ocr[l_class].load(get_filename('secondary', l_class))

def train_on_lexigraph(font, manchu):
    slices, row_labels = lexigraphy.get_slices(font, manchu)
    lexigraph_class = classifier.classify(font)
    predictions = []

    predictions = primary_ocr[lexigraph_class].train_on_lexigraph(slices, row_labels)
    primary_ocr[lexigraph_class].save(get_filename('primary', lexigraph_class))

    return predictions, lexigraph_class

@app.route("/corpus/count")
def get_corpus_count():
    '''
    returns how many unique words are in the corpus
    '''
    return { "word_count": corpus.get_corpus_size() }

@app.route("/corpus/random")
def get_random_word():
    '''
    returns:
    - manchu text
    - romanized text
    '''
    return corpus.get_random_word()

@app.route("/corpus/autocorrect/<manchu>")
def get_autocorrect(manchu):
    return corpus.autocorrect(manchu)

@app.route("/lexigraphy/fonts/dict")
def get_fonts_dict():
    return lexigraphy.get_fonts_dict()

@app.route("/lexigraphy/fonts/<filename>")
def get_font_file(filename):
    return send_file('fonts/' + filename, mimetype="font/ttf")

@app.route("/lexigraphy/new/<font>/<manchu>")
def create_lexigraph(font, manchu):
    '''
    given:
    - manchu text
    - index of a font
    returns:
    - image of that text
    '''
    lexigraph = lexigraphy.create_lexigraph(manchu, int(font))
    image_io = io.BytesIO()
    lexigraph.save(image_io, 'PNG')
    image_io.seek(0)
    return send_file(image_io, mimetype="image/png")

@app.route("/lexigraphy/predict/<font>/<manchu>")
def predict_lexigraph(font, manchu):
    slices, row_labels = lexigraphy.get_slices(font, manchu)

    lexigraph_class = classifier.classify(font)
    primary_predictions = primary_ocr[lexigraph_class].predict_lexigraph(slices)

    secondary_inputs = []
    for index, primary_prediction in enumerate(primary_predictions):
        secondary_input = [0] * 21
        for delta in range(21):
            pred_index = index + delta - 10
            if pred_index < 0: continue
            if pred_index >= len(primary_predictions): continue
            secondary_input[delta] = primary_predictions[pred_index]['character']

        secondary_input = CTC_class_A.digits_array_to_x(secondary_input)
        secondary_inputs.append(secondary_input)

    secondary_predictions = secondary_ocr[lexigraph_class].predict_tokens(secondary_inputs)

    return { "primary_predictions": primary_predictions,
             "secondary_predictions": secondary_predictions,
             "class": lexigraph_class }

@app.route("/lexigraphy/save/<font>/<manchu>", methods=["PUT"])
def save_lexigraph(font, manchu):
    boundaries = request.get_json()["boundaries"]
    success = lexigraphy.save_lexigraph(font, manchu, boundaries)

    predictions, lexigraph_class = train_on_lexigraph(font, manchu)

    return predictions

@app.route("/lexigraphy/get/page")
def get_lexigraph():
    start = int(request.args.get("start"))
    end = int(request.args.get("end"))
    page = lexigraphy.get_lexigraph_page(start, end)
    return page

def train_secondary_ocr(trials):
    inputs = { 'A': [], 'B': [], 'C': [], 'D': [] }
    labels = { 'A': [], 'B': [], 'C': [], 'D': [] }
    for index, trial in enumerate(trials):
        predictions = [0] * 21
        for delta in range(21):
            pred_index = index + delta - 10
            if pred_index < 0: continue
            if pred_index >= len(trials): continue
            predictions[delta] = trials[pred_index]['prediction']

        actual = trial['actual']
        lexigraph_class = trial['class']

        predictions = CTC_class_A.digits_array_to_x(predictions)
        inputs[lexigraph_class].append(predictions)
        labels[lexigraph_class].append(actual)

    successes = 0
    for l_class in ['A', 'B', 'C', 'D']:
        predictions = secondary_ocr[l_class].train_on_tokens(inputs[l_class], labels[l_class])
    
        for index, prediction in enumerate(predictions):
            character = prediction['character']
            if character == labels[l_class][index]: successes += 1

        secondary_ocr[l_class].save(get_filename('secondary', l_class))

    accuracy = successes / len(trials) * 100
    print("CTC Accuracy: {0}%".format(accuracy))

@app.route("/train", methods=["PUT"])
def train_100_times():
    trials = []
    successes = 0
    for i in range(100):
        font, manchu, slices, row_labels = lexigraphy.get_random_marked_lexigraph()
        predictions, lexigraph_class = train_on_lexigraph(font, manchu)
        for index in range(len(row_labels)):
            prediction = predictions[index]["character"]
            answer = constants.ALPHABET.index(row_labels[index])
            if prediction == answer: successes += 1
            trial = {
                    "prediction": prediction,
                    "actual": answer,
                    "correct": prediction == answer,
                    "font": font,
                    "manchu": manchu,
                    "class": lexigraph_class
                    }
            trials.append(trial)
    train_secondary_ocr(trials)
    accuracy = successes / len(trials) * 100
    print("OCR Accuracy: {0}%".format(accuracy))
    return { "accuracy": accuracy, "trials": trials }

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

