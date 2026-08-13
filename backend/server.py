from flask import Flask, send_file, request
from flask_cors import CORS
import io

import random
import numpy as np

import corpus
import lexigraphy
import constants
from primary_ocr import NeuralNetwork
from secondary_ocr import CTCNeuralNetwork
from classifier import Classifier

app = Flask(__name__)
CORS(app)

CLASSIFIER_FILENAME = './saved_nns/classifier.json'
PRIMARY_FILENAME = './saved_nns/primary.json'
SECONDARY_FILENAME = './saved_nns/secondary.json'

num_hidden_nodes = 256

primary_ocr = NeuralNetwork(num_hidden_nodes)
primary_ocr.load(PRIMARY_FILENAME)

secondary_ocr = CTCNeuralNetwork()
secondary_ocr.load(SECONDARY_FILENAME)

classifier = Classifier()
classifier.load(CLASSIFIER_FILENAME)

def train_on_lexigraph(font, manchu):
    slices, row_labels = lexigraphy.get_slices(font, manchu)
    predictions = []

    predictions = primary_ocr.train_on_lexigraph(slices, row_labels)
    primary_ocr.save(PRIMARY_FILENAME)

    return predictions

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

    image_array = lexigraphy.get_lexigraph_array(font, manchu)
    primary_predictions = primary_ocr.predict_lexigraph(slices)

    secondary_inputs = []
    for index, primary_prediction in enumerate(primary_predictions):
        secondary_input = [0] * 21
        for delta in range(21):
            pred_index = index + delta - 10
            if pred_index < 0: continue
            if pred_index >= len(primary_predictions): continue
            secondary_input[delta] = primary_predictions[pred_index]['character']

        secondary_input = secondary_ocr.digits_array_to_x(secondary_input)
        secondary_inputs.append(secondary_input)

    secondary_predictions = secondary_ocr.predict_tokens(secondary_inputs)

    return { "primary_predictions": primary_predictions,
             "secondary_predictions": secondary_predictions }

@app.route("/lexigraphy/save/<font>/<manchu>", methods=["PUT"])
def save_lexigraph(font, manchu):
    boundaries = request.get_json()["boundaries"]
    success = lexigraphy.save_lexigraph(font, manchu, boundaries)

    predictions = train_on_lexigraph(font, manchu)

    return predictions

@app.route("/lexigraphy/get/page")
def get_lexigraph():
    start = int(request.args.get("start"))
    end = int(request.args.get("end"))
    page = lexigraphy.get_lexigraph_page(start, end)
    return page

def train_secondary_ocr(trials):
    inputs = []
    labels = []
    training_data = []
    for index, trial in enumerate(trials):
        primary_output = [0] * 21
        for delta in range(21):
            pred_index = index + delta - 10
            if pred_index < 0: continue
            if pred_index >= len(trials): continue
            primary_output[delta] = trials[pred_index]['prediction']

        actual = trial['actual']

        primary_output = secondary_ocr.digits_array_to_x(primary_output)
        inputs.append(primary_output)
        labels.append(actual)

    successes = 0
    predictions = secondary_ocr.train_on_tokens(inputs, labels)
    
    for index, prediction in enumerate(predictions):
        character = prediction['character']
        if character == labels[index]: successes += 1

    secondary_ocr.save(SECONDARY_FILENAME)
    accuracy = successes / len(trials) * 100
    print("Secondary Accuracy: {0}%".format(int(accuracy)))

def train_primary_ocr():
    trials = []
    successes = 0

    for i in range(100):
        font, manchu, slices, row_labels = lexigraphy.get_random_marked_lexigraph()

        predictions = train_on_lexigraph(font, manchu)
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
                    }
            trials.append(trial)
    accuracy = successes / len(trials) * 100
    print("Primary Accuracy: {0}%".format(int(accuracy)))
    return trials, accuracy

def train_classifier():
    trials = []
    successes = 0
    total_loss = 0

    for i in range(100):
        font = lexigraphy.get_random_font_index()
        manchu = corpus.get_random_word()["manchu"]
        slices, row_labels = lexigraphy.get_slices(font, manchu)

        label = 'A'
        if font in [2, 5]: label = 'B'
        if font in [3, 8]: label = 'C'
        if font == 6: label = 'D'

        prediction, success, loss = classifier.train_on_lexigraph(slices, label)
        successes += int(success)
        total_loss += loss
        trials.append({ "label": label, "prediction": prediction })

    classifier.save(CLASSIFIER_FILENAME)
    accuracy = successes / len(trials) * 100
    print("Classifier Accuracy: {0}%".format(int(accuracy)))
    print("Classifier Average Loss: {0}%".format(total_loss / len(trials)))
    return trials, accuracy

@app.route("/train", methods=["PUT"])
def train_100_times():
    train_classifier()
    trials, accuracy = train_primary_ocr()
    train_secondary_ocr(trials)

    return { "accuracy": accuracy, "trials": trials }

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

