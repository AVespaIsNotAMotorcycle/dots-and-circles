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

l_classes = ["A", "B", "C", "D"]
primary_ocr = {}
secondary_ocr = {}

def classify_font(font):
    if font in [2, 5]: return 'B'
    if font in [3, 8]: return 'C'
    if font == 6: return 'D'
    return 'A'

def get_filename(rank, l_class):
    valid_ranks = ["primary", "secondary"]
    assert rank in valid_ranks, \
        f"get_filename expected rank to be one of {valid_ranks}, was {rank} instead"

    assert l_class in l_classes, \
        f"get_filename expected l_class to be one of {l_classes}, was {l_class} instead"

    return "./saved_nns/{0}_{1}.json".format(rank, l_class)

for l_class in l_classes:
    primary_ocr[l_class] = NeuralNetwork(num_hidden_nodes)
    primary_ocr[l_class].load(get_filename("primary", l_class))

    secondary_ocr[l_class] = CTCNeuralNetwork()
    secondary_ocr[l_class].load(get_filename("secondary", l_class))

classifier = Classifier()
classifier.load(CLASSIFIER_FILENAME)

def load_all():
    for l_class in l_classes:
        primary_ocr[l_class].load(get_filename("primary", l_class))

        secondary_ocr[l_class].load(get_filename("secondary", l_class))

def train_on_lexigraph(font, manchu):
    slices, row_labels = lexigraphy.get_slices(font, manchu)
    predictions = []
    l_class = classify_font(font)

    predictions = primary_ocr[l_class].train_on_lexigraph(slices, row_labels)

    return predictions

@app.route("/loadall"):
    try:
        loadall()
        return "Loaded saved weights."
    except Exception:
        return Exception

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
    lexigraph = lexigraphy.create_lexigraph(manchu, int(font), crop=True)
    image_io = io.BytesIO()
    lexigraph.save(image_io, 'PNG')
    image_io.seek(0)
    return send_file(image_io, mimetype="image/png")

@app.route("/lexigraphy/predict/<font>/<manchu>")
def predict_lexigraph(font, manchu):
    slices, row_labels = lexigraphy.get_slices(font, manchu)

    l_class = classifier.predict_lexigraph(slices)
    image_array = lexigraphy.get_lexigraph_array(font, manchu)
    primary_predictions = primary_ocr[l_class].predict_lexigraph(slices)

    secondary_inputs = []
    for index, primary_prediction in enumerate(primary_predictions):
        secondary_input = [0] * 21
        for delta in range(21):
            pred_index = index + delta - 10
            if pred_index < 0: continue
            if pred_index >= len(primary_predictions): continue
            secondary_input[delta] = primary_predictions[pred_index]['character']

        secondary_input = secondary_ocr[l_class].digits_array_to_x(secondary_input)
        secondary_inputs.append(secondary_input)

    secondary_predictions = secondary_ocr[l_class].predict_tokens(secondary_inputs)

    return { "primary_predictions": primary_predictions,
             "secondary_predictions": secondary_predictions,
             "l_class": l_class }

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

        l_class = trial['l_class']
        primary_output = secondary_ocr[l_class].digits_array_to_x(primary_output)
        inputs.append(primary_output)
        labels.append(actual)

    successes = 0
    predictions = secondary_ocr[l_class].train_on_tokens(inputs, labels)
    
    for index, prediction in enumerate(predictions):
        character = prediction['character']
        if character == labels[index]: successes += 1

    for l_class in l_classes:
        secondary_ocr[l_class].save(get_filename('secondary', l_class))
    accuracy = successes / len(trials) * 100
    print("Secondary Accuracy: {0}%".format(int(accuracy)))

def train_primary_ocr():
    trials = []
    successes = 0

    for i in range(100):
        font, manchu, slices, row_labels = lexigraphy.get_random_marked_lexigraph()

        l_class = classify_font(font)
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
                    "l_class": l_class
                    }
            trials.append(trial)
    for l_class in l_classes:
        primary_ocr[l_class].save(get_filename('primary', l_class))
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
    print("Classifier Average Loss: {0}".format(total_loss / len(trials)))
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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
