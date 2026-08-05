from flask import Flask, send_file, request
from flask_cors import CORS
import io

import corpus
import lexigraphy
import constants
from ocr import NeuralNetwork
from classifier import Classifier

app = Flask(__name__)
CORS(app)
    
classifier = Classifier()

OCR_class_A = NeuralNetwork(128)
OCR_class_A.load()
OCR_class_B = NeuralNetwork(128)
OCR_class_B.load()
OCR_class_C = NeuralNetwork(128)
OCR_class_C.load()
OCR_class_D = NeuralNetwork(128)
OCR_class_D.load()

def train_on_lexigraph(font, manchu):
    slices, row_labels = lexigraphy.get_slices(font, manchu)
    lexigraph_class = classifier.classify(font)
    predictions = []
    if lexigraph_class == 'A':
        predictions = OCR_class_A.train_on_lexigraph(slices, row_labels)
        OCR_class_A.save()
    if lexigraph_class == 'B':
        predictions = OCR_class_B.train_on_lexigraph(slices, row_labels)
        OCR_class_B.save()
    if lexigraph_class == 'C':
        predictions = OCR_class_C.train_on_lexigraph(slices, row_labels)
        OCR_class_C.save()
    if lexigraph_class == 'D':
        predictions = OCR_class_D.train_on_lexigraph(slices, row_labels)
        OCR_class_D.save()

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
    predictions = []
    if lexigraph_class == 'A': predictions = OCR_class_A.predict_lexigraph(slices)
    if lexigraph_class == 'B': predictions = OCR_class_B.predict_lexigraph(slices)
    if lexigraph_class == 'C': predictions = OCR_class_C.predict_lexigraph(slices)
    if lexigraph_class == 'D': predictions = OCR_class_D.predict_lexigraph(slices)

    return { "predictions": predictions, "class": lexigraph_class }

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
                    }
            trials.append(trial)
    accuracy = successes / len(trials) * 100
    print("Accuracy: {0}%".format(accuracy))
    return { "accuracy": accuracy, "trials": trials }

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

