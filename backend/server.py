from flask import Flask, send_file, request
from flask_cors import CORS
import io

import corpus
import lexigraphy
from ocr import NeuralNetwork, ALPHABET

app = Flask(__name__)
CORS(app)
    
nn = NeuralNetwork(128)
nn._load()

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

@app.route("/lexigraphy/save/<font>/<manchu>", methods=["PUT"])
def save_lexigraph(font, manchu):
    boundaries = request.get_json()["boundaries"]
    success = lexigraphy.save_lexigraph(font, manchu, boundaries)

    slices, row_labels = lexigraphy.get_slices(font, manchu)
    predictions = nn.train_on_lexigraph(slices, row_labels)

    nn.save()
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
        predictions = nn.train_on_lexigraph(slices, row_labels)
        for index in range(len(row_labels)):
            prediction = predictions[index]["character"]
            answer = ALPHABET.index(row_labels[index])
            if prediction == answer: successes += 1
            trial = { "prediction": prediction, "actual": answer, "correct": prediction == answer }
            trials.append(trial)
    accuracy = successes / len(trials) * 100
    print("Accuracy: {0}%".format(accuracy))
    return { "accuracy": accuracy, "trials": trials }

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

