from flask import Flask, send_file
import io

import corpus
import lexigraphy

app = Flask(__name__)


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

@app.route("/lexigraphy/new/<manchu>")
def create_random_lexigraph(manchu):
    '''
    given:
    - manchu text
    returns:
    - image of that text, generated with a random font
    '''
    lexigraph = lexigraphy.create_lexigraph(manchu)
    image_io = io.BytesIO()
    lexigraph.save(image_io, 'PNG')
    image_io.seek(0)
    return send_file(image_io, mimetype="image/png")

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

