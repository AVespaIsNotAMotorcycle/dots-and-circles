from flask import Flask

import corpus

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
    - image
    - manchu text
    - font used
    '''
    return corpus.get_random_word()

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

