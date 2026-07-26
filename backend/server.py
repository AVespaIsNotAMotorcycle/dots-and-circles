from flask import Flask

app = Flask(__name__)


@app.route("/corpus/count")
def get_corpus_count():
    '''
    returns how many unique words are in the corpus
    '''
    return { "word_count": 0 }

@app.route("/corpus/random")
def get_random_word():
    '''
    returns:
    - image
    - manchu text
    - font used
    '''

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

