'''
Flask server for the frontend: reads photographed Manchu words with the
models listed in config.json. Run from the repository root with:
  .venv/bin/python -m flask --app server.app run
On Heroku, the Procfile runs it with gunicorn.
'''

import os
# The dyno has little memory and CPU, so numpy's BLAS shouldn't start threads
# of its own. This only works if it's set before numpy is first imported.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import base64
import importlib
import io
import json
import mimetypes
import random

import numpy as np
from flask import Flask, jsonify

from ocr.photos import load_photo, manchu_to_abkai
from ocr.train import (CONTEXT_LENGTH, END, NUM_CLASSES, TOKEN_DIM, alphabet,
                       character_accuracy, ctc_decode, image_to_array,
                       predict, token_count)
from utils.softmax import softmax

directory = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(directory, "config.json")

def load_model(entry, root):
    '''
    Builds the model a config.json entry describes and loads its weights.
    Paths in the entry are relative to root, the config's folder.
    '''
    name = entry["class"]
    module = importlib.import_module(f"ocr.{name[0].lower()}{name[1:]}")
    model = getattr(module, name)(token_dim=TOKEN_DIM,
                                  context_length=CONTEXT_LENGTH,
                                  num_classes=NUM_CLASSES, blank=END,
                                  **entry.get("params", {}))
    if not getattr(model, "uses_ctc", False):
        raise ValueError(f"{entry['name']}: only CTC models can be served, "
                         "since the response includes the CTC string")
    model.load_weights(os.path.join(root, entry["weights"]))
    model.set_training(False)
    return model

def read_word(entry, model, image, label):
    '''
    Runs model on a prepared photo, returning the response's object for it.
    ctc and probs have one element per real token: the most likely class
    (None for the blank) and its probability.
    '''
    entries = [(image_to_array(image), None, token_count(image))]
    _, y = predict(model, entries)
    length = entries[0][2]
    probabilities = softmax(y[0][:length])
    indices = np.argmax(probabilities, axis=-1)
    guess = ctc_decode(y[0], length, model.blank)
    abkai = manchu_to_abkai(guess)
    return {
        "name": entry["name"],
        "model": entry["model"],
        "prediction-manchu": guess,
        "prediction-abkai": abkai,
        "ctc": [None if index == model.blank else alphabet[index]
                for index in indices],
        "probs": [float(probabilities[token, index])
                  for token, index in enumerate(indices)],
        "score": float(character_accuracy(abkai, label)),
    }

def data_url(path):
    # The file at path as a data: URL, so the frontend needn't fetch it
    mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
    with open(path, "rb") as file:
        encoded = base64.b64encode(file.read()).decode("ascii")
    return f"data:{mime};base64,{encoded}"

def image_data_url(image):
    # A PIL image as a PNG data: URL
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"

def create_app(config_path=CONFIG_PATH):
    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)
    root = os.path.dirname(os.path.abspath(config_path))
    models = [(entry, load_model(entry, root)) for entry in config["models"]]
    photo_directory = os.path.join(root, config["photos"])
    with open(os.path.join(photo_directory, "manifest.json"), "r",
              encoding="utf-8") as file:
        photos = json.load(file)

    app = Flask(__name__)
    # Send Manchu as itself rather than as \u escapes
    app.json.ensure_ascii = False

    @app.after_request
    def allow_origin(response):
        response.headers["Access-Control-Allow-Origin"] = \
            config["allowed_origin"]
        return response

    @app.get("/awake")
    def awake():
        # Lets the frontend know the dyno is up before it asks for more
        return "", 200

    @app.get("/random")
    def random_word():
        photo = random.choice(photos)
        path = os.path.join(photo_directory, photo["path"])
        results = []
        for entry, model in models:
            preprocess = entry.get("preprocess", {})
            image = load_photo(path, preprocess.get("crop", True),
                               preprocess.get("whiten", True))
            results.append(read_word(entry, model, image, photo["label"]))
        # The photo as models with the default preprocessing see it
        preprocessed = load_photo(path)
        return jsonify({"label": photo["label"], "manchu": photo["manchu"],
                        "image": data_url(path),
                        "preprocessed-image": image_data_url(preprocessed),
                        "results": results})

    return app
