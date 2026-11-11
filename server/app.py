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
import binascii
import importlib
import io
import json
import mimetypes
import random

import numpy as np
from flask import Flask, jsonify, request
from PIL import Image, ImageOps, UnidentifiedImageError

from ocr.box_classifier import load_classifier
from ocr.evaluate_real_pages import pages as real_pages
from ocr.evaluate_real_pages import read_transcription
from ocr.evaluate_real_pages import score_page as score_real_page
from ocr.page_reader import page_text, read_page
from ocr.photos import (abkai_to_manchu, load_photo, manchu_to_abkai,
                        manchu_to_roman, spell_in_manchu)
from ocr.preprocess import prepare
from ocr.train import (CONTEXT_LENGTH, END, NUM_CLASSES, TOKEN_DIM, alphabet,
                       character_accuracy, ctc_decode, image_to_array,
                       predict, token_count)
from utils.softmax import softmax

directory = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(directory, "config.json")
# The largest image POST /upload/single takes, in bytes. A request may be a
# third bigger than this, for a base64 image in a JSON body.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
# Uploads longer than this on either side are shrunk to it before
# preprocessing, to save the dyno's memory. Words are read at WIDTH (32)
# pixels wide, so this costs nothing the models see.
MAX_UPLOAD_SIDE = 2000
# Uploads with more pixels than this are refused before they're decoded
MAX_UPLOAD_PIXELS = 50_000_000

class RequestError(Exception):
    '''
    A bad request, with the message and HTTP status to send back.
    '''
    def __init__(self, message, status=400):
        super().__init__(message)
        self.message = message
        self.status = status

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
    (None for the blank) and its probability. label is in the photo
    dataset's romanization (see ocr.photos.MANCHU_TO_ROMAN), so score is
    computed as in the run reports; with no label, score is None.
    '''
    entries = [(image_to_array(image), None, token_count(image))]
    _, y = predict(model, entries)
    length = entries[0][2]
    probabilities = softmax(y[0][:length])
    indices = np.argmax(probabilities, axis=-1)
    guess = ctc_decode(y[0], length, model.blank)
    return {
        "name": entry["name"],
        "model": entry["model"],
        "prediction-manchu": guess,
        "prediction-abkai": manchu_to_abkai(guess),
        "ctc": [None if index == model.blank else alphabet[index]
                for index in indices],
        "probs": [float(probabilities[token, index])
                  for token, index in enumerate(indices)],
        "score": (None if label is None
                  else float(character_accuracy(manchu_to_roman(guess),
                                                label))),
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

def read_upload():
    '''
    Returns (image bytes, label) from a POST /upload/single request: either
    multipart/form-data with the file in an "image" field, or a JSON body
    {"image": a base64 data: URL or bare base64}. Either may also have a
    "label", the word in Abkai (any of ' ’ ‘ ʻ for its marks). label is
    None if it's left out or blank.
    '''
    if "image" in request.files:
        data = request.files["image"].read()
        label = request.form.get("label")
    else:
        body = request.get_json(silent=True)
        if not isinstance(body, dict) or not isinstance(body.get("image"),
                                                        str):
            raise RequestError("Send an image, either as a file in an "
                              "\"image\" form field or as base64 in a JSON "
                              "body's \"image\".")
        encoded = body["image"]
        if encoded.startswith("data:"):
            encoded = encoded.partition(",")[2]
        try:
            data = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError):
            raise RequestError("\"image\" isn't valid base64.")
        label = body.get("label")
        if label is not None and not isinstance(label, str):
            raise RequestError("\"label\" must be a string.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise RequestError(f"The image is over {MAX_UPLOAD_BYTES // 2**20} "
                          "MB.", 413)
    label = label.strip() if label else None
    return data, label or None

def open_upload(data):
    '''
    Opens uploaded image bytes as an RGB PIL image: turned upright by its
    EXIF orientation (as phones save photos), transparency made white, and
    shrunk to fit MAX_UPLOAD_SIDE.
    '''
    try:
        image = Image.open(io.BytesIO(data))
        if image.width * image.height > MAX_UPLOAD_PIXELS:
            raise RequestError(f"The image is over {MAX_UPLOAD_PIXELS:,} "
                              "pixels.", 413)
        # JPEGs can decode at a smaller scale straight away
        image.draft("RGB", (MAX_UPLOAD_SIDE, MAX_UPLOAD_SIDE))
        image = ImageOps.exif_transpose(image)
        if image.mode in ("RGBA", "LA", "PA") or (
                image.mode == "P" and "transparency" in image.info):
            image = image.convert("RGBA")
            background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, image)
        image = image.convert("RGB")
    except RequestError:
        raise
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError,
            ValueError):
        raise RequestError("The upload isn't an image that can be read.")
    image.thumbnail((MAX_UPLOAD_SIDE, MAX_UPLOAD_SIDE), Image.LANCZOS)
    return image

def page_response(entry, page, original):
    '''
    The JSON for POST /upload/multiple: the page, straightened as it was
    read, its text in Manchu and Abkai (a line per column with Manchu in
    it), and each box: where it is in the straightened page, the image in
    it, and the model's reading. A box's "line" is its line of the text, or
    None if its column has no Manchu and so no line. original is the image
    as read, in colour.
    '''
    lines = {}
    for word in page.words:
        if word["kind"] == "manchu":
            lines.setdefault(word["column"], None)
    lines = {column: line for line, column in enumerate(sorted(lines))}
    straight = page.straighten(original)
    boxes = []
    for word in page.words:
        left, top, right, bottom = word["box"]
        manchu = word["kind"] == "manchu"
        boxes.append({
            "x": left, "y": top, "w": right - left, "h": bottom - top,
            "image": image_data_url(straight.crop(word["box"])),
            "line": lines.get(word["column"]),
            "position": word["position"],
            "kind": word["kind"],
            "manchu-probability": word["manchu"],
            "prediction-manchu": word["text"] if manchu else None,
            "prediction-abkai": word["abkai"] if manchu else None,
            "confidence": word["confidence"] if manchu else None,
        })
    return {"model": entry["name"],
            "image": image_data_url(straight),
            "angle": page.angle,
            "manchu": page_text(page),
            "abkai": page_text(page, abkai=True),
            "boxes": boxes}

def create_app(config_path=CONFIG_PATH):
    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)
    root = os.path.dirname(os.path.abspath(config_path))
    models = [(entry, load_model(entry, root)) for entry in config["models"]]
    # /upload/multiple reads with one model, by default default_model
    default_model = config.get("default_model", config["models"][-1]["name"])
    # Optional: without it, /upload/multiple gets a 404
    classifier = (load_classifier(os.path.join(root, config["box_classifier"]))
                  if "box_classifier" in config else None)
    # The real pages GET /random?source=page picks from, with their
    # transcriptions: optional in config.json
    pages_folder = (os.path.join(root, config["pages"]) if "pages" in config
                    else None)
    page_sources = {}
    if pages_folder and os.path.exists(os.path.join(pages_folder,
                                                    "sources.json")):
        with open(os.path.join(pages_folder, "sources.json"), "r",
                  encoding="utf-8") as file:
            page_sources = json.load(file)
    # The images GET /random picks from, by source: a folder and its
    # manifest.json for each. "synthetic" is optional in config.json.
    sources = {}
    for source, key in (("photo", "photos"), ("synthetic", "synthetic")):
        if key in config:
            folder = os.path.join(root, config[key])
            with open(os.path.join(folder, "manifest.json"), "r",
                      encoding="utf-8") as file:
                sources[source] = (folder, json.load(file))

    app = Flask(__name__)
    # Room for a MAX_UPLOAD_BYTES image as base64, plus the rest of the body
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES * 4 // 3 + 2**16
    # Send Manchu as itself rather than as \u escapes
    app.json.ensure_ascii = False

    @app.after_request
    def allow_origin(response):
        response.headers["Access-Control-Allow-Origin"] = \
            config["allowed_origin"]
        # A JSON upload from another origin is preflighted, and the browser
        # only sends it if Content-Type is allowed
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response

    @app.errorhandler(RequestError)
    def upload_error(error):
        return jsonify({"error": error.message}), error.status

    @app.errorhandler(413)
    def too_large(error):
        return jsonify({"error": "The request is too large."}), 413

    def read_all(original, label):
        # Every model's reading of a PIL image, each with its preprocessing
        results = []
        for entry, model in models:
            preprocess = entry.get("preprocess", {})
            image = prepare(original, preprocess.get("crop", True),
                            preprocess.get("whiten", True))
            results.append(read_word(entry, model, image, label))
        return results

    @app.get("/awake")
    def awake():
        # Lets the frontend know the dyno is up before it asks for more
        return "", 200

    def chosen_model():
        # The (entry, model) ?model= names, default_model by default
        name = request.args.get("model") or default_model
        chosen = [(entry, model) for entry, model in models
                  if entry["name"] == name]
        if not chosen:
            raise RequestError(f"Unknown model {name!r}: use one of "
                               f"{', '.join(e['name'] for e, _ in models)}.")
        return chosen[0]

    def read_original(entry, model, original):
        # read_page with the entry's preprocessing
        preprocess = entry.get("preprocess", {})
        return read_page(original, model, classifier,
                         preprocess.get("crop", True),
                         preprocess.get("whiten", True))

    def random_page():
        '''
        /random?source=page: a real page with its transcription, read as
        /upload/multiple reads an upload (see page_response). A random one,
        or the one ?name= names. Adds the page's
        name, the original image, the transcription, how the reading scores
        against it, and the page's credit and link.
        '''
        entry, model = chosen_model()
        available = real_pages(pages_folder)
        # ?name= asks for one page by name (e.g. page003)
        name = request.args.get("name")
        if name:
            available = [page for page in available if page[0] == name]
            if not available:
                raise RequestError(f"Unknown page {name!r}: use one of "
                                   f"{', '.join(n for n, _, _ in real_pages(pages_folder))}.")
        name, image_path, transcription_path = random.choice(available)
        transcription = read_transcription(transcription_path)
        with Image.open(image_path) as image:
            original = image.convert("RGB")
        original.thumbnail((MAX_UPLOAD_SIDE, MAX_UPLOAD_SIDE), Image.LANCZOS)
        page = read_original(entry, model, original)
        result = score_real_page(page, transcription)
        response = page_response(entry, page, original)
        source = page_sources.get(name, {})
        response.update({
            "source": "page", "name": name,
            "original-image": data_url(image_path),
            "transcription": "\n".join(transcription),
            "score": {"characters": result["characters"],
                      "words": result["words"]},
            "credit": source.get("credit"), "link": source.get("link"),
        })
        return response

    @app.get("/pages")
    def list_pages():
        # The names of the real pages /random?source=page&name= takes, in
        # order; none if the server doesn't read pages
        if not pages_folder or classifier is None:
            return jsonify({"pages": []})
        return jsonify({"pages": [name for name, _, _
                                  in real_pages(pages_folder)]})

    @app.get("/random")
    def random_word():
        # ?source=photo (the default) for a photographed word,
        # ?source=synthetic for a rendered one, or ?source=page for a real
        # page (see random_page)
        source = request.args.get("source") or "photo"
        if source == "page" and pages_folder and classifier is not None:
            return jsonify(random_page())
        if source not in sources:
            available = list(sources) + (
                ["page"] if pages_folder and classifier is not None else [])
            raise RequestError(f"Unknown source {source!r}: use one of "
                               f"{', '.join(available)}.")
        folder, entries = sources[source]
        photo = random.choice(entries)
        path = os.path.join(folder, photo["path"])
        # The manifest's label is in the dataset's romanization, for scoring;
        # people are shown Abkai
        with Image.open(path) as original:
            results = read_all(original, photo["label"])
        # The photo as models with the default preprocessing see it
        preprocessed = load_photo(path)
        return jsonify({"source": source, "font": photo.get("font"),
                        "label": manchu_to_abkai(photo["manchu"]),
                        "manchu": photo["manchu"],
                        "image": data_url(path),
                        "preprocessed-image": image_data_url(preprocessed),
                        "results": results})

    @app.post("/upload/single")
    def upload_single():
        # Like /random, but for an uploaded image (see read_upload)
        data, label = read_upload()
        roman = manchu = None
        if label is not None:
            try:
                # Through the dataset's romanization, to use the lexicon's
                # spelling where it has the word
                roman = manchu_to_roman(abkai_to_manchu(label))
            except ValueError:
                raise RequestError(f"{label!r} isn't a word in Abkai.")
            manchu = spell_in_manchu(roman)
            label = manchu_to_abkai(manchu)
        original = open_upload(data)
        return jsonify({"source": "upload", "font": None,
                        "label": label, "manchu": manchu,
                        "image": image_data_url(original),
                        "preprocessed-image": image_data_url(
                            prepare(original)),
                        "results": read_all(original, roman)})

    @app.post("/upload/multiple")
    def upload_multiple():
        '''
        Reads a page of words: an image sent as for /upload/single, read
        with the model named by ?model= (default_model in config.json by
        default). See page_response.
        '''
        if classifier is None:
            raise RequestError("This server doesn't read pages.", 404)
        entry, model = chosen_model()
        data, _ = read_upload()
        original = open_upload(data)
        page = read_original(entry, model, original)
        return jsonify(page_response(entry, page, original))

    return app
