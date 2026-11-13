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
import math
import mimetypes
import random
import tempfile
from urllib.parse import urlsplit

import numpy as np
from flask import Flask, jsonify, request
from PIL import Image, ImageOps, UnidentifiedImageError

from ocr.box_classifier import load_classifier
from ocr.dictionary import Dictionary
from ocr.layout import straighten
from server.visitors import VisitorCounter
from ocr.evaluate_real_pages import pages as real_pages
from ocr.evaluate_real_pages import read_transcription
from ocr.evaluate_real_pages import score_page as score_real_page
from ocr.page_reader import (in_text, low_confidence, page_text, read_boxes,
                             read_page)
from ocr.photos import (abkai_to_manchu, load_photo, manchu_to_abkai,
                        manchu_to_norman, manchu_to_roman, spell_in_manchu)
from ocr.preprocess import prepare
from ocr.training_data.pages import OTHER
from ocr.train import (CONTEXT_LENGTH, TOKEN_DIM, alphabet,
                       character_accuracy, classes_in, ctc_decode,
                       image_to_array, predict, token_count)
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
# POST /upload/multiple/boxes takes back the straightened page
# /upload/multiple returned, which turning can make up to √2 times longer
# than MAX_UPLOAD_SIDE. It isn't shrunk, as that would move the boxes.
MAX_STRAIGHT_SIDE = math.ceil(MAX_UPLOAD_SIDE * math.sqrt(2))
# The most boxes POST /upload/multiple/boxes reads at once, and the longest
# reading it takes back for one
MAX_BOXES = 2000
MAX_READING_LENGTH = 500
# Dictionary entries given for each Manchu box of a page
CLOSEST_WORDS = 2
# Whole pages are sent back as JPEGs of this quality, to show: a tenth of
# the PNG's size. They aren't for reading again, which JPEG changes (see
# POST /upload/multiple/boxes, which takes the upload and angle instead).
PAGE_JPEG_QUALITY = 85
# The largest angle POST /upload/multiple/boxes turns a page by
MAX_ANGLE = 45
# Words per page of GET /lexigraphs, by default and at most
DEFAULT_PER_PAGE = 50
MAX_PER_PAGE = 100

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
    weights = os.path.join(root, entry["weights"])
    # Older models were trained before punctuation joined the alphabet, so
    # they have fewer classes, with the blank last (see classes_in)
    classes = classes_in(weights)
    model = getattr(module, name)(token_dim=TOKEN_DIM,
                                  context_length=CONTEXT_LENGTH,
                                  num_classes=classes, blank=classes - 1,
                                  **entry.get("params", {}))
    if not getattr(model, "uses_ctc", False):
        raise ValueError(f"{entry['name']}: only CTC models can be served, "
                         "since the response includes the CTC string")
    model.load_weights(weights)
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
        "prediction-norman": manchu_to_norman(guess),
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

def jpeg_data_url(image, quality=PAGE_JPEG_QUALITY):
    # A PIL image as a JPEG data: URL, for whole pages
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"

def open_page_file(path):
    # A real page from its file, as /random?source=page reads it
    with Image.open(path) as image:
        original = image.convert("RGB")
    original.thumbnail((MAX_UPLOAD_SIDE, MAX_UPLOAD_SIDE), Image.LANCZOS)
    return original

def crops_wanted():
    # ?crops=: whether page boxes carry their own image (true by default)
    value = (request.args.get("crops") or "true").lower()
    if value not in ("true", "1", "yes", "false", "0", "no"):
        raise RequestError("crops must be true or false.")
    return value in ("true", "1", "yes")

def read_angle(body):
    # The "angle" of a JSON body or form, or None if it's left out or null
    if "angle" in request.form:
        try:
            angle = float(request.form["angle"])
        except ValueError:
            raise RequestError("\"angle\" must be a number.")
    else:
        angle = body.get("angle") if isinstance(body, dict) else None
        if angle is None:
            return None
        if isinstance(angle, bool) or not isinstance(angle, (int, float)):
            raise RequestError("\"angle\" must be a number.")
    if not math.isfinite(angle) or abs(angle) > MAX_ANGLE:
        raise RequestError(f"\"angle\" must be from -{MAX_ANGLE} to "
                          f"{MAX_ANGLE} degrees.")
    return float(angle)

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

def open_upload(data, shrink=True):
    '''
    Opens uploaded image bytes as an RGB PIL image: turned upright by its
    EXIF orientation (as phones save photos), transparency made white, and
    shrunk to fit MAX_UPLOAD_SIDE. With shrink=False it's kept at its size,
    and refused if it's longer than MAX_STRAIGHT_SIDE.
    '''
    try:
        image = Image.open(io.BytesIO(data))
        if image.width * image.height > MAX_UPLOAD_PIXELS:
            raise RequestError(f"The image is over {MAX_UPLOAD_PIXELS:,} "
                              "pixels.", 413)
        if not shrink and max(image.size) > MAX_STRAIGHT_SIDE:
            raise RequestError(f"The image is over {MAX_STRAIGHT_SIDE} "
                              "pixels on a side.", 413)
        if shrink:
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
    if shrink:
        image.thumbnail((MAX_UPLOAD_SIDE, MAX_UPLOAD_SIDE), Image.LANCZOS)
    return image

def read_boxes_field(size):
    '''
    The boxes of a POST /upload/multiple/boxes request, for read_boxes: a
    JSON body's "boxes", or a form field "boxes" holding them as JSON. Each
    is {"x", "y", "w", "h"} in pixels of the image, of the given (width,
    height), an optional "kind", "manchu" or "other" (null or left out
    for the classifier to decide), and optionally the "prediction-manchu"
    and "confidence" an earlier response gave it, to keep rather than read
    it again (both or neither; null counts as left out). Other keys are
    ignored, so a box can be sent back as it came. Boxes reaching past the
    image are cut to it.
    '''
    if "boxes" in request.form:
        try:
            boxes = json.loads(request.form["boxes"])
        except ValueError:
            raise RequestError("\"boxes\" isn't valid JSON.")
    else:
        body = request.get_json(silent=True)
        boxes = body.get("boxes") if isinstance(body, dict) else None
    if not isinstance(boxes, list):
        raise RequestError("Send \"boxes\", a list of {\"x\", \"y\", "
                          "\"w\", \"h\"} and optionally \"kind\".")
    if len(boxes) > MAX_BOXES:
        raise RequestError(f"Send at most {MAX_BOXES} boxes.")
    width, height = size
    result = []
    for number, box in enumerate(boxes):
        if not isinstance(box, dict):
            raise RequestError(f"Box {number} isn't an object.")
        values = [box.get(key) for key in ("x", "y", "w", "h")]
        if not all(isinstance(value, (int, float))
                   and not isinstance(value, bool) and math.isfinite(value)
                   for value in values):
            raise RequestError(f"Box {number} needs numbers \"x\", "
                              "\"y\", \"w\" and \"h\".")
        x, y, w, h = values
        if w <= 0 or h <= 0:
            raise RequestError(f"Box {number} must have a positive width "
                              "and height.")
        left, top = max(0, round(x)), max(0, round(y))
        right, bottom = min(width, round(x + w)), min(height, round(y + h))
        if right <= left or bottom <= top:
            raise RequestError(f"Box {number} is outside the image.")
        kind = box.get("kind")
        if kind not in (None, "manchu", "other"):
            raise RequestError(f"Box {number}'s \"kind\" must be "
                              "\"manchu\", \"other\" or null.")
        text, confidence = box.get("prediction-manchu"), box.get("confidence")
        reading = None
        if text is not None or confidence is not None:
            if not (isinstance(text, str)
                    and len(text) <= MAX_READING_LENGTH
                    and isinstance(confidence, (int, float))
                    and not isinstance(confidence, bool)
                    and 0 <= confidence <= 1):
                raise RequestError(
                    f"Box {number} needs both \"prediction-manchu\" (a "
                    f"string of at most {MAX_READING_LENGTH} characters) and "
                    "\"confidence\" (from 0 to 1), or neither.")
            reading = (text, confidence)
        result.append({"box": [left, top, right, bottom], "kind": kind,
                       "reading": reading})
    return result

def page_response(entry, page, original, dictionary=None, crops=True):
    '''
    The JSON for POST /upload/multiple: the page, straightened as it was
    read, its text in Manchu and Abkai (a line per column with Manchu in
    it), and each box: where it is in the straightened page, the image in
    it, and the model's reading. A box's "line" is its line of the text, or
    None if its column has no Manchu (read with enough confidence) and so
    no line. Manchu read with low confidence is left out of the text, and
    its box marked "low-confidence". original is the image as read, in
    colour. With a Dictionary, each Manchu box has its "closest-words"
    (None without one, and for other boxes). The page is a JPEG, and
    without crops, boxes' "image" is None.
    '''
    lines = {}
    for word in page.words:
        if word["kind"] == "manchu" and in_text(word):
            lines.setdefault(word["column"], None)
    lines = {column: line for line, column in enumerate(sorted(lines))}
    straight = page.straighten(original)
    boxes = []
    for word in page.words:
        left, top, right, bottom = word["box"]
        manchu = word["kind"] == "manchu"
        boxes.append({
            "x": left, "y": top, "w": right - left, "h": bottom - top,
            "image": (image_data_url(straight.crop(word["box"])) if crops
                      else None),
            "line": lines.get(word["column"]),
            "position": word["position"],
            "kind": word["kind"],
            "manchu-probability": word["manchu"],
            "prediction-manchu": word["text"] if manchu else None,
            "prediction-abkai": word["abkai"] if manchu else None,
            "prediction-norman": manchu_to_norman(word["text"])
                                 if manchu else None,
            "confidence": word["confidence"] if manchu else None,
            "low-confidence": low_confidence(word) if manchu else None,
            "closest-words": closest_words(dictionary, word["text"])
                             if manchu and dictionary is not None else None,
        })
    return {"model": entry["name"],
            "image": jpeg_data_url(straight),
            "angle": page.angle,
            "manchu": page_text(page),
            "abkai": page_text(page, abkai=True),
            "norman": norman_text(page_text(page)),
            "boxes": boxes}

def closest_words(dictionary, text):
    # The CLOSEST_WORDS dictionary entries closest to a reading, for the
    # frontend to show beside it
    return [{"manchu": match["manchu"], "abkai": match["abkai"],
             "norman": manchu_to_norman(match["manchu"]),
             "distance": match["distance"], "sources": match["sources"]}
            for match in dictionary.closest(text, CLOSEST_WORDS)]

def norman_text(text):
    # A page's text (page_text) in Norman's romanization, OTHER kept
    return "\n".join(" ".join(word if word == OTHER
                              else manchu_to_norman(word)
                              for word in line.split(" "))
                     for line in text.split("\n"))

def is_localhost(origin):
    # Whether a request's Origin is a page served from this machine, on any
    # port: localhost, 127.0.0.1 or ::1
    try:
        parts = urlsplit(origin)
    except ValueError:
        return False
    return (parts.scheme in ("http", "https")
            and parts.hostname in ("localhost", "127.0.0.1", "::1"))

def cors_origin(origin, allowed, local):
    '''
    The Access-Control-Allow-Origin to answer a request from origin with,
    or None to send none (so a browser keeps the response from the page):
    "*" if allowed is "*", origin if it's allowed, and when the server runs
    locally (local), any localhost origin too.
    '''
    if allowed == "*":
        return "*"
    if origin and (origin == allowed or (local and is_localhost(origin))):
        return origin
    return None

def create_app(config_path=CONFIG_PATH, visitors_folder=None):
    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)
    root = os.path.dirname(os.path.abspath(config_path))
    models = [(entry, load_model(entry, root)) for entry in config["models"]]
    # /upload/multiple reads with one model, by default default_model
    default_model = config.get("default_model", config["models"][-1]["name"])
    # Optional: without it, /upload/multiple gets a 404
    classifier = (load_classifier(os.path.join(root, config["box_classifier"]))
                  if "box_classifier" in config else None)
    # The entries each page box's closest words come from: optional
    dictionary = (Dictionary.load(os.path.join(root, config["dictionary"]))
                  if "dictionary" in config else None)
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
    # Heroku sets DYNO on its dynos. Anywhere else the server is running
    # locally, and answers pages on localhost too (see cors_origin).
    local = "DYNO" not in os.environ
    # Room for a MAX_UPLOAD_BYTES image as base64, plus the rest of the body
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES * 4 // 3 + 2**16
    # Send Manchu as itself rather than as \u escapes
    app.json.ensure_ascii = False

    @app.after_request
    def allow_origin(response):
        allowed = cors_origin(request.headers.get("Origin"),
                              config["allowed_origin"], local)
        if allowed:
            response.headers["Access-Control-Allow-Origin"] = allowed
        if allowed != "*":
            # The answer depends on the Origin, so caches must keep them apart
            response.vary.add("Origin")
        # A JSON upload from another origin is preflighted, and the browser
        # only sends it if Content-Type is allowed
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response

    # Requests from the frontend are counted (see server/visitors.py); none
    # are while any origin is allowed, as in testing
    counter = (VisitorCounter(visitors_folder or tempfile.gettempdir())
               if config["allowed_origin"] != "*" else None)
    reading_views = {"random_word", "lexigraphs", "upload_single",
                     "upload_multiple", "upload_multiple_boxes"}

    @app.after_request
    def count_visitor(response):
        if (counter is None or request.method == "OPTIONS"
                or request.headers.get("Origin") != config["allowed_origin"]):
            return response
        # Heroku's router adds the address it saw last; earlier ones can be
        # made up by the client
        forwarded = request.headers.get("X-Forwarded-For", "")
        ip = forwarded.split(",")[-1].strip() or request.remote_addr
        try:
            counter.record(ip, request.headers.get("User-Agent", ""),
                           request.endpoint in reading_views
                           and response.status_code < 400)
        except OSError as error:
            # Counting must never break a request
            print(f"visitor count failed: {error}", flush=True)
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
        # read_page with the entry's preprocessing and stem-break setting
        preprocess = entry.get("preprocess", {})
        return read_page(original, model, classifier,
                         preprocess.get("crop", True),
                         preprocess.get("whiten", True),
                         entry.get("stem_breaks", True))

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
        original = open_page_file(image_path)
        page = read_original(entry, model, original)
        result = score_real_page(page, transcription)
        response = page_response(entry, page, original, dictionary,
                                 crops_wanted())
        source = page_sources.get(name, {})
        response.update({
            "source": "page", "name": name,
            "original-image": jpeg_data_url(original),
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
                        "label-norman": manchu_to_norman(photo["manchu"]),
                        "manchu": photo["manchu"],
                        "image": data_url(path),
                        "preprocessed-image": image_data_url(preprocessed),
                        "results": results})

    # Readings for /lexigraphs, by (model name, source, index in the
    # manifest), so each word is read once per worker
    readings = {}

    @app.get("/lexigraphs")
    def lexigraphs():
        '''
        Every word of a source (?source=photo, the default, or synthetic) in
        manifest order, a page at a time (?page=, from 1, and ?per_page=,
        up to MAX_PER_PAGE), each with one model's reading (?model=,
        default_model by default).
        '''
        entry, model = chosen_model()
        source = request.args.get("source") or "photo"
        if source not in sources:
            raise RequestError(f"Unknown source {source!r}: use one of "
                               f"{', '.join(sources)}.")
        try:
            page = int(request.args.get("page", 1))
            per_page = int(request.args.get("per_page", DEFAULT_PER_PAGE))
        except ValueError:
            raise RequestError("page and per_page must be whole numbers.")
        if page < 1 or not 1 <= per_page <= MAX_PER_PAGE:
            raise RequestError("page must be at least 1, and per_page from "
                               f"1 to {MAX_PER_PAGE}.")
        folder, entries = sources[source]
        start = (page - 1) * per_page
        words = []
        for index in range(start, min(start + per_page, len(entries))):
            photo = entries[index]
            path = os.path.join(folder, photo["path"])
            key = (entry["name"], source, index)
            if key not in readings:
                preprocess = entry.get("preprocess", {})
                with Image.open(path) as original:
                    image = prepare(original, preprocess.get("crop", True),
                                    preprocess.get("whiten", True))
                readings[key] = read_word(entry, model, image,
                                          photo["label"])
            reading = readings[key]
            words.append({
                "index": index,
                "label": manchu_to_abkai(photo["manchu"]),
                "label-norman": manchu_to_norman(photo["manchu"]),
                "manchu": photo["manchu"],
                "font": photo.get("font"),
                "image": data_url(path),
                "prediction-manchu": reading["prediction-manchu"],
                "prediction-abkai": reading["prediction-abkai"],
                "prediction-norman": reading["prediction-norman"],
                "score": reading["score"],
            })
        return jsonify({"source": source, "model": entry["name"],
                        "page": page, "per-page": per_page,
                        "total": len(entries),
                        "pages": -(-len(entries) // per_page),
                        "words": words})

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
                        "label": label,
                        "label-norman": (manchu_to_norman(manchu)
                                         if manchu is not None else None),
                        "manchu": manchu,
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
        return jsonify(page_response(entry, page, original, dictionary,
                                     crops_wanted()))

    @app.post("/upload/multiple/boxes")
    def upload_multiple_boxes():
        '''
        Reads a page in boxes the user gave, with the boxes as edited (see
        read_boxes_field). The page is either the image sent to
        /upload/multiple, sent again with the "angle" that response gave
        (turned here exactly as it was then); a real page by its "name",
        with its "angle"; or, with no angle, a page already straightened.
        Returns the same JSON as /upload/multiple, with the boxes in
        reading order and the angle the page was turned by.
        '''
        if classifier is None:
            raise RequestError("This server doesn't read pages.", 404)
        entry, model = chosen_model()
        body = request.get_json(silent=True)
        name = body.get("name") if isinstance(body, dict) else None
        if name is not None:
            if not pages_folder or not isinstance(name, str):
                raise RequestError("\"name\" must be one of the pages "
                                  "GET /pages lists.")
            paths = [path for page_name, path, _ in real_pages(pages_folder)
                     if page_name == name]
            if not paths:
                raise RequestError(f"Unknown page {name!r}: use one of "
                                   f"{', '.join(n for n, _, _ in real_pages(pages_folder))}.")
            original = open_page_file(paths[0])
            angle = read_angle(body) or 0.0
        else:
            data, _ = read_upload()
            angle = read_angle(body)
            # Sent with its angle, the upload is opened as /upload/multiple
            # opened it; without, it's the straightened page, kept as it is
            original = open_upload(data, shrink=angle is not None)
            angle = angle or 0.0
        straight = straighten(original, angle)
        boxes = read_boxes_field(straight.size)
        preprocess = entry.get("preprocess", {})
        page = read_boxes(straight, boxes, model, classifier,
                          preprocess.get("crop", True),
                          preprocess.get("whiten", True))
        # The response turns original by the page's angle, as for
        # /upload/multiple, so the next request can send the same again
        page.angle = angle
        return jsonify(page_response(entry, page, original, dictionary,
                                     crops_wanted()))

    return app
