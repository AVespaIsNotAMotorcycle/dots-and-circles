'''
Tells Manchu boxes from others (Chinese characters, mostly) on a page found
by ocr.layout, with a small MLP. Hand-set rules can't do it: short Manchu
words such as ᠪᡝ, ᡩᡝ and ᡳ are as square as Chinese characters, and no
threshold on stems or ink separates them well.

The MLP reads a box's ink, padded to a square and shrunk to SIZE x SIZE,
along with its shape: log(height / width), and its width and height in the
page's unit. train_box_classifier.py trains it on synthetic pages.

label_words then marks each word of a Page with its kind, and splits boxes
of other text taller than they are wide into square characters, since runs
of Chinese characters are often too close together for ocr.layout to part.
'''

import math

import numpy as np
from PIL import Image

from nn.GELU import GELU
from nn.Linear import Linear
from nn.Sequential import Sequential
from ocr.layout import stretches
from utils.softmax import softmax

SIZE = 24
SHAPE_FEATURES = 3
HIDDEN = 64
KINDS = ["manchu", "other"]
# A box of other text this many times taller than wide is split into
# characters
RUN_ASPECT = 1.4


def make_classifier():
    return Sequential([Linear(SIZE * SIZE + SHAPE_FEATURES, HIDDEN), GELU(),
                       Linear(HIDDEN, len(KINDS))])

def load_classifier(path):
    model = make_classifier()
    model.load_weights(path)
    model.set_training(False)
    return model

def box_features(ink, box, unit):
    '''
    The MLP's input for a box of a page's ink (a boolean array): the ink,
    centred in a square and shrunk to SIZE x SIZE, then the box's shape.
    '''
    left, top, right, bottom = box
    width, height = right - left, bottom - top
    side = max(width, height)
    square = np.zeros((side, side), dtype=np.uint8)
    x, y = (side - width) // 2, (side - height) // 2
    square[y:y + height, x:x + width] = ink[top:bottom, left:right] * 255
    pixels = np.asarray(Image.fromarray(square).resize((SIZE, SIZE),
                                                       Image.BOX),
                        dtype=np.float64) / 255
    shape = [math.log(height / width), width / unit, height / unit]
    return np.concatenate([pixels.ravel(), shape])

def classify(model, ink, boxes, unit):
    # The probability that each box is Manchu
    if not boxes:
        return np.zeros(0)
    features = np.array([box_features(ink, box, unit) for box in boxes])
    return softmax(model(features))[:, KINDS.index("manchu")]

def split_characters(ink, box):
    '''
    Splits a box of other text, taller than it is wide, into roughly square
    characters. The box is cut into pieces at rows without ink; a piece much
    taller than wide (characters touching) is cut at its thinnest rows; then
    the pieces are grouped, in order, so each group is as near a square as
    can be. Returns the characters' boxes, top to bottom.
    '''
    left, top, right, bottom = box
    size = right - left
    profile = ink[top:bottom, left:right].sum(axis=1)
    pieces = []
    for start, end in stretches(profile > 0):
        count = round((end - start) / size)
        if count < 2:
            pieces.append((start, end))
            continue
        # Cut at the thinnest row near each place a character should end
        cuts = [start]
        for index in range(1, count):
            expected = start + index * (end - start) // count
            window = max(1, size // 4)
            low, high = max(cuts[-1] + 1, expected - window), min(
                end - 1, expected + window)
            if low >= high:
                continue
            cuts.append(low + int(np.argmin(profile[low:high])))
        cuts.append(end)
        pieces.extend(zip(cuts, cuts[1:]))
    if not pieces:
        return [box]

    # Grouping with the least total squared distance from square shapes
    count = len(pieces)
    best = [0.0] + [math.inf] * count
    previous = [0] * (count + 1)
    for end in range(1, count + 1):
        for start in range(end):
            height = pieces[end - 1][1] - pieces[start][0]
            cost = best[start] + (height / size - 1) ** 2
            if cost < best[end]:
                best[end], previous[end] = cost, start
    groups = []
    end = count
    while end > 0:
        groups.append((previous[end], end))
        end = previous[end]
    characters = []
    for start, end in reversed(groups):
        first, last = top + pieces[start][0], top + pieces[end - 1][1]
        columns = np.nonzero(ink[first:last, left:right].any(axis=0))[0]
        characters.append([left + int(columns[0]), first,
                           left + int(columns[-1]) + 1, last])
    return characters

def label_words(page, model):
    '''
    Gives each word of a Page (from ocr.layout.find_words) a "kind",
    "manchu" or "other", and a "manchu" probability, splitting tall boxes of
    other text into characters and renumbering positions. Changes page in
    place and returns it.
    '''
    probabilities = classify(model, page.ink,
                             [word["box"] for word in page.words], page.unit)
    words = []
    for word, probability in zip(page.words, probabilities):
        kind = "manchu" if probability >= 0.5 else "other"
        box = word["box"]
        tall = (box[3] - box[1]) > RUN_ASPECT * (box[2] - box[0])
        boxes = (split_characters(page.ink, box)
                 if kind == "other" and tall else [box])
        for part in boxes:
            words.append({"box": [int(value) for value in part],
                          "column": word["column"],
                          "kind": kind, "manchu": float(probability)})
    position = {}
    for word in words:
        word["position"] = position.get(word["column"], 0)
        position[word["column"]] = word["position"] + 1
    page.words = words
    return page
