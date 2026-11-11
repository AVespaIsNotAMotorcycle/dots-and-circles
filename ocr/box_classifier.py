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
from ocr.layout import longest_stroke, stretches
from utils.softmax import softmax

SIZE = 24
SHAPE_FEATURES = 3
HIDDEN = 64
KINDS = ["manchu", "other"]
# A box of other text this many times taller than wide is split into
# characters
RUN_ASPECT = 1.4
# A column at least this share (by height) other text is all other text,
# and its boxes less than RUN_GAP units apart are joined before splitting
# into characters
# Boxes the classifier is unsure of (a Manchu probability between UNSURE
# and 1 - UNSURE) are split into left and right parts parted by a blank band
# at least SIDE_GAP units wide, each at least SIDE_HEIGHT units tall
UNSURE = 0.05
SIDE_GAP = 0.08
SIDE_HEIGHT = 0.4
OTHER_COLUMN = 0.5
RUN_GAP = 0.5
# A column at least SOME_OTHER other text, with no stroke as long as
# LONG_STROKE units, is all other text too
SOME_OTHER = 0.1
LONG_STROKE = 1.5


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

def other_runs(words, unit):
    '''
    The boxes of a column of other text, its words' boxes joined into runs
    wherever they're less than RUN_GAP units apart.
    '''
    runs = []
    for word in sorted(words, key=lambda word: word["box"][1]):
        box = list(word["box"])
        if runs and box[1] - runs[-1][3] < RUN_GAP * unit:
            last = runs[-1]
            runs[-1] = [min(last[0], box[0]), last[1], max(last[2], box[2]),
                        max(last[3], box[3])]
        else:
            runs.append(box)
    return runs

def side_parts(ink, box, unit):
    '''
    A box's ink as a left and a right part, if a blank band at least
    SIDE_GAP units wide parts them and both are at least SIDE_HEIGHT units
    tall; otherwise None. A Manchu word never falls apart like this (the
    marks beside its stem are small, its tails attached), but a word with a
    gloss set close beside it does.
    '''
    left, top, right, bottom = box
    region = ink[top:bottom, left:right]
    inked = region.any(axis=0)
    gaps = [(start, end) for start, end in stretches(~inked)
            if start > 0 and end < len(inked)
            and end - start >= SIDE_GAP * unit]
    if not gaps:
        return None
    start, end = max(gaps, key=lambda gap: gap[1] - gap[0])
    parts = []
    for first, last in ((0, start), (end, len(inked))):
        rows = np.nonzero(region[:, first:last].any(axis=1))[0]
        columns = np.nonzero(region[:, first:last].any(axis=0))[0]
        if len(rows) == 0 or rows[-1] - rows[0] + 1 < SIDE_HEIGHT * unit:
            return None
        parts.append([left + first + int(columns[0]), top + int(rows[0]),
                      left + first + int(columns[-1]) + 1,
                      top + int(rows[-1]) + 1])
    return parts

def separate_side_by_side(page, model):
    '''
    The page's words, with each the classifier is unsure of (a Manchu
    probability between UNSURE and 1 - UNSURE) split into its left and right
    parts when side_parts finds them, each classified again. A part of
    other text split from a Manchu one is a gloss set beside it, and goes
    in a column of its own, just after its Manchu one's, so it's left out
    of the Manchu line. Columns are renumbered.
    '''
    words = []
    for word in page.words:
        parts = (side_parts(page.ink, word["box"], page.unit)
                 if UNSURE < word["manchu"] < 1 - UNSURE else None)
        if not parts:
            words.append(dict(word, column=float(word["column"])))
            continue
        probabilities = classify(model, page.ink, parts, page.unit)
        kinds = ["manchu" if probability >= 0.5 else "other"
                 for probability in probabilities]
        for part, probability, kind in zip(parts, probabilities, kinds):
            column = float(word["column"])
            if kind == "other" and "manchu" in kinds:
                column += 0.5
            words.append(dict(word, box=part, manchu=float(probability),
                              kind=kind, column=column))
    order = {column: index for index, column in
             enumerate(sorted({word["column"] for word in words}))}
    for word in words:
        word["column"] = order[word["column"]]
    return words

def label_words(page, model):
    '''
    Gives each word of a Page (from ocr.layout.find_words) a "kind",
    "manchu" or "other", and a "manchu" probability, splitting tall boxes of
    other text into characters and renumbering positions. Changes page in
    place and returns it.

    A box the classifier is unsure of may hold a Manchu word with a gloss
    set close beside it; see separate_side_by_side.

    A column mostly (by height) of other text, or of more than one box,
    partly other text and with no stroke as long as LONG_STROKE units (no
    Manchu stem), is taken to be all other text, such as small Chinese glosses between Manchu columns:
    those are often cut at blank rows inside their characters, and the
    pieces can look like Manchu. Its boxes are joined into runs and split
    into characters again.
    '''
    probabilities = classify(model, page.ink,
                             [word["box"] for word in page.words], page.unit)
    for word, probability in zip(page.words, probabilities):
        word["manchu"] = float(probability)
        word["kind"] = "manchu" if probability >= 0.5 else "other"
    page.words = separate_side_by_side(page, model)
    columns = {}
    for word in page.words:
        columns.setdefault(word["column"], []).append(word)

    words = []
    for column in sorted(columns):
        column_words = columns[column]
        height = lambda word: word["box"][3] - word["box"][1]
        other = sum(height(word) for word in column_words
                    if word["kind"] == "other")
        extent = [min(word["box"][0] for word in column_words),
                  min(word["box"][1] for word in column_words),
                  max(word["box"][2] for word in column_words),
                  max(word["box"][3] for word in column_words)]
        total = sum(map(height, column_words))
        stemless = (len(column_words) > 1 and other >= SOME_OTHER * total
                    and longest_stroke(page.ink, extent, page.unit)
                    < LONG_STROKE * page.unit)
        if stemless or other >= OTHER_COLUMN * total:
            probability = min(word["manchu"] for word in column_words)
            for run in other_runs(column_words, page.unit):
                for part in split_characters(page.ink, run):
                    words.append({"box": part, "column": column,
                                  "kind": "other", "manchu": probability})
            continue
        for word in column_words:
            box = word["box"]
            tall = (box[3] - box[1]) > RUN_ASPECT * (box[2] - box[0])
            boxes = (split_characters(page.ink, box)
                     if word["kind"] == "other" and tall else [box])
            for part in boxes:
                words.append({"box": part, "column": column,
                              "kind": word["kind"],
                              "manchu": word["manchu"]})
    position = {}
    for word in words:
        word["box"] = [int(value) for value in word["box"]]
        word["position"] = position.get(word["column"], 0)
        position[word["column"]] = word["position"] + 1
    page.words = words
    return page
