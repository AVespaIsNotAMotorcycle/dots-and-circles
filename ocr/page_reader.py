'''
Reads a whole page: ocr.layout finds its words, ocr.box_classifier tells
Manchu from other text, and an OCR model reads each Manchu word, a batch
at a time.

read_page returns the Page, its words in reading order each with its
"kind", and for Manchu words the model's reading and confidence.
read_boxes does the same for boxes given by hand, on a page already
straightened, keeping them as they are. page_text gives the page's text,
a line per column (see training_data.pages.page_lines), and draw_page the
page with its boxes.
'''

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ocr.box_classifier import classify, label_words
from ocr.layout import find_words, stem_breaks
from ocr.photos import manchu_to_abkai
from ocr.preprocess import prepare
from ocr.train import (ROWS_PER_TOKEN, alphabet, image_to_array, predict,
                       token_count)
from ocr.training_data import HEIGHT
from ocr.training_data.pages import page_lines
from utils.softmax import softmax

# Words read at once. The models keep every layer's activations, so memory
# grows with this: Run 30 needs about 12 MB a word. Batches of 4 read as
# fast as bigger ones, in a quarter of the memory of 16, which matters with
# two workers on a 512 MB dyno.
BATCH_SIZE = 4
# Manchu read with a confidence (see read_crops) under this is left out of
# the page's text. On the real pages, every correct reading scored at
# least 0.7 and most junk at most 0.57.
MIN_CONFIDENCE = 0.65
# How many times split_at_stem_breaks may split one box in turn
MAX_SPLITS = 6
# read_boxes: a box is in a column if at least this share of the narrower
# of the two's width overlaps
COLUMN_OVERLAP = 0.5
# Drawing: outline colours, and the white laid over Manchu boxes (its
# opacity, 0-255) so their Abkai shows over the print
MANCHU_COLOUR = (0, 170, 0)
OTHER_COLOUR = (220, 0, 0)
FILL_OPACITY = 170
TEXT_COLOUR = (20, 20, 120)
MIN_TEXT_SIZE = 8
# Pillow's built-in font has no ʻ (U+02BB), so it's drawn as the
# near-identical ‘ on the image. The text keeps ʻ.
DRAWN_AS = {"ʻ": "‘"}


def decode_words(output, length, blank):
    '''
    Reads a CTC model's output for one image, as ctc_decode does, and finds
    where it puts spaces. Returns the reading and, for each run of spaces,
    the token in its middle.
    '''
    classes = np.argmax(output[:length], axis=-1)
    text = ""
    spaces = []
    start = 0
    # Each run of one class is one letter (or nothing, for the blank)
    for token in range(1, length + 1):
        if token < length and classes[token] == classes[start]:
            continue
        index = classes[start]
        if index != blank:
            text += alphabet[index]
            if alphabet[index] == " ":
                spaces.append((start + token - 1) / 2)
        start = token
    return text, spaces

def read_crops(model, crops, crop=True, whiten=True):
    '''
    Reads greyscale PIL word images with a CTC model. Returns (reading,
    confidence, spaces) for each. The confidence is the mean, over the
    tokens where the model reads a letter, of that letter's probability (0
    if it reads none): the blanks between letters are left out, as the
    model is nearly always sure of them, even when what it reads is junk.
    spaces are where the model put spaces between words, as shares of the
    prepared image's height.
    '''
    images = [prepare(image, crop, whiten) for image in crops]
    lengths = [token_count(image) for image in images]
    # Read in order of length, each batch cut to its longest image's tokens:
    # the models attend to no padding token, so the readings are the same,
    # but short words (most of them) cost far less
    order = sorted(range(len(images)), key=lambda index: lengths[index])
    readings = [None] * len(images)
    for start in range(0, len(order), BATCH_SIZE):
        batch = order[start:start + BATCH_SIZE]
        longest = max(lengths[index] for index in batch)
        entries = [(image_to_array(images[index])[:longest], None,
                    lengths[index]) for index in batch]
        _, y = predict(model, entries)
        for index, output, (_, _, length) in zip(batch, y, entries):
            image = images[index]
            probabilities = softmax(output[:length])
            text, spaces = decode_words(output, length, model.blank)
            # Images taller than HEIGHT are squeezed into it
            rows = min(image.height, HEIGHT)
            letters = probabilities.argmax(axis=-1) != model.blank
            confidence = (float(probabilities.max(axis=-1)[letters].mean())
                          if letters.any() else 0.0)
            readings[index] = (text, confidence,
                               [((token + 0.5) * ROWS_PER_TOKEN) / rows
                                for token in spaces])
    return readings

def split_reading(ink, box, text, spaces):
    '''
    Splits a box the model read as several words where it put the spaces,
    as (box, word) pairs: the model knows where words end better than
    gaps do, since words in print are often closer than a blank row.
    '''
    words = text.split(" ")
    if len(words) != len(spaces) + 1:
        # Spaces at the ends, or doubled, don't split anything
        words = [word for word in words if word]
        return [(box, " ".join(words))] if words else [(box, "")]
    left, top, right, bottom = box
    edges = [top] + [top + round(share * (bottom - top))
                     for share in spaces] + [bottom]
    parts = []
    for word, first, last in zip(words, edges, edges[1:]):
        region = ink[first:last, left:right]
        rows = np.nonzero(region.any(axis=1))[0]
        columns = np.nonzero(region.any(axis=0))[0]
        if not len(rows) or not word:
            continue
        parts.append(([left + int(columns[0]), first + int(rows[0]),
                       left + int(columns[-1]) + 1, first + int(rows[-1]) + 1],
                      word))
    return parts or [(box, text.strip())]

def tight_box(ink, left, right, first, last):
    # The box of the ink in columns left to right of rows first to last,
    # or None if there's none
    left, right, first, last = int(left), int(right), int(first), int(last)
    region = ink[first:last, left:right]
    rows = np.nonzero(region.any(axis=1))[0]
    columns = np.nonzero(region.any(axis=0))[0]
    if not len(rows):
        return None
    return [left + int(columns[0]), first + int(rows[0]),
            left + int(columns[-1]) + 1, first + int(rows[-1]) + 1]

def split_at_stem_breaks(model, page, box, reading, crop=True, whiten=True,
                         depth=0, cache=None):
    '''
    Splits a box of Manchu, read as reading (text, confidence), into words
    where it holds several set so tight that no row is blank between them.
    Each candidate break (ocr.layout.stem_breaks) is tried by reading the
    two parts; the break whose parts' lower confidence is highest is taken
    if it's at least the whole's, and the parts are split again in turn.
    A word cut inside reads as two pieces of nonsense, with less
    confidence; two words read as words. Returns [(box, (text,
    confidence))]. cache holds readings by box, as the halves of a split
    are tried again inside each other.
    '''
    cache = {} if cache is None else cache
    text, confidence = reading
    if depth >= MAX_SPLITS:
        return [(box, reading)]
    left, top, right, bottom = box
    pairs = []
    for cut in stem_breaks(page.ink, box, page.unit):
        above = tight_box(page.ink, left, right, top, top + cut)
        below = tight_box(page.ink, left, right, top + cut, bottom)
        if above and below:
            pairs.append((above, below))
    if not pairs:
        return [(box, reading)]
    unread = list({tuple(part) for pair in pairs for part in pair
                   if tuple(part) not in cache})
    for part, part_reading in zip(unread, read_crops(
            model, [page.image.crop(part) for part in unread], crop, whiten)):
        cache[part] = part_reading
    best = None
    for above, below in pairs:
        first, second = cache[tuple(above)], cache[tuple(below)]
        if not first[0] or not second[0]:
            continue
        score = min(first[1], second[1])
        if score >= confidence and (best is None or score > best[0]):
            best = (score, above, first, below, second)
    if best is None:
        return [(box, reading)]
    _, above, first, below, second = best
    return (split_at_stem_breaks(model, page, above, first[:2], crop, whiten,
                                 depth + 1, cache)
            + split_at_stem_breaks(model, page, below, second[:2], crop,
                                   whiten, depth + 1, cache))

def read_page(image, model, classifier, crop=True, whiten=True,
              stem_breaks=True):
    '''
    Reads a page, given as a PIL image, with a CTC model and a box
    classifier. crop and whiten are the model's preprocessing. Returns the
    Page (see ocr.layout), whose words each have "box", "column",
    "position", "kind" and "manchu" (the classifier's probability that it's
    Manchu), and for Manchu words "text" (the reading in Manchu),
    "abkai" and "confidence". A box the model reads as several words is
    split into one box per word, and so is one holding several words set
    too tight to part by gaps (see split_at_stem_breaks), unless
    stem_breaks is false. Models trained on tight phrases (Run 31 on)
    write those spaces themselves, and the stem breaks only cut lone
    letters off them (b i for bi).
    '''
    page = label_words(find_words(image), classifier)
    manchu = [word for word in page.words if word["kind"] == "manchu"]
    crops = [page.image.crop(word["box"]) for word in manchu]
    readings = dict(zip(map(id, manchu), read_crops(model, crops, crop,
                                                    whiten)))
    words = []
    for word in page.words:
        if word["kind"] != "manchu":
            words.append(word)
            continue
        text, confidence, spaces = readings[id(word)]
        for box, part in split_reading(page.ink, word["box"], text, spaces):
            # A single word as read may still be several set tight
            pieces = ([(box, (part, confidence))]
                      if not stem_breaks or not part or " " in part
                      else split_at_stem_breaks(model, page, box,
                                                (part, confidence), crop,
                                                whiten))
            for piece, (piece_text, piece_confidence) in pieces:
                words.append(dict(word, box=piece, text=piece_text,
                                  abkai=manchu_to_abkai(piece_text),
                                  confidence=piece_confidence))
    position = {}
    for word in words:
        word["position"] = position.get(word["column"], 0)
        position[word["column"]] = word["position"] + 1
    page.words = words
    return page

def group_columns(boxes):
    '''
    The column (from 0 at the left) and position (from 0 at the top) of
    each box: boxes overlapping by at least COLUMN_OVERLAP of the narrower
    one's width share a column. Boxes are [left, top, right, bottom].
    '''
    columns = []
    column_of = [None] * len(boxes)
    order = sorted(range(len(boxes)),
                   key=lambda index: boxes[index][0] + boxes[index][2])
    for index in order:
        left, _, right, _ = boxes[index]
        best, best_overlap = None, 0.0
        for number, (start, end) in enumerate(columns):
            overlap = min(right, end) - max(left, start)
            share = overlap / max(1, min(right - left, end - start))
            if share >= COLUMN_OVERLAP and share > best_overlap:
                best, best_overlap = number, share
        if best is None:
            columns.append([left, right])
            best = len(columns) - 1
        else:
            columns[best] = [min(left, columns[best][0]),
                             max(right, columns[best][1])]
        column_of[index] = best
    renumber = {old: new for new, old in enumerate(sorted(
        range(len(columns)), key=lambda number: sum(columns[number])))}
    places = [None] * len(boxes)
    for column in range(len(columns)):
        members = sorted((index for index in range(len(boxes))
                          if renumber[column_of[index]] == column),
                         key=lambda index: (boxes[index][1], boxes[index][0]))
        for position, index in enumerate(members):
            places[index] = (column, position)
    return places

def read_boxes(image, boxes, model, classifier, crop=True, whiten=True):
    '''
    Reads a page, given as a PIL image already straight (as a Page's
    straighten gives it), in boxes given by hand rather than found: each a
    dict with "box" ([left, top, right, bottom] in the image), "kind"
    ("manchu", "other" or None for the classifier to decide) and
    optionally "reading", (text, confidence) from an earlier reading of
    the same box. A Manchu box with a reading isn't read again: the boxes
    read_page splits at the model's spaces read differently alone, so this
    keeps the boxes a user hasn't changed as they were (and saves reading
    them). Returns a Page as read_page does, its words in reading order
    (columns found by group_columns). The boxes are kept as given: a
    Manchu box the model reads as several words keeps its spaces, and none
    is split at stem breaks. Words given a kind and read here, rather than
    given a reading, have "kind_given" true, and so stay in the text however
    unsure the model is (see in_text).
    '''
    page = find_words(image, deskew=False)
    if not page.words and boxes:
        # Nothing found to measure the unit by: a word is about a unit wide
        page.unit = float(np.median([box["box"][2] - box["box"][0]
                                     for box in boxes]))
    probabilities = (classify(classifier, page.ink,
                              [box["box"] for box in boxes], page.unit)
                     if boxes else [])
    places = group_columns([box["box"] for box in boxes])
    words = []
    for box, probability, (column, position) in zip(boxes, probabilities,
                                                    places):
        kind = box.get("kind") or ("manchu" if probability >= 0.5
                                   else "other")
        word = {"box": [int(value) for value in box["box"]],
                "column": column, "position": position,
                "manchu": float(probability), "kind": kind}
        reading = box.get("reading")
        if kind == "manchu" and reading is not None:
            text, confidence = reading
            word.update(text=text, abkai=manchu_to_abkai(text),
                        confidence=float(confidence), kind_given=False)
        else:
            word["kind_given"] = bool(box.get("kind"))
        words.append(word)
    manchu = [word for word in words
              if word["kind"] == "manchu" and "text" not in word]
    for word, (text, confidence, _) in zip(manchu, read_crops(
            model, [page.image.crop(word["box"]) for word in manchu], crop,
            whiten)):
        word.update(text=text.strip(), abkai=manchu_to_abkai(text.strip()),
                    confidence=confidence)
    page.words = sorted(words, key=lambda word: (word["column"],
                                                 word["position"]))
    return page

def low_confidence(word):
    # Whether a word of read_page's is Manchu read with a confidence under
    # MIN_CONFIDENCE, too likely junk to put in the text
    return (word["kind"] == "manchu"
            and word.get("confidence", 1.0) < MIN_CONFIDENCE)

def in_text(word):
    # Whether a word goes in the page's text: all but Manchu read with low
    # confidence, unless it was marked Manchu by hand (see read_boxes)
    return not low_confidence(word) or word.get("kind_given", False)

def page_text(page, abkai=False):
    '''
    The page's text, as page_lines gives it, from read_page's words,
    leaving out Manchu read with low confidence unless it was marked Manchu
    by hand (a column of nothing else, such as a page's edge read as
    Manchu, leaves no line).
    '''
    return page_lines([word for word in page.words if in_text(word)], abkai)

def draw_text(image, text, box):
    '''
    Writes text in box on an RGB image, turned to read top to bottom as the
    print does, as large as fits.
    '''
    left, top, right, bottom = box
    width, height = right - left, bottom - top
    text = "".join(DRAWN_AS.get(letter, letter) for letter in text)
    size = max(MIN_TEXT_SIZE, round(0.7 * width))
    font = ImageFont.load_default(size=size)
    length = font.getlength(text)
    if length > 0.95 * height:
        size = max(MIN_TEXT_SIZE, int(size * 0.95 * height / length))
        font = ImageFont.load_default(size=size)
    text_left, text_top, text_right, text_bottom = font.getbbox(text)
    mask = Image.new("L", (text_right - text_left, text_bottom - text_top), 0)
    ImageDraw.Draw(mask).text((-text_left, -text_top), text, font=font,
                              fill=255)
    mask = mask.rotate(-90, expand=True)
    position = (left + (width - mask.width) // 2,
                top + (height - mask.height) // 2)
    image.paste(TEXT_COLOUR, position + (position[0] + mask.width,
                                         position[1] + mask.height), mask)

def draw_page(page):
    '''
    The deskewed page in colour with its words outlined: Manchu in green,
    filled with a translucent white and its reading in Abkai, and other
    text in red, left as it is.
    '''
    image = page.image.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for word in page.words:
        if word["kind"] == "manchu":
            left, top, right, bottom = word["box"]
            draw.rectangle((left, top, right - 1, bottom - 1),
                           fill=(255, 255, 255, FILL_OPACITY))
    image = Image.alpha_composite(image, overlay).convert("RGB")
    draw = ImageDraw.Draw(image)
    line = max(2, round(page.unit / 30))
    for word in page.words:
        left, top, right, bottom = word["box"]
        manchu = word["kind"] == "manchu"
        if manchu and word.get("abkai"):
            draw_text(image, word["abkai"], word["box"])
        # Outlined just outside the box, so the ink stays visible
        draw.rectangle((left - line, top - line, right - 1 + line,
                        bottom - 1 + line),
                       outline=MANCHU_COLOUR if manchu else OTHER_COLOUR,
                       width=line)
    return image
