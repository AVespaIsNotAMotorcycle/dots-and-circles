'''
Reads a whole page: ocr.layout finds its words, ocr.box_classifier tells
Manchu from other text, and an OCR model reads each Manchu word, a batch
at a time.

read_page returns the Page, its words in reading order each with its
"kind", and for Manchu words the model's reading and confidence.
page_text gives the page's text, a line per column (see
training_data.pages.page_lines), and draw_page the page with its boxes.
'''

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ocr.box_classifier import label_words
from ocr.layout import find_words
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
    confidence, spaces) for each: the confidence is the mean, over the
    image's tokens, of the probability of the class read there, and spaces
    are where the model put spaces between words, as shares of the
    prepared image's height.
    '''
    readings = []
    for start in range(0, len(crops), BATCH_SIZE):
        images = [prepare(image, crop, whiten)
                  for image in crops[start:start + BATCH_SIZE]]
        entries = [(image_to_array(image), None, token_count(image))
                   for image in images]
        _, y = predict(model, entries)
        for image, output, (_, _, length) in zip(images, y, entries):
            probabilities = softmax(output[:length])
            text, spaces = decode_words(output, length, model.blank)
            # Images taller than HEIGHT are squeezed into it
            rows = min(image.height, HEIGHT)
            readings.append((text, float(probabilities.max(axis=-1).mean()),
                             [((token + 0.5) * ROWS_PER_TOKEN) / rows
                              for token in spaces]))
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

def read_page(image, model, classifier, crop=True, whiten=True):
    '''
    Reads a page, given as a PIL image, with a CTC model and a box
    classifier. crop and whiten are the model's preprocessing. Returns the
    Page (see ocr.layout), whose words each have "box", "column",
    "position", "kind" and "manchu" (the classifier's probability that it's
    Manchu), and for Manchu words "text" (the reading in Manchu),
    "abkai" and "confidence". A box the model reads as several words is
    split into one box per word.
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
            words.append(dict(word, box=box, text=part,
                              abkai=manchu_to_abkai(part),
                              confidence=confidence))
    position = {}
    for word in words:
        word["position"] = position.get(word["column"], 0)
        position[word["column"]] = word["position"] + 1
    page.words = words
    return page

def page_text(page, abkai=False):
    # The page's text, as page_lines gives it, from read_page's words
    return page_lines(page.words, abkai)

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
