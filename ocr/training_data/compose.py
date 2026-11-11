'''
Training items made of more than one lexicon entry, for what pages hold
and single words don't teach:

- Phrases of two or three words, set from far apart to tight (each word as
  close to the one before as its ink allows, its tail beside the next
  word's start). On a page the layout often can't part such words, so the
  model has to write the space between them itself.
- The genitive i, printed apart from its word as a hook: after words
  ending in a consonant in phrases, and alone, as the page reader finds it
  when it's boxed on its own.
- Punctuation: the Manchu comma and full stop, after a word, or between
  two words of a phrase.

Each item is a Composite, made by make_composites from the training
words only, so the train/test split by word still holds. Like a lexicon
seed, it's always rendered the same way, and generate_composite augments
it like generate does a word.
'''

import functools
import random
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageChops, ImageFont

from ocr.training_data import (HEIGHT, RENDER_SIZE, SUFFIX_SPACE, WIDTH,
                               fonts, lexicon, pad_top)
from ocr.training_data.pages import (GENITIVE, NIRUGU,
                                     ends_in_consonant, ink_columns,
                                     render_manchu, tightest_gap)

COMMA = "᠈"
FULL_STOP = "᠉"
# The parts of a Composite that aren't lexicon entries: the genitive i as
# the hook print uses (drawn after U+202F) or in its isolated form, with a
# crossbar, and the punctuation marks
HOOK = "hook"
ISOLATED = "isolated"
MARKS = (COMMA, FULL_STOP)

# Each kind's share of the items made, against the number of training
# seeds of single lexicon entries
PHRASE_SHARE = 0.15
LONE_I_SHARE = 0.03
PUNCTUATED_SHARE = 0.10
# Phrases: how many words, and how often each length
PHRASE_LENGTHS = [2, 3]
PHRASE_WEIGHTS = [0.6, 0.4]
# A phrase's words have at most this many letters between them, to fit
# the model's input
MAX_PHRASE_LETTERS = 18
# Chance a consonant-final word in a phrase is followed by the genitive i,
# and that a phrase has a punctuation mark (after one of its words)
GENITIVE_CHANCE = 0.2
PHRASE_PUNCTUATION_CHANCE = 0.1
# Chance a lone i is the hook, rather than the isolated form
HOOK_CHANCE = 0.75
# Chance a punctuation mark is a comma, rather than a full stop
COMMA_CHANCE = 0.7
# How the parts of an item are spaced, in the font's unit (the width of
# its words' ink), and how often: "tight" as in tightly set pages (see
# pages.py), "close" a little apart, "normal" as words are usually set
SPACINGS = ["tight", "close", "normal"]
SPACING_WEIGHTS = [0.4, 0.3, 0.3]
CLOSE_GAP = (0.0, 0.25)
NORMAL_GAP = (0.25, 0.8)
# Tight setting, as in pages.py: the least clearance between two words'
# ink, random slack added to it, and the most two words can overlap
TIGHT_CLEARANCE = 0.06
TIGHT_SLACK = 0.06
TIGHT_LIMIT = 0.6
# The gap before a punctuation mark
MARK_GAP = (0.02, 0.3)
# Trailing parts are dropped until an item, scaled to WIDTH, is at most
# this share of HEIGHT tall. Augmenting can stretch it up to a fifth
# taller, and the model's input then squashes it to fit, as it does a tall
# box on a page.
MAX_HEIGHT_SHARE = 1.0
# Seeds for augmenting composites start here, well past any lexicon seed
SEED_BASE = 10 ** 8


@dataclass(frozen=True)
class Composite:
    '''
    One training item: its parts in reading order (lexicon indices, HOOK,
    ISOLATED, COMMA or FULL_STOP), the index of its font, how its parts
    are spaced (one of SPACINGS) and its number, which seeds its gaps and
    augmentation.
    '''
    parts: tuple
    font: int
    spacing: str
    number: int

    @property
    def seed(self):
        return SEED_BASE + self.number


def part_text(part):
    # What a part contributes to the label, and the text to draw for it
    if isinstance(part, int):
        return lexicon[part], lexicon[part]
    if part == HOOK:
        return GENITIVE, SUFFIX_SPACE + GENITIVE
    if part == ISOLATED:
        return GENITIVE, GENITIVE
    return part, part

def label(parts):
    '''
    The label for parts: words separated by spaces, with punctuation
    straight after the word before it, as Manchu text writes it.
    '''
    text = ""
    for part in parts:
        written, _ = part_text(part)
        if text and part not in MARKS:
            text += " "
        text += written
    return text

@functools.lru_cache(maxsize=None)
def load_font(index):
    return ImageFont.truetype(fonts[index], RENDER_SIZE)

@functools.lru_cache(maxsize=None)
def font_unit(index):
    # The width of a word's ink in this font, as pages.py measures it
    columns = ink_columns(render_manchu("ᠠᠪᡴᠠ ᠨᠠ", load_font(index)))
    return max(1, columns[1] - columns[0])

def gap(rng, spacing, above, below, unit, mark):
    # The gap in pixels between two rendered parts (negative overlaps)
    if mark:
        return round(unit * rng.uniform(*MARK_GAP))
    if spacing == "tight":
        return (tightest_gap(above, below,
                             max(1, round(TIGHT_CLEARANCE * unit)),
                             round(TIGHT_LIMIT * unit))
                + rng.randint(0, round(TIGHT_SLACK * unit)))
    return round(unit * rng.uniform(*(CLOSE_GAP if spacing == "close"
                                      else NORMAL_GAP)))

def stack(images, gaps):
    '''
    The images, one under the next with the gaps between them, their ink
    laid over each other where they overlap, cropped to the ink.
    '''
    tops = [0]
    for image, space in zip(images, gaps):
        tops.append(tops[-1] + image.height + space)
    top = min(tops)
    height = max(t + image.height for t, image in zip(tops, images)) - top
    width = max(image.width for image in images)
    canvas = Image.new("L", (width, height), 255)
    for t, image in zip(tops, images):
        layer = Image.new("L", (width, height), 255)
        layer.paste(image, (0, t - top))
        canvas = ImageChops.darker(canvas, layer)
    # Inverted, the ink is the non-zero part getbbox finds
    box = ImageChops.invert(canvas).getbbox()
    return canvas.crop(box) if box else canvas

def fits(image):
    return image.height * WIDTH / image.width <= MAX_HEIGHT_SHARE * HEIGHT

def render_composite(composite):
    '''
    (canvas, label) for a Composite: its parts rendered in its font and
    stacked with its spacing, cropped to the ink and reading top to bottom,
    as training_data.render gives a word. Parts are dropped from the end
    until it fits the model's input (marks and a genitive go with their
    word), so the label is of the parts kept.
    '''
    font = load_font(composite.font)
    unit = font_unit(composite.font)
    rng = random.Random(composite.seed)
    parts = list(composite.parts)
    images = [render_manchu(part_text(part)[1], font) for part in parts]
    gaps = [gap(rng, composite.spacing, above, below, unit, part in MARKS)
            for above, below, part in zip(images, images[1:], parts[1:])]
    while True:
        canvas = stack(images, gaps)
        if fits(canvas) or len(parts) == 1:
            return canvas, label(parts)
        # Drop the last word, along with anything after it
        last = max(index for index, part in enumerate(parts)
                   if isinstance(part, int) or part == ISOLATED)
        last = last if last > 0 else len(parts) - 1
        parts, images, gaps = parts[:last], images[:last], gaps[:last - 1]

def generate_composite(composite, epoch=None):
    '''
    (image, label) for a Composite, as training_data.generate gives them
    for a seed: clean with epoch=None, otherwise augmented for that epoch.
    '''
    canvas, text = render_composite(composite)
    if epoch is not None:
        # Imported here, as in training_data.generate
        from ocr.training_data.augment import augment
        return augment(canvas, text, composite.seed, epoch), text
    height = max(1, round(canvas.height * WIDTH / canvas.width))
    image = canvas.resize((WIDTH, height), Image.LANCZOS)
    return pad_top(image, random.Random(composite.seed)), text

def single_words(indices):
    # The lexicon entries among indices that are single words, not
    # phrases or dictionary forms of suffixes
    return [index for index in indices
            if " " not in lexicon[index] and NIRUGU not in lexicon[index]]

def letters(index):
    return len(lexicon[index])

def make_composites(indices, seed_count, seed=0, start=0):
    '''
    Composites made from the lexicon entries at indices (one set of the
    split, so they never mix train and test words), in the numbers
    PHRASE_SHARE, LONE_I_SHARE and PUNCTUATED_SHARE of seed_count. The same
    arguments always give the same items. Their numbers count from start.
    '''
    rng = random.Random(seed)
    words = single_words(indices)
    number = start
    composites = []

    def add(parts, spacing=None):
        nonlocal number
        composites.append(Composite(
            tuple(parts), rng.randrange(len(fonts)),
            spacing or rng.choices(SPACINGS, SPACING_WEIGHTS)[0], number))
        number += 1

    def mark():
        return COMMA if rng.random() < COMMA_CHANCE else FULL_STOP

    for _ in range(round(PHRASE_SHARE * seed_count)):
        length = rng.choices(PHRASE_LENGTHS, PHRASE_WEIGHTS)[0]
        chosen = []
        while len(chosen) < length:
            word = rng.choice(words)
            if sum(map(letters, chosen)) + letters(word) <= MAX_PHRASE_LETTERS:
                chosen.append(word)
            elif not chosen:
                continue
            else:
                break
        parts = []
        for word in chosen:
            parts.append(word)
            if (ends_in_consonant(lexicon[word])
                    and rng.random() < GENITIVE_CHANCE):
                parts.append(HOOK)
        if rng.random() < PHRASE_PUNCTUATION_CHANCE:
            # After any word but the last, or at the end
            parts.insert(rng.randint(1, len(parts)), mark())
            # Never straight before a genitive: it follows its word
            parts = fix_mark_order(parts)
        add(parts)
    for _ in range(round(LONE_I_SHARE * seed_count)):
        add([HOOK if rng.random() < HOOK_CHANCE else ISOLATED])
    for _ in range(round(PUNCTUATED_SHARE * seed_count)):
        add([rng.choice(words), mark()])
    return composites

def fix_mark_order(parts):
    # Moves a mark that landed between a word and its genitive after the
    # genitive, and drops a mark at the very start
    parts = list(parts)
    for index in range(len(parts) - 1):
        if parts[index] in MARKS and parts[index + 1] == HOOK:
            parts[index], parts[index + 1] = parts[index + 1], parts[index]
    while parts and parts[0] in MARKS:
        parts.pop(0)
    return parts
