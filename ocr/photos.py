'''
Photographed words from printed Manchu books, for testing the models on real
pages: used by ocr/evaluate_non_synthetic.py, and by ocr/train.py to score a
fixed set of photos after each epoch.
'''

import os
import random

from PIL import Image

from ocr.preprocess import prepare

directory = os.path.dirname(os.path.abspath(__file__))
# One folder per word. Each folder is named after its word in the dataset's
# romanization (see MANCHU_TO_ROMAN); the file names inside sometimes use
# another, so the folder name is the label.
# Source: https://www.scidb.cn/en/detail?dataSetId=b45491b63d694534a9323acf14846586
DATA_DIRECTORY = os.path.join(directory, "non_synthetic_data")

# The photo dataset's romanization, used inside the project for labels and
# scoring: Möllendorff written in ASCII (ts', dz, k', c'y, sy), but with x for
# ᡧ and v for ᡡ as in Abkai. It isn't Abkai, which writes ᠴ as q (see
# MANCHU_TO_ABKAI). Variation selectors and the other invisible marks have
# no spelling, so they're dropped. The Mongolian letters (ᠢ, ᠤ, ᠭ, ᠳ) take
# the spelling of their Manchu counterparts. ᡰ is very rare, and r' is a
# best guess.
MANCHU_TO_ROMAN = {
    "ᠠ": "a", "ᡝ": "e", "ᡳ": "i", "ᠢ": "i", "ᠣ": "o", "ᡠ": "u", "ᠤ": "u",
    "ᡡ": "v", "ᠨ": "n", "ᠩ": "ng", "ᠪ": "b", "ᡦ": "p", "ᡤ": "g", "ᠭ": "g",
    "ᡥ": "h", "ᡴ": "k", "ᠺ": "k'", "ᡬ": "g'", "ᡭ": "h'", "ᠮ": "m", "ᠯ": "l",
    "ᠰ": "s", "ᡧ": "x", "ᡨ": "t", "ᡩ": "d", "ᠳ": "d", "ᠴ": "c", "ᠵ": "j",
    "ᠶ": "y", "ᠸ": "w", "ᡵ": "r", "ᡶ": "f", "ᡮ": "ts'", "ᡯ": "dz",
    "ᡰ": "r'", "ᡱ": "c'", "ᡷ": "z", "ᡟ": "y", " ": " ",
    "᠋": "", "᠌": "", "᠍": "", "᠏": "", "᠇": "",
    "᠊": "",
}

def manchu_to_roman(word):
    '''
    Romanizes a word in Manchu script as the photo dataset does, e.g.
    ᠠᠪᡴᠠ -> abka, ᠠᠴᠠᠮᠪᡳ -> acambi.
    '''
    return "".join(MANCHU_TO_ROMAN[letter] for letter in word)

def reverse(table):
    # Manchu for each spelling in table. Where several letters share one (ᡳ
    # and ᠢ, say), the Manchu one comes first in table, so it's the one kept.
    reversed_table = {}
    for letter, spelling in table.items():
        if spelling:
            reversed_table.setdefault(spelling, letter)
    return reversed_table

def to_manchu(word, table):
    '''
    Spells word in Manchu script with table (spelling -> Manchu), reading
    the longest spelling it can at each step (so ng is ᠩ, not ᠨᡤ).
    '''
    longest = max(len(spelling) for spelling in table)
    manchu = ""
    position = 0
    while position < len(word):
        for length in range(longest, 0, -1):
            letter = table.get(word[position:position + length])
            if letter:
                manchu += letter
                position += length
                break
        else:
            raise ValueError(f"No Manchu for {word[position:]!r} in {word!r}")
    return manchu

ROMAN_TO_MANCHU = reverse(MANCHU_TO_ROMAN)

def roman_to_manchu(word):
    '''
    Spells a word in the dataset's romanization in Manchu script, e.g.
    abka -> ᠠᠪᡴᠠ.
    '''
    return to_manchu(word, ROMAN_TO_MANCHU)

# Abkai, as in the chart at
# https://en.wikipedia.org/wiki/Transliterations_of_Manchu, for showing words
# to people. The loanword letters are marked with ‘ (U+2018) after k, g and
# h, and ʻ (U+02BB) elsewhere.
ABKAI_QUOTE = "\u2018"
ABKAI_MARK = "\u02bb"
MANCHU_TO_ABKAI = {
    **MANCHU_TO_ROMAN,
    "ᠴ": "q", "ᡮ": "c", "ᡯ": "z", "ᡱ": "q", "ᡷ": "j",
    "ᠺ": "k" + ABKAI_QUOTE, "ᡬ": "g" + ABKAI_QUOTE, "ᡭ": "h" + ABKAI_QUOTE,
    "ᡰ": "r" + ABKAI_MARK, "ᡟ": "y" + ABKAI_MARK,
}
# Pairs Abkai spells as a whole: in these, ᡳ is the apical vowel (as in
# Chinese zi, zhi, chi), written yʻ. ᡯᡳ only counts with FVS2, as the
# lexicon spells it; plain ᡯᡳ stays zi.
ABKAI_PAIRS = {
    "ᡱᡳ": "qy" + ABKAI_MARK,
    "ᡷᡳ": "jy" + ABKAI_MARK,
    "ᡯᡳ\u180c": "zy" + ABKAI_MARK,
}

def manchu_to_abkai(word):
    '''
    Transliterates a word in Manchu script to Abkai, e.g. ᠠᠪᡴᠠ -> abka,
    ᠠᠴᠠᠮᠪᡳ -> aqambi.
    '''
    abkai = ""
    position = 0
    while position < len(word):
        for pair, spelling in ABKAI_PAIRS.items():
            if word.startswith(pair, position):
                abkai += spelling
                position += len(pair)
                break
        else:
            abkai += MANCHU_TO_ABKAI[word[position]]
            position += 1
    return abkai

ABKAI_TO_MANCHU = {**reverse(MANCHU_TO_ABKAI),
                   **{spelling: pair for pair, spelling in ABKAI_PAIRS.items()}}
# People type the marks in many ways, so all of these are read as ʻ
ABKAI_MARKS = "'\u2019" + ABKAI_QUOTE + ABKAI_MARK
TYPED_ABKAI_TO_MANCHU = {spelling.replace(ABKAI_QUOTE, ABKAI_MARK): letter
                         for spelling, letter in ABKAI_TO_MANCHU.items()}

def abkai_to_manchu(word):
    '''
    Spells a word typed in Abkai in Manchu script, e.g. aqambi -> ᠠᠴᠠᠮᠪᡳ.
    Any of ' ’ ‘ ʻ is accepted as the mark on loanword letters.
    '''
    for mark in ABKAI_MARKS:
        word = word.replace(mark, ABKAI_MARK)
    return to_manchu(word, TYPED_ABKAI_TO_MANCHU)

_spellings = None

def spell_in_manchu(label):
    '''
    A label in the dataset's romanization in Manchu script: the lexicon's
    spelling where it has the word, since it knows which letters each word
    really uses, or else roman_to_manchu's. Suffixes (᠊ᠨ and so on) aren't
    matched, since a photo is of a whole word.
    '''
    global _spellings
    if _spellings is None:
        # Imported here: the lexicon is only needed once a label is spelled
        from ocr.training_data import lexicon
        _spellings = {}
        for word in lexicon:
            if not word.startswith("᠊"):
                _spellings.setdefault(manchu_to_roman(word), word)
    return _spellings.get(label) or roman_to_manchu(label)

def available():
    # The photos aren't in the repository, so they may not be here
    return os.path.isdir(DATA_DIRECTORY)

def sample_images(count, seed=0):
    '''
    Picks count random (path, label) pairs from DATA_DIRECTORY, the same ones
    for the same seed. The label is the folder name, in the dataset's
    romanization (see MANCHU_TO_ROMAN).
    '''
    images = []
    for label in sorted(os.listdir(DATA_DIRECTORY)):
        folder = os.path.join(DATA_DIRECTORY, label)
        for name in sorted(os.listdir(folder)):
            images.append((os.path.join(folder, name), label))
    return random.Random(seed).sample(images, min(count, len(images)))

def load_photo(path, crop=True, whiten=True):
    '''
    Opens a photographed word as a greyscale image the models can read; see
    ocr.preprocess.prepare. Training images are augmented to look like
    photos prepared with the defaults, crop=True and whiten=True.
    '''
    return prepare(Image.open(path), crop, whiten)
