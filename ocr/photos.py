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
# One folder per word. Each folder is named after its word in Abkai
# transliteration; the file names inside sometimes use another romanization,
# so the folder name is the label.
# Source: https://www.scidb.cn/en/detail?dataSetId=b45491b63d694534a9323acf14846586
DATA_DIRECTORY = os.path.join(directory, "non_synthetic_data")

# Abkai for each character the models can output. Variation selectors and
# the other invisible marks have no Abkai spelling, so they're dropped. The
# Mongolian letters (ᠢ, ᠤ, ᠭ, ᠳ) take the spelling of their Manchu
# counterparts. ᡰ is very rare, and r' is a best guess.
MANCHU_TO_ABKAI = {
    "ᠠ": "a", "ᡝ": "e", "ᡳ": "i", "ᠢ": "i", "ᠣ": "o", "ᡠ": "u", "ᠤ": "u",
    "ᡡ": "v", "ᠨ": "n", "ᠩ": "ng", "ᠪ": "b", "ᡦ": "p", "ᡤ": "g", "ᠭ": "g",
    "ᡥ": "h", "ᡴ": "k", "ᠺ": "k'", "ᡬ": "g'", "ᡭ": "h'", "ᠮ": "m", "ᠯ": "l",
    "ᠰ": "s", "ᡧ": "x", "ᡨ": "t", "ᡩ": "d", "ᠳ": "d", "ᠴ": "c", "ᠵ": "j",
    "ᠶ": "y", "ᠸ": "w", "ᡵ": "r", "ᡶ": "f", "ᡮ": "ts'", "ᡯ": "dz",
    "ᡰ": "r'", "ᡱ": "c'", "ᡷ": "z", "ᡟ": "y", " ": " ",
    "᠋": "", "᠌": "", "᠍": "", "᠏": "", "᠇": "",
    "᠊": "",
}

def manchu_to_abkai(word):
    '''
    Transliterates a word in Manchu script to Abkai, e.g. ᠠᠪᡴᠠ -> abka.
    '''
    return "".join(MANCHU_TO_ABKAI[letter] for letter in word)

# Manchu for each Abkai spelling, for words not in the lexicon. Where several
# letters share a spelling (ᡳ and ᠢ, say), the Manchu one comes first in
# MANCHU_TO_ABKAI, so it's the one kept.
ABKAI_TO_MANCHU = {}
for letter, abkai in MANCHU_TO_ABKAI.items():
    if abkai:
        ABKAI_TO_MANCHU.setdefault(abkai, letter)
LONGEST_SPELLING = max(len(abkai) for abkai in ABKAI_TO_MANCHU)

def abkai_to_manchu(word):
    '''
    Transliterates a word in Abkai to Manchu script, e.g. abka -> ᠠᠪᡴᠠ,
    reading the longest spelling it can at each step (so ng is ᠩ, not ᠨᡤ).
    '''
    manchu = ""
    position = 0
    while position < len(word):
        for length in range(LONGEST_SPELLING, 0, -1):
            letter = ABKAI_TO_MANCHU.get(word[position:position + length])
            if letter:
                manchu += letter
                position += length
                break
        else:
            raise ValueError(f"No Manchu for {word[position:]!r} in {word!r}")
    return manchu

_spellings = None

def spell_in_manchu(label):
    '''
    A photo's Abkai label in Manchu script: the lexicon's spelling where it
    has the word, since it knows which letters each word really uses, or
    else abkai_to_manchu's. Suffixes (᠊ᠨ and so on) aren't matched, since a
    photo is of a whole word.
    '''
    global _spellings
    if _spellings is None:
        # Imported here: the lexicon is only needed once a label is spelled
        from ocr.training_data import lexicon
        _spellings = {}
        for word in lexicon:
            if not word.startswith("᠊"):
                _spellings.setdefault(manchu_to_abkai(word), word)
    return _spellings.get(label) or abkai_to_manchu(label)

def available():
    # The photos aren't in the repository, so they may not be here
    return os.path.isdir(DATA_DIRECTORY)

def sample_images(count, seed=0):
    '''
    Picks count random (path, label) pairs from DATA_DIRECTORY, the same ones
    for the same seed. The label is the Abkai folder name.
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
