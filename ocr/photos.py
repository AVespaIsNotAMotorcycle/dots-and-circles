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
