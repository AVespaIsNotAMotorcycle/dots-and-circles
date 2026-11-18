'''
Handwritten cursive Manchu word crops (mic7ch's cursive_v1 and cursive_v2
on Hugging Face; see ocr.training_data.fetch_cursive). They're kept like
the print photos (ocr.photos): one folder per word, named in the project's
romanization, under ocr/cursive_data/v1/ and v2/.

v2 is split by word like the photos (ocr.split.split_cursive_words, listed
in training_data/cursive_words.json): ocr/train.py trains on its training
crops and scores its monitor set after each epoch, and
ocr/evaluate_non_synthetic.py --source cursive tests on its benchmark. v1,
from another document in another hand, is only tested on (--source
cursive_v1).
'''

import os

from ocr.photos import load_photo

directory = os.path.dirname(os.path.abspath(__file__))
DATA_DIRECTORY = os.path.join(directory, "cursive_data")


def version_directory(version):
    return os.path.join(DATA_DIRECTORY, version)

def available(version="v2"):
    # The crops aren't in the repository, so they may not be here
    return os.path.isdir(version_directory(version))

def all_images(version="v2", words=None):
    '''
    Every (path, label) pair of one version, in order, or only those of
    words if given. The label is the folder name.
    '''
    images = []
    root = version_directory(version)
    for label in sorted(os.listdir(root)):
        if words is not None and label not in words:
            continue
        folder = os.path.join(root, label)
        for name in sorted(os.listdir(folder)):
            images.append((os.path.join(folder, name), label))
    return images

def set_images(name):
    '''
    Every (path, label) pair of one set of the v2 split: "train", "monitor"
    or "benchmark" (see ocr.split.split_cursive_words).
    '''
    # Imported here, since ocr.split imports ocr.photos
    from ocr.split import load_cursive_split
    return all_images("v2", load_cursive_split()[name])

def load_crop(path):
    # Prepared as the print photos are (see ocr.preprocess.prepare)
    return load_photo(path)
