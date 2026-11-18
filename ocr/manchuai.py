'''
The two handwritten test sets of mic7ch's ManchuAI-OCR (Chung & Choi,
arXiv 2507.06761; see ocr.training_data.fetch_manchuai), kept like the
print photos (ocr.photos): one folder per word, named in the project's
romanization, under ocr/manchuai_data/218/ and 753/.

- 218: the paper's test set (Hugging Face mic7ch/test_data_long, its train
  and validation splits together).
- 753: the repository's newer test set (mic7ch/manchu, test split).

Both are only tested on: ocr/evaluate_non_synthetic.py --source
manchuai_218 or manchuai_753.
'''

import os

directory = os.path.dirname(os.path.abspath(__file__))
DATA_DIRECTORY = os.path.join(directory, "manchuai_data")
SETS = ("218", "753")


def set_directory(name):
    return os.path.join(DATA_DIRECTORY, name)

def available(name="218"):
    # The images aren't in the repository, so they may not be here
    return os.path.isdir(set_directory(name))

def all_images(name):
    '''
    Every (path, label) pair of one set, in order. The label is the folder
    name.
    '''
    images = []
    root = set_directory(name)
    for label in sorted(os.listdir(root)):
        folder = os.path.join(root, label)
        for file_name in sorted(os.listdir(folder)):
            images.append((os.path.join(folder, file_name), label))
    return images
