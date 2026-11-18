'''
Rebuilds cursive_words.json, the words of the cursive_v2 crops
(ocr/cursive_data/v2/; see ocr.cursive) split into training, monitor and
benchmark words by ocr.split.split_cursive_words. A word keeps its set in
photo_words.json, or else its set in the lexicon split, so the lexicon split
is unchanged apart from words it doesn't have. The crops are gitignored but
the list isn't, so the split is the same on every machine.

Run from the repository root, after the cursive data, the photo split or
the lexicon changes:
    python -m ocr.training_data.list_cursive_words
'''

import collections
import json
import os

from ocr import cursive
from ocr.photos import manchu_to_roman
from ocr.split import (CURSIVE_WORDS_PATH, load_photo_split,
                       split_cursive_words, split_words)
from ocr.train import CONTEXT_LENGTH
from ocr.training_data import lexicon
from ocr.training_data.augment import tokens_needed


def lexicon_sets(photo_split):
    '''
    The romanized lexicon words in train and in test, split as train.py
    splits them but with only the photo words fixed: the cursive words
    can't decide the split they're placed by.
    '''
    kept = [word for word in lexicon if tokens_needed(word) <= CONTEXT_LENGTH]
    train, _, test = split_words(
        kept, photo_words=photo_split["monitor"] | photo_split["benchmark"],
        train_photo_words=photo_split["train"])
    return ({manchu_to_roman(kept[index]) for index in train},
            {manchu_to_roman(kept[index]) for index in test})


def main():
    if not cursive.available():
        raise SystemExit(f"No cursive data at {cursive.DATA_DIRECTORY}; "
                         f"see ocr/training_data/fetch_cursive.py")
    counts = collections.Counter(label for _, label
                                 in cursive.all_images("v2"))
    photo_split = load_photo_split()
    lexicon_train, lexicon_test = lexicon_sets(photo_split)
    test_words = (photo_split["monitor"] | photo_split["benchmark"]
                  | (lexicon_test - photo_split["train"]))
    train_words = photo_split["train"] | lexicon_train
    sets = split_cursive_words(counts, train_words, test_words)
    data = {
        "description": "The words (folder names) of the handwritten cursive "
                       "crops in ocr/cursive_data/v2/, in the project's "
                       "romanization (see ocr.photos.MANCHU_TO_ROMAN), split "
                       "by ocr.split.split_cursive_words: train is trained "
                       "on, monitor is scored after every epoch, and "
                       "benchmark is the held-out test",
        "source": "https://huggingface.co/datasets/mic7ch/cursive_v2",
        "license": "CC BY 4.0",
        "crops": {name: sum(counts[word] for word in words)
                  for name, words in sets.items()},
        **sets,
    }
    with open(CURSIVE_WORDS_PATH, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=1)
        file.write("\n")
    for name, words in sets.items():
        print(f"{name}: {len(words)} words, {data['crops'][name]} crops")
    print(f"Wrote {len(counts)} words to {CURSIVE_WORDS_PATH}")


if __name__ == "__main__":
    main()
