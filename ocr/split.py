'''
Splits words into train and test sets (and optionally validation), so that
no word is ever in more than one. train.py then gives each set every font of
its words, so a test word is never seen in training, in any font.

- Words that differ only in invisible marks (free variation selectors, the
  Mongolian vowel separator, ZWJ/ZWNJ) are grouped and always share a set,
  since they can render the same or nearly the same. Only the split uses
  this grouping: the labels keep their marks.
- Every word in the photographed test data (training_data/photo_words.json)
  goes to test, so the photo score is always on words the models haven't
  trained on. Photos are named in the dataset's romanization (see
  ocr.photos.MANCHU_TO_ROMAN), so a word goes to test if its romanization
  matches.
- Then more groups are added to test, in order of a hash of their text,
  until it holds test_proportion of the words. Validation is filled the
  same way, after test. The hash means the split is the same every run, and
  mostly the same when the lexicon changes: only words near the cut-off can
  move between sets.
'''

import hashlib
import json
import os

from ocr.photos import manchu_to_roman

directory = os.path.dirname(os.path.abspath(__file__))
PHOTO_WORDS_PATH = os.path.join(directory, "training_data", "photo_words.json")

# FVS1-4, the Mongolian vowel separator, ZWNJ and ZWJ
INVISIBLE_MARKS = "\u180b\u180c\u180d\u180f\u180e\u200c\u200d"


def group_key(word):
    '''
    The word without invisible marks: words with the same key share a set.
    '''
    return "".join(letter for letter in word if letter not in INVISIBLE_MARKS)

def rank(key):
    '''
    Where a group falls in the order test and validation are filled in. A
    hash, so it depends only on the group's text.
    '''
    return hashlib.sha256(key.encode("utf-8")).hexdigest()

def load_photo_words():
    with open(PHOTO_WORDS_PATH, "r", encoding="utf-8") as file:
        return set(json.load(file)["words"])

def split_words(words, test_proportion=0.15, valid_proportion=0.0,
                photo_words=None):
    '''
    Returns (train, valid, test), lists of indices into words. photo_words
    is a set of romanized words that must go to test (by default, the ones
    in photo_words.json).
    '''
    if photo_words is None:
        photo_words = load_photo_words()

    groups = {}
    for index, word in enumerate(words):
        groups.setdefault(group_key(word), []).append(index)

    test = []
    remaining = []
    for key, members in groups.items():
        if any(manchu_to_roman(words[index]) in photo_words
               for index in members):
            test.extend(members)
        else:
            remaining.append(key)
    remaining.sort(key=rank)

    test_target = round(test_proportion * len(words))
    if len(test) > test_target:
        print(f"The photo words alone are {len(test)} words, more than the "
              f"{test_target} test_proportion asks for.")
    valid_target = round(valid_proportion * len(words))
    valid = []
    train = []
    for key in remaining:
        if len(test) < test_target:
            test.extend(groups[key])
        elif len(valid) < valid_target:
            valid.extend(groups[key])
        else:
            train.extend(groups[key])

    return sorted(train), sorted(valid), sorted(test)
