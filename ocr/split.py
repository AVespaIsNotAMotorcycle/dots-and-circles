'''
Splits words into train and test sets (and optionally validation), so that
no word is ever in more than one. train.py then gives each set every font of
its words, so a test word is never seen in training, in any font.

- Words that differ only in invisible marks (free variation selectors, the
  Mongolian vowel separator, ZWJ/ZWNJ) are grouped and always share a set,
  since they can render the same or nearly the same. Only the split uses
  this grouping: the labels keep their marks.
- The photographed words (training_data/photo_words.json) are split by
  word too, by split_photo_words: about 85% of the photos are trained on,
  and the rest are test photos, a small monitor set scored after every
  epoch and the benchmark. A lexicon word whose romanization matches a test
  photo word goes to test, and one matching a training photo word goes to
  train, so a word is in the same set as a rendering and as a photo. Photos
  are named in the dataset's romanization (see ocr.photos.MANCHU_TO_ROMAN).
- The handwritten cursive crops (training_data/cursive_words.json; see
  ocr.cursive) are split by split_cursive_words to agree with both: a
  cursive word keeps its photo set, or else its lexicon set, and only words
  in neither are placed by hash. Lexicon words matching cursive test or
  training words then go to test or train, as for the photos.
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
CURSIVE_WORDS_PATH = os.path.join(directory, "training_data",
                                  "cursive_words.json")

# The share of the photos held out from training (the monitor set and the
# benchmark), and how many of those make up the monitor set
PHOTO_TEST_PROPORTION = 0.15
MONITOR_PHOTOS = 500
# The share of the cursive crops held out, as for the photos, and how many
# of those make up the cursive monitor set
CURSIVE_TEST_PROPORTION = 0.15
MONITOR_CURSIVE = 100

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

def split_photo_words(counts):
    '''
    Splits the photographed words, given as {word: number of photos}, into
    {"train": [...], "monitor": [...], "benchmark": [...]}. Words are taken
    in order of a hash of their text: the monitor set until it holds
    MONITOR_PHOTOS photos, then the benchmark until the two together hold
    PHOTO_TEST_PROPORTION of the photos. The rest are trained on.
    '''
    total = sum(counts.values())
    sets = {"train": [], "monitor": [], "benchmark": []}
    monitor_photos = test_photos = 0
    for word in sorted(counts, key=rank):
        if monitor_photos < MONITOR_PHOTOS:
            sets["monitor"].append(word)
            monitor_photos += counts[word]
            test_photos += counts[word]
        elif test_photos < PHOTO_TEST_PROPORTION * total:
            sets["benchmark"].append(word)
            test_photos += counts[word]
        else:
            sets["train"].append(word)
    return {name: sorted(words) for name, words in sets.items()}

def load_photo_split():
    '''
    The photo words in each set, as {"train", "monitor", "benchmark"} ->
    set of words, from photo_words.json.
    '''
    with open(PHOTO_WORDS_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)
    return {name: set(data[name]) for name in ("train", "monitor",
                                                "benchmark")}

def split_cursive_words(counts, train_words, test_words):
    '''
    Splits the cursive words, given as {word: number of crops}, into
    {"train": [...], "monitor": [...], "benchmark": [...]}. Words in
    test_words (the photo test words, then the lexicon test words) are held
    out, and those in train_words (the photo, then lexicon, training words)
    are trained on, so every word keeps the set it has elsewhere. The rest
    are held out in order of a hash of their text until
    CURSIVE_TEST_PROPORTION of the crops are, and trained on after that.
    Then the held-out words, in hash order, fill the monitor set until it
    holds MONITOR_CURSIVE crops, and the rest are the benchmark.
    '''
    total = sum(counts.values())
    train, test, remaining = [], [], []
    for word in counts:
        if word in test_words:
            test.append(word)
        elif word in train_words:
            train.append(word)
        else:
            remaining.append(word)
    test_crops = sum(counts[word] for word in test)
    for word in sorted(remaining, key=rank):
        if test_crops < CURSIVE_TEST_PROPORTION * total:
            test.append(word)
            test_crops += counts[word]
        else:
            train.append(word)
    monitor, benchmark = [], []
    monitor_crops = 0
    for word in sorted(test, key=rank):
        if monitor_crops < MONITOR_CURSIVE:
            monitor.append(word)
            monitor_crops += counts[word]
        else:
            benchmark.append(word)
    return {"train": sorted(train), "monitor": sorted(monitor),
            "benchmark": sorted(benchmark)}

def load_cursive_split():
    '''
    The cursive words in each set, as {"train", "monitor", "benchmark"} ->
    set of words, from cursive_words.json, or empty sets if it isn't there.
    '''
    if not os.path.exists(CURSIVE_WORDS_PATH):
        return {name: set() for name in ("train", "monitor", "benchmark")}
    with open(CURSIVE_WORDS_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)
    return {name: set(data[name]) for name in ("train", "monitor",
                                                "benchmark")}

def load_photo_words():
    # Every photographed word, in any set
    return set().union(*load_photo_split().values())

def split_words(words, test_proportion=0.15, valid_proportion=0.0,
                photo_words=None, train_photo_words=None):
    '''
    Returns (train, valid, test), lists of indices into words. photo_words
    is a set of romanized words that must go to test, and train_photo_words
    one of words that must go to train (by default, the test and training
    words of photo_words.json and cursive_words.json).
    '''
    if photo_words is None or train_photo_words is None:
        photo_split = load_photo_split()
        cursive_split = load_cursive_split()
        if photo_words is None:
            photo_words = (photo_split["monitor"] | photo_split["benchmark"]
                           | cursive_split["monitor"]
                           | cursive_split["benchmark"])
        if train_photo_words is None:
            train_photo_words = photo_split["train"] | cursive_split["train"]

    groups = {}
    for index, word in enumerate(words):
        groups.setdefault(group_key(word), []).append(index)

    test = []
    train = []
    remaining = []
    for key, members in groups.items():
        romanized = {manchu_to_roman(words[index]) for index in members}
        if romanized & photo_words:
            test.extend(members)
        elif romanized & train_photo_words:
            train.extend(members)
        else:
            remaining.append(key)
    remaining.sort(key=rank)

    test_target = round(test_proportion * len(words))
    if len(test) > test_target:
        print(f"The photo words alone are {len(test)} words, more than the "
              f"{test_target} test_proportion asks for.")
    valid_target = round(valid_proportion * len(words))
    valid = []
    for key in remaining:
        if len(test) < test_target:
            test.extend(groups[key])
        elif len(valid) < valid_target:
            valid.extend(groups[key])
        else:
            train.extend(groups[key])

    return sorted(train), sorted(valid), sorted(test)
