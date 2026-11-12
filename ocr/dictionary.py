'''
Finds the dictionary entries closest to a reading, for showing beside it
(the reading itself is never replaced: the models must read words in no
dictionary). The entries are training_data/dictionary_words.json, from
Wiktionary and Norman's dictionary.

Closeness is the edit distance in the photo dataset's romanization
(ocr.photos.manchu_to_roman), as the scores use, with the reading's
punctuation left out.
'''

import json
import os

import numpy as np

from ocr.photos import manchu_to_abkai, manchu_to_roman

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "training_data", "dictionary_words.json")

# Romanized Manchu comma and full stop, left out of readings
PUNCTUATION = ",."


def distances(query, words):
    '''
    The edit distance from query, a list of letter codes, to each row of
    words, an array of letter codes all the same length: the usual dynamic
    programme, a row of it at a time for every word at once.
    '''
    count, length = words.shape
    previous = np.tile(np.arange(length + 1, dtype=np.int32), (count, 1))
    for i, letter in enumerate(query, start=1):
        # Substituting (or keeping) a letter, or deleting one from query...
        best = np.minimum(previous[:, :-1] + (words != letter),
                          previous[:, 1:] + 1)
        current = np.empty_like(previous)
        current[:, 0] = i
        # ...or inserting one, which depends on the row's previous cell
        for j in range(1, length + 1):
            current[:, j] = np.minimum(best[:, j - 1], current[:, j - 1] + 1)
        previous = current
    return previous[:, -1]


class Dictionary:
    '''
    Dictionary entries, each a (manchu, sources) pair, searched by
    closest. The romanized entries are kept grouped by length, as an entry
    n letters longer or shorter than a reading is at least n edits away.
    '''

    def __init__(self, entries):
        self.entries = [(manchu, list(sources)) for manchu, sources in entries]
        romans = [manchu_to_roman(manchu) for manchu, _ in self.entries]
        self.codes = {letter: code for code, letter in enumerate(
            sorted({letter for roman in romans for letter in roman}))}
        # Length -> (indices of its entries, their letter codes)
        groups = {}
        for index, roman in enumerate(romans):
            groups.setdefault(len(roman), []).append(index)
        self.groups = {length: (np.array(indices), np.array(
            [[self.codes[letter] for letter in romans[index]]
             for index in indices], dtype=np.int32).reshape(len(indices),
                                                            length))
            for length, indices in groups.items()}
        self.romans = romans

    @classmethod
    def load(cls, path=PATH):
        with open(path, "r", encoding="utf-8") as file:
            return cls(json.load(file))

    def closest(self, reading, count=2):
        '''
        The count entries closest to reading (Manchu script), closest
        first, as dicts with "manchu", "abkai", "roman", "distance" and
        "sources". Ties go to the entry nearer the reading's length, then
        in romanized order. Empty for a reading with no letters.
        '''
        roman = " ".join(manchu_to_roman(reading).translate(
            str.maketrans(PUNCTUATION, " " * len(PUNCTUATION))).split())
        if not roman:
            return []
        # Letters in no entry match nothing
        query = [self.codes.get(letter, -1) for letter in roman]
        found = []  # (distance, length difference, roman, index)
        for length in sorted(self.groups,
                             key=lambda length: abs(length - len(roman))):
            gap = abs(length - len(roman))
            # Every entry from here on is at least gap edits away
            if len(found) >= count and found[count - 1][0] <= gap:
                break
            indices, words = self.groups[length]
            # Only the entries that could beat what's been found need
            # sorting in
            new = distances(query, words)
            limit = found[count - 1][0] if len(found) >= count else None
            keep = np.nonzero(new <= limit)[0] if limit is not None \
                else np.argsort(new, kind="stable")[:count]
            found.extend((int(new[k]), gap, self.romans[indices[k]],
                          int(indices[k])) for k in keep)
            found = sorted(found)[:count]
        results = []
        for distance, _, roman_entry, index in found:
            manchu, sources = self.entries[index]
            results.append({"manchu": manchu,
                            "abkai": manchu_to_abkai(manchu),
                            "roman": roman_entry, "distance": distance,
                            "sources": sources})
        return results
