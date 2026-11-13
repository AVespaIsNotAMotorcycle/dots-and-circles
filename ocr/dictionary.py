'''
Finds the dictionary entries closest to a reading, for showing beside it
(the reading itself is never replaced: the models must read words in no
dictionary). The entries are training_data/dictionary_words.json, from
Wiktionary and Norman's dictionary.

Closeness is the edit distance in the photo dataset's romanization
(ocr.photos.manchu_to_roman), as the scores use, with punctuation left
off the end of each of the reading's words (aljara, is aljara). Inside a
word it stays (al.jara), and costs an edit, as no entry has any.

The dictionaries list verbs by their -mbi form (arambi), so a verb's other
forms are matched too: its stem with up to two derivational infixes and a
suffix (ara-bu-ha), from ocr.affixes. The edit distance from a reading q
to stem + ending is the least, over every way of splitting q in two, of
the first part's distance to the stem plus the second's to the ending.
The second doesn't depend on the stem, so the best ending for each split
is found once per reading, and the stems are searched like the entries.
'''

import json
import os

import numpy as np

from ocr.affixes import LEMMA_ENDING, endings as verb_endings
from ocr.photos import manchu_to_abkai, manchu_to_roman, roman_to_manchu

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "training_data", "dictionary_words.json")

# Romanized Manchu comma and full stop, left off the ends of words
PUNCTUATION = ",."

# The Manchu spelling of LEMMA_ENDING, which every verb entry ends with
MANCHU_LEMMA_ENDING = "\u182e\u182a\u1873"


def prefix_distances(query, words):
    '''
    The edit distance from each prefix of query, a list of letter codes,
    to each row of words, an array of letter codes all the same length: an
    array with a row per word and a column per prefix length, 0 to
    len(query). The usual dynamic programme, a row of it at a time for
    every word at once.
    '''
    count, length = words.shape
    previous = np.tile(np.arange(length + 1, dtype=np.int32), (count, 1))
    found = np.empty((count, len(query) + 1), dtype=np.int32)
    found[:, 0] = length
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
        found[:, i] = current[:, -1]
    return found


def distances(query, words):
    '''
    The edit distance from query, a list of letter codes, to each row of
    words, an array of letter codes all the same length.
    '''
    return prefix_distances(query, words)[:, -1]


def group_by_length(romans, codes):
    '''
    Length -> (indices into romans, their letters as an array of codes),
    for searching every word of a length at once.
    '''
    groups = {}
    for index, roman in enumerate(romans):
        groups.setdefault(len(roman), []).append(index)
    return {length: (np.array(indices), np.array(
        [[codes[letter] for letter in romans[index]] for index in indices],
        dtype=np.int32).reshape(len(indices), length))
        for length, indices in groups.items()}


class Dictionary:
    '''
    Dictionary entries, each a (manchu, sources) pair, searched by
    closest. The romanized entries are kept grouped by length, as an entry
    n letters longer or shorter than a reading is at least n edits away.
    '''

    def __init__(self, entries, endings=None):
        '''
        endings are the verb endings, as ocr.affixes.endings gives them
        (the default), as (roman, affixes) pairs.
        '''
        self.entries = [(manchu, list(sources)) for manchu, sources in entries]
        romans = [manchu_to_roman(manchu) for manchu, _ in self.entries]
        self.endings = verb_endings() if endings is None else endings
        letters = {letter for roman in romans for letter in roman}
        letters |= {letter for roman, _ in self.endings for letter in roman}
        self.codes = {letter: code
                      for code, letter in enumerate(sorted(letters))}
        self.groups = group_by_length(romans, self.codes)
        self.romans = romans
        self.known = set(romans)
        # Verbs: the entries ending in -mbi, as their stems
        self.verbs = [index for index, roman in enumerate(romans)
                      if roman.endswith(LEMMA_ENDING)
                      and len(roman) > len(LEMMA_ENDING)]
        self.stems = [romans[index][:-len(LEMMA_ENDING)]
                      for index in self.verbs]
        self.stem_groups = group_by_length(self.stems, self.codes)
        # The endings backwards, as their distance to the end of a
        # reading is found from the end
        self.ending_groups = group_by_length(
            [roman[::-1] for roman, _ in self.endings], self.codes)

    @classmethod
    def load(cls, path=PATH):
        with open(path, "r", encoding="utf-8") as file:
            return cls(json.load(file))

    def closest(self, reading, count=2):
        '''
        The count entries or verb forms closest to reading (Manchu script),
        closest first, as dicts with "manchu", "abkai", "roman",
        "distance" and "sources". A verb form also has "lemma", its verb's
        entry (Manchu script), and "affixes", the Norman affixes put on its
        stem (["-bu-", "-ha"]); an entry has None for both, and a form
        spelled like an entry is that entry. Ties go to entries, then to
        forms with fewer infixes, then to the nearer the reading's length,
        then in romanized order. Punctuation ending a word is left off.
        Empty for a reading with nothing else.
        '''
        words = (word.rstrip(PUNCTUATION)
                 for word in manchu_to_roman(reading).split())
        roman = " ".join(word for word in words if word)
        if not roman:
            return []
        # Letters in no entry match nothing
        query = [self.codes.get(letter, -1) for letter in roman]
        entries = self.closest_entries(query, count)
        # Forms tied with an entry go after it, so only closer ones count
        limit = entries[-1][0] if len(entries) == count else None
        found = sorted(entries
                       + self.closest_forms(query, count, limit))[:count]
        results = []
        for distance, _, _, _, roman_found, index, ending in found:
            manchu, sources = self.entries[index]
            lemma, affixes = None, None
            if ending is not None:
                lemma, affixes = manchu, self.endings[ending][1]
                manchu = (manchu[:-len(MANCHU_LEMMA_ENDING)]
                          + roman_to_manchu(self.endings[ending][0]))
            results.append({"manchu": manchu,
                            "abkai": manchu_to_abkai(manchu),
                            "roman": roman_found, "distance": distance,
                            "sources": sources, "lemma": lemma,
                            "affixes": affixes})
        return results

    def closest_entries(self, query, count):
        '''
        The count entries closest to query (letter codes), as (distance, 0,
        0, length difference, roman, index, None), closest first.
        '''
        found = []  # (distance, length difference, roman, index)
        for length in sorted(self.groups,
                             key=lambda length: abs(length - len(query))):
            gap = abs(length - len(query))
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
        return [(distance, 0, 0, gap, roman, index, None)
                for distance, gap, roman, index in found]

    def best_endings(self, query, limit=None):
        '''
        For each split of query (letter codes) into a stem's part and an
        ending's, query[:j] and query[j:] for j from 0 to len(query): the
        index of the ending closest to query[j:] (fewest infixes, then
        nearest its length, on a tie) and its distance, as two arrays.
        Endings at least limit edits from all of query's ends are skipped
        (an ending n letters longer than query is at least n edits away);
        if all are, None.
        '''
        backwards = query[::-1]
        size = len(query) + 1
        keys, indices = [], []
        for length, (group, words) in self.ending_groups.items():
            if limit is not None and length - len(query) >= limit:
                continue
            # Column i is the distance to the last i letters of query, so
            # reversed, column j is the distance to query[j:]
            found = prefix_distances(backwards, words)[:, ::-1]
            infixes = np.array([len(self.endings[index][1]) - 1
                                for index in group])
            gaps = np.abs(length - (len(query) - np.arange(size)))
            keys.append(found.astype(np.int64) * 1000000
                        + infixes[:, None] * 1000 + gaps[None, :])
            indices.append(group)
        if not keys:
            return None
        keys, indices = np.concatenate(keys), np.concatenate(indices)
        best = np.argmin(keys, axis=0)
        return indices[best], keys[best, np.arange(size)] // 1000000

    def closest_forms(self, query, count, limit=None):
        '''
        The count verb forms (stem and ending) closest to query (letter
        codes) that aren't spelled like an entry, as (distance, 1, number
        of infixes, length difference, roman, index of the verb's entry,
        index of the ending), closest first: one form a verb, its closest.
        Only forms closer than limit edits, if given.
        '''
        if not self.verbs:
            return []
        best = self.best_endings(query, limit)
        if best is None:
            return []
        ending_index, ending_distance = best
        shortest = min(len(roman) for roman, _ in self.endings)
        ending_lengths = np.array([len(self.endings[index][0])
                                   for index in ending_index])
        ending_infixes = np.array([len(self.endings[index][1]) - 1
                                   for index in ending_index])
        stems, keys, ending = [], [], []
        for length, (group, codes) in self.stem_groups.items():
            # A form n letters longer than query is at least n edits away
            if limit is not None and length + shortest - len(query) >= limit:
                continue
            total = prefix_distances(query, codes) + ending_distance[None, :]
            gaps = np.abs(length + ending_lengths - len(query))
            # Each stem's best split: the closest, then the nearest the
            # reading's length
            split = np.argmin(total * 1000 + gaps[None, :], axis=1)
            stems.append(group)
            keys.append(total[np.arange(len(group)), split].astype(np.int64)
                        * 1000000 + ending_infixes[split] * 1000
                        + gaps[split])
            ending.append(ending_index[split])
        if not keys:
            return []
        stems, keys, ending = (np.concatenate(values)
                               for values in (stems, keys, ending))
        # The forms in order of (distance, infixes, length difference),
        # each tie in romanized order, until count are found
        found = []
        seen = set()
        for key in np.unique(keys):
            tied = sorted((self.stems[stems[k]] + self.endings[ending[k]][0],
                           int(k)) for k in np.nonzero(keys == key)[0])
            for roman, k in tied:
                if roman in self.known or roman in seen:
                    continue
                seen.add(roman)
                found.append((int(key // 1000000), 1,
                              int(key // 1000 % 1000), int(key % 1000),
                              roman, self.verbs[stems[k]], int(ending[k])))
                if len(found) == count:
                    return found
            if limit is not None and key // 1000000 >= limit:
                break
        return found
