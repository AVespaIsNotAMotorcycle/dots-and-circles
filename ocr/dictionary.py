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
is found once per reading (once for each vowel class of stem, as on a tie
the ending keeping vowel harmony is preferred), and the stems are searched
like the entries. The irregular forms Norman's dictionary gives (jengke,
of jembi; jio) are searched as entries.
'''

import json
import os

import numpy as np

from ocr.affixes import (INFIXES as INFIX_AFFIXES, LEMMA_ENDING, VARIANTS,
                         endings as verb_endings, ending_breaks, harmony,
                         norman_to_roman)
from ocr.photos import manchu_to_abkai, manchu_to_roman, roman_to_manchu

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "training_data", "dictionary_words.json")

# Romanized Manchu comma and full stop, left off the ends of words
PUNCTUATION = ",."

# A form's place in the order, as one number: its distance, then whether
# it breaks vowel harmony, then its number of infixes, then how far its
# length is from the reading's (under 1000)
DISTANCE, BREAKS, INFIXES = 10 ** 7, 10 ** 6, 10 ** 3

# The infixes with one spelling (-bu-, -nji-), in the dataset's
# romanization, longest first
INVARIANT_INFIXES = sorted((norman_to_roman(infix) for infix in INFIX_AFFIXES
                            if infix not in VARIANTS), key=len, reverse=True)

# Stems' vowel classes (ocr.affixes.harmony), each with a stem in it
VOWEL_CLASSES = {"a": "a", "e": "e", "o": "o"}

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
        entries are [manchu, sources], or [manchu, sources, note] for a verb
        with a note in Norman's dictionary, as in dictionary_words.json.
        endings are the verb endings, as ocr.affixes.endings gives them
        (the default), as (roman, affixes) pairs.
        '''
        self.entries = [(entry[0], list(entry[1])) for entry in entries]
        romans = [manchu_to_roman(manchu) for manchu, _ in self.entries]
        # Each verb's perfective endings for ocr.affixes.ending_breaks:
        # None if it isn't in Norman
        self.perfective = [
            entry[2]["perfective"] if len(entry) > 2
            else [] if "norman" in entry[1] else None for entry in entries]
        # The irregular forms in notes, searched as entries after them,
        # each with its verb's index and its affixes (None for entries)
        self.form_of = [None] * len(self.entries)
        known = set(romans)
        for index, entry in enumerate(entries):
            for manchu, affixes in (entry[2]["forms"] if len(entry) > 2
                                    else []):
                roman = manchu_to_roman(manchu)
                if roman not in known:
                    known.add(roman)
                    self.entries.append((manchu, list(entry[1])))
                    romans.append(roman)
                    self.form_of.append((index, affixes))
        self.endings = verb_endings() if endings is None else endings
        letters = {letter for roman in romans for letter in roman}
        letters |= {letter for roman, _ in self.endings for letter in roman}
        self.codes = {letter: code
                      for code, letter in enumerate(sorted(letters))}
        self.groups = group_by_length(romans, self.codes)
        self.romans = romans
        self.known = set(romans)
        # Verbs: the entries ending in -mbi, as their stems, grouped by
        # vowel class then length
        self.verbs = [index for index, roman in enumerate(romans)
                      if self.form_of[index] is None
                      and roman.endswith(LEMMA_ENDING)
                      and len(roman) > len(LEMMA_ENDING)]
        self.stems = [romans[index][:-len(LEMMA_ENDING)]
                      for index in self.verbs]
        # Each verb's perfective endings inherited from a verb it's derived
        # from by an infix with one spelling (ilibumbi from ilimbi)
        noted = {romans[index]: self.perfective[index]
                 for index in range(len(entries)) if self.perfective[index]}
        self.inherited = {}
        for index, stem in zip(self.verbs, self.stems):
            for infix in INVARIANT_INFIXES:
                root = stem[:-len(infix)] + LEMMA_ENDING
                if stem.endswith(infix) and len(stem) > len(infix) \
                        and root in noted:
                    self.inherited[index] = noted[root]
                    break
        self.stem_groups = {}
        for vowel_class in VOWEL_CLASSES:
            members = [k for k, stem in enumerate(self.stems)
                       if harmony(stem) == vowel_class]
            for length, (group, codes) in group_by_length(
                    [self.stems[k] for k in members], self.codes).items():
                self.stem_groups[vowel_class, length] = (
                    np.array(members)[group], codes)
        # The endings backwards, as their distance to the end of a
        # reading is found from the end
        self.ending_groups = group_by_length(
            [roman[::-1] for roman, _ in self.endings], self.codes)
        self.ending_lengths = np.array([len(roman)
                                        for roman, _ in self.endings])
        self.ending_infixes = np.array([len(affixes) - 1
                                        for _, affixes in self.endings])
        # Whether each ending breaks vowel harmony after each class of stem
        self.ending_breaks = {
            vowel_class: np.array([bool(ending_breaks(stem, affixes))
                                   for _, affixes in self.endings])
            for vowel_class, stem in VOWEL_CLASSES.items()}

    @classmethod
    def load(cls, path=PATH):
        with open(path, "r", encoding="utf-8") as file:
            return cls(json.load(file))

    def closest(self, reading, count=2):
        '''
        The count entries or verb forms closest to reading (Manchu script),
        closest first, as dicts with "manchu", "abkai", "roman",
        "distance" and "sources". A verb form also has "lemma", its verb's
        entry (Manchu script), "affixes", the Norman affixes put on its
        stem (["-bu-", "-ha"]), and "breaks", those that are wrong
        (ocr.affixes.ending_breaks, empty if none); an entry has None for
        all three, and a form spelled like an entry is that entry. An
        irregular form from a note in Norman's dictionary (jengke) has its
        affixes there (["-ngke"], or [] for a whole word, like jio) and no
        breaks. Ties go to entries and irregular forms, then to forms
        keeping vowel harmony,
        then with fewer infixes, then to the nearer the reading's length,
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
        for distance, _, roman_found, index, ending in found:
            manchu, sources = self.entries[index]
            lemma, affixes, breaks = None, None, None
            if ending is not None:
                lemma, affixes = manchu, self.endings[ending][1]
                breaks = ending_breaks(
                    self.romans[index][:-len(LEMMA_ENDING)], affixes,
                    self.perfective[index], self.inherited.get(index))
                manchu = (manchu[:-len(MANCHU_LEMMA_ENDING)]
                          + roman_to_manchu(self.endings[ending][0]))
            elif self.form_of[index] is not None:
                verb, affixes = self.form_of[index]
                lemma, breaks = self.entries[verb][0], []
            results.append({"manchu": manchu,
                            "abkai": manchu_to_abkai(manchu),
                            "roman": roman_found, "distance": distance,
                            "sources": sources, "lemma": lemma,
                            "affixes": affixes, "breaks": breaks})
        return results

    def closest_entries(self, query, count):
        '''
        The count entries closest to query (letter codes), as (distance, 0,
        roman, index, None), closest first: in the form of closest_forms,
        but entries go before forms on a tie, then by length difference.
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
        return [(distance, gap, roman, index, None)
                for distance, gap, roman, index in found]

    def best_endings(self, query, limit=None):
        '''
        For each split of query (letter codes) into a stem's part and an
        ending's, query[:j] and query[j:] for j from 0 to len(query), and
        each vowel class of stem: the index of the ending closest to
        query[j:] (keeping vowel harmony, then with fewest infixes, then
        nearest its length, on a tie) and its distance, as {vowel class:
        (indices, distances)}. Endings at least limit edits from all of
        query's ends are skipped (an ending n letters longer than query is
        at least n edits away); if all are, None.
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
            gaps = np.abs(length - (len(query) - np.arange(size)))
            keys.append(found.astype(np.int64) * DISTANCE
                        + self.ending_infixes[group][:, None] * INFIXES
                        + gaps[None, :])
            indices.append(group)
        if not keys:
            return None
        keys, indices = np.concatenate(keys), np.concatenate(indices)
        best = {}
        for vowel_class, breaks in self.ending_breaks.items():
            ranked = keys + breaks[indices][:, None] * BREAKS
            chosen = np.argmin(ranked, axis=0)
            best[vowel_class] = (indices[chosen],
                                 ranked[chosen, np.arange(size)] // DISTANCE)
        return best

    def closest_forms(self, query, count, limit=None):
        '''
        The count verb forms (stem and ending) closest to query (letter
        codes) that aren't spelled like an entry, as (distance, key, roman,
        index of the verb's entry, index of the ending), closest first:
        one form a verb, its closest. key orders ties (see DISTANCE), and
        is at least 1 so entries come first. Only forms closer than limit
        edits, if given.
        '''
        if not self.verbs:
            return []
        best = self.best_endings(query, limit)
        if best is None:
            return []
        shortest = self.ending_lengths.min()
        stems, keys, ending = [], [], []
        for (vowel_class, length), (group, codes) in self.stem_groups.items():
            # A form n letters longer than query is at least n edits away
            if limit is not None and length + shortest - len(query) >= limit:
                continue
            ending_index, ending_distance = best[vowel_class]
            total = prefix_distances(query, codes) + ending_distance[None, :]
            ranked = (total.astype(np.int64) * DISTANCE
                      + (self.ending_breaks[vowel_class][ending_index]
                         * BREAKS + self.ending_infixes[ending_index]
                         * INFIXES + np.abs(length - len(query) + self
                                            .ending_lengths[ending_index])
                         )[None, :])
            # Each stem's best split
            split = np.argmin(ranked, axis=1)
            stems.append(group)
            keys.append(ranked[np.arange(len(group)), split])
            ending.append(ending_index[split])
        if not keys:
            return []
        stems, keys, ending = (np.concatenate(values)
                               for values in (stems, keys, ending))
        # The forms in order, each tie in romanized order, until count
        # are found
        found = []
        seen = set()
        for key in np.unique(keys):
            distance = int(key // DISTANCE)
            if limit is not None and distance >= limit:
                break
            tied = sorted((self.stems[stems[k]] + self.endings[ending[k]][0],
                           int(k)) for k in np.nonzero(keys == key)[0])
            for roman, k in tied:
                if roman in self.known or roman in seen:
                    continue
                seen.add(roman)
                found.append((distance, int(key % DISTANCE) + 1, roman,
                              self.verbs[stems[k]], int(ending[k])))
                if len(found) == count:
                    return found
        return found
