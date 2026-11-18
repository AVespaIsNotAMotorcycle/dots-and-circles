import collections
import os

import pytest

from ocr import cursive, photos
from ocr.photos import manchu_to_roman
from ocr.split import (CURSIVE_TEST_PROPORTION, MONITOR_CURSIVE,
                       MONITOR_PHOTOS, PHOTO_TEST_PROPORTION, group_key,
                       load_cursive_split, load_photo_split, load_photo_words,
                       split_cursive_words, split_photo_words, split_words)
from ocr.training_data import lexicon


@pytest.fixture(scope="module")
def lexicon_split():
    return split_words(lexicon)


class TestSplitWords:
    def test_every_word_in_exactly_one_set(self, lexicon_split):
        train, valid, test = lexicon_split
        assert not set(train) & set(test)
        assert not valid
        assert sorted(train + test) == list(range(len(lexicon)))

    def test_test_is_about_fifteen_percent(self, lexicon_split):
        _, _, test = lexicon_split
        assert abs(len(test) / len(lexicon) - 0.15) < 0.001

    def test_photo_words_follow_the_photo_split(self, lexicon_split):
        train, _, test = lexicon_split
        photo_split = load_photo_split()
        test_photo_words = photo_split["monitor"] | photo_split["benchmark"]
        assert not [lexicon[index] for index in train
                    if manchu_to_roman(lexicon[index]) in test_photo_words]
        assert not [lexicon[index] for index in test
                    if manchu_to_roman(lexicon[index]) in photo_split["train"]]

    def test_cursive_words_follow_the_cursive_split(self, lexicon_split):
        train, _, test = lexicon_split
        cursive_split = load_cursive_split()
        test_words = cursive_split["monitor"] | cursive_split["benchmark"]
        assert not [lexicon[index] for index in train
                    if manchu_to_roman(lexicon[index]) in test_words]
        assert not [lexicon[index] for index in test
                    if manchu_to_roman(lexicon[index])
                    in cursive_split["train"]]

    def test_training_photo_words_go_to_train(self):
        words = [f"ᠠ{'ᠨ' * n}" for n in range(1, 40)]
        romanized = {manchu_to_roman(word) for word in words}
        train, _, test = split_words(words, photo_words=set(),
                                     train_photo_words=romanized)
        assert sorted(train) == list(range(len(words)))
        assert not test

    def test_variants_share_a_set(self):
        words = ["ᠠᠮᠠ", "ᠠ᠋ᠮᠠ", "ᠠᠮ᠌ᠠ"] + [f"ᠠ{'ᠨ' * n}" for n in range(1, 40)]
        train, _, test = split_words(words, photo_words=set(),
                                     train_photo_words=set())
        variants = {0, 1, 2}
        assert variants <= set(train) or variants <= set(test)
        assert group_key(words[1]) == group_key(words[2]) == words[0]

    def test_validation_doesnt_change_test(self, lexicon_split):
        _, _, test = lexicon_split
        train, valid, same_test = split_words(lexicon, valid_proportion=0.15)
        assert same_test == test
        assert not set(train) & set(valid)
        assert sorted(train + valid + test) == list(range(len(lexicon)))

    def test_is_repeatable(self, lexicon_split):
        assert split_words(lexicon) == lexicon_split


class TestSplitPhotoWords:
    counts = {f"word{number}": 1 + number % 12 for number in range(2000)}

    def test_every_word_in_exactly_one_set(self):
        sets = split_photo_words(self.counts)
        words = sets["train"] + sets["monitor"] + sets["benchmark"]
        assert sorted(words) == sorted(self.counts)

    def test_set_sizes(self):
        sets = split_photo_words(self.counts)
        photos_in = {name: sum(self.counts[word] for word in words)
                     for name, words in sets.items()}
        total = sum(self.counts.values())
        # Each set stops at the first word that takes it to its size
        assert MONITOR_PHOTOS <= photos_in["monitor"] < MONITOR_PHOTOS + 12
        test_share = (photos_in["monitor"] + photos_in["benchmark"]) / total
        assert PHOTO_TEST_PROPORTION <= test_share < PHOTO_TEST_PROPORTION + 0.01

    def test_is_repeatable(self):
        assert split_photo_words(self.counts) == split_photo_words(
            dict(reversed(list(self.counts.items()))))

    def test_committed_sets_are_disjoint(self):
        sets = load_photo_split()
        assert not sets["train"] & sets["monitor"]
        assert not sets["train"] & sets["benchmark"]
        assert not sets["monitor"] & sets["benchmark"]


class TestSplitCursiveWords:
    counts = {f"word{number}": 1 + number % 7 for number in range(600)}
    fixed_train = {f"word{number}" for number in range(0, 600, 3)}
    fixed_test = {f"word{number}" for number in range(1, 60, 3)}

    def split(self, counts=None):
        return split_cursive_words(counts or self.counts, self.fixed_train,
                                   self.fixed_test)

    def test_every_word_in_exactly_one_set(self):
        sets = self.split()
        words = sets["train"] + sets["monitor"] + sets["benchmark"]
        assert sorted(words) == sorted(self.counts)

    def test_fixed_words_keep_their_sets(self):
        sets = self.split()
        assert self.fixed_train <= set(sets["train"])
        assert self.fixed_test <= set(sets["monitor"] + sets["benchmark"])

    def test_set_sizes(self):
        sets = self.split()
        crops = {name: sum(self.counts[word] for word in words)
                 for name, words in sets.items()}
        total = sum(self.counts.values())
        assert MONITOR_CURSIVE <= crops["monitor"] < MONITOR_CURSIVE + 7
        test_share = (crops["monitor"] + crops["benchmark"]) / total
        assert (CURSIVE_TEST_PROPORTION <= test_share
                < CURSIVE_TEST_PROPORTION + 0.01)

    def test_is_repeatable(self):
        assert self.split() == self.split(
            dict(reversed(list(self.counts.items()))))

    def test_committed_split_agrees_with_the_photo_split(self):
        sets = load_cursive_split()
        photo_sets = load_photo_split()
        test = sets["monitor"] | sets["benchmark"]
        photo_test = photo_sets["monitor"] | photo_sets["benchmark"]
        assert not sets["train"] & test
        assert not sets["monitor"] & sets["benchmark"]
        assert not sets["train"] & photo_test
        assert not test & photo_sets["train"]


@pytest.mark.skipif(not cursive.available(), reason="no cursive data")
def test_cursive_word_list_matches_the_crops():
    # If this fails, rebuild it: python -m ocr.training_data.list_cursive_words
    counts = collections.Counter(label for _, label
                                 in cursive.all_images("v2"))
    assert set().union(*load_cursive_split().values()) == set(counts)


def photo_counts():
    return {name: len(os.listdir(os.path.join(photos.DATA_DIRECTORY, name)))
            for name in os.listdir(photos.DATA_DIRECTORY)
            if os.path.isdir(os.path.join(photos.DATA_DIRECTORY, name))}


@pytest.mark.skipif(not photos.available(), reason="no photo data")
def test_photo_word_list_matches_the_photos():
    assert load_photo_words() == set(photo_counts())


@pytest.mark.skipif(not photos.available(), reason="no photo data")
def test_committed_photo_split_is_up_to_date():
    # If this fails, rebuild it: python -m ocr.training_data.list_photo_words
    committed = {name: sorted(words)
                 for name, words in load_photo_split().items()}
    assert committed == split_photo_words(photo_counts())
