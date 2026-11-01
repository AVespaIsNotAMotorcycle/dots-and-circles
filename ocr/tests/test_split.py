import os

import pytest

from ocr import photos
from ocr.photos import manchu_to_roman
from ocr.split import group_key, load_photo_words, split_words
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

    def test_photo_words_are_all_in_test(self, lexicon_split):
        train, _, _ = lexicon_split
        photo_words = load_photo_words()
        assert not [lexicon[index] for index in train
                    if manchu_to_roman(lexicon[index]) in photo_words]

    def test_variants_share_a_set(self):
        words = ["ᠠᠮᠠ", "ᠠ᠋ᠮᠠ", "ᠠᠮ᠌ᠠ"] + [f"ᠠ{'ᠨ' * n}" for n in range(1, 40)]
        train, _, test = split_words(words, photo_words=set())
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


@pytest.mark.skipif(not photos.available(), reason="no photo data")
def test_photo_word_list_matches_the_photos():
    folders = {name for name in os.listdir(photos.DATA_DIRECTORY)
               if os.path.isdir(os.path.join(photos.DATA_DIRECTORY, name))}
    assert load_photo_words() == folders
