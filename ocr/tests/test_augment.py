import random

import numpy as np
from PIL import Image

from ocr.photos import (abkai_to_manchu, manchu_to_abkai, manchu_to_norman,
                        manchu_to_roman,
                        roman_to_manchu)
from ocr.preprocess import crop_to_ink, prepare, stretch
from ocr.training_data import HEIGHT, WIDTH, generate, lexicon, pad_top
from ocr.training_data.augment import tokens_needed


class TestGenerate:
    def test_clean_is_repeatable(self):
        first, word = generate(1234)
        second, same_word = generate(1234)
        assert word == same_word
        assert first.tobytes() == second.tobytes()

    def test_augmented_is_repeatable_per_epoch(self):
        first, _ = generate(1234, epoch=3)
        second, _ = generate(1234, epoch=3)
        assert first.tobytes() == second.tobytes()

    def test_augmented_differs_between_epochs(self):
        first, _ = generate(1234, epoch=3)
        second, _ = generate(1234, epoch=4)
        assert first.tobytes() != second.tobytes()

    def test_augmenting_keeps_the_word(self):
        _, clean_word = generate(777)
        _, word = generate(777, epoch=0)
        assert word == clean_word

    def test_augmented_images_are_model_inputs(self):
        for seed in range(0, 21879, 997):
            image, _ = generate(seed, epoch=1)
            assert image.mode == "L"
            assert image.width == WIDTH

    def test_augmented_images_have_enough_tokens_for_ctc(self):
        for seed in range(0, 21879, 211):
            for epoch in range(3):
                image, word = generate(seed, epoch)
                tokens = -(-min(image.height, HEIGHT) // 4)
                assert tokens >= tokens_needed(word)

    def test_padding_keeps_words_that_fit_inside_height(self):
        # pad_top only adds rows a word has room for
        word = Image.new("L", (WIDTH, 100), 0)
        for seed in range(50):
            padded = pad_top(word, random.Random(seed))
            assert 100 <= padded.height <= HEIGHT


class TestPreprocess:
    def grey_page(self):
        # Grey paper (200) with a darker speck, and a black bar of ink
        pixels = np.full((100, 60), 200, dtype=np.uint8)
        pixels[0, 0] = 230
        pixels[30:70, 20:30] = 20
        return Image.fromarray(pixels)

    def test_plain_stretch_leaves_paper_grey(self):
        pixels = np.asarray(stretch(self.grey_page()))
        assert pixels.min() == 0 and pixels.max() == 255
        assert 150 < pixels[50, 50] < 255

    def test_whitening_makes_paper_white(self):
        pixels = np.asarray(stretch(self.grey_page(), whiten=True))
        assert pixels[50, 50] == 255
        assert pixels[50, 25] == 0

    def test_crop_to_ink(self):
        cropped = crop_to_ink(stretch(self.grey_page(), whiten=True))
        assert cropped.size == (10, 40)

    def test_prepare_scales_to_width(self):
        image = prepare(self.grey_page())
        assert image.width == WIDTH
        # The 10 x 40 bar of ink, scaled to WIDTH wide
        assert image.height == 4 * WIDTH


class TestManchuToRoman:
    def test_letters(self):
        assert manchu_to_roman("ᠠᠪᡴᠠ") == "abka"
        assert manchu_to_roman("ᡧᡠᠸᡝ") == "xuwe"
        assert manchu_to_roman("ᠠᠴᠠᠮᠪᡳ") == "acambi"

    def test_variation_selectors_are_dropped(self):
        assert manchu_to_roman("ᡩ᠋ᡝ") == "de"

    def test_loanword_letters(self):
        assert manchu_to_roman("ᡮᠠᡳ") == "ts'ai"
        assert manchu_to_roman("ᡯᡳ") == "dzi"

    def test_round_trip(self):
        assert roman_to_manchu("inenggi") == "ᡳᠨᡝᠩᡤᡳ"
        assert roman_to_manchu("ts'ai") == "ᡮᠠᡳ"


class TestManchuToNorman:
    def test_letters(self):
        assert manchu_to_norman("ᠠᠪᡴᠠ") == "abka"
        assert manchu_to_norman("ᡧᡠᠸᡝ") == "šuwe"
        assert manchu_to_norman("ᠠᠴᠠᠮᠪᡳ") == "acambi"
        assert manchu_to_norman("ᠠᡴᡡ") == "akū"
        assert manchu_to_norman("ᠺᠣ") == "k'o"
        assert manchu_to_norman("ᡰᡳᠨ") == "žin"
        assert manchu_to_norman("ᡮᠠᡳ") == "tsai"

    def test_apical_vowel(self):
        # As Norman prints them: sy, cy, jy, and ts and dz alone
        assert manchu_to_norman("ᠰᡟ") == "sy"
        assert manchu_to_norman("ᡱᡳ") == "cy"
        assert manchu_to_norman("ᡷᡳ") == "jy"
        assert manchu_to_norman("ᡨᠠᡳᡯᡳ\u180c") == "taidz"
        assert manchu_to_norman("ᠮᡳᠶᠠᠨᡮᡟ") == "miyants"
        # ᡯᡳ without FVS2 is dzi, and ᡱ before another vowel c'
        assert manchu_to_norman("ᡯᡳ") == "dzi"
        assert manchu_to_norman("ᡱᠠ") == "c'a"

    def test_round_trips_norman_headwords(self):
        # Norman as add_norman.py reads it comes back as he printed it
        from ocr.training_data.add_norman import to_manchu
        for word in ("hoošan", "šanyan", "gūnin", "jahūdai", "g'aosy",
                     "cyming", "jyši", "dzung", "taidz", "k'amduri"):
            assert manchu_to_norman(to_manchu(word)) == word

    def test_punctuation_and_marks(self):
        assert manchu_to_norman("ᠠᠯᠵᠠᡵᠠ\u1808") == "aljara,"
        assert manchu_to_norman("ᡳ\u180bᠨᡝ") == "ine"


class TestManchuToAbkai:
    def test_letters(self):
        assert manchu_to_abkai("ᠠᠪᡴᠠ") == "abka"
        assert manchu_to_abkai("ᡧᡠᠸᡝ") == "xuwe"
        assert manchu_to_abkai("ᠠᠴᠠᠮᠪᡳ") == "aqambi"
        assert manchu_to_abkai("ᡤᡡᠨᡳᠮᠪᡳ") == "gvnimbi"

    def test_variation_selectors_are_dropped(self):
        assert manchu_to_abkai("ᡩ᠋ᡝ") == "de"

    def test_loanword_letters(self):
        assert manchu_to_abkai("ᡮᠠᡳ") == "cai"
        assert manchu_to_abkai("ᠺᠠᠮᡩᡠᡵᡳ") == "k\u2018amduri"
        assert manchu_to_abkai("ᠪᠠᡳᡰᡳᠨ") == "bair\u02bbin"

    def test_apical_vowel(self):
        assert manchu_to_abkai("ᠰᡟ") == "sy\u02bb"
        assert manchu_to_abkai("ᡮᡟ") == "cy\u02bb"
        assert manchu_to_abkai("ᡱᡳ") == "qy\u02bb"
        assert manchu_to_abkai("ᡷᡳ") == "jy\u02bb"
        assert manchu_to_abkai("ᡯᡳ\u180c") == "zy\u02bb"
        # Without FVS2, ᡯᡳ is an ordinary zi
        assert manchu_to_abkai("ᡯᡳ") == "zi"

    def test_typed_marks(self):
        for mark in "'\u2019\u2018\u02bb":
            assert abkai_to_manchu(f"k{mark}amduri") == "ᠺᠠᠮᡩᡠᡵᡳ"
            assert abkai_to_manchu(f"sy{mark}") == "ᠰᡟ"
        assert abkai_to_manchu("qy'") == "ᡱᡳ"
        assert abkai_to_manchu("aqambi") == "ᠠᠴᠠᠮᠪᡳ"

    def test_round_trips_the_lexicon(self):
        for word in lexicon:
            abkai = manchu_to_abkai(word)
            assert manchu_to_abkai(abkai_to_manchu(abkai)) == abkai
