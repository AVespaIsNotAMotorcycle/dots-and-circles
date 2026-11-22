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


class TestHandwriting:
    '''Bending rendered words and augmenting real crops (train(handwriting=True)).'''

    def word_image(self):
        # A rendered word at WORK_WIDTH, as photograph bends it
        from ocr.training_data import fonts, render
        from ocr.training_data.augment import WORK_WIDTH, scale_to_width
        return scale_to_width(render(lexicon[1234], fonts[0]), WORK_WIDTH)

    def photo(self):
        # A grey "photo": dark strokes weaving down a light page
        pixels = np.full((120, 40), 200, dtype=np.uint8)
        for row in range(10, 110):
            column = 18 + round(4 * np.sin(row / 12))
            pixels[row, column:column + 4] = 30
        pixels[30:34, 8:32] = 30
        return Image.fromarray(pixels)

    def test_bend_keeps_all_the_ink(self):
        from ocr.training_data.augment import bend
        image = self.word_image()
        ink = (255 - np.asarray(image, dtype=np.float64)).sum()
        for seed in range(5):
            bent = bend(image, np.random.default_rng(seed))
            assert bent.height == image.height
            assert bent.width > image.width
            bent_ink = (255 - np.asarray(bent, dtype=np.float64)).sum()
            assert abs(bent_ink - ink) / ink < 0.02

    def test_bend_moves_rows_sideways(self):
        from ocr.training_data.augment import bend
        image = self.word_image()
        bent = bend(image, np.random.default_rng(3))
        # Each row's ink centre, relative to the image's middle
        def centres(picture):
            ink = 255 - np.asarray(picture, dtype=np.float64)
            rows = ink.sum(axis=1) > 0
            return ((ink[rows] * np.arange(picture.width)).sum(axis=1)
                    / ink[rows].sum(axis=1) - picture.width / 2)
        assert np.abs(centres(bent) - centres(image)).max() > 1

    def test_words_without_bending_are_as_before(self):
        # bend=False draws no extra random numbers, so earlier runs'
        # images are unchanged
        for seed in (5, 1234, 9999):
            plain, _ = generate(seed, epoch=2)
            unbent, _ = generate(seed, epoch=2, bend=False)
            assert plain.tobytes() == unbent.tobytes()

    def test_bent_words_are_model_inputs(self):
        for seed in range(0, 21879, 499):
            image, word = generate(seed, epoch=1, bend=True)
            assert image.mode == "L" and image.width == WIDTH
            assert -(-min(image.height, HEIGHT) // 4) >= tokens_needed(word)

    def test_some_words_are_bent(self):
        differ = sum(generate(seed, 1)[0].tobytes()
                     != generate(seed, 1, bend=True)[0].tobytes()
                     for seed in range(0, 21879, 997))
        assert differ > 5

    def test_augmented_crop_is_repeatable_and_varies(self):
        from ocr.training_data.augment import augment_crop
        photo, word = self.photo(), "ᠠᠪᡴᠠ"
        first = augment_crop(photo, word, [7, 0, 3])
        assert first.tobytes() == augment_crop(photo, word,
                                               [7, 0, 3]).tobytes()
        others = [augment_crop(photo, word, [7, copy, epoch]).tobytes()
                  for copy, epoch in [(1, 3), (0, 4)]]
        assert all(other != first.tobytes() for other in others)

    def test_augmented_crops_are_model_inputs(self):
        from ocr.training_data.augment import augment_crop
        photo, word = self.photo(), "ᠠᠪᡴᠠ"
        for seed in range(20):
            image = augment_crop(photo, word, [seed, 0, 0])
            assert image.mode == "L" and image.width == WIDTH
            assert image.height >= tokens_needed(word) * 4

    def test_crops_too_short_for_ctc_fall_back_to_the_prepared_crop(self):
        from ocr.training_data.augment import augment_crop
        photo = self.photo()
        word = "ᠠ" * 40
        assert (augment_crop(photo, word, [1, 0, 0]).tobytes()
                == prepare(photo).tobytes())
