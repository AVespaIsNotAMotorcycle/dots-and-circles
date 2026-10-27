import random

import numpy as np
from PIL import Image

from ocr.photos import manchu_to_abkai
from ocr.preprocess import crop_to_ink, prepare, stretch
from ocr.training_data import HEIGHT, WIDTH, generate, pad_top
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


class TestManchuToAbkai:
    def test_letters(self):
        assert manchu_to_abkai("ᠠᠪᡴᠠ") == "abka"
        assert manchu_to_abkai("ᡧᡠᠸᡝ") == "xuwe"

    def test_variation_selectors_are_dropped(self):
        assert manchu_to_abkai("ᡩ᠋ᡝ") == "de"

    def test_loanword_letters(self):
        assert manchu_to_abkai("ᡮᠠᡳ") == "ts'ai"
        assert manchu_to_abkai("ᡯᡳ") == "dzi"
