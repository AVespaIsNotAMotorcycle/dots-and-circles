import numpy as np
import pytest
from PIL import Image, ImageDraw

from ocr.layout import (close_gaps, despeckle, find_skew, find_words, otsu,
                        remove_rules, unrotate_point)
from ocr.training_data import fonts
from ocr.training_data.pages import chinese_fonts, generate_page


def test_otsu_splits_two_levels():
    levels = np.array([30] * 50 + [220] * 200)
    threshold = otsu(levels)
    assert 30 < threshold <= 220

def test_close_gaps():
    mask = np.array([[1, 0, 0, 1, 0, 0, 0, 0, 1, 0]], dtype=bool)
    closed = close_gaps(mask, 2)
    # The 2-pixel gap is bridged, the 4-pixel one and the open end aren't
    assert closed.astype(int).tolist() == [[1, 1, 1, 1, 0, 0, 0, 0, 1, 0]]

def test_despeckle_keeps_strokes():
    ink = np.zeros((20, 20), dtype=bool)
    ink[2:18, 5:7] = True       # a stroke two pixels wide
    ink[10, 15] = True          # a speck
    cleaned = despeckle(ink)
    assert cleaned[2:18, 5:7].all()
    assert not cleaned[10, 15]

def test_unrotate_point_inverts_pil_rotate():
    image = Image.new("L", (300, 200), 0)
    image.putpixel((250, 40), 255)
    for angle in (3.0, -4.5):
        rotated = image.rotate(angle, resample=Image.NEAREST, expand=True)
        ys, xs = np.nonzero(np.asarray(rotated))
        x, y = unrotate_point(xs.mean() + 0.5, ys.mean() + 0.5, angle,
                              image.size, rotated.size)
        assert abs(x - 250.5) < 1.5 and abs(y - 40.5) < 1.5

def comb(height=200, left=50):
    # A Manchu-like stem with teeth on its left, as ink
    ink = np.zeros((300, 300), dtype=bool)
    ink[50:50 + height, left:left + 3] = True
    for y in range(55, 50 + height, 12):
        ink[y:y + 3, left - 10:left] = True
    return ink

def test_remove_rules_keeps_stems():
    ink = comb()
    assert (remove_rules(ink) == ink).all()

def test_remove_rules_removes_lines():
    ink = comb()
    ink[20:23, 10:290] = True      # a horizontal rule
    ink[20:290, 150:152] = True    # a vertical one
    cleaned = remove_rules(ink)
    assert not cleaned[20:23, 100:140].any()
    assert not cleaned[100:280, 150:152].any()
    assert cleaned[50:250, 50:53].all()

def test_find_skew():
    # Columns of combs
    ink = np.zeros((600, 600), dtype=bool)
    for left in range(60, 560, 70):
        ink[50:550, left + 20:left + 23] = True
        for y in range(55, 550, 12):
            ink[y:y + 3, left + 10:left + 20] = True
    image = Image.fromarray(ink.astype(np.uint8) * 255)
    for angle in (0.0, 2.0, -3.0):
        rotated = np.asarray(image.rotate(angle, expand=True)) > 127
        assert abs(find_skew(rotated) + angle) <= 0.15

@pytest.mark.skipif(not fonts or not chinese_fonts,
                    reason="needs the fonts in training_data/fonts/")
@pytest.mark.parametrize("photo", [False, True])
def test_finds_the_words_of_a_manchu_page(photo):
    from ocr.evaluate_pages import match
    image, items = generate_page(0, layout="manchu", photo=photo)
    page = find_words(image)
    found = [page.to_input(word["box"]) for word in page.words]
    pairing = match(found, [item["box"] for item in items])
    assert len(pairing) >= 0.95 * len(items)
    assert len(found) <= 1.05 * len(items)
    # Reading order: columns left to right, words top to bottom
    order = [(word["column"], word["position"]) for word in page.words]
    assert order == sorted(order)

def test_blank_page():
    page = find_words(Image.new("L", (200, 300), 230))
    assert page.words == []

def test_split_characters():
    from ocr.box_classifier import split_characters
    ink = np.zeros((400, 100), dtype=bool)
    # Three 40-pixel characters: the first in two parts (like 二), the
    # second touching the third
    ink[10:20, 30:70] = True
    ink[35:50, 30:70] = True
    ink[60:100, 30:70] = True
    ink[100:140, 32:68] = True
    characters = split_characters(ink, [30, 10, 70, 140])
    assert len(characters) == 3
    assert characters[0][1] == 10 and characters[0][3] == 50
    assert abs(characters[1][3] - 100) <= 3
    assert characters[2][3] == 140
