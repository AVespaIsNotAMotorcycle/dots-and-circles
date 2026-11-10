import numpy as np
import pytest

pytest.importorskip("matplotlib")  # ocr.train imports it

from ocr.page_reader import decode_words, split_reading
from ocr.train import alphabet


def outputs(classes, blank):
    # One-hot model outputs for a list of classes
    y = np.zeros((len(classes), blank + 1))
    y[np.arange(len(classes)), classes] = 1
    return y

def test_decode_words_finds_spaces():
    blank = len(alphabet)
    a, b, space = alphabet.index("ᠠ"), alphabet.index("ᠪ"), alphabet.index(" ")
    classes = [a, a, blank, b, space, space, space, blank, a, blank]
    text, spaces = decode_words(outputs(classes, blank), len(classes), blank)
    assert text == "ᠠᠪ ᠠ"
    assert spaces == [5.0]

def test_split_reading_at_spaces():
    ink = np.zeros((100, 20), dtype=bool)
    ink[0:40, 5:15] = True
    ink[60:100, 6:14] = True
    parts = split_reading(ink, [0, 0, 20, 100], "ᠠᠪ ᠠ", [0.5])
    assert [word for _, word in parts] == ["ᠠᠪ", "ᠠ"]
    assert parts[0][0] == [5, 0, 15, 40]
    assert parts[1][0] == [6, 60, 14, 100]

def test_split_reading_without_spaces():
    ink = np.ones((50, 20), dtype=bool)
    assert split_reading(ink, [0, 0, 20, 50], "ᠠᠪ", []) == [
        ([0, 0, 20, 50], "ᠠᠪ")]
