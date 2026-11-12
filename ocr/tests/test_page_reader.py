import numpy as np
import pytest

pytest.importorskip("matplotlib")  # ocr.train imports it

from ocr.page_reader import (MIN_CONFIDENCE, decode_words, low_confidence,
                              page_text, split_reading)
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

def test_low_confidence_words_are_left_out_of_the_text():
    class Page:
        words = [
            {"kind": "manchu", "text": "ᠠᠪᡴᠠ", "column": 0, "position": 0,
             "confidence": 0.95},
            {"kind": "manchu", "text": "ᡴᡴᡴ", "column": 0, "position": 1,
             "confidence": MIN_CONFIDENCE - 0.1},
            # A column of nothing but junk leaves no line
            {"kind": "manchu", "text": "ᡴᡴ", "column": 1, "position": 0,
             "confidence": 0.3},
            {"kind": "other", "text": "", "column": 1, "position": 1,
             "manchu": 0.0},
        ]
    assert low_confidence(Page.words[1]) and not low_confidence(Page.words[0])
    assert not low_confidence(Page.words[3])
    assert page_text(Page()) == "ᠠᠪᡴᠠ"

def test_split_at_stem_breaks(monkeypatch):
    import ocr.page_reader as reader
    from PIL import Image
    ink = np.zeros((200, 60), dtype=bool)
    ink[0:95, 30:34] = True
    ink[98:200, 30:34] = True

    class Page:
        unit = 40.0
        image = Image.new("L", (60, 200), 255)
    Page.ink = ink

    def read_crops(model, crops, crop=True, whiten=True):
        # Parts read as words, confidently
        return [("ᠠᠪᡴᠠ", 0.95, []) for _ in crops]
    monkeypatch.setattr(reader, "read_crops", read_crops)
    # Split when the parts read more confidently than the whole...
    parts = reader.split_at_stem_breaks(None, Page(), [0, 0, 60, 200],
                                        ("ᠠᠪᡴᠠᠠᠪᡴᠠ", 0.8))
    assert [box for box, _ in parts] == [[30, 0, 34, 95], [30, 98, 34, 200]]
    # ...not when they read less so
    parts = reader.split_at_stem_breaks(None, Page(), [0, 0, 60, 200],
                                        ("ᠠᠪᡴᠠᠠᠪᡴᠠ", 0.99))
    assert parts == [([0, 0, 60, 200], ("ᠠᠪᡴᠠᠠᠪᡴᠠ", 0.99))]

def test_read_page_can_leave_out_stem_breaks(monkeypatch):
    import ocr.page_reader as reader
    from PIL import Image

    class Page:
        ink = np.ones((100, 20), dtype=bool)
        image = Image.new("L", (20, 100), 255)
        words = [{"kind": "manchu", "box": [0, 0, 20, 100], "column": 0}]

    monkeypatch.setattr(reader, "find_words", lambda image: Page())
    monkeypatch.setattr(reader, "label_words", lambda page, classifier: page)
    monkeypatch.setattr(reader, "read_crops", lambda model, crops, crop=True,
                        whiten=True: [("ᠪᡳ", 0.9, []) for _ in crops])
    # A splitter that always cuts a lone letter off, as Run 31's did
    def split_at_stem_breaks(model, page, box, reading, crop=True,
                             whiten=True):
        return [([0, 0, 20, 50], ("ᠪ", 1.0)), ([0, 50, 20, 100], ("ᡳ", 1.0))]
    monkeypatch.setattr(reader, "split_at_stem_breaks", split_at_stem_breaks)
    texts = lambda page: [word["text"] for word in page.words]
    assert texts(reader.read_page(None, None, None)) == ["ᠪ", "ᡳ"]
    assert texts(reader.read_page(None, None, None, stem_breaks=False)) == [
        "ᠪᡳ"]
