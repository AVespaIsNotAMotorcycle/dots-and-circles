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

def test_group_columns():
    from ocr.page_reader import group_columns
    boxes = [
        [100, 200, 140, 300],  # right column, lower
        [10, 50, 50, 150],     # left column, upper
        [102, 20, 138, 120],   # right column, upper
        [12, 180, 48, 260],    # left column, lower
        [55, 10, 65, 20],      # a narrow mark between them: its own column
    ]
    assert group_columns(boxes) == [(2, 1), (0, 0), (2, 0), (0, 1), (1, 0)]
    assert group_columns([]) == []

def test_read_boxes_keeps_boxes_and_given_kinds(monkeypatch):
    import ocr.page_reader as reader
    from PIL import Image

    class Page:
        ink = np.zeros((200, 100), dtype=bool)
        image = Image.new("L", (100, 200), 255)
        unit = 40.0
        words = []

    calls = {}
    def find_words(image, deskew=True):
        calls["deskew"] = deskew
        return Page()
    monkeypatch.setattr(reader, "find_words", find_words)
    # The classifier calls every box other text
    monkeypatch.setattr(reader, "classify", lambda model, ink, boxes, unit:
                        np.zeros(len(boxes)))
    # The model reads two words, unsure of them
    monkeypatch.setattr(reader, "read_crops", lambda model, crops, crop=True,
                        whiten=True: [("ᠠᠪᡴᠠ ᠨᠠ", 0.3, [0.5])
                                      for _ in crops])
    page = reader.read_boxes(None, [
        {"box": [50, 10, 90, 190], "kind": "manchu"},
        {"box": [5, 100, 45, 150], "kind": None},
        {"box": [5, 10, 45, 90], "kind": "manchu"},
    ], None, None)
    assert calls["deskew"] is False
    # In reading order, each box as given, not split at the spaces
    assert [word["box"] for word in page.words] == [
        [5, 10, 45, 90], [5, 100, 45, 150], [50, 10, 90, 190]]
    assert [word["kind"] for word in page.words] == [
        "manchu", "other", "manchu"]
    assert page.words[0]["text"] == "ᠠᠪᡴᠠ ᠨᠠ"
    # Marked Manchu by hand, so in the text despite low confidence
    assert low_confidence(page.words[0])
    assert page_text(page) == "ᠠᠪᡴᠠ ᠨᠠ □\nᠠᠪᡴᠠ ᠨᠠ"

def test_read_boxes_reuses_readings(monkeypatch):
    import ocr.page_reader as reader
    from PIL import Image

    class Page:
        ink = np.zeros((200, 100), dtype=bool)
        image = Image.new("L", (100, 200), 255)
        unit = 40.0
        words = []

    monkeypatch.setattr(reader, "find_words", lambda image, deskew=True:
                        Page())
    monkeypatch.setattr(reader, "classify", lambda model, ink, boxes, unit:
                        np.ones(len(boxes)))
    read = []
    def read_crops(model, crops, crop=True, whiten=True):
        read.extend(crops)
        return [("ᠨᠠ", 0.9, []) for _ in crops]
    monkeypatch.setattr(reader, "read_crops", read_crops)
    page = reader.read_boxes(None, [
        # Sent back unchanged, with a low confidence: left out of the text
        {"box": [5, 10, 45, 90], "kind": "manchu",
         "reading": ("ᡴᡴ", 0.3)},
        {"box": [5, 100, 45, 150], "kind": "manchu", "reading": ("ᠪᡝ", 0.8)},
        # New
        {"box": [5, 160, 45, 190], "kind": None},
    ], None, None)
    assert len(read) == 1
    assert [word["text"] for word in page.words] == ["ᡴᡴ", "ᠪᡝ", "ᠨᠠ"]
    assert page.words[1]["abkai"] == "be"
    assert page_text(page) == "ᠪᡝ ᠨᠠ"
