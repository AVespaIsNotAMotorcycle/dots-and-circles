import numpy as np
import pytest

from ocr.photos import manchu_to_abkai
from ocr.training_data import fonts
from ocr.training_data.pages import (INK_THRESHOLD, OTHER, chinese_fonts,
                                     generate_page, page_lines, unseen_words)

pytestmark = pytest.mark.skipif(
    not fonts or not chinese_fonts,
    reason="needs the Manchu and Chinese fonts in training_data/fonts/")


def test_repeatable():
    first, items = generate_page(7, photo=True)
    second, same_items = generate_page(7, photo=True)
    assert items == same_items
    assert first.tobytes() == second.tobytes()

@pytest.mark.parametrize("photo", [False, True])
def test_boxes_are_on_the_page_and_hold_ink(photo):
    for seed in range(5):
        page, items = generate_page(seed, photo=photo)
        pixels = np.asarray(page)
        assert page.mode == "L"
        for item in items:
            left, top, right, bottom = item["box"]
            assert 0 <= left < right <= page.width
            assert 0 <= top < bottom <= page.height
            if not photo:
                assert (pixels[top:bottom, left:right] < INK_THRESHOLD).any()

def test_all_ink_is_boxed_without_rules():
    page, items = generate_page(3, rules=False)
    ink = np.asarray(page) < INK_THRESHOLD
    for item in items:
        left, top, right, bottom = item["box"]
        ink[top:bottom, left:right] = False
    assert not ink.any()

def test_reading_order():
    page, items = generate_page(11, layout="mixed", rules=False)
    assert [(item["column"], item["position"]) for item in items] == sorted(
        (item["column"], item["position"]) for item in items)
    columns = {}
    for item in items:
        columns.setdefault(item["column"], []).append(item["box"])
    for boxes in columns.values():
        # Top to bottom, without overlapping
        for above, below in zip(boxes, boxes[1:]):
            assert above[3] <= below[1]
    # Left to right, without overlapping
    ordered = [columns[column] for column in sorted(columns)]
    for left, right in zip(ordered, ordered[1:]):
        assert max(box[2] for box in left) <= min(box[0] for box in right)

def test_layouts():
    _, items = generate_page(1, layout="manchu")
    assert {item["kind"] for item in items} == {"manchu"}

    _, items = generate_page(1, layout="alternating")
    kinds = {}
    for item in items:
        kinds.setdefault(item["column"], set()).add(item["kind"])
    assert set(map(frozenset, kinds.values())) == {
        frozenset({"manchu"}), frozenset({"other"})}

def test_words_are_single_unseen_words():
    words = set(unseen_words())
    _, items = generate_page(2, layout="mixed")
    for item in items:
        if item["kind"] == "manchu":
            assert item["text"] in words
            assert " " not in item["text"]

def test_page_lines():
    items = [
        {"kind": "other", "text": "天", "column": 0, "position": 0},
        {"kind": "manchu", "text": "ᠠᠪᡴᠠ", "column": 1, "position": 0},
        {"kind": "other", "text": "天", "column": 1, "position": 1},
        {"kind": "other", "text": "地", "column": 1, "position": 2},
        {"kind": "manchu", "text": "ᠨᠠ", "column": 1, "position": 3},
        {"kind": "manchu", "text": "ᠪᠠ", "column": 2, "position": 0},
    ]
    # The all-Chinese column is left out, and each character is a □
    assert page_lines(items) == f"ᠠᠪᡴᠠ {OTHER} {OTHER} ᠨᠠ\nᠪᠠ"
    assert page_lines(items, abkai=True) == (
        f"{manchu_to_abkai('ᠠᠪᡴᠠ')} {OTHER} {OTHER} "
        f"{manchu_to_abkai('ᠨᠠ')}\n{manchu_to_abkai('ᠪᠠ')}")
