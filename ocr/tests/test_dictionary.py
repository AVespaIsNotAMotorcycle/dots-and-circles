import random

import pytest

pytest.importorskip("matplotlib")  # ocr.train imports it

from ocr.dictionary import Dictionary
from ocr.photos import manchu_to_roman
from ocr.train import edit_distance

ENTRIES = [
    ["ᠠᠪᡴᠠ", ["kaikki", "norman"]],
    ["ᠠᠪᠰᠠ", ["norman"]],
    ["ᠨᠠ", ["kaikki"]],
    ["ᠪᠠ ᠨᠠ", ["norman"]],
    ["ᠠᠪᡴᠠᡳ", ["kaikki"]],
]


def test_closest_finds_the_word_and_its_neighbour():
    found = Dictionary(ENTRIES).closest("ᠠᠪᡴᠠᡳ")
    assert [(match["abkai"], match["distance"]) for match in found] == [
        ("abkai", 0), ("abka", 1)]
    found = Dictionary(ENTRIES).closest("ᠠᠪᡴᠠ")
    assert found[0]["manchu"] == "ᠠᠪᡴᠠ"
    assert found[0]["sources"] == ["kaikki", "norman"]

def test_closest_ties_go_to_the_nearer_length():
    # absa and abkai are both one edit from abka; absa is its length
    found = Dictionary(ENTRIES[1:]).closest("ᠠᠪᡴᠠ")
    assert [match["abkai"] for match in found] == ["absa", "abkai"]

def test_closest_ignores_punctuation_and_matches_phrases():
    dictionary = Dictionary(ENTRIES)
    assert dictionary.closest("ᠠᠪᡴᠠ᠈")[0]["distance"] == 0
    assert dictionary.closest("ᠪᠠ  ᠨᠠ᠉")[0]["abkai"] == "ba na"
    assert dictionary.closest("᠉") == [] and dictionary.closest("") == []

def test_closest_with_letters_in_no_entry():
    # ᠺ is k' in the romanization, and no entry has '
    found = Dictionary(ENTRIES).closest("ᠺᠠ")
    assert (found[0]["abkai"], found[0]["distance"]) == ("na", 2)

def test_closest_agrees_with_brute_force():
    random.seed(0)
    letters = "ᠠᡝᡳᠣᡠᠨᡴᡤᠪᠰᡨᡩᠯᠮᠴᠵᠶᡵᡶ"
    words = sorted({"".join(random.choice(letters)
                            for _ in range(random.randint(1, 8)))
                    for _ in range(300)})
    dictionary = Dictionary([[word, ["kaikki"]] for word in words])
    for _ in range(50):
        query = "".join(random.choice(letters)
                        for _ in range(random.randint(1, 10)))
        expected = sorted(edit_distance(manchu_to_roman(query),
                                        manchu_to_roman(word))
                          for word in words)[:3]
        assert [match["distance"] for match in
                dictionary.closest(query, count=3)] == expected

def test_the_committed_dictionary_loads():
    found = Dictionary.load().closest("ᠠᠪᡴᠠ")
    assert found[0]["abkai"] == "abka" and found[0]["distance"] == 0
    assert set(found[0]["sources"]) <= {"kaikki", "norman"}
