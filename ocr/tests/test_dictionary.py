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

def test_closest_ignores_punctuation_ending_words_and_matches_phrases():
    dictionary = Dictionary(ENTRIES)
    assert dictionary.closest("ᠠᠪᡴᠠ᠈")[0]["distance"] == 0
    assert dictionary.closest("ᠠᠪᡴᠠ᠈᠉")[0]["distance"] == 0
    assert dictionary.closest("ᠪᠠ᠈  ᠨᠠ᠉")[0]["abkai"] == "ba na"
    assert dictionary.closest("᠉") == [] and dictionary.closest("") == []

def test_closest_keeps_punctuation_inside_words():
    dictionary = Dictionary(ENTRIES)
    # ab.ka is one edit from abka (deleting the .), not the two words ab ka
    found = dictionary.closest("ᠠᠪ᠉ᡴᠠ")
    assert (found[0]["abkai"], found[0]["distance"]) == ("abka", 1)
    # So is punctuation starting a word
    assert dictionary.closest("᠈ᠠᠪᡴᠠ")[0]["distance"] == 1

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

VERBS = [
    ["ᠠᡵᠠᠮᠪᡳ", ["kaikki", "norman"]],
    ["ᠠᡵᠠᠪᡠᠮᠪᡳ", ["norman"]],
    ["ᠠᡩᠠᡥᠠ", ["norman"]],
]


def test_closest_reads_verb_forms():
    found = Dictionary(VERBS).closest("ᠠᡵᠠᡥᠠ")
    assert found[0] == {"manchu": "ᠠᡵᠠᡥᠠ", "abkai": "araha",
                        "roman": "araha", "distance": 0,
                        "sources": ["kaikki", "norman"],
                        "lemma": "ᠠᡵᠠᠮᠪᡳ", "affixes": ["-ha"],
                        "breaks": []}
    # Entries have no lemma
    assert found[1]["abkai"] == "adaha" and found[1]["lemma"] is None
    assert found[1]["breaks"] is None

def test_closest_flags_and_prefers_vowel_harmony():
    dictionary = Dictionary(VERBS + [["ᡤᡝᠨᡝᠮᠪᡳ", ["norman"]]])
    # araho is still read as itself, but breaks harmony
    found = dictionary.closest("ᠠᡵᠠᡥᠣ")[0]
    assert (found["roman"], found["distance"], found["breaks"]) == (
        "araho", 0, [{"affix": "-ho", "expected": ["-ha"],
                      "reason": "harmony"}])
    # genehu is one edit from genehe, geneha and geneho: genehe keeps it
    found = dictionary.closest("ᡤᡝᠨᡝᡥᡠ")[0]
    assert (found["roman"], found["breaks"]) == ("genehe", [])

def test_closest_reads_infixes_and_norman_suffixes():
    dictionary = Dictionary(VERBS)
    # Up to two infixes
    found = dictionary.closest("ᠠᡵᠠᠨᠵᡳᠪᡠᡥᠠ")[0]
    assert (found["lemma"], found["affixes"], found["distance"]) == (
        "ᠠᡵᠠᠮᠪᡳ", ["-nji-", "-bu-", "-ha"], 0)
    # -rahū is rahv in the dataset's romanization
    found = dictionary.closest("ᠠᡵᠠᡵᠠᡥᡡ")[0]
    assert (found["roman"], found["affixes"]) == ("ararahv", ["-rahū"])

def test_closest_verb_forms_prefer_entries_and_fewer_infixes():
    dictionary = Dictionary(VERBS)
    # arabumbi is an entry: arabuha is its -ha, not arambi's -bu-ha
    found = dictionary.closest("ᠠᡵᠠᠪᡠᡥᠠ")[0]
    assert (found["lemma"], found["affixes"]) == ("ᠠᡵᠠᠪᡠᠮᠪᡳ", ["-ha"])
    # And arabumbi is the entry, not arambi with -bu- and -mbi
    found = dictionary.closest("ᠠᡵᠠᠪᡠᠮᠪᡳ")
    assert found[0]["lemma"] is None and found[0]["distance"] == 0
    assert all(match["roman"] != "arabumbi" for match in found[1:])
    # Ties go to entries
    # (arahu is one edit from arahi, and from araha, arahe and araho)
    found = Dictionary(VERBS[:1] + [["ᠠᡵᠠᡥᡳ", ["norman"]]]).closest("ᠠᡵᠠᡥᡠ")
    assert [(match["roman"], match["distance"]) for match in found] == [
        ("arahi", 1), ("araha", 1)]
    assert found[1]["lemma"] == "ᠠᡵᠠᠮᠪᡳ"

def test_closest_uses_norman_notes():
    dictionary = Dictionary([
        ["ᠠᡵᠠᠮᠪᡳ", ["norman"]],
        ["ᡳᠯᡳᠮᠪᡳ", ["norman"], {"perfective": ["-ha"], "forms": [
            ["ᡳᠯᡳᡥᠠ", ["-ha"]]]}],
        ["ᡳᠯᡳᠪᡠᠮᠪᡳ", ["norman"]],
        ["ᠵᡝᠮᠪᡳ", ["norman"], {"perfective": ["-ngke"], "forms": [
            ["ᠵᡝᠩᡴᡝ", ["-ngke"]], ["ᠵᡝᡶᡠ", []]]}],
    ])
    # Irregular forms are read as themselves, with their affixes
    found = dictionary.closest("ᠵᡝᠩᡴᡝ")[0]
    assert (found["roman"], found["distance"], found["lemma"],
            found["affixes"], found["breaks"]) == (
        "jengke", 0, "ᠵᡝᠮᠪᡳ", ["-ngke"], [])
    found = dictionary.closest("ᠵᡝᡶᡠ")[0]
    assert (found["lemma"], found["affixes"]) == ("ᠵᡝᠮᠪᡳ", [])
    # A regular jeke is checked against the note
    assert dictionary.closest("ᠵᡝᡴᡝ")[0]["breaks"] == [
        {"affix": "-ke", "expected": ["-ngke"], "reason": "norman"}]
    # iliha, against Porter's e for ili-, as Norman notes it; ilibuha
    # takes ilimbi's a
    assert dictionary.closest("ᡳᠯᡳᡥᠠ")[0]["breaks"] == []
    found = dictionary.closest("ᡳᠯᡳᠪᡠᡥᠠ")[0]
    assert (found["lemma"], found["breaks"]) == ("ᡳᠯᡳᠪᡠᠮᠪᡳ", [])
    # araka: arambi has no note giving -ka
    assert dictionary.closest("ᠠᡵᠠᡴᠠ")[0]["breaks"] == [
        {"affix": "-ka", "expected": ["-ha"], "reason": "unmarked"}]

def test_closest_verb_forms_agree_with_brute_force():
    # Endings of x, z and y and stems without them, so no form is spelled
    # like an entry or like another verb's form
    random.seed(1)
    letters = "ᠠᡝᡳᠣᡠᠨᡴᡤᠪᠰᡨᡩᠯᠮ"
    endings = [("x", ["-x"]), ("zx", ["-z-", "-x"]), ("yzx", ["-y-", "-zx"]),
               ("xxzz", ["-xxzz"]), ("zy", ["-zy"])]
    words = sorted({"".join(random.choice(letters)
                            for _ in range(random.randint(1, 7)))
                    for _ in range(200)})
    verbs = [word + "ᠮᠪᡳ" for word in words[::3]]
    dictionary = Dictionary([[word, ["norman"]] for word in words + verbs],
                            endings=endings)
    romans = [manchu_to_roman(word) for word in words + verbs]
    stems = [manchu_to_roman(verb)[:-3] for verb in verbs]
    to_manchu = {"x": "ᡧ", "z": "ᡷ", "y": "ᠶ"}
    for _ in range(80):
        query = "".join(random.choice(letters + "ᡧᡷᠶ")
                        for _ in range(random.randint(1, 10)))
        roman = manchu_to_roman(query)
        expected = sorted(
            [edit_distance(roman, word) for word in romans]
            + [min(edit_distance(roman, stem + ending)
                   for ending, _ in endings) for stem in stems])[:3]
        found = dictionary.closest(query, count=3)
        assert [match["distance"] for match in found] == expected
        for match in found:
            assert edit_distance(roman, match["roman"]) == match["distance"]
            if match["lemma"] is not None:
                assert match["manchu"] == "".join(
                    to_manchu.get(letter, letter) for letter in
                    match["lemma"][:-3] + "".join(
                        affix.strip("-") for affix in match["affixes"]))

def test_the_committed_dictionary_loads():
    found = Dictionary.load().closest("ᠠᠪᡴᠠ")
    assert found[0]["abkai"] == "abka" and found[0]["distance"] == 0
    assert set(found[0]["sources"]) <= {"kaikki", "norman"}
    found = Dictionary.load().closest("ᡤᡝᠨᡝᡶᡳ")[0]
    assert found["breaks"] == []
    assert (found["abkai"], found["distance"], found["affixes"]) == (
        "genefi", 0, ["-fi"])
    assert found["lemma"] == "ᡤᡝᠨᡝᠮᠪᡳ"
